import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn
import lightning as L
import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import ray.train.lightning
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import pickle
from scipy.optimize import root

logger = TensorBoardLogger("tb_logs", name="my_model")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Defining the neural network
class BasicLightning(pl.LightningModule):
    """
    This Neural Network takes the input of density and temperature and predicts the helmholtz free energy.
    """
    def __init__(self):
        super(BasicLightning,self).__init__() 
        self.lr = 1e-6
        self.batch_size = 6000
        self.layer_size = 45

        # Creating a sequential stack of Linear layers all of the same width with Tanh activation function 
        self.layers_stack = nn.Sequential(
          nn.Linear(2,self.layer_size),
          nn.Tanh(),
          nn.Linear(self.layer_size,self.layer_size),
          nn.Tanh(),
          nn.Linear(self.layer_size,self.layer_size),
          nn.Tanh(),
          nn.Linear(self.layer_size,self.layer_size),
          nn.Tanh(),
          nn.Linear(self.layer_size,self.layer_size),
          nn.Tanh(),
          nn.Linear(self.layer_size,1),
        )

    def forward(self,x):
        '''
        Passes the input x through the neural network and returns the output
        '''
        out = self.layers_stack(x)
        return out
    
    def configure_optimizers(self):
        '''
        Configures optimiser
        '''
        optimiser = torch.optim.AdamW(self.parameters(),lr = self.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser,500)


        # output_dict = {
        # "optimizer": optimiser,
        # "lr_scheduler": {"scheduler":scheduler}
        # }
        return optimiser

    
    def training_step(self,train_batch,batch_index):
        '''
        Performs a training step.

        Unpacks the batch, passes the batch through the NN, calculates the loss, and logs the loss.
        '''

        # Unpack validation batch
        train_input_i, train_target_i = train_batch
        rho = train_input_i[:,0]
        T = train_input_i[:,1]

        cv_target = train_target_i[:,0]
        gammaV_target = train_target_i[:,1]
        cp_target = train_target_i[:,2]
        alphaP_target = train_target_i[:,3]
        betaT_target = train_target_i[:,4]
        U_target = train_target_i[:,5]
        P_target = train_target_i[:,6]
        mu_jt_target = train_target_i[:,7]
        Z_target = train_target_i[:,8]

        # cp_target = cv_target + (T/rho)*((alphaP_target**2)/betaT_target)
        # mu_jt_target = (1/(rho*cp_target))*((T*alphaP_target)-1)
        adiabatic_index_target = cp_target/cv_target

        var_cv = self.calculate_variance(cv_target)
        var_Z = self.calculate_variance(Z_target)
        var_U = self.calculate_variance(U_target)
        var_gammaV = self.calculate_variance(gammaV_target)
        var_rho_betaT = self.calculate_variance(rho*betaT_target)
        var_alphaP = self.calculate_variance(alphaP_target)
        var_adiabatic_index = self.calculate_variance(adiabatic_index_target)
        var_mu_jt = self.calculate_variance(mu_jt_target)
        var_P = self.calculate_variance(P_target)

        # Ensures that the DAG is created for the input so that the gradient and hessian can be computed 
        train_input_i.requires_grad = True

        # Pass input through NN to get the output
        zero_densities = torch.zeros_like(rho)
        input_at_zero_densities = torch.stack((zero_densities,T),dim=-1)

        #A is the residual helmholtz free energy
        A = self.forward(train_input_i)-self.forward(input_at_zero_densities)

        # Computes gradient and hessian
        train_gradient = self.compute_gradient(train_input_i,input_at_zero_densities)
        train_hessian = self.compute_hessian(train_input_i)

        dA_drho = train_gradient[:,0]

        dA_dT = train_gradient[:,1]

        d2A_drho2= train_hessian[:, # In all of the hessians in the batch ...
                                 :, # In all of the heassians in the batch ...
                                 0, # in the first row ...
                                 0] # return the value in the first column
        
        d2A_dT2 = train_hessian[:, # In all of the hessians in the batch ...
                                :, # In all of the heassians in the batch ...
                                1, # in the second row ...
                                1] # return the value in the second column
        
        d2A_dT_drho = train_hessian[:, # In all of the hessians in the batch ...
                                :, # In all of the heassians in the batch ...
                                1, # in the second row ...
                                0] # return the value in the first column
        
        T = torch.reshape(T,(-1,))
        rho = torch.reshape(rho,(-1,))
        A = torch.reshape(A,(-1,))
        dA_dT = torch.reshape(dA_dT,(-1,))
        dA_drho = torch.reshape(dA_drho,(-1,))
        d2A_dT2 = torch.reshape(d2A_dT2,(-1,))
        d2A_drho2 = torch.reshape(d2A_drho2,(-1,))
        d2A_dT_drho = torch.reshape(d2A_dT_drho,(-1,))

        S = -dA_dT
        P_predicted = (rho**2)*dA_drho

        U_predicted = A+(T*S)
        U_predicted += (2/2)*T  # Adding ideal gas contribution

        P_by_rho = rho*dA_drho
        P_by_rho += T # Adding ideal gas contribution

        Z_predicted= (P_by_rho)/T

        cv_predicted = -T*d2A_dT2
        cv_predicted += (2/2) # Adding ideal gas contribution

        dP_dT = (rho**2)*d2A_dT_drho
        dP_dT += rho    # Adding ideal gas contribution

        dP_drho = 2*rho*dA_drho + (rho**2)*d2A_drho2
        dP_drho += T    # Adding ideal gas contribution

        alphaP_predicted = (dP_dT)/(rho*dP_drho)


        rho_betaT = torch.reciprocal(dP_drho)
        betaT_predicted = torch.divide(rho_betaT,rho)

        gammaV_predicted = alphaP_predicted/betaT_predicted

        cp_aux1 = T*(alphaP_predicted**2)
        cp_aux2 = rho*betaT_predicted
        cp_predicted = cv_predicted+(cp_aux1/cp_aux2)
        
        mu_jt_predicted = (1/(rho*cp_predicted))*((T*alphaP_predicted)-1)
        
        # Z_predicted = P_predicted/(rho*T)

        adiabatic_index_predicted = cp_predicted/cv_predicted

        # Calculates the loss

        cv_predicted = torch.clamp(cv_predicted,0,torch.inf)

        # Don't perform the loss calculations for any of the properties where the values were deemed non converged from the data collation script
        non_nan_alphaP_index = ~torch.isnan(alphaP_target)
        non_nan_Z_index = ~torch.isnan(Z_target)        
        non_nan_U_index = ~torch.isnan(U_target)
        non_nan_adiabatic_index_index = ~torch.isnan(adiabatic_index_target)
        non_nan_gammaV_index = ~torch.isnan(gammaV_target)
        non_nan_cv_index = ~torch.isnan(cv_target)        
        non_nan_betaT_index = ~torch.isnan(rho*betaT_target)
        non_nan_mu_jt_index = ~torch.isnan(mu_jt_target )

        # If a property was deemed non converged in the collation script it was set to Nan, this indexes all of the properties where they arent Nan's.
        # This means that we can still use the properties that were converged in a given simulation and disregard the non converged properties

        alphaP_target = alphaP_target[non_nan_alphaP_index]
        alphaP_predicted = alphaP_predicted[non_nan_alphaP_index]

        Z_target = Z_target[non_nan_Z_index]
        Z_predicted = Z_predicted[non_nan_Z_index]

        U_target = U_target[non_nan_U_index]
        U_predicted = U_predicted[non_nan_U_index]
        
        adiabatic_index_target = adiabatic_index_target[non_nan_adiabatic_index_index]
        adiabatic_index_predicted = adiabatic_index_predicted[non_nan_adiabatic_index_index]

        gammaV_target = gammaV_target[non_nan_gammaV_index]
        gammaV_predicted = gammaV_predicted[non_nan_gammaV_index]

        cv_target = cv_target[non_nan_cv_index]
        cv_predicted = cv_predicted[non_nan_cv_index]

        betaT_target = betaT_target[non_nan_betaT_index]
        betaT_predicted = betaT_predicted[non_nan_betaT_index]

        mu_jt_target = mu_jt_target[non_nan_mu_jt_index]
        mu_jt_predicted = mu_jt_predicted[non_nan_mu_jt_index]


        Z_loss = ((Z_target-Z_predicted)**2)/var_Z
        U_loss = ((U_target-U_predicted)**2)/var_U
        alphaP_loss = 1/20*((alphaP_target-alphaP_predicted)**2)/var_alphaP
        adiabatic_index_loss = 1/20*((adiabatic_index_target-adiabatic_index_predicted)**2)/var_adiabatic_index
        gammmaV_loss = 1/20*((gammaV_target-gammaV_predicted)**2)/var_gammaV
        cv_loss = 1/20*((cv_target-cv_predicted)**2)/var_cv
        rho_betaT_loss = 1/20*((rho[non_nan_betaT_index]*betaT_target-rho[non_nan_betaT_index]*betaT_predicted)**2)/var_rho_betaT
        mu_jt_loss = 1/20*((mu_jt_target-mu_jt_predicted)**2)/var_mu_jt

        Z_loss = torch.mean(Z_loss)
        U_loss = torch.mean(U_loss)
        alphaP_loss = torch.mean(alphaP_loss)
        adiabatic_index_loss = torch.mean(adiabatic_index_loss)
        gammmaV_loss = torch.mean(gammmaV_loss)
        cv_loss = torch.mean(cv_loss)
        rho_betaT_loss = torch.mean(rho_betaT_loss)
        mu_jt_loss = torch.mean(mu_jt_loss)



        # DDP training strategy requires that the output of the forward pass of the ANN be in the loss function, 
        # this just fudges the output to 0 so that it doesn't error.

        A_loss_required_for_DDP = A*torch.zeros_like(A)
        A_loss_required_for_DDP = torch.mean(A_loss_required_for_DDP)

        

        # loss = Z_loss+U_loss+cv_loss+gammmaV_loss+adiabatic_index_loss+alphaP_loss+A*torch.zeros_like(A)
        loss = Z_loss+U_loss+cv_loss+alphaP_loss+gammmaV_loss+rho_betaT_loss+adiabatic_index_loss+mu_jt_loss+A_loss_required_for_DDP




        self.log("train_P_loss",torch.mean((P_predicted-P_target)**2)) 
        self.log("train_cv_loss",torch.mean(((cv_target-cv_predicted)**2)))
        self.log("train_gammaV_loss",torch.mean((gammaV_target-gammaV_predicted)**2))
        self.log("train_rhoBetaT_loss",torch.mean((rho[non_nan_betaT_index]*betaT_target-rho[non_nan_betaT_index]*betaT_predicted)**2))
        self.log("train_alphaP_loss",torch.mean((alphaP_target-alphaP_predicted)**2))
        self.log("train_mu_jt_loss",torch.mean((mu_jt_predicted-mu_jt_target)**2))
        self.log("train_cp_predicted",torch.mean((cp_predicted-cp_target)**2))
        self.log("train_adiabatic_index_loss",torch.mean((adiabatic_index_target-adiabatic_index_predicted)**2))
        self.log("train_U_loss",torch.mean((U_target-U_predicted)**2))
        self.log("train_Z_loss",torch.mean((Z_predicted-Z_target)**2))  
        self.log("train_loss",loss)
        return {"loss": loss}
    
    def validation_step(self, val_batch, batch_idx):

        # Unpack validation batch
        val_input_i, val_target_i = val_batch
        rho = val_input_i[:,0]
        T = val_input_i[:,1]
        cv_target = val_target_i[:,0]
        gammaV_target = val_target_i[:,1]
        cp_target = val_target_i[:,2]
        alphaP_target = val_target_i[:,3]
        betaT_target = val_target_i[:,4]
        U_target = val_target_i[:,5]
        P_target = val_target_i[:,6]
        mu_jt_target = val_target_i[:,7]
        Z_target = val_target_i[:,8]


        rho_betaT_target = rho*betaT_target
        adiabatic_index_target = cp_target/cv_target

        var_cv = self.calculate_variance(cv_target)
        var_Z = self.calculate_variance(Z_target)
        var_U = self.calculate_variance(U_target)
        var_gammaV = self.calculate_variance(gammaV_target)
        var_rho_betaT = self.calculate_variance(rho*betaT_target)
        var_alphaP = self.calculate_variance(alphaP_target)
        var_adiabatic_index = self.calculate_variance(adiabatic_index_target)
        var_mu_jt = self.calculate_variance(mu_jt_target)
        var_P = self.calculate_variance(P_target)

        # Ensures that the DAG is created for the input so that the gradient and hessian can be computed 
        val_input_i.requires_grad = True

        # Pass input through NN to get the output
        zero_densities = torch.zeros_like(rho)

        input_at_zero_densities = torch.stack((zero_densities,T),dim=-1)

        # A is the residual helmholtz free energy
        A = self.forward(val_input_i)-self.forward(input_at_zero_densities)

        # Computes gradient and hessian
        val_gradient = self.compute_gradient(val_input_i,input_at_zero_densities)
        val_hessian = self.compute_hessian(val_input_i)

        dA_drho = val_gradient[:,0]

        dA_dT = val_gradient[:,1]

        d2A_drho2= val_hessian[:, # In all of the hessians in the batch ...
                                 :, # In all of the heassians in the batch ...
                                 0, # in the first row ...
                                 0] # return the value in the first column
        
        d2A_dT2 = val_hessian[:, # In all of the hessians in the batch ...
                                :, # In all of the heassians in the batch ...
                                1, # in the second row ...
                                1] # return the value in the second column
        
        d2A_dT_drho = val_hessian[:, # In all of the hessians in the batch ...
                                :, # In all of the heassians in the batch ...
                                1, # in the second row ...
                                0] # return the value in the first column
        
        T = torch.reshape(T,(-1,))
        rho = torch.reshape(rho,(-1,))
        A = torch.reshape(A,(-1,))
        dA_dT = torch.reshape(dA_dT,(-1,))
        dA_drho = torch.reshape(dA_drho,(-1,))
        d2A_dT2 = torch.reshape(d2A_dT2,(-1,))
        d2A_drho2 = torch.reshape(d2A_drho2,(-1,))
        d2A_dT_drho = torch.reshape(d2A_dT_drho,(-1,))

        S = -dA_dT
        P_predicted = (rho**2)*dA_drho

        U_predicted = A+(T*S)
        U_predicted += (2/2)*T  # Adding ideal gas contribution

        P_by_rho = rho*dA_drho
        P_by_rho += T # Adding ideal gas contribution

        Z_predicted= (P_by_rho)/T

        cv_predicted = -T*d2A_dT2
        cv_predicted += (2/2) # Adding ideal gas contribution

        dP_dT = (rho**2)*d2A_dT_drho
        dP_dT += rho    # Adding ideal gas contribution

        dP_drho = 2*rho*dA_drho + (rho**2)*d2A_drho2
        dP_drho += T    # Adding ideal gas contribution

        alphaP_predicted = (dP_dT)/(rho*dP_drho)


        rho_betaT = torch.reciprocal(dP_drho)
        betaT_predicted = torch.divide(rho_betaT,rho)

        gammaV_predicted = alphaP_predicted/betaT_predicted

        cp_aux1 = T*(alphaP_predicted**2)
        cp_aux2 = rho*betaT_predicted
        cp_predicted = cv_predicted+(cp_aux1/cp_aux2)
        
        mu_jt_predicted = (1/(rho*cp_predicted))*((T*alphaP_predicted)-1)
        
        # Z_predicted = P_predicted/(rho*T)

        adiabatic_index_predicted = cp_predicted/cv_predicted

        # Calculates the loss

        cv_predicted = torch.clamp(cv_predicted,0,torch.inf)

        # Don't perform the loss calculations for any of the properties where the values were deemed non converged from the data collation script
        non_nan_alphaP_index = ~torch.isnan(alphaP_target)
        non_nan_Z_index = ~torch.isnan(Z_target)        
        non_nan_U_index = ~torch.isnan(U_target)
        non_nan_adiabatic_index_index = ~torch.isnan(adiabatic_index_target)
        non_nan_gammaV_index = ~torch.isnan(gammaV_target)
        non_nan_cv_index = ~torch.isnan(cv_target)        
        non_nan_betaT_index = ~torch.isnan(rho*betaT_target)
        non_nan_mu_jt_index = ~torch.isnan(mu_jt_target )

        # If a property was deemed non converged in the collation script it was set to Nan, this indexes all of the properties where they arent Nan's.
        # This means that we can still use the properties that were converged in a given simulation and disregard the non converged properties

        alphaP_target = alphaP_target[non_nan_alphaP_index]
        alphaP_predicted = alphaP_predicted[non_nan_alphaP_index]

        Z_target = Z_target[non_nan_Z_index]
        Z_predicted = Z_predicted[non_nan_Z_index]

        U_target = U_target[non_nan_U_index]
        U_predicted = U_predicted[non_nan_U_index]
        
        adiabatic_index_target = adiabatic_index_target[non_nan_adiabatic_index_index]
        adiabatic_index_predicted = adiabatic_index_predicted[non_nan_adiabatic_index_index]

        gammaV_target = gammaV_target[non_nan_gammaV_index]
        gammaV_predicted = gammaV_predicted[non_nan_gammaV_index]

        cv_target = cv_target[non_nan_cv_index]
        cv_predicted = cv_predicted[non_nan_cv_index]

        betaT_target = betaT_target[non_nan_betaT_index]
        betaT_predicted = betaT_predicted[non_nan_betaT_index]

        mu_jt_target = mu_jt_target[non_nan_mu_jt_index]
        mu_jt_predicted = mu_jt_predicted[non_nan_mu_jt_index]

        Z_loss = ((Z_target-Z_predicted)**2)/var_Z
        U_loss = ((U_target-U_predicted)**2)/var_U
        alphaP_loss = 1/20*((alphaP_target-alphaP_predicted)**2)/var_alphaP
        adiabatic_index_loss = 1/20*((adiabatic_index_target-adiabatic_index_predicted)**2)/var_adiabatic_index
        gammmaV_loss = 1/20*((gammaV_target-gammaV_predicted)**2)/var_gammaV
        cv_loss = 1/20*((cv_target-cv_predicted)**2)/var_cv
        rho_betaT_loss = 1/20*((rho[non_nan_betaT_index]*betaT_target-rho[non_nan_betaT_index]*betaT_predicted)**2)/var_rho_betaT
        mu_jt_loss = 1/20*((mu_jt_target-mu_jt_predicted)**2)/var_mu_jt
        Z_loss = torch.mean(Z_loss)
        U_loss = torch.mean(U_loss)
        alphaP_loss = torch.mean(alphaP_loss)
        adiabatic_index_loss = torch.mean(adiabatic_index_loss)
        gammmaV_loss = torch.mean(gammmaV_loss)
        cv_loss = torch.mean(cv_loss)
        rho_betaT_loss = torch.mean(rho_betaT_loss)
        mu_jt_loss = torch.mean(mu_jt_loss)



        # DDP training strategy requires that the output of the forward pass of the ANN be in the loss function, 
        # this just fudges the output to 0 so that it doesn't error.

        A_loss_required_for_DDP = A*torch.zeros_like(A)
        A_loss_required_for_DDP = torch.mean(A_loss_required_for_DDP)

        

        # loss = Z_loss+U_loss+cv_loss+gammmaV_loss+adiabatic_index_loss+alphaP_loss+A*torch.zeros_like(A)
        loss = Z_loss+U_loss+cv_loss+alphaP_loss+gammmaV_loss+rho_betaT_loss+adiabatic_index_loss+mu_jt_loss+A_loss_required_for_DDP





        self.log("val_loss",loss) 
        return {"val_loss": loss}
    
    def backward(self, loss):

        # The backward method normally destroys the DAG once its finished.
        # This modifies the backward method to retain the DAG so that the gradient and hessian can be computed
        loss.backward(retain_graph=True)

    def on_validation_model_eval(self, *args, **kwargs):

        # PyTorch Lightning by default sets the model to eval mode when performing the validation loop to conserve memory,
        # However this means that the DAG isn't computed so the gradients cannot be calculated.
        # This overrides that default

        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def compute_gradient(self,inputs,input_at_zero_densities):
        # Compute the gradient of the output of the forward pass wrt the input, grad_outputs is d(forward)/d(forward) which is 1 , See https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
        gradient = torch.autograd.grad(self.forward(inputs)-self.forward(input_at_zero_densities),
                                       inputs,
                                       grad_outputs=torch.ones_like(self.forward(inputs)-self.forward(input_at_zero_densities)),
                                       retain_graph=True,
                                       create_graph=True
                                       )[0]
        return gradient
    def compute_residual_helmholtz_for_hessian(self,x):
        rho = x[0]
        T = x[1]
        zero_densities = torch.zeros_like(rho)
        input_at_zero_densities = torch.stack((zero_densities,T),dim=-1)
        return self.forward(x)-self.forward(input_at_zero_densities)
    
    def compute_hessian(self,input):
        # Compute the hessian of the output wrt the input   
        hessians = torch.vmap(torch.func.hessian(self.compute_residual_helmholtz_for_hessian), (0))(input)
        return hessians
    
    def calculate_variance(self, Tensor_to_calculate_variance):
        Tensor_to_calculate_variance = Tensor_to_calculate_variance[~torch.isnan(Tensor_to_calculate_variance)]
        variance = torch.var(torch.reshape(Tensor_to_calculate_variance,(-1,)))
        return variance
    
    def calculate_cv(self,input):
        T, rho = self.extract_T_and_rho(input)
        d2A_dT2 = self.calculate_d2A_dT2(input)
        d2A_dT2 = torch.reshape(d2A_dT2,(-1,))
        Cv = -T*d2A_dT2
        Cv += 1
        return Cv
    
    def calculate_d2A_dT2(self,input):
        hessians = self.compute_hessian(input)
        d2A_dT2 = hessians[:, # In all of the hessians in the batch ...
                        :, # In all of the heassians in the batch ...
                        1, # in the second row ...
                        1] # return the value in the second column
        return d2A_dT2
    
    def calculate_d2A_drho2(self,input):
        hessians = self.compute_hessian(input)
        d2A_drho2= hessians[:, # In all of the hessians in the batch ...
                                 :, # In all of the heassians in the batch ...
                                 0, # in the first row ...
                                 0] # return the value in the first column
        return d2A_drho2
    
    def extract_T_and_rho(self,input):
        rho = input[:,0]
        T = input[:,1]
        return T,rho

    def calculate_Z(self,input):
        T,rho = self.extract_T_and_rho(input)
        dA_drho = self.calculate_dA_drho(input)
        dA_drho = torch.reshape(dA_drho,(-1,))
        P_by_rho = rho*dA_drho
        P_by_rho += T # Adding ideal gas contribution
        Z= (P_by_rho)/T
        return Z
    
    def calculate_dA_drho(self,input):
        T, rho =  self.extract_T_and_rho(input)
        zero_densities = torch.zeros_like(rho)
        input_at_zero_densities = torch.stack((zero_densities,T),dim=-1)
        gradients = self.compute_gradient(input,input_at_zero_densities)
        dA_drho = gradients[:,0]
        return dA_drho
    
    def calculate_dA_dT(self,input):
        T, rho =  self.extract_T_and_rho(input)
        zero_densities = torch.zeros_like(rho)
        input_at_zero_densities = torch.stack((zero_densities,T),dim=-1)
        gradients = self.compute_gradient(input,input_at_zero_densities)
        dA_dT = gradients[:,1]
        return dA_dT
    
    def calculate_dP_dT(self,input):
        T, rho = self.extract_T_and_rho(input)
        d2A_dT_drho = self.calculate_d2A_dT_drho(input)
        d2A_dT_drho = torch.reshape(d2A_dT_drho,(-1,))
        dP_dT = (rho**2)*d2A_dT_drho
        dP_dT += rho    # Adding ideal gas contribution
        return dP_dT
    
    def calculate_dP_drho(self,input):
        T, rho = self.extract_T_and_rho(input)
        dA_drho = self.calculate_dA_drho(input)
        dA_drho = torch.reshape(dA_drho,(-1,))
        d2A_drho2 = self.calculate_d2A_drho2(input)
        d2A_drho2 = torch.reshape(d2A_drho2,(-1,))
        dP_drho = 2*rho*dA_drho + (rho**2)*d2A_drho2
        dP_drho += T    # Adding ideal gas contribution
        return dP_drho


    def calculate_d2A_dT_drho(self,input):
        hessians = self.compute_hessian(input)
        d2A_dT_drho = hessians[:, # In all of the hessians in the batch ...
                                :, # In all of the hessians in the batch ...
                                1, # in the second row ...
                                0] # return the value in the first column
        return d2A_dT_drho

    def calculate_U(self,input):
        T, rho = self.extract_T_and_rho(input)
        zero_densities = torch.zeros_like(rho)
        input_at_zero_densities = torch.stack((zero_densities,T),dim=-1)

        #A is the residual helmholtz free energy
        A = self.forward(input)-self.forward(input_at_zero_densities)
        A= torch.reshape(A,(-1,))
        S = self.calculate_S(input)
        U=A+(T*S)
        U += T
        return U

    
    def calculate_S(self,input):
        T,rho = self.extract_T_and_rho(input)
        dA_dT = self.calculate_dA_dT(input)
        dA_dT = torch.reshape(dA_dT,(-1,))
        S = -dA_dT
        return S
    
    def calculate_gammaV(self,input):
        alphaP = self.calculate_alphaP(input)
        betaT = self.calculate_betaT(input)
        gammaV = alphaP/betaT
        return gammaV
    
    def calculate_betaT(self,input):
        T, rho = self.extract_T_and_rho(input)
        dP_drho = self.calculate_dP_drho(input)
        dP_drho = torch.reshape(dP_drho,(-1,))
        rho_betaT = torch.reciprocal(dP_drho)
        betaT = torch.divide(rho_betaT,rho)
        return betaT
    
    def calculate_alphaP(self,input):
        T, rho = self.extract_T_and_rho(input)
        dP_dT = self.calculate_dP_dT(input)
        dP_drho = self.calculate_dP_drho(input)
        alphaP = (dP_dT)/(rho*dP_drho)
        return alphaP

    def calculate_P(self,input):
        T,rho = self.extract_T_and_rho(input)
        dA_drho = self.calculate_dA_drho(input)
        dA_drho = torch.reshape(dA_drho,(-1,))
        P_by_rho = rho*dA_drho
        P_by_rho += T # Adding ideal gas contribution
        P = P_by_rho*rho
        return P

    def calculate_cp(self,input):
        T,rho = self.extract_T_and_rho(input)
        cv = self.calculate_cv(input)
        alphaP = self.calculate_alphaP(input)
        betaT = self.calculate_betaT(input)
        return (cv + (T/rho)*((alphaP**2)/betaT))
    
    def calculate_mu_jt(self,input):
        T,rho = self.extract_T_and_rho(input)
        cp = self.calculate_cp(input)
        alphaP = self.calculate_alphaP(input)
        return (1/(rho*cp))*((T*alphaP)-1)
    
    def calculate_mu(self,input):
        T, rho = self.extract_T_and_rho(input)
        zero_densities = torch.zeros_like(rho)
        input_at_zero_densities = torch.stack((zero_densities,T),dim=-1)
        A = self.forward(input)-self.forward(input_at_zero_densities)
        A = torch.reshape(A,(-1,))
        dA_drho = self.calculate_dA_drho(input)
        dA_drho = torch.reshape(dA_drho,(-1,))
        mu = A+rho*dA_drho
        return mu
    
    def calculate_adiabatic_index(self,input):
        cp = self.calculate_cp(input)
        cv = self.calculate_cv(input)
        adiabatic_index = cp/cv
        return adiabatic_index
    def calculate_d2P_drho2(self,input):
        input.requires_grad=True
        T, rho = self.extract_T_and_rho(input)
        dA_drho = self.calculate_dA_drho(input)
        dA_drho = torch.reshape(dA_drho,(-1,))
        d2A_drho2 = self.calculate_d2A_drho2(input)
        d2A_drho2 = torch.reshape(d2A_drho2,(-1,))
        d3A_drho3 = torch.autograd.grad(d2A_drho2,input,grad_outputs=torch.ones_like(d2A_drho2),create_graph=True)[0][:,0]
        d2P_drho2 = 2*dA_drho+4*rho*d2A_drho2+rho**2*d3A_drho3
        return d2P_drho2


path_to_training_data = r"models\training_data_for_current_ANN.txt"
path_to_validation_data = r"models\validation_data_for_current_ANN.txt"
model = BasicLightning.load_from_checkpoint(r"models\Model.ckpt")
model = model.double()
model.eval()

train_arr = np.loadtxt(path_to_training_data)
val_arr = np.loadtxt(path_to_validation_data)

# Splitting the preprocessed data into the inputs and targets
density_column = 4
temperature_column = 2
pressure_column = 3
internal_energy_column = 1
cp_column = 20
alphaP_column = cp_column +1
betaT_column = alphaP_column +1
mu_jt_column = betaT_column + 1
Z_column = mu_jt_column+1
cv_column = Z_column+1
gammaV_column = cv_column+1
target_columns = [pressure_column]
train_inputs = torch.tensor(train_arr[:,[density_column,temperature_column]])
train_targets = torch.tensor(train_arr[:,target_columns])
val_inputs = torch.tensor(val_arr[:,[density_column,temperature_column]])
val_targets = torch.tensor(val_arr[:,target_columns])
train_inputs = train_inputs.double()
train_targets = train_targets.double()
val_inputs = val_inputs.double()
val_targets = val_targets.double()
input = torch.tensor([[0.5,2.0]])
input.requires_grad = True
train_inputs.requires_grad=True
val_inputs.requires_grad = True

predicted_cv = model.calculate_cv(val_inputs).detach().numpy()[~np.isnan(val_arr[:,cv_column])]
predicted_z = model.calculate_Z(val_inputs).detach().numpy()[~np.isnan(val_arr[:,Z_column])]
predicted_U = model.calculate_U(val_inputs).detach().numpy()[~np.isnan(val_arr[:,internal_energy_column])]
predicted_P = model.calculate_P(val_inputs).detach().numpy()[~np.isnan(val_arr[:,pressure_column])]
predicted_alphaP = model.calculate_alphaP(val_inputs).detach().numpy()[~np.isnan(val_arr[:,alphaP_column])]
predicted_betaT = model.calculate_betaT(val_inputs).detach().numpy()[~np.isnan(val_arr[:,betaT_column])]
predicted_mujt = model.calculate_mu_jt(val_inputs).detach().numpy()[~np.isnan(val_arr[:,mu_jt_column])]
predicted_gammaV = model.calculate_gammaV(val_inputs).detach().numpy()[~np.isnan(val_arr[:,gammaV_column])]
predicted_cp = model.calculate_cp(val_inputs).detach().numpy()[~np.isnan(val_arr[:,cp_column])]
predicted_adiabatic_index = model.calculate_adiabatic_index(val_inputs).detach().numpy()
target_Z = val_arr[:,Z_column][~np.isnan(val_arr[:,Z_column])]
target_cv = val_arr[:,cv_column][~np.isnan(val_arr[:,cv_column])]
target_U = val_arr[:,internal_energy_column][~np.isnan(val_arr[:,internal_energy_column])]
target_P = val_arr[:,pressure_column][~np.isnan(val_arr[:,pressure_column])]
target_alphaP = val_arr[:,alphaP_column][~np.isnan(val_arr[:,alphaP_column])]
target_betaT = val_arr[:,betaT_column][~np.isnan(val_arr[:,betaT_column])]
target_mujt = val_arr[:,mu_jt_column][~np.isnan(val_arr[:,mu_jt_column])]
target_gammaV = val_arr[:,gammaV_column][~np.isnan(val_arr[:,gammaV_column])]
target_cp = val_arr[:,cp_column][~np.isnan(val_arr[:,cp_column])]
target_adiabatic_index = target_cp/val_arr[:,cv_column][~np.isnan(val_arr[:,cp_column])]
predicted_adiabatic_index = predicted_adiabatic_index[~np.isnan(val_arr[:,cp_column])]
targets = [target_Z,target_U,target_cv,target_cp,target_gammaV,target_betaT,target_alphaP,target_mujt,target_adiabatic_index]
predicteds = [predicted_z,predicted_U,predicted_cv,predicted_cp,predicted_gammaV,predicted_betaT,predicted_alphaP,predicted_mujt,predicted_adiabatic_index]

import matplotlib 
sns.set_style('ticks')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Palatino Linotype'
plt.rcParams["mathtext.default"] = 'it'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
plt.rcParams.update({'font.size': 14})

plasma_big = matplotlib.colormaps['plasma']
newcmp = matplotlib.colors.ListedColormap(plasma_big(np.linspace(0.1, 0.8, 128)))
#Error heatplot
figsize=(10, 4)
fig, ax = plt.subplots(1,2,figsize=figsize,constrained_layout=True)
fig.get_layout_engine().set(wspace=0.1)
max_error_clipping = 0.1
marker_size = 10

error = (target_Z-predicted_z)**2
scatter = ax[0].scatter(val_arr[:,density_column][~np.isnan(val_arr[:,Z_column])], val_arr[:,temperature_column][~np.isnan(val_arr[:,Z_column])], c=error, cmap=newcmp,norm = matplotlib.colors.LogNorm(vmin = 1e-3),s = marker_size)
ax[0].set_xlabel(r'$\mathit{\rho^*}$', style='italic',fontsize = 14)
ax[0].set_ylabel(r"$T^*$",style="italic",fontsize = 14)
ax[0].set(xlim=(0, 1),ylim=(0.35,10))
ax[0].set_title('(a)')

cbar = fig.colorbar(scatter)
cbar.ax.get_xaxis().get_major_formatter().labelOnlyBase = False
tick_labels = cbar.ax.get_yticklabels()
tick_labels[-1] = '> {}'.format(max_error_clipping)
cbar.ax.set_yticklabels(tick_labels)


error = (target_U-predicted_U)**2
scatter = ax[1].scatter(val_arr[:,density_column][~np.isnan(val_arr[:,internal_energy_column])], val_arr[:,temperature_column][~np.isnan(val_arr[:,internal_energy_column])], c=error, cmap=newcmp,norm = matplotlib.colors.LogNorm(vmin = 1e-3),s = marker_size)
ax[1].set_xlabel(r'$\mathit{\rho^*}$', style='italic',fontsize = 14)
ax[1].set_ylabel(r"$T^*$",style="italic",fontsize = 14)
ax[1].set(xlim=(0, 1),ylim=(0.35,10))
ax[1].set_title('(b)')

cbar = fig.colorbar(scatter)
tick_labels = cbar.ax.get_yticklabels()
tick_labels[-1] = '> {}'.format(max_error_clipping)
cbar.ax.set_yticklabels(tick_labels)


plt.savefig("Z_U_error_heatmap.svg")
plt.show()




fig, axs = plt.subplots(2, 5, figsize=(15, 10),constrained_layout=True)

sns.lineplot(x =[0,8],y=[0,8],ax = axs[0, 0], color = "k",)
sns.scatterplot(x = predicted_z.flatten(),y=target_Z.flatten(), ax = axs[0, 0],alpha =0.2)
axs[0, 0].set_xlabel('Predicted_Z')
axs[0, 0].set_ylabel('Target_Z')
axs[0, 0].set_title('Z_parity')

sns.scatterplot(x = predicted_U.flatten(),y = target_U.flatten(), ax = axs[0, 1],alpha =0.2)
sns.lineplot(x =[-2,20],y=[-2,20], ax = axs[0, 1], color = "k",)
axs[0, 1].set_xlabel('Predicted_U')
axs[0, 1].set_ylabel('Target_U')
axs[0, 1].set_title('U_parity')

sns.scatterplot(x = predicted_P.flatten(),y = target_P.flatten(), ax = axs[0, 2],alpha =0.2)
sns.lineplot(x =[0,65],y=[0,65], ax = axs[0, 2], color = "k",)
axs[0, 2].set_xlabel('Predicted_P')
axs[0, 2].set_ylabel('Target_P')
axs[0, 2].set_title('P_parity')

sns.scatterplot(x=predicted_cv.flatten(),y=target_cv.flatten(), ax = axs[0, 3],alpha =0.2)
axs[0, 3].set_xlabel('Predicted_cv')
axs[0, 3].set_ylabel('Target_cv')
axs[0, 3].set_title('cv_parity')
sns.lineplot(x =[1,6],y=[1,6], ax = axs[0, 3], color = "k",)

sns.scatterplot(x=predicted_betaT.flatten(),y=target_betaT.flatten(), ax = axs[1, 0],alpha =0.2)
sns.lineplot(x=[0,400],y=[0,400], ax = axs[1, 0], color = "k",)
axs[1, 0].set_xlabel('Predicted_betaT')
axs[1, 0].set_ylabel('Target_betaT')
axs[1, 0].set_title('betaT_parity')
# axs[1, 0].set_xlim(0,20)
# axs[1, 0].set_ylim(0,20)

sns.scatterplot(x=predicted_cp.flatten(),y=target_cp.flatten(), ax = axs[0, 4],alpha =0.2)
axs[0, 4].set_xlabel('predicted_Cp')
axs[0, 4].set_ylabel('target_Cp')
axs[0, 4].set_title('Cp_parity')
sns.lineplot(x =[1.5,36],y=[1.5,36], ax = axs[0, 4], color = "k",)

sns.scatterplot(x=predicted_alphaP.flatten(),y=target_alphaP.flatten(), ax = axs[1, 2],alpha =0.2)
axs[1, 2].set_xlabel('Predicted_alphaP')
axs[1, 2].set_ylabel('Target_alphaP')
axs[1, 2].set_title('alphaP_parity')
sns.lineplot(x =[0,37],y=[0,37], ax = axs[1, 2], color = "k",)

sns.scatterplot(x=predicted_gammaV.flatten(),y=target_gammaV.flatten(), ax = axs[1, 1],alpha =0.2)
axs[1, 1].set_xlabel('Predicted_gammaV')
axs[1, 1].set_ylabel('Target_gammaV')
axs[1, 1].set_title('gammaV_parity')
sns.lineplot(x =[-5,10],y=[-5,10], ax = axs[1, 1], color = "k",)

sns.scatterplot(x=predicted_mujt.flatten(),y=target_mujt.flatten(), ax = axs[1, 3],alpha =0.2)
axs[1, 3].set_xlabel('predicted_mujt')
axs[1, 3].set_ylabel('target_mujt')
axs[1, 3].set_title('mujt_parity')
sns.lineplot(x =[-1,17],y=[-1,17], ax = axs[1, 3], color = "k",)

sns.scatterplot(x=predicted_adiabatic_index.flatten(),y=target_adiabatic_index.flatten(), ax = axs[1, 4],alpha =0.2)
axs[1, 4].set_xlabel('predicted_Cp/Cv')
axs[1, 4].set_ylabel('target_Cp/Cv')
axs[1, 4].set_title('Cp/Cv_parity')
sns.lineplot(x =[0,10],y=[0,10], ax = axs[1, 4], color = "k",)

# plt.show()




##########
# Isotherms
##########
# [0.511,0.522,1.09,3.38]
crit_isotherm_array = [0.45,0.511,0.522,2.247764,4.504468]
isotherm_figure, isotherm_plots = plt.subplots(2, 3, figsize=(12, 7),constrained_layout=True)
i = 0
plt.rcParams.update({'font.size': 18})
for temperature in crit_isotherm_array:

    colour_list = ['r','b','g','m','k']
    def find_closest(array, value, n=100):
        array = np.asarray(array)
        idx = np.argsort(np.abs(array - value))[:n]
        return idx
    
    # Find the closest points since the MD result temperature isnt numerically exactly the input temperature
    index_of_points_close_to_temp = find_closest(train_arr[:,temperature_column],temperature,50)

    # Exclude datapoints which are within the closest N points but are clearly not from the isotherm
    upper_limit = temperature+0.005
    lower_limit = temperature-0.005

    mask = (train_arr[index_of_points_close_to_temp,temperature_column]<upper_limit)&(lower_limit<train_arr[index_of_points_close_to_temp,temperature_column])
    isotherm = train_arr[index_of_points_close_to_temp,:][mask]
    
    # Index the isotherm data to get the properties for each plot
    P_isotherm_MD = isotherm[:,pressure_column]
    cv_isotherm_MD = isotherm[:,cv_column]
    cp_isotherm_MD = isotherm[:,cp_column]
    gammaV_isotherm_MD = isotherm[:,gammaV_column]
    betaT_isotherm_MD = isotherm[:,betaT_column]
    alphaP_isotherm_MD = isotherm[:,alphaP_column]
    mu_jt_isotherm_MD = isotherm[:,mu_jt_column]
    density_MD = isotherm[:,density_column]


    # Create a numpy array with values from 0 to 0.9
    density = torch.linspace(0,0.9, 1000)

    # Create a numpy array with a constant value
    temperature = torch.full((1000,), temperature)

    # Concatenate the two numpy arrays horizontally
    tensor = torch.stack((density, temperature), dim=1).double()
    tensor.requires_grad=True
    predicted_cv = model.calculate_cv(tensor).detach().numpy()
    predicted_P = model.calculate_P(tensor).detach().numpy()
    predicted_alphaP = model.calculate_alphaP(tensor).detach().numpy()
    predicted_betaT = model.calculate_betaT(tensor).detach().numpy()
    predicted_mujt = model.calculate_mu_jt(tensor).detach().numpy()
    predicted_gammaV = model.calculate_gammaV(tensor).detach().numpy()
    predicted_cp = model.calculate_cp(tensor).detach().numpy()

    sns.lineplot(x=density.numpy().flatten(),y=predicted_cv,ax=isotherm_plots[0,0],color = colour_list[i])
    sns.scatterplot(x =density_MD.flatten(),y=cv_isotherm_MD.flatten(),ax=isotherm_plots[0,0],color = colour_list[i])
    isotherm_plots[0,0].set_xlabel(r'$\mathit{\rho^*}$', style='italic',fontsize = 18)
    isotherm_plots[0,0].set_ylabel(r'$\mathit{C_v^*}$', style='italic',fontsize = 18)
    isotherm_plots[0,0].set(ylim=(1,6),xlim=(0,0.9))
    isotherm_plots[0,0].tick_params(axis="both", labelsize=18)
    plt.xticks(fontsize=18)
    sns.lineplot(x=density.numpy().flatten(),y=predicted_cp,ax=isotherm_plots[0,1],color = colour_list[i])
    sns.scatterplot(x =density_MD.flatten(),y=cp_isotherm_MD.flatten(),ax=isotherm_plots[0,1],color = colour_list[i])
    isotherm_plots[0,1].set_xlabel(r'$\mathit{\rho^*}$', style='italic',fontsize = 18)
    isotherm_plots[0,1].set_ylabel(r'$\mathit{C_p^*}$', style='italic',fontsize = 18)
    isotherm_plots[0,1].set(ylim=(0,20),xlim=(0,0.9))
    isotherm_plots[0,1].tick_params(axis="both", labelsize=18)
    plt.xticks(fontsize=18)
    sns.lineplot(x=density.numpy().flatten(),y=predicted_mujt,ax=isotherm_plots[0,2],color = colour_list[i])
    sns.scatterplot(x =density_MD.flatten(),y=mu_jt_isotherm_MD.flatten(),ax=isotherm_plots[0,2],color = colour_list[i])
    isotherm_plots[0,2].set_xlabel(r'$\mathit{\rho^*}$', style='italic',fontsize = 18)
    isotherm_plots[0,2].set_ylabel(r'$\mathit{\mu_{JT}^*}$', style='italic',fontsize = 18)
    isotherm_plots[0,2].set(ylim=(-1.0,12.5),xlim=(0,0.9))
    isotherm_plots[0,2].tick_params(axis="both", labelsize=18)
    plt.xticks(fontsize=18)
    sns.lineplot(x=density.numpy().flatten(),y=predicted_gammaV,ax=isotherm_plots[1,0],color = colour_list[i])
    sns.scatterplot(x =density_MD.flatten(),y=gammaV_isotherm_MD.flatten(),ax=isotherm_plots[1,0],color = colour_list[i])
    isotherm_plots[1,0].set_xlabel(r'$\mathit{\rho^*}$', style='italic',fontsize = 18)
    isotherm_plots[1,0].set_ylabel(r'$\mathit{\gamma_v^*}$', style='italic',fontsize = 18)
    isotherm_plots[1,0].set(ylim=(0.0,6.5),xlim=(0,0.9))
    isotherm_plots[1,0].tick_params(axis="both", labelsize=18)
    plt.xticks(fontsize=18)
    sns.lineplot(x=density.numpy().flatten(),y=predicted_alphaP,ax=isotherm_plots[1,1],color = colour_list[i])
    sns.scatterplot(x =density_MD.flatten(),y=alphaP_isotherm_MD.flatten(),ax=isotherm_plots[1,1],color = colour_list[i])
    isotherm_plots[1,1].set_xlabel(r'$\mathit{\rho^*}$', style='italic',fontsize = 18)
    isotherm_plots[1,1].set_ylabel(r'$\mathit{\alpha_p^*}$', style='italic',fontsize = 18)
    isotherm_plots[1,1].set(ylim=(-0.5,20),xlim=(0,0.9))
    isotherm_plots[1,2].tick_params(axis="both", labelsize=18)
    plt.xticks(fontsize=18)

    sns.lineplot(x=density.numpy().flatten(),y=density.numpy()*predicted_betaT,ax=isotherm_plots[1,2],color = colour_list[i])
    sns.scatterplot(x =density_MD.flatten(),y=density_MD.flatten()*betaT_isotherm_MD.flatten(),ax=isotherm_plots[1,2],color = colour_list[i])
    isotherm_plots[1,2].set_xlabel(r'$\mathit{\rho^*}$', style='italic',fontsize = 18)
    isotherm_plots[1,2].set_ylabel(r'$\mathit{\rho^* \beta_T^*}$', style='italic',fontsize = 18)
    isotherm_plots[1,2].set(ylim=(-0.5,25),xlim=(0.0,0.9))
    isotherm_plots[1,2].tick_params(axis="both", labelsize=18)
    plt.xticks(fontsize=18)
    i +=1
plt.savefig("Derivative_properties_isotherms.svg")
plt.show()
plt.rcParams.update({'font.size': 14})
figsize = (4, 6)
gridspec_kw = dict(
    nrows=2, ncols=1,
    width_ratios=[1],
    height_ratios=[3, 1],
)
subplot_kw = dict(sharey="row")
p_isotherm_fig = plt.figure(figsize=figsize, constrained_layout=True)
P_isotherm_plots = p_isotherm_fig.add_gridspec(**gridspec_kw).subplots(**subplot_kw)
i = 0
for temperature in crit_isotherm_array:

    colour_list = ['r','b','g','m','k']
    def find_closest(array, value, n=100):
        array = np.asarray(array)
        idx = np.argsort(np.abs(array - value))[:n]
        return idx
    
    # Find the closest points since the MD result temperature isnt numerically exactly the input temperature

    index_of_points_close_to_temp = find_closest(train_arr[:,temperature_column],temperature,100)

    # Exclude datapoints which are within the closest N points but are clearly not from the isotherm
    upper_limit = temperature+0.001
    lower_limit = temperature-0.001

    mask = (train_arr[index_of_points_close_to_temp,temperature_column]<upper_limit)&(lower_limit<train_arr[index_of_points_close_to_temp,temperature_column])
    isotherm = train_arr[index_of_points_close_to_temp,:][mask]
    
    # Index the isotherm data to get the properties for each plot
    P_isotherm_MD = isotherm[:,pressure_column]
    cv_isotherm_MD = isotherm[:,cv_column]
    cp_isotherm_MD = isotherm[:,cp_column]
    gammaV_isotherm_MD = isotherm[:,gammaV_column]
    betaT_isotherm_MD = isotherm[:,betaT_column]
    alphaP_isotherm_MD = isotherm[:,alphaP_column]
    mu_jt_isotherm_MD = isotherm[:,mu_jt_column]
    density_MD = isotherm[:,density_column]


    # Create a numpy array with values from 0 to 0.9
    density = torch.linspace(0,0.9, 1000)

    # Create a numpy array with a constant value
    temperature = torch.full((1000,), temperature)

    # Concatenate the two numpy arrays horizontally
    tensor = torch.stack((density, temperature), dim=1).double()
    tensor.requires_grad=True
    predicted_cv = model.calculate_cv(tensor).detach().numpy()
    predicted_P = model.calculate_P(tensor).detach().numpy()
    predicted_alphaP = model.calculate_alphaP(tensor).detach().numpy()
    predicted_betaT = model.calculate_betaT(tensor).detach().numpy()
    predicted_mujt = model.calculate_mu_jt(tensor).detach().numpy()
    predicted_gammaV = model.calculate_gammaV(tensor).detach().numpy()
    predicted_cp = model.calculate_cp(tensor).detach().numpy()
    sns.lineplot(x=density.numpy().flatten(),y=predicted_P,ax=P_isotherm_plots[0],color = colour_list[i])
    sns.scatterplot(x =density_MD.flatten(),y=P_isotherm_MD.flatten(),ax=P_isotherm_plots[0],color = colour_list[i])
    P_isotherm_plots[0].set_xlabel(r'$\mathit{\rho^*}$', style='italic',fontsize = 14)
    P_isotherm_plots[0].set_ylabel(r'$\mathit{P^*}$', style='italic',fontsize = 14)
    P_isotherm_plots[0].set(ylim=(0,2),xlim=(0,0.9))

    sns.lineplot(x=density.numpy().flatten(),y=predicted_P,ax=P_isotherm_plots[1],color = colour_list[i])
    sns.scatterplot(x =density_MD.flatten(),y=P_isotherm_MD.flatten(),ax=P_isotherm_plots[1],color = colour_list[i])
    P_isotherm_plots[1].set_xlabel(r'$\mathit{\rho^*}$', style='italic',fontsize = 14)
    P_isotherm_plots[1].set_ylabel(r'$\mathit{P^*}$', style='italic',fontsize = 14)
    P_isotherm_plots[1].set(ylim=(0.02,0.04),xlim=(0,0.9))
    # isotherm_plots.set_ylabel(r'$\mathit{P^*}$', style='italic')
    # isotherm_plots.set_xlabel(r"$\rho^*$",style="italic")
    i += 1
plt.savefig("Pressure_isotherms.svg")
plt.show()


# Adapted from:
#  https://github.com/gustavochm/fe-ann-eos-mie/blob/main/examples/2.%20Computing%20fluid%20phase%20equilibria%20with%20the%20FE-ANN%20EoS.ipynb
#  Gustavo Chaparro,2023
def fobj_crit(input, model):
    rhoad, Tad = np.split(np.asarray(input), 2)
    input = torch.tensor([input])
    input.requires_grad=True
    input = input.double()

    press = model.calculate_P(input).detach().numpy()
    dpress = model.calculate_dP_drho(input).detach().numpy()
    d2press = model.calculate_d2P_drho2(input).detach().numpy()
    
    fo = np.hstack([dpress, 2*dpress/rhoad + d2press])
    return fo

# initial guess
Tc0 = 0.522
rhoc0 = 0.4
# solving fobj_crit
sol_crit = root(fobj_crit, x0=[rhoc0, Tc0], args=(model),tol = 1e-80)
rhoc_model, Tc_model = sol_crit.x
print(sol_crit.x)
# computing critical pressure
# Pc_model = model.calculate_P(torch.tensor([rhoc_model,Tc_model]))
# print(Pc_model)

# Objetive function to compute VLE.
# This function compute the difference between the pressure and chemical potential each phase.
def fobj_vle(rhoad,Tad,model):
    Tad = Tad[0]
    Tad = torch.full((len(rhoad),),Tad)
    rhoad = torch.tensor(rhoad)
    input = torch.stack((rhoad,Tad),dim=1)
    input.requires_grad=True
    input = input.double()
    press = model.calculate_P(input).detach().numpy()
    aideal = Tad*(np.log(rhoad) - 1.)
    # daideal_drho = Tad/rhoad
    # chempot_ideal = aideal + rhoad*daideal_drho
    chempot_ideal = aideal + Tad
    chempot_res = model.calculate_mu(input).detach().numpy()
    chempot_res = torch.tensor(chempot_res)
    chempot = (chempot_res + chempot_ideal)
    fo = np.hstack([np.diff(press), np.diff(chempot)])   
    return fo

# number of points to compute the phase envelop
n = 200
rhol = np.zeros(n) # array to store liquid density
rhov = np.zeros(n) # array to store vapour density
T = np.linspace(0.2, Tc_model, n) # array with the saturation temperatures

# solving for the first point at a low temperature
i = 0 
Tad = np.ones(1)*T[i]
# initial guesses for the vapour and liquid phase, respectively
rhoad0 = np.array([1e-5, 0.9])
sol_vle = root(fobj_vle, x0=[rhoad0], args=(Tad,model))
rhov[i], rhol[i] = sol_vle.x

print("First point VLE results")
print("Equilibrium liquid density at T={Tsat}: ".format(Tsat=T[i]), np.round(rhol[i], 5))
print("Equilibrium vapour density at T={Tsat}: ".format(Tsat=T[i]), np.round(rhov[i], 5))

# solving VLE for other temperatues using previous solution as initial guess
for i in range(1, n):
    Tad = np.ones(1)*T[i]
    rhoad0 = np.array([rhov[i-1], rhol[i-1]])
    sol_vle = root(fobj_vle, x0=rhoad0, args=(Tad, model),tol = 1e-10)
    rhov[i], rhol[i] = sol_vle.x


# number of points to compute the phase envelop
n = 9
rhol = np.zeros(n) # array to store liquid density
rhov = np.zeros(n) # array to store vapour density
T = np.array([0.415,0.425,0.435,0.445,0.450,0.455,0.460,0.465,0.468]) # array with the saturation temperatures

# solving for the first point at a low temperature
i = 0 
Tad = np.ones(1)*T[i]
# initial guesses for the vapour and liquid phase, respectively
rhoad0 = np.array([1e-5, 0.9])
sol_vle = root(fobj_vle, x0=[rhoad0], args=(Tad,model))
rhov[i], rhol[i] = sol_vle.x

print("First point VLE results")
print("Equilibrium liquid density at T={Tsat}: ".format(Tsat=T[i]), np.round(rhol[i], 5))
print("Equilibrium vapour density at T={Tsat}: ".format(Tsat=T[i]), np.round(rhov[i], 5))

# solving VLE for other temperatues using previous solution as initial guess
T_gl = np.array([0.450,0.460,0.470,0.480,0.490,0.495,0.500,0.505,0.515])
density_gas = np.array([0.030, 0.036, 0.05, 0.053, 0.064, 0.07, 0.09, 0.09, 0.28])
density_liq = np.array([0.722,0.72,0.70,0.68,0.65,0.65,0.65,0.64,0.43])

for i in range(1, n):
    Tad = np.ones(1)*T[i]
    rhoad0 = np.array([rhov[i-1], rhol[i-1]])
    sol_vle = root(fobj_vle, x0=rhoad0, args=(Tad, model),tol = 1e-10)
    rhov[i], rhol[i] = sol_vle.x
print(np.array(rhov)-density_gas)
print(np.array(rhol)-density_liq)


##########################################
# End of code adapted from Gustavo Chaparro
##########################################
# Literature VLE EOS data

ROS_EOS_VLE_df = pd.read_csv(r"C:\Users\Daniel.000\Documents\New_research_project\Neural_Net_EOS\Lit_Data\RO_EOS_VLE_DATA.txt",delimiter=" ")
CMO_EOS_VLE_df = pd.read_csv(r"C:\Users\Daniel.000\Documents\New_research_project\Neural_Net_EOS\Lit_Data\CMO_EOS_VLE_DATA.txt",delimiter =" ")
# Literature VLE simulation Dataw

vle_temp_2019_data = np.array([0.4,0.41,0.42,0.43,0.44])
inverse_gas_density_2019_data = np.array([35.40806914,28.16798258,21.91249159,16.245217,11.06553708])
inverse_liq_density_2019_data = np.array([1.331229281,1.361101621,1.39302346,1.436343071,1.480891538])
vle_density_gas_2019_data = np.reciprocal(inverse_gas_density_2019_data)
vle_density_liq_2019_data = np.reciprocal(inverse_liq_density_2019_data)

density_f = np.array([0.786,0.801,0.829,0.856])
density_s = np.array([0.855,0.870,0.880,0.903])
T_sf = np.array([0.45,0.55,0.70,1])

critical_point_density = np.array([0.366])
critical_point_temperature = np.array([0.522])

T_gl = np.array([0.450,0.460,0.470,0.480,0.490,0.495,0.500,0.505,0.515])
density_gas = np.array([0.030, 0.036, 0.05, 0.053, 0.064, 0.07, 0.09, 0.09, 0.28])
density_liq = np.array([0.722,0.72,0.70,0.68,0.65,0.65,0.65,0.64,0.43])

T_vl = np.array([0.415,0.425,0.435,0.445,0.450,0.455,0.460,0.465,0.468])
density_gas2 = np.array([0.0163,0.0173,0.0235,0.0274,0.0285,0.0339,0.0406,0.0473,0.0420])
density_liq2 = np.array([0.763,0.757,0.738,0.725,0.712,0.706,0.698,0.695,0.603])

liq_vle_density = np.concatenate((critical_point_density,density_liq,density_liq2))
liq_vle_temp = np.concatenate((critical_point_temperature,T_gl,T_vl))
gas_vle_density = np.concatenate((density_gas2,density_gas,critical_point_density))
gas_vle_temperature = np.concatenate((T_vl,T_gl,critical_point_temperature))

vle_liq_df = pd.DataFrame({
    'Densities' : liq_vle_density,
    'Temperatures' : liq_vle_temp})
vle_liq_df_sorted = vle_liq_df.sort_values(by=['Densities'])
vle_liq_df_sorted
vle_liq_den = vle_liq_df_sorted['Densities'].to_numpy()
vle_liq_tem = vle_liq_df_sorted['Temperatures'].to_numpy()

vle_gas_df = pd.DataFrame({
    'Densities' : gas_vle_density,
    'Temperatures' : gas_vle_temperature})
vle_gas_df_sorted = vle_gas_df.sort_values(by=['Densities'])
vle_gas_den = vle_gas_df_sorted['Densities'].to_numpy()
vle_gas_tem = vle_gas_df_sorted['Temperatures'].to_numpy()

vle_densities = np.concatenate([vle_gas_den,vle_liq_den])
vle_temperatures = np.concatenate([vle_gas_tem,vle_liq_tem])

liquid_density_range = np.linspace(min(vle_liq_den),min(density_f),100)
gas_density_range = np.linspace(min(gas_vle_density),max(gas_vle_density),100)


figsize=(5, 4)

fig, plots = plt.subplots(constrained_layout=True,figsize=figsize)

plots.scatter(x = critical_point_density,y = critical_point_temperature,facecolors='r',edgecolors='r', marker='*',zorder=1)
plots.scatter(x = liq_vle_density[1:-2],y = liq_vle_temp[1:-2],facecolors='none',edgecolors='r', marker='^',zorder=1)
plots.scatter(x = vle_gas_den[:-2],y = vle_gas_tem[:-2],facecolors='none',edgecolors='r', marker='^',zorder=1)
# plots.scatter(density_f,T_sf,facecolors='none',edgecolors='r', marker='^')
# plots.scatter(x=ROS_EOS_VLE_df["rhoV"],y=ROS_EOS_VLE_df["T"])
# plots.scatter(data = ROS_EOS_VLE_df,x='rhoV',y='T')
plots.plot(rhov, T, color='k')
plots.plot(rhol, T, color='k')
plots.plot(rhoc_model, Tc_model, '*',color='k')

plots.plot(ROS_EOS_VLE_df["rhoV"].values,ROS_EOS_VLE_df["T"].values,label = "RO EOS",color = "b")
plots.plot(ROS_EOS_VLE_df["rhoL"].values,ROS_EOS_VLE_df["T"].values,label = "RO EOS",color = "b")
plots.plot(CMO_EOS_VLE_df["rhoV"].values,CMO_EOS_VLE_df["T"].values,label = "CMO EOS",color = "g")
plots.plot(CMO_EOS_VLE_df["rhoL"].values,CMO_EOS_VLE_df["T"].values,label = "CMO EOS",color = "g")
plots.set_xlim([0, 0.8])
plots.set_ylim([0.4, 0.55])
plots.set_xlabel(r'$\mathit{\rho^*}$', style='italic')
plots.set_ylabel(r"$T^*$",style="italic")
plt.savefig("Predicted_VLE.svg")


plt.show()








