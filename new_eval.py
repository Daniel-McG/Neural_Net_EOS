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
        self.layer_size = 46

        # Creating a sequential stack of Linear layers all of the same width with Tanh activation function 
        self.layers_stack = nn.Sequential(
          nn.Linear(2,self.layer_size),
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
        optimiser = torch.optim.Adam(self.parameters(),lr = self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser,500)
        output_dict = {
        "optimizer": optimiser,
        "lr_scheduler": {"scheduler":scheduler}
        }
        return output_dict
    
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
        adiabatic_index_target = cp_target/cv_target
        var_cv = self.calculate_variance(cv_target)
        var_Z = self.calculate_variance(Z_target)
        var_U = self.calculate_variance(U_target)
        var_gammaV = self.calculate_variance(gammaV_target)
        var_rho_betaT = self.calculate_variance(rho*betaT_target)
        var_alphaP = self.calculate_variance(alphaP_target)
        var_adiabatic_index = self.calculate_variance(adiabatic_index_target)
        var_mu_jt = self.calculate_variance(mu_jt_target)

        # Ensures that the DAG is created for the input so that the gradient and hessian can be computed 
        train_input_i.requires_grad = True

        # Pass input through NN to get the output
        A = self.forward(train_input_i)

        # Computes gradient and hessian
        train_gradient = self.compute_gradient(train_input_i)
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
        # print(T,rho,A,dA_dT,dA_drho,d2A_drho2,d2A_dT2,d2A_drho2,d2A_dT_drho)
        S = -dA_dT
        P_predicted = (rho**2)*dA_drho
        U_predicted = A+(T*S)
        Z = (rho*dA_drho)/T
        cv_predicted = -T*d2A_dT2
        dP_dT = (rho**2)*d2A_dT_drho
        dP_drho = 2*rho*dA_drho + (rho**2)*d2A_drho2
        alphaP_predicted = (dP_dT)/(rho*dP_drho)
        rho_betaT_predicted = 1/dP_drho
        betaT_predicted = torch.reciprocal(rho*dP_drho)
        gammaV_predicted = alphaP_predicted/betaT_predicted
        cp_predicted = cv_predicted + (T/rho)*((alphaP_predicted**2)/betaT_predicted)
        mu_jt_predicted = (1/(rho*cp_predicted))*((T*alphaP_predicted)-1)
        Z_predicted = P_predicted/(rho*T)
        adiabatic_index_predicted = cp_predicted/cv_predicted

        
        # Calculates the loss

        loss = A*torch.zeros_like(A) \
            + ((Z_target-Z_predicted)**2)/var_Z + ((U_target-U_predicted)**2)/var_U \
            + 1/10*((cv_target-cv_predicted)**2)/var_cv \
            + 1/10*((gammaV_target-gammaV_predicted)**2)/var_gammaV \
            + 1/10*((alphaP_target-alphaP_predicted)**2)/var_alphaP \
        
        mean_train_loss = torch.mean(loss)

        self.log("train_loss",mean_train_loss,sync_dist=True)
        return {"loss": mean_train_loss}
    
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
        adiabatic_index_target = cp_target/cv_target


        var_cv = self.calculate_variance(cv_target)
        var_Z = self.calculate_variance(Z_target)
        var_U = self.calculate_variance(U_target)
        var_gammaV = self.calculate_variance(gammaV_target)
        var_rho_betaT = self.calculate_variance(rho*betaT_target)
        var_alphaP = self.calculate_variance(alphaP_target)
        var_adiabatic_index = self.calculate_variance(adiabatic_index_target)
        var_mu_jt = self.calculate_variance(mu_jt_target)

        # Ensures that the DAG is created for the input so that the gradient and hessian can be computed 
        val_input_i.requires_grad = True

        # Pass input through NN to get the output
        A = self.forward(val_input_i)

        # Computes gradient and hessian
        val_gradient = self.compute_gradient(val_input_i)
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
        Z = (rho*dA_drho)/T
        cv_predicted = -T*d2A_dT2
        dP_dT = (rho**2)*d2A_dT_drho
        dP_drho = 2*rho*dA_drho + (rho**2)*d2A_drho2
        alphaP_predicted = (dP_dT)/(rho*dP_drho)
        rho_betaT_predicted = 1/dP_drho
        betaT_predicted = torch.reciprocal(rho*dP_drho)
        gammaV_predicted = alphaP_predicted/betaT_predicted
        cp_predicted = cv_predicted + (T/rho)*((alphaP_predicted**2)/betaT_predicted)
        mu_jt_predicted = (1/(rho*cp_predicted))*((T*alphaP_predicted)-1)
        Z_predicted = P_predicted/(rho*T)
        adiabatic_index_predicted = cp_predicted/cv_predicted
        # Calculates the loss
        loss = A*torch.zeros_like(A) \
            + ((Z_target-Z_predicted)**2)/var_Z \
            + ((U_target-U_predicted)**2)/var_U \
            + 1/10*((cv_target-cv_predicted)**2)/var_cv \
            + 1/10*((gammaV_target-gammaV_predicted)**2)/var_gammaV \
            + 1/10*((alphaP_target-alphaP_predicted)**2)/var_alphaP \
        
        mean_val_loss = torch.mean(loss)
        self.log("val_P_loss",torch.mean((P_predicted-P_target)**2),sync_dist=True) 
        self.log("val_cv_loss",torch.mean(((cv_target-cv_predicted)**2)),sync_dist=True)
        self.log("val_gammaV_loss",torch.mean((gammaV_target-gammaV_predicted)**2),sync_dist=True)
        self.log("val_U_loss",torch.mean((U_target-U_predicted)**2),sync_dist=True)  
        self.log("val_loss",mean_val_loss,sync_dist=True) 
        return {"val_loss": mean_val_loss}
    
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

    def compute_gradient(self,inputs):
        # Compute the gradient of the output of the forward pass wrt the input, grad_outputs is d(forward)/d(forward) which is 1 , See https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
        gradient = torch.autograd.grad(self.forward(inputs),
                                       inputs,
                                       grad_outputs=torch.ones_like(self.forward(inputs)),
                                       retain_graph=True,
                                       create_graph=True
                                       )[0]
        return gradient
    
    def compute_hessian(self, x):
        
        # Compute the hessian of the output wrt the input
        hessians = torch.vmap(torch.func.hessian(self.forward), (0))(x)
        return hessians
    
    def calculate_variance(self, Tensor_to_calculate_variance):
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
        gradients = self.compute_gradient(input)
        dA_drho = gradients[:,0]
        return dA_drho
    
    def calculate_dA_dT(self,input):
        gradient = self.compute_gradient(input)
        dA_dT = gradient[:,1]
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


path_to_training_data = r"C:\Users\Daniel.000\Documents\New_research_project\Neural_Net_EOS\models\TrainedNaNs_LowLR\training_data_for_current_ANN.txt" 
path_to_validation_data = r"C:\Users\Daniel.000\Documents\New_research_project\Neural_Net_EOS\models\TrainedNaNs_LowLR\validation_data_for_current_ANN.txt"
model = BasicLightning.load_from_checkpoint(r"C:\Users\Daniel.000\Documents\New_research_project\Neural_Net_EOS\models\TrainedNaNs_LowLR\lightning_logs\version_0\checkpoints\epoch=18509-step=462750.ckpt")
model = model.double()
# model.eval()
# data_df = pd.read_csv('cleaned_coallated_results.txt',delimiter=" ")
# Preprocessing the data

# The data was not MinMax scaled as the gradient and hessian had to be computed wrt the input e.g. temperature , not scaled temperature.
# It may be possible to write the min max sacaling in PyTorch so that the DAG is retained all the way to the input data but im not sure if
# the TensorDataset and DataLoader would destroy the DAG.
# Since the density is already in ~0-1 scale and the temperature is only on a ~0-10 scale, it will be okay.
# Problems would occur if non-simulated experimental data was used as pressures are typically ~ 100kPa and temperatures ~ 298K,
# very far from the typical 0-1 range we want for training a neural network

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

# def zeno_curve(density,temperature):
#     density.requires_grad=True
#     temperature.requires_grad = True
#     zero_densities = torch.zeros_like(density)
#     zero_densities.requires_grad = True
#     ideal_input = torch.stack((zero_densities,temperature),dim=-1)
#     ideal_input = ideal_input.double()
#     A_ideal = model.forward(ideal_input)
#     A_ideal = A_ideal/temperature
#     dAideal_drho = torch.autograd.grad(A_ideal,zero_densities,grad_outputs=torch.ones_like(A_ideal),create_graph=True)[0]
#     A_zeno = density*dAideal_drho
#     dA_zeno_drho = torch.autograd.grad(A_zeno,density,grad_outputs=torch.ones_like(A_zeno),create_graph=True)[0]
#     P_by_rho = density*dA_zeno_drho
#     P = P_by_rho*density
#     return P

# r = torch.tensor([0.3,0.5])
# T = torch.tensor([1.0,2.0])
# zeno_curve(r,T)


# def amagat_curve(T):
#     def find_dZ_dT(rho,T):
#         temperature = torch.tensor([T])
#         temperature = temperature.double()
#         temperature.requires_grad = True
#         rho = torch.tensor(rho)
#         rho = rho.double()
#         input_at_constant_density = torch.stack((rho,temperature),dim=-1)
#         input_at_constant_density = input_at_constant_density.double()
#         model.calculate_Z(input_at_constant_density)
#         dZ_dT= torch.autograd.grad(model.calculate_Z(input_at_constant_density),input_at_constant_density, grad_outputs=torch.ones_like(model.calculate_Z(input_at_constant_density)), create_graph=True)[0][:,0].detach().numpy()
#         return dZ_dT.item()

#     rho_at_dZ_dT_is__zero_for_T = scipy.optimize.root(find_dZ_dT,args=(T),x0=0.5).x

#     T = torch.tensor([T])
#     T = T.double()
#     T.requires_grad = True
#     rho_at_dZ_dT_is__zero_for_T = torch.tensor(rho_at_dZ_dT_is__zero_for_T)
#     rho_at_dZ_dT_is__zero_for_T.requires_grad = True
#     input_at_dZ_dT_is_zero= torch.stack((rho_at_dZ_dT_is__zero_for_T,T),dim=-1)
#     pressure = model.calculate_P(input_at_dZ_dT_is_zero)
#     return pressure.detach().numpy(),rho_at_dZ_dT_is__zero_for_T



# number_of_points_to_evaluate = 100
# amagat_pressure_list = []
# amagat_density_list = []
# amagat_temperature_range = np.linspace(0.01,1.5,number_of_points_to_evaluate)
# for temperature in amagat_temperature_range:
#     pressure, density =amagat_curve(temperature)
#     amagat_pressure_list.append(pressure)
#     amagat_density_list.append(density)
# sns.scatterplot(x=amagat_temperature_range.flatten(),y = np.array(amagat_pressure_list).flatten())
# plt.xlim(0.4,1.5)
# plt.ylim(0,0.21)
# # plt.show()

# zeno_density_ranage = torch.tensor(amagat_density_list)
# zeno_temperature_range = torch.linspace(0.1,1.5,number_of_points_to_evaluate)
# zeno_inputs = torch.stack((zeno_density_ranage,zeno_temperature_range),dim=-1)
# zeno_inputs = zeno_inputs.double()
# zeno_inputs.requires_grad = True
# zeno_pressure = model.calculate_P(zeno_inputs)
# sns.scatterplot(x=zeno_temperature_range.detach().numpy().flatten(),y = zeno_pressure.detach().numpy().flatten())
# plt.xlim(0.4,1.5)
# plt.ylim(0,0.21)
# plt.show()


# sns.scatterplot(y = torch.autograd.grad(model.calculate_Z(input_at_constant_density),input_at_constant_density, grad_outputs=torch.ones_like(model.calculate_Z(input_at_constant_density)), create_graph=True)[0][:,1].detach().numpy(),x = temperature.detach().numpy())
# plt.show()
# model.calculate_P()
# target_adiabatic_index = val_arr[:,cp_column]/val_arr[:,cv_column]

# predicted_cv = model.calculate_cv(train_inputs).detach().numpy()
# predicted_z = model.calculate_Z(train_inputs).detach().numpy()
# predicted_U = model.calculate_U(train_inputs).detach().numpy()
# predicted_P = model.calculate_P(train_inputs).detach().numpy()
# predicted_alphaP = model.calculate_alphaP(train_inputs).detach().numpy()
# predicted_betaT = model.calculate_betaT(train_inputs).detach().numpy()
# predicted_mujt = model.calculate_mu_jt(train_inputs).detach().numpy()
# predicted_gammaV = model.calculate_gammaV(train_inputs).detach().numpy()
# predicted_cp = model.calculate_cp(train_inputs).detach().numpy()
# predicted_adiabatic_index = model.calculate_adiabatic_index(train_inputs).detach().numpy()
# target_Z = train_arr[:,Z_column]
# target_cv = train_arr[:,cv_column]
# target_U = train_arr[:,internal_energy_column]
# target_P = train_arr[:,pressure_column]
# target_alphaP = train_arr[:,alphaP_column]
# target_betaT = train_arr[:,betaT_column]
# target_mujt = train_arr[:,mu_jt_column]
# target_gammaV = train_arr[:,gammaV_column]
# target_cp = train_arr[:,cp_column]
# target_adiabatic_index = train_arr[:,cp_column]/train_arr[:,cv_column]
# sns.set_style("ticks")
# sns.lineplot(x =[0,8],y=[0,8])
# sns.scatterplot(x = predicted_z.flatten(),y=target_Z.flatten())
# plt.xlabel('Predicted_Z')
# plt.ylabel('Target_Z')
# plt.title('Z_parity')
# plt.show()

# sns.set_style("ticks")
# sns.scatterplot(x = predicted_U.flatten(),y = target_U.flatten())
# sns.lineplot(x =[0,14],y=[0,14])
# plt.xlabel('Predicted_U')
# plt.ylabel('Target_U')
# plt.title('U_parity')
# plt.show()

# sns.set_style("ticks")
# sns.scatterplot(x = predicted_P.flatten(),y = target_P.flatten())
# sns.lineplot(x =[0,20],y=[0,20])
# plt.xlabel('Predicted_P')
# plt.ylabel('Target_P')
# plt.title('P_parity')
# plt.show()

# sns.set_style("ticks")
# sns.scatterplot(x=predicted_cv.flatten(),y=target_cv.flatten())
# plt.xlabel('Predicted_cv')
# plt.ylabel('Target_cv')
# plt.title('cv_parity')
# sns.lineplot(x =[0,10],y=[0,10])
# plt.show()

# sns.set_style("ticks")
# sns.scatterplot(x=val_arr[:,density_column]*predicted_betaT.flatten(),y=val_arr[:,density_column]*target_betaT.flatten())
# sns.lineplot(x=[0,20],y=[0,20])
# plt.xlabel('Predicted_Rho*betaT')
# plt.ylabel('Target_Rho*betaT')
# plt.title('Rho*betaT_parity')
# plt.xlim(0,20)
# plt.ylim(0,20)
# plt.show()

# sns.set_style("ticks")
# sns.scatterplot(x=predicted_alphaP.flatten(),y=target_alphaP.flatten())
# plt.xlabel('Predicted_alphaP')
# plt.ylabel('Target_alphaP')
# plt.title('alpahP_parity')
# sns.lineplot(x =[0,10],y=[0,10])
# plt.show()

# sns.set_style("ticks")
# sns.scatterplot(x=predicted_mujt.flatten(),y=target_mujt.flatten())
# plt.xlabel('Predicted_mujt')
# plt.ylabel('Target_mujt')
# plt.title('mujt_parity')
# sns.lineplot(x =[-100,0],y=[-100,0])
# plt.xlim(-100,0)
# plt.ylim(-100,0)
# plt.show()


sns.set_style("ticks")
fig, axs = plt.subplots(2, 5, figsize=(15, 10),constrained_layout=True)

sns.lineplot(x =[0,8],y=[0,8],ax = axs[0, 0])
sns.scatterplot(x = predicted_z.flatten(),y=target_Z.flatten(), ax = axs[0, 0])
axs[0, 0].set_xlabel('Predicted_Z')
axs[0, 0].set_ylabel('Target_Z')
axs[0, 0].set_title('Z_parity')

sns.scatterplot(x = predicted_U.flatten(),y = target_U.flatten(), ax = axs[0, 1])
sns.lineplot(x =[0,14],y=[0,14], ax = axs[0, 1])
axs[0, 1].set_xlabel('Predicted_U')
axs[0, 1].set_ylabel('Target_U')
axs[0, 1].set_title('U_parity')

sns.scatterplot(x = predicted_P.flatten(),y = target_P.flatten(), ax = axs[0, 2])
sns.lineplot(x =[0,20],y=[0,20], ax = axs[0, 2])
axs[0, 2].set_xlabel('Predicted_P')
axs[0, 2].set_ylabel('Target_P')
axs[0, 2].set_title('P_parity')

sns.scatterplot(x=predicted_cv.flatten(),y=target_cv.flatten(), ax = axs[0, 3])
axs[0, 3].set_xlabel('Predicted_cv')
axs[0, 3].set_ylabel('Target_cv')
axs[0, 3].set_title('cv_parity')
sns.lineplot(x =[0,4],y=[0,4], ax = axs[0, 3])

sns.scatterplot(x=val_arr[:,density_column][~np.isnan(val_arr[:,betaT_column])]*predicted_betaT.flatten(),y=val_arr[:,density_column][~np.isnan(val_arr[:,betaT_column])]*target_betaT.flatten(), ax = axs[1, 0])
sns.lineplot(x=[0,20],y=[0,20], ax = axs[1, 0])
axs[1, 0].set_xlabel('Predicted_Rho*betaT')
axs[1, 0].set_ylabel('Target_Rho*betaT')
axs[1, 0].set_title('Rho*betaT_parity')
# axs[1, 0].set_xlim(0,20)
# axs[1, 0].set_ylim(0,20)

sns.scatterplot(x=predicted_cp.flatten(),y=target_cp.flatten(), ax = axs[0, 4])
axs[0, 4].set_xlabel('predicted_Cp')
axs[0, 4].set_ylabel('target_Cp')
axs[0, 4].set_title('Cp_parity')
sns.lineplot(x =[0,10],y=[0,10], ax = axs[0, 4])

sns.scatterplot(x=predicted_alphaP.flatten(),y=target_alphaP.flatten(), ax = axs[1, 2])
axs[1, 2].set_xlabel('Predicted_alphaP')
axs[1, 2].set_ylabel('Target_alphaP')
axs[1, 2].set_title('alphaP_parity')
sns.lineplot(x =[0,10],y=[0,10], ax = axs[1, 2])

sns.scatterplot(x=predicted_gammaV.flatten(),y=target_gammaV.flatten(), ax = axs[1, 1])
axs[1, 1].set_xlabel('Predicted_gammaV')
axs[1, 1].set_ylabel('Target_gammaV')
axs[1, 1].set_title('gammaV_parity')
sns.lineplot(x =[0,10],y=[0,10], ax = axs[1, 1])

sns.scatterplot(x=predicted_mujt.flatten(),y=target_mujt.flatten(), ax = axs[1, 3])
axs[1, 3].set_xlabel('predicted_mujt')
axs[1, 3].set_ylabel('target_mujt')
axs[1, 3].set_title('mujt_parity')
sns.lineplot(x =[0,10],y=[0,10], ax = axs[1, 3])

# sns.scatterplot(x=predicted_adiabatic_index.flatten(),y=target_adiabatic_index.flatten(), ax = axs[1, 4])
# axs[1, 4].set_xlabel('predicted_Cp/Cv')
# axs[1, 4].set_ylabel('target_Cp/Cv')
# axs[1, 4].set_title('Cp/Cv_parity')
# sns.lineplot(x =[0,10],y=[0,10], ax = axs[1, 4])

# plt.show()

# def zeno_curve(density,temperature):
#     """
#     Calculates pressure from density and temperature for zeno curve

#     Z = 1 for Z = P/(Rho*T)
#     """
#     rho = density
#     T = temperature
#     P = rho*T
#     return P


# ##########
# # Isotherm
# ##########
# def find_closest(array, value, n=100):
#     array = np.asarray(array)
#     idx = np.argsort(np.abs(array - value))[:n]
#     return idx
# temperature = 0.4927
# index_of_points_close_to_temp = find_closest(train_arr[:,temperature_column],temperature,50)

# isotherm = train_arr[index_of_points_close_to_temp,:]

# P_isotherm_MD = isotherm[:,pressure_column]
# density_MD = isotherm[:,density_column]


# # Create a numpy array with values from 0 to 1
# density = torch.linspace(0,0.8, 1000)

# # Create a numpy array with a constant value
# temperature = torch.full((1000,), temperature)

# # Concatenate the two numpy arrays horizontally
# tensor = torch.stack((density, temperature), dim=1).double()
# tensor.requires_grad=True
# P_isotherm = model.calculate_P(tensor)
# sns.lineplot(x=density.numpy().flatten(),y=P_isotherm.detach().numpy().flatten(),label="ANN")
# sns.scatterplot(x =density_MD.flatten(),y=P_isotherm_MD.flatten(),label = "MD" )
# plt.xlabel('Density')
# plt.ylabel('Reduced Pressure')
# plt.show()

###################
# Helmholtz surface
####################
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Generate random data
# x = val_arr[:,density_column].flatten()
# y = val_arr[:,temperature_column].flatten()
# z = model.forward(val_inputs).detach().numpy().flatten()

# # Create a 3D scatter plot
# ax.scatter(x, y, z, c=z)

# # Set labels and title
# ax.set_xlabel('Density')
# ax.set_ylabel('Temperature')
# ax.set_zlabel('Helmholtz Free Energy')
# plt.title('Helmholtz Free Energy Surface')

# plt.show()

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

# number of points to compute the phase envelope
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
semi_final_fig, plots = plt.subplots()
# plots.plot(gas_density_range,vapour_curve(gas_density_range,*vapour_curve_parameters),'b')
# plots.plot(liquid_density_range,liquid_curve(liquid_density_range,*liquid_curve_parameters),'b')
# plots.plot(solid_density_range,solid_curve(solid_density_range,*solid_curve_parameters),'b')

# sns.lineplot(x = gas_density_range,y = vapour_curve(gas_density_range,*vapour_curve_parameters),color = '#1f77b4')
# sns.lineplot(x = liquid_density_range[:-5],y = liquid_curve(liquid_density_range[:-5],*liquid_curve_parameters),color = '#1f77b4')
# sns.lineplot(x = solid_density_range[500:-5],y = solid_curve(solid_density_range[500:-5],*solid_curve_parameters),color = '#1f77b4')
plots.scatter(x = vle_densities,y = vle_temperatures,facecolors='none',edgecolors='r', marker='^')
plots.scatter(density_f,T_sf,facecolors='none',edgecolors='r', marker='^')
# plots.scatter(density_triple_point,temperature_triple_point,facecolors='none',edgecolors='g', marker='^')
plt.title("VLSE curve with datapoints")
plt.xlabel("Reduced Density")
plt.ylabel("Reduced Temperature")


plots.plot(rhov, T, color='k')
plots.plot(rhol, T, color='k')
plots.plot(rhoc_model, Tc_model, '*',color='k')
plots.set_xlim([0, 1.0])
plots.set_ylim([0.1, 0.6])
plots.set_xlabel(r"$\rho$")
plots.set_ylabel(r"$T$")

plt.show()
# Computing saturation pressure
# pressure_l = model.pressure(alpha_n, rhol, T) + rhol * T
# pressure_v = model.pressure(alpha_n, rhov, T) +rhov * T
# Computing vaporisation enthalpy 
# enthalpy_l =  model.enthalpy(alpha_n, rhol, T)
# enthalpy_v =  model.enthalpy(alpha_n, rhov, T)
# hvap = enthalpy_v - enthalpy_l
################
# Phase Envelope
################
# num_points = 1000000
# density = torch.linspace(0,1,num_points)
# temperature = torch.full((num_points,), 0.5115)

# input_tensor = torch.stack((density, temperature), dim=1).double()
# input_tensor.requires_grad=True
# print(input_tensor)
# chemical_potential = model.calculate_mu(input_tensor)
# pressure = model.calculate_P(input_tensor)



# diff_chemcial_potential = torch.abs(chemical_potential[:len(chemical_potential)//2] - torch.flip(chemical_potential[len(chemical_potential)//2:],dims=(0,)))

# for value in diff_chemcial_potential[diff_chemcial_potential<1e-2].flatten():
#     mask = diff_chemcial_potential == value
#     index_diff_zero = mask.nonzero().flatten()

#     if (torch.abs(pressure[index_diff_zero]-pressure[-index_diff_zero-1])<1e-3):
#         if pressure[index_diff_zero]>0:
#             print("success")
#             print((pressure[index_diff_zero],pressure[-index_diff_zero-1]))
#             print((chemical_potential[index_diff_zero],chemical_potential[-index_diff_zero-1]))
#             print(density[index_diff_zero],density[-index_diff_zero-1])

# plt.show()