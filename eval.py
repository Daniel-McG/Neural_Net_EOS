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
        d2A_dT_drho = torch.reshape(d2A_drho2,(-1,))
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
        d2A_dT_drho = torch.reshape(d2A_drho2,(-1,))

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
        return Cv
    def calculate_d2A_dT2(self,input):
        hessians = self.compute_hessian(input)
        d2A_dT2 = hessians[:, # In all of the hessians in the batch ...
                        :, # In all of the heassians in the batch ...
                        1, # in the second row ...
                        1] # return the value in the second column
        return d2A_dT2
    
    def extract_T_and_rho(self,input):
        rho = input[:,0]
        T = input[:,1]
        return T,rho

    def calculate_Z(self,input):
        T,rho = self.extract_T_and_rho(input)
        dA_drho = self.calculate_dA_drho(input)
        dA_drho = torch.reshape(dA_drho,(-1,))
        P = (rho**2)*dA_drho
        Z = P/(rho*T)
        return Z
    
    def calculate_dA_drho(self,input):
        gradients = self.compute_gradient(input)
        dA_drho = gradients[:,0]
        return dA_drho
    
    def calculate_dA_dT(self,input):
        gradient = self.compute_gradient(input)
        dA_dT = gradient[:,1]
        return dA_dT
    
    def calculate_U(self,input):
        T, rho = self.un
        A = self.forward(input)
        A= torch.reshape(A,(-1,))
        S = self.calculate_S(input)
        U=A+(T*S)
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
    
    
    
    def calculate_alphaP(self,input):
        T, rho = self.extract_T_and_rho(input)
        dP_dT = self.calculate_dP_dT(input)
        dP_drho = self.calcualte_dP_drho(input)
        alphaP = (dP_dT)/(rho*dP_drho)
        return alphaP







    
model = BasicLightning().load_from_checkpoint("/home/daniel/ray_results/TorchTrainer_2023-11-17_13-38-47/TorchTrainer_ff53aa03_1_layer_size=45,lr=0.0000,weight_decay_coefficient=0.0000_2023-11-17_13-38-47/lightning_logs/version_0/checkpoints/epoch=4600-step=4601.ckpt")
model.eval()
data_df = pd.read_csv('coallated_results.txt',delimiter=" ")
# Preprocessing the data

# The data was not MinMax scaled as the gradient and hessian had to be computed wrt the input e.g. temperature , not scaled temperature.
# It may be possible to write the min max sacaling in PyTorch so that the DAG is retained all the way to the input data but im not sure if
# the TensorDataset and DataLoader would destroy the DAG.
# Since the density is already in ~0-1 scale and the temperature is only on a ~0-10 scale, it will be okay.
# Problems would occur if non-simulated experimental data was used as pressures are typically ~ 100kPa and temperatures ~ 298K,
# very far from the typical 0-1 range we want for training a neural network

train_df,test_df = train_test_split(data_df,train_size=0.7)

train_arr = train_df.values
val_arr = test_df.values

# Splitting the preprocessed data into the inputs and targets
density_column = 4
temperature_column = 2
pressure_column = 3
internal_energy_column = 1
cv_column = 20
gammaV_column = cv_column + 1
cp_column = gammaV_column + 1
alphaP_column = cp_column + 1
betaT_column = alphaP_column + 1
mu_jt_column = betaT_column + 1
Z_column = mu_jt_column + 1
target_columns = [pressure_column]
train_inputs = torch.tensor(train_arr[:,[density_column,temperature_column]])
train_targets = torch.tensor(train_arr[:,target_columns])
val_inputs = torch.tensor(val_arr[:,[density_column,temperature_column]])
val_targets = torch.tensor(val_arr[:,target_columns])
train_inputs = train_inputs.float()
train_targets = train_targets.float()
val_inputs = val_inputs.float()
val_targets = val_targets.float()
input = torch.tensor([[0.5,2.0]])
input.requires_grad = True
train_inputs.requires_grad=True
predicted_cv = model.calculate_cv(train_inputs).detach().numpy()
predicted_z = model.calculate_Z(train_inputs).detach().numpy()
target_Z = train_arr[:,Z_column]
target_cv = train_arr[:,cv_column]
sns.regplot(x = predicted_z.flatten(),y=target_Z.flatten())
plt.show()
sns.scatterplot(x = predicted_cv.flatten(),y = target_cv.flatten())
plt.show()


