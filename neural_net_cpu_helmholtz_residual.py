import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn
import lightning as L
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import ray.train.lightning
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
import pickle
import nevergrad as ng
from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search import Repeater
from ray.train.torch import TorchConfig
from ray.train.lightning import (RayDDPStrategy,
                                 RayLightningEnvironment,
                                 RayTrainReportCallback,
                                 prepare_trainer)
from ray.tune import CLIReporter


max_number_of_training_epochs = 2000000


reporter = CLIReporter(max_progress_rows=5)

ray.init(log_to_driver=True)
data_scaling = False
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
#Defining the neural 

class BasicLightning(pl.LightningModule):
    """
    This Neural Network takes the input of density and temperature and predicts the helmholtz free energy.
    """
    def __init__(self,config):
        super(BasicLightning,self).__init__() 
        self.lr = config["lr"]
        self.batch_size = 6000
        self.layer_size = config["layer_size"]
        self.weight_decay_coefficient = config["weight_decay_coefficient"]

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

        Z_loss = ((Z_target-Z_predicted)**2)/var_Z
        U_loss = ((U_target-U_predicted)**2)/var_U
        alphaP_loss = 1/20*((alphaP_target-alphaP_predicted)**2)/var_alphaP
        adiabatic_index_loss = 1/20*((adiabatic_index_target-adiabatic_index_predicted)**2)/var_adiabatic_index
        gammmaV_loss = 1/20*((gammaV_target-gammaV_predicted)**2)/var_gammaV
        cv_loss = 1/20*((cv_target-cv_predicted)**2)/var_cv
        rho_betaT_loss = 1/20*((rho*betaT_target-rho*betaT_predicted)**2)/var_rho_betaT
        mu_jt_loss = 1/20*((mu_jt_target-mu_jt_predicted)**2)/var_mu_jt


        # loss = Z_loss+U_loss+cv_loss+gammmaV_loss+adiabatic_index_loss+alphaP_loss+A*torch.zeros_like(A)
        loss = Z_loss+U_loss+A*torch.zeros_like(A)


        mean_train_loss = torch.mean(loss)
        self.log("val_P_loss",torch.mean((P_predicted-P_target)**2)) 
        self.log("val_cv_loss",torch.mean(((cv_target-cv_predicted)**2)))
        self.log("val_gammaV_loss",torch.mean((gammaV_target-gammaV_predicted)**2))
        self.log("val_rhoBetaT_loss",torch.mean((rho*betaT_target-rho*betaT_predicted)**2))
        self.log("val_alphaP_loss",torch.mean((alphaP_target-alphaP_predicted)**2))
        self.log("val_mu_jt_loss",torch.mean((mu_jt_predicted-mu_jt_target)**2))
        self.log("val_cp_predicted",torch.mean((cp_predicted-cp_target)**2))
        self.log("val_adiabatic_index_loss",torch.mean((adiabatic_index_target-adiabatic_index_predicted)**2))
        self.log("val_U_loss",torch.mean((U_target-U_predicted)**2))
        self.log("val_Z_loss",torch.mean((Z_predicted-Z_target)**2))  
        self.log("train_loss",mean_train_loss)
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


        # gammaV_target = alphaP_target/betaT_target
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

        Z_loss = ((Z_target-Z_predicted)**2)/var_Z
        U_loss = ((U_target-U_predicted)**2)/var_U
        alphaP_loss = 1/20*((alphaP_target-alphaP_predicted)**2)/var_alphaP
        adiabatic_index_loss = 1/20*((adiabatic_index_target-adiabatic_index_predicted)**2)/var_adiabatic_index
        gammmaV_loss = 1/20*((gammaV_target-gammaV_predicted)**2)/var_gammaV
        cv_loss = 1/20*((cv_target-cv_predicted)**2)/var_cv
        rho_betaT_loss = 1/20*((rho*betaT_target-rho*betaT_predicted)**2)/var_rho_betaT
        mu_jt_loss = 1/20*((mu_jt_target-mu_jt_predicted)**2)/var_mu_jt


        # loss = Z_loss+U_loss+cv_loss+gammmaV_loss+adiabatic_index_loss+alphaP_loss+A*torch.zeros_like(A)
        loss = Z_loss+U_loss+A*torch.zeros_like(A)


        
        mean_val_loss = torch.mean(loss)

        self.log("val_loss",mean_val_loss) 
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
        variance = torch.var(torch.reshape(Tensor_to_calculate_variance,(-1,)))
        return variance
                                                                                                                           

def train_func(config):
    # Read data from csv
    data_df = pd.read_csv('/home/daniel/Documents/Research Project/Neural_Net_EOS/coallated_results_debug.txt',delimiter=" ")
    # Preprocessing the data
    data_df = data_df.dropna()
    # The data was not MinMax scaled as the gradient and hessian had to be computed wrt the input e.g. temperature , not scaled temperature.
    # It may be possible to write the min max sacaling in PyTorch so that the DAG is retained all the way to the input data but im not sure if
    # the TensorDataset and DataLoader would destroy the DAG.
    # Since the density is already in ~0-1 scale and the temperature is only on a ~0-10 scale, it will be okay.
    # Problems would occur if non-simulated experimental data was used as pressures are typically ~ 100kPa and temperatures ~ 298K,
    # very far from the typical 0-1 range we want for training a neural network

    train_df,test_df = train_test_split(data_df,train_size=0.7)

    train_arr = train_df.values
    val_arr = test_df.values
    np.savetxt("training_data_for_current_ANN.txt",train_arr)
    np.savetxt("validation_data_for_current_ANN.txt",val_arr)
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
    target_columns = [cv_column,gammaV_column,cp_column,alphaP_column,betaT_column,internal_energy_column,pressure_column,mu_jt_column,Z_column]
    train_inputs = torch.tensor(train_arr[:,[density_column,temperature_column]])
    train_targets = torch.tensor(train_arr[:,target_columns])
    val_inputs = torch.tensor(val_arr[:,[density_column,temperature_column]])
    val_targets = torch.tensor(val_arr[:,target_columns])
    train_inputs = train_inputs.double()
    train_targets = train_targets.double()
    val_inputs = val_inputs.double()
    val_targets = val_targets.double()

    # Loading inputs and targets into the dataloaders
    train_dataset = TensorDataset(train_inputs,train_targets)
    val_Dataset = TensorDataset(val_inputs,val_targets)
    train_dataloader = DataLoader(train_dataset,batch_size = 256)
    val_dataloader = DataLoader(val_Dataset,batch_size = 256)

    # Instantiating the neural network
    model = BasicLightning(config)
    model = model.double()
    trainer = pl.Trainer(
        # Define the max number of epochs for the trainer, this is also enforced by the scheduler.
        max_epochs=max_number_of_training_epochs,

        # Use GPU if available
        devices="auto",
        accelerator="auto",

        # Selects distributed data paralell training strategy. URL explaining DDP : https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#distributed-data-parallel
        # DDP is described as being used for "Strategy for multi-process single-device training on one or multiple nodes". URL: https://lightning.ai/docs/pytorch/stable/extensions/strategy.html
        strategy=RayDDPStrategy(),

        # End of epoch reporting to ray train for metrics and to pytorch lightning for early stopping
        callbacks=[
                   RayTrainReportCallback(),

                   # Monitor the validation loss and if its not decreasing for more than 500 epochs, terminate the training.
                   EarlyStopping(monitor="val_loss",mode="min",patience=500000)
                   ],
        plugins=[RayLightningEnvironment()],

        # Don't print progress bar to terminal when training.
        enable_progress_bar=False,
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
                )

scaling_config = ScalingConfig(num_workers=1, use_gpu=False,resources_per_worker={"CPU":11})

run_config = RunConfig(progress_reporter=reporter,
                       checkpoint_config=CheckpointConfig(
                                                          num_to_keep=2,
                                                          checkpoint_score_attribute="val_loss",
                                                          checkpoint_score_order="min"
                                                          ),
                        )

trainer = TorchTrainer(
                       train_func, 
                       scaling_config=scaling_config,
                       run_config=run_config
                       )



def tune_asha(num_samples,max_number_of_training_epochs):

    lower_limit_of_neurons_per_layer = 45
    upper_limit_of_neurons_per_layer = 47

    # Create distribution of integer values for the number of neurons per layer
    layer_size_dist = tune.randint(lower_limit_of_neurons_per_layer,upper_limit_of_neurons_per_layer)
    
    # Create search space dict
    search_space = {
                    "layer_size":layer_size_dist,
                    "lr": tune.loguniform(1e-4, 1.1e-4),
                    "weight_decay_coefficient":tune.uniform(1e-6,1.1e-6)
                    }

    # Use Asynchronus Successive Halving to schedule concurrent trails. Paper url = {https://proceedings.mlsys.org/paper_files/paper/2020/file/a06f20b349c6cf09a6b171c71b88bbfc-Paper.pdf}
    scheduler = ASHAScheduler(max_t= max_number_of_training_epochs, grace_period=1000000, reduction_factor=2)

    # Use Particle swarm optimisation for hyperparameter tuning from the Nevergrad package
    algo = NevergradSearch(optimizer=ng.optimizers.PSO,
                           metric="val_loss",
                           mode="min",
                           )

    # Instantiate the Tuner
    tuner = tune.Tuner(
                        trainer,
                        param_space={"train_loop_config": search_space},
                        tune_config=tune.TuneConfig(
                                                    metric="val_loss",
                                                    mode="min",
                                                    search_alg=algo,
                                                    num_samples=num_samples,
                                                    scheduler=scheduler
                                                    )
                        )
    return tuner.fit()


# Define the number of tuning experiments to run
num_samples = 1000

results = tune_asha(num_samples,max_number_of_training_epochs)
results.get_best_result(metric="val_loss", mode="min")
