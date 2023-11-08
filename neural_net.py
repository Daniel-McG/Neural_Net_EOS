import torch
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


max_number_of_training_epochs = 20000


reporter = CLIReporter(max_progress_rows=5)

ray.init(log_to_driver=False)
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
          nn.Linear(self.layer_size ,self.layer_size),
          nn.Tanh(),
          nn.Linear(self.layer_size ,self.layer_size),
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
        optimiser = torch.optim.AdamW(self.parameters(),lr = self.lr,weight_decay=self.weight_decay_coefficient)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser,500)
        current_lr = scheduler.get_last_lr()
        print(current_lr)
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

        S = -dA_dT
        P_predicted = (rho**2)*dA_drho
        U_predicted = A+T*S
        Z = (rho*dA_drho)/T
        cv_predicted = -T*d2A_dT2
        dP_dT = (rho**2)*d2A_dT_drho
        dP_drho = 2*rho*dA_drho + (rho**2)*d2A_drho2
        alphaP_predicted = (dP_dT/rho)/dP_drho
        rho_betaT = 1/dP_drho
        betaT_predicted = rho_betaT/rho
        gammaV_predicted = alphaP_predicted/betaT_predicted
        cp_predicted = cv_predicted + (T*(alphaP_predicted**2))/(betaT_predicted*rho)

        # Calculates the loss
        loss = A*torch.zeros_like(A) + ((P_predicted-P_target)/P_target)**2 + ((cv_target-cv_predicted)/cv_target)**2 + ((gammaV_target-gammaV_predicted)/gammaV_target)**2 + ((U_target-U_predicted)/U_target)**2  + ((alphaP_target - alphaP_predicted)/alphaP_predicted)**2 #+ (betaT_predicted-betaT_target)**2 +(cp_target-cp_predicted)**2       
        mean_train_loss = torch.mean(loss)

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
        U_predicted = A+T*S
        Z = (rho*dA_drho)/T
        cv_predicted = -T*d2A_dT2
        dP_dT = (rho**2)*d2A_dT_drho
        dP_drho = 2*rho*dA_drho + (rho**2)*d2A_drho2
        alphaP_predicted = (dP_dT/rho)/dP_drho
        rho_betaT = 1/dP_drho
        betaT_predicted = rho_betaT/rho
        gammaV_predicted = alphaP_predicted/betaT_predicted
        cp_predicted = cv_predicted + (T*(alphaP_predicted**2))/(betaT_predicted*rho)

        # Calculates the loss

        loss = A*torch.zeros_like(A) + ((P_predicted-P_target)/P_target)**2 + ((cv_target-cv_predicted)/cv_target)**2 + ((gammaV_target-gammaV_predicted)/gammaV_target)**2 + ((U_target-U_predicted)/U_target)**2  + ((alphaP_target - alphaP_predicted)/alphaP_predicted)**2 #+ (betaT_predicted-betaT_target)**2 +(cp_target-cp_predicted)**2   
        mean_val_loss = torch.mean(loss)
        self.log("val_P_loss",torch.mean((P_predicted-P_target)**2)) 
        self.log("val_cv_loss",torch.mean(((cv_target-cv_predicted)**2)))
        self.log("val_gammaV_loss",torch.mean((gammaV_target-gammaV_predicted)**2))
        self.log("val_U_loss",torch.mean((U_target-U_predicted)**2))  
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
    
    def cv_from_ann(temperature,d2A_drho2):
        -temperature*d2A_drho2
                                                                                                                           

def train_func(config):
    # Read data from csv
    data_df = pd.read_csv('/home/daniel/Documents/Research Project/Neural_Net_EOS/coallated_results.txt',delimiter=" ")
    # Preprocessing the data

    # The data was not MinMax scaled as the gradient and hessian had to be computed wrt the input e.g. temperature , not scaled temperature.
    # It may be possible to write the min max sacaling in PyTorch so that the DAG is retained all the way to the input data but im not sure if
    # the TensorDataset and DataLoader would destroy the DAG.
    # Since the density is already in ~0-1 scale and the temperature is only on a ~0-10 scale, it will be okay.
    # Problems would occur if non-simulated experimental data was used as pressures are typically ~ 100kPa and temperatures ~ 298K,
    # very far from the typical 0-1 range we want for training a neural network

    train_df,test_df = train_test_split(data_df,train_size=0.6)

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
    target_columns = [cv_column,gammaV_column,cp_column,alphaP_column,betaT_column,internal_energy_column,pressure_column]
    train_inputs = torch.tensor(train_arr[:,[density_column,temperature_column]])
    train_targets = torch.tensor(train_arr[:,target_columns])
    val_inputs = torch.tensor(val_arr[:,[density_column,temperature_column]])
    val_targets = torch.tensor(val_arr[:,target_columns])
    train_inputs = train_inputs.float()
    train_targets = train_targets.float()
    val_inputs = val_inputs.float()
    val_targets = val_targets.float()

    # Loading inputs and targets into the dataloaders
    train_dataset = TensorDataset(train_inputs,train_targets)
    val_Dataset = TensorDataset(val_inputs,val_targets)
    train_dataloader = DataLoader(train_dataset,batch_size = 6000)
    val_dataloader = DataLoader(val_Dataset,batch_size = 6000)

    # Instantiating the neural network
    model = BasicLightning(config)

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
                   EarlyStopping(monitor="val_loss",mode="min",patience=500)
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

scaling_config = ScalingConfig(num_workers=1, use_gpu=True)

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

    lower_limit_of_neurons_per_layer = 200
    upper_limit_of_neurons_per_layer = 1000

    # Create distribution of integer values for the number of neurons per layer
    layer_size_dist = tune.randint(lower_limit_of_neurons_per_layer,upper_limit_of_neurons_per_layer)
    
    # Create search space dict
    search_space = {
                    "layer_size":layer_size_dist,
                    "lr": tune.loguniform(1e-5, 1e-3),
                    "weight_decay_coefficient":tune.uniform(1e-6,1e-2)
                    }

    # Use Asynchronus Successive Halving to schedule concurrent trails. Paper url = {https://proceedings.mlsys.org/paper_files/paper/2020/file/a06f20b349c6cf09a6b171c71b88bbfc-Paper.pdf}
    scheduler = ASHAScheduler(max_t= max_number_of_training_epochs, grace_period=1000, reduction_factor=2)

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
num_samples = 100000

results = tune_asha(num_samples,max_number_of_training_epochs)
results.get_best_result(metric="val_loss", mode="min")