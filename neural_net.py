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
        return torch.optim.AdamW(self.parameters(),lr = self.lr,weight_decay=self.weight_decay_coefficient)
    
    def training_step(self,train_batch,batch_index):
        '''
        Performs a training step.

        Unpacks the batch, passes the batch through the NN, calculates the loss, and logs the loss.
        '''

        # Unpacks trauning batch
        input_i,target_i = train_batch
        temperature = input_i[:,0]
        density = input_i[:,1]
        cv_target = target_i[:,0]
        # print(cv_target)  
        # temperature = input_i[:,0]
        # Ensures that the DAG is created for the input so that the gradient and hessian can be computed 
        input_i.requires_grad = True

        # Passes input through the neural net
        output_i = self.forward(input_i)

        # Computes gradient and hessian
        train_gradient = self.compute_gradient(input_i)
        train_hessian = self.compute_hessian(input_i)

        # The train hessian is returned as a 4D tensor, it can be thought of a batch_size*1 matrix of 2*2 matricies
        # print("start")
        # print(train_hessian[:,:])

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
        temperature = torch.reshape(temperature,(-1,))
        d2A_drho2 = torch.reshape(d2A_drho2,(-1,))

        cv_predicted = -temperature*d2A_drho2


        # print("first {}".format(train_hessian[:,0]))
        # print("second {}".format(train_hessian[:,0,0]))
        # print("thrid {}".format(train_hessian[:,0,0,0]))
        # Calculates loss
        
        loss = (cv_target-cv_predicted)**2 + output_i*torch.zeros_like(output_i)      
        mean_train_loss = torch.mean(loss)

        self.log("train_loss",mean_train_loss)
        return {"loss": mean_train_loss}
    
    def validation_step(self, val_batch, batch_idx):

        # Unpack validation batch
        val_input_i, val_target_i = val_batch

        # Ensures that the DAG is created for the input so that the gradient and hessian can be computed 
        val_input_i.requires_grad = True

        # Pass input through NN to get the output
        val_output_i = self.forward(val_input_i)

        # Computes gradient and hessian
        # val_gradient = self.compute_gradient(val_input_i)
        # val_hessian = self.compute_hessian(val_input_i)

        # Calculates the loss
        loss = (val_output_i-val_target_i)**2
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

    def compute_gradient(self,inputs):
        # Compute the gradient of the output of the forward pass wrt the input, grad_outputs is d(forward)/d(forward) which is 1 , See https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
        gradient = torch.autograd.grad(self.forward(inputs),
                                       inputs,
                                       grad_outputs=torch.ones_like(self.forward(inputs)),
                                       retain_graph=True,
                                       create_graph=True
                                       )
        return gradient
    
    def compute_hessian(self, x):
        
        # Compute the hessian of the output wrt the input
        hessians = torch.vmap(torch.func.hessian(self.forward), (0))(x)
        return hessians
                                                                                                                           

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
    cv_column = len
    train_inputs = torch.tensor(train_arr[:,[density_column,temperature_column]])
    train_targets = torch.tensor(train_arr[:,[20]])
    val_inputs = torch.tensor(val_arr[:,[density_column,temperature_column]])
    val_targets = torch.tensor(val_arr[:,[20]])
    train_inputs = train_inputs.float()
    train_targets = train_targets.float()
    val_inputs = val_inputs.float()
    val_targets = val_targets.float()

    # Loading inputs and targets into the dataloaders
    train_dataset = TensorDataset(train_inputs,train_targets)
    val_Dataset = TensorDataset(val_inputs,val_targets)
    train_dataloader = DataLoader(train_dataset,batch_size = 6000)
    val_dataloader = DataLoader(val_Dataset,batch_size =6000)

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

    lower_limit_of_neurons_per_layer = 10
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
    scheduler = ASHAScheduler(max_t= max_number_of_training_epochs, grace_period=500, reduction_factor=2)

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
num_samples = 1

results = tune_asha(num_samples,max_number_of_training_epochs)
results.get_best_result(metric="val_loss", mode="min")