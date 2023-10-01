import torch
import pandas as pd
import torch.nn as nn
import lightning as L
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, CheckpointConfig
import ray.train.lightning
from ray import tune
from ray.tune.schedulers import ASHAScheduler
logger = TensorBoardLogger("tb_logs", name="my_model")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Defining the neural network

class BasicLightning(pl.LightningModule):
    def __init__(self):
        super(BasicLightning,self).__init__() 

        # Creating a sequential stack of Linear layers with Tanh activation function 

        self.s1 = nn.Sequential(
          nn.Linear(2,2),
          nn.Tanh(),
          nn.Linear(2,2),
          nn.Tanh(),
          nn.Linear(2,2),
          nn.Tanh()
        )
    def forward(self,x):
        out = self.s1(x)
        return out
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = 0.001 )
    def training_step(self,train_batch,batch_index):

        input_i,target_i = train_batch            #Unpacking data from a batch
        output_i = self.forward(input_i)    #Putting input data frm the batch through the neural network
        loss = (output_i-target_i)**2       #Calculating loss
        mean_loss = torch.mean(loss)
        print(mean_loss)
        self.log("train_loss",mean_loss)         #Logging the training loss
        return {'train_loss': mean_loss}
    
    def validation_step(self, val_batch, batch_idx):
        val_input_i, val_target_i = val_batch
        val_output_i = self.forward(val_input_i)
        loss = (val_output_i-val_target_i)**2
        mean_loss = torch.mean(loss)
        self.log("val_loss",mean_loss) 
        return {"val_loss": mean_loss}

def train_func(config):
    # Read data from csv
    data_df = pd.read_csv('/home/daniel/Downloads/MSc_data.csv')
    # Create and load data into dataoader
    data_df = pd.read_csv('/home/daniel/Downloads/MSc_data.csv',names=['rho','T','P','U'])
    input_df = data_df[['rho','T']]
    target_df = data_df[['P','U']]
    input_values = input_df.values
    target_values= target_df.values
    input_tensor = torch.tensor(input_values)
    target_tensor = torch.tensor(target_values)
    input_tensor = input_tensor.float()
    target_tensor = target_tensor.float()
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)

    dataset = TensorDataset(input_tensor,target_tensor)

    train_tensor_size = int(len(input_tensor) * 0.8)
    val_tensor_size = len(input_tensor) - train_tensor_size
    seed = torch.Generator().manual_seed(584)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_tensor_size, val_tensor_size], generator=seed)

    train_dataloader = DataLoader(train_set)
    val_dataloader = DataLoader(val_set)
    model = BasicLightning()
    trainer = pl.Trainer(
        max_epochs=100000,
        devices="auto",
        accelerator="auto",
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=[ray.train.lightning.RayTrainReportCallback()],
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


scaling_config = ScalingConfig(num_workers=1, use_gpu=True)

trainer = TorchTrainer(train_func, scaling_config=scaling_config)
result = trainer.fit()

