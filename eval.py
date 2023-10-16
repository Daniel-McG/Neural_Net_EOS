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
logger = TensorBoardLogger("tb_logs", name="my_model")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Defining the neural network

class BasicLightning(pl.LightningModule):
    def __init__(self):
        super(BasicLightning,self).__init__() 

        # Creating a sequential stack of Linear layers with Tanh activation function 

        self.s1 = nn.Sequential(
          nn.Linear(2,6),
          nn.Tanh(),
          nn.Linear(6,4),
          nn.Tanh(),
          nn.Linear(4,2),
          nn.Tanh()
        )

    def forward(self,x):
        out = self.s1(x)
        return out
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = 0.000001 )
    
    def training_step(self,train_batch,batch_index):
        input_i,target_i = train_batch            #Unpacking data from a batch
        output_i = self.forward(input_i)    #Putting input data frm the batch through the neural network
        loss = (output_i-target_i)**2       #Calculating loss
        mean_loss = torch.mean(loss)
        self.log("train_loss",mean_loss)         #Logging the training loss
        return {"loss": mean_loss}
    
    def validation_step(self, val_batch, batch_idx):
        val_input_i, val_target_i = val_batch
        val_output_i = self.forward(val_input_i)
        loss = (val_output_i-val_target_i)**2
        mean_loss = torch.mean(loss)
        self.log("val_loss",mean_loss) 
        return {"val_loss": mean_loss}
    
model = BasicLightning().load_from_checkpoint("/home/daniel/ray_results/TorchTrainer_2023-10-16_09-24-18/TorchTrainer_67f3d_00000_0_2023-10-16_09-24-21/lightning_logs/version_0/checkpoints/epoch=28414-step=56830.ckpt")
model.eval()

with torch.no_grad():
    # density=np.arange(0.4,0.7,0.01)
    # temp = np.full((1,len(density)),2)
    # Temp_and_density = np.vstack((density,temp[0])).T
    x = torch.tensor([0.7,1.5025])
    x = x.float()
    print(x)
    x = x.to(device)
    y_hat = model(x).detach()
    print(y_hat)
    y_hat = y_hat.cpu()
    y_hat = y_hat.numpy()
    # y_hat = list(np.concatenate(y_hat).flat)
    # print(y_hat)
