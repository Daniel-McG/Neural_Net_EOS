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
    def __init__(self):
        super(BasicLightning,self).__init__() 

        # Creating a sequential stack of Linear layers with Tanh activation function 

        self.s1 = nn.Sequential(
          nn.Linear(2,1000),
          nn.ReLU(),
          nn.Linear(1000,1000),
          nn.ReLU(),
          nn.Linear(1000,1000),
          nn.ReLU(),
          nn.Linear(1000,1000),
          nn.ReLU(),
          nn.Linear(1000,1000),
          nn.ReLU(),
          nn.Linear(1000,1000),
          nn.ReLU(),
          nn.Linear(1000,1000),
          nn.ReLU(),
          nn.Linear(1000,2),
          nn.ReLU()
        )

    def forward(self,x):
        out = self.s1(x)
        return out
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = 0.00005 )
    
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
    
model = BasicLightning().load_from_checkpoint("/home/daniel/ray_results/TorchTrainer_2023-10-17_10-49-17/TorchTrainer_7250b_00000_0_2023-10-17_10-49-21/lightning_logs/version_0/checkpoints/epoch=1727-step=3456.ckpt")
model.eval()

with torch.no_grad():
    # density=np.arange(0.4,0.7,0.01)
    # temp = np.full((1,len(density)),2)
    # Temp_and_density = np.vstack((density,temp[0])).T

    data_df = pd.read_csv('/home/daniel/Downloads/MSc_data.csv',names=['rho','T','P','U'])

    #Preprocessing the data
    train_df,test_df = train_test_split(data_df,train_size=0.6)
    print(train_df)
    scaler = MinMaxScaler(feature_range =(0,1))
    train_arr= scaler.fit_transform(train_df)
    val_arr = scaler.transform(test_df)
    #Plotting distribution of train and test data
    for column in range(0,4,1):
        plt.clf()
        train_histplot = sns.histplot(data = train_arr[:,column])
        train_fig = train_histplot.get_figure()
        train_fig.savefig("/home/daniel/Pictures/BS_32_train_col{}.png".format(str(column)))

    for column in range(0,4,1):
        plt.clf()
        val_histplot = sns.histplot(data = val_arr[:,column])
        val_fig = val_histplot.get_figure()
        val_fig.savefig("/home/daniel/Pictures/BS_32_val_col{}.png".format(str(column)))

    #Splitting the preprocessed data into the inputs and targets
    train_inputs = torch.tensor(train_arr[:,[0,1]])

    x = train_inputs

    x = x.float()
    x = x.to(device)
    y_hat = model(x).detach()
    scaler = pickle.load(open("/home/daniel/ray_results/TorchTrainer_2023-10-17_10-49-17/TorchTrainer_7250b_00000_0_2023-10-17_10-49-21/scaler.pkl", "rb"))
    

    y_hat = y_hat.cpu()
    y_hat = y_hat.numpy()
    y_hat = y_hat.reshape(-1,2)

    print(np.zeros((2,len(y_hat))).T)
    y_hat = np.hstack((np.zeros((2,len(y_hat))).T,y_hat))

    y_hat = scaler.inverse_transform(y_hat)
    print(y_hat)
    # # y_hat = list(np.concatenate(y_hat).flat)
    # # print(y_hat)
