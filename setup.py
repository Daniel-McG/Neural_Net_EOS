import torch
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

def train_func(config):
    # Read data from csv
    data_df = pd.read_csv('/home/daniel/Downloads/MSc_data.csv',names=['rho','T','P','U'])

    #Preprocessing the data
    train_df,test_df = train_test_split(data_df,train_size=0.9)
    scaler = MinMaxScaler(feature_range =(0,1))
    train_arr= scaler.fit_transform(train_df)
    val_arr = scaler.transform(test_df)
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
    train_targets = torch.tensor(train_arr[:,[2,3]])
    val_inputs = torch.tensor(val_arr[:,[0,1]])
    val_targets = torch.tensor(val_arr[:,[2,3]])
    train_inputs = train_inputs.float()
    train_targets = train_targets.float()
    val_inputs = val_inputs.float()
    val_targets = val_targets.float()
    print(train_inputs)
    # Loading inputs and targets into the dataloaders
    train_dataset = TensorDataset(train_inputs,train_targets)
    val_Dataset = TensorDataset(val_inputs,val_targets)
    train_dataloader = DataLoader(train_dataset,batch_size = 64)
    val_dataloader = DataLoader(val_Dataset,batch_size = 64)
    model = BasicLightning()
    trainer = pl.Trainer(
        max_epochs=200000,
        devices="auto",
        accelerator="auto",
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()]
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


scaling_config = ScalingConfig(num_workers=1, use_gpu=True)

trainer = TorchTrainer(train_func, scaling_config=scaling_config)
result = trainer.fit()


