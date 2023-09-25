from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
import seaborn as sns
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import ray.train.lightning
logger = TensorBoardLogger("tb_logs", name="my_model")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class BasicLightning(pl.LightningModule):
    def __init__(self):
        super(BasicLightning,self).__init__() 
        self.s1 = nn.Sequential(
          nn.Linear(1,2),
          nn.Tanh(),
          nn.Linear(2,2),
          nn.Tanh(),
          nn.Linear(2,1),
          nn.Tanh()
        )
    def forward(self,x):
        out = self.s1(x)
        return out
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = 0.001 )
    def training_step(self,batch,batch_index):
        input_i,target_i = batch
        output_i = self.forward(input_i)
        # target_i = torch.unsqueeze(target_i,1)
        loss = (output_i-target_i)**2
        self.log("train_loss",loss)
        return {'loss': loss}


def train_func(config):
    # Create and load data into dataoader
    inputs = torch.tensor([0.,0.25, 0.5,0.75, 1.] * 10000)
    labels = torch.tensor([0.,0.5, 1.,0.5, 0.] * 10000)
    inputs = inputs.to(device)
    labels = labels.to(device)
    dataset = TensorDataset(inputs,labels)
    train_dataloader = DataLoader(dataset)
    
    model = BasicLightning()
    trainer = pl.Trainer(
        max_epochs=10,
        devices="auto",
        accelerator="auto",
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=[ray.train.lightning.RayTrainReportCallback()],
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_dataloader)


scaling_config = ScalingConfig(num_workers=2, use_gpu=False)

trainer = TorchTrainer(train_func, scaling_config=scaling_config)
result = trainer.fit()




# model = BasicLightning()
# model = model.to(device)
# trainer = L.Trainer(max_epochs=1000,accelerator="auto",default_root_dir="/home/daniel/Documents/Development/Logging",callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0, patience=3, verbose=True, mode="min")])
# print(dataset.tensors)
# trainer.fit(model,dataloader)
