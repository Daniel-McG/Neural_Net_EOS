from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
import seaborn as sns
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
logger = TensorBoardLogger("tb_logs", name="my_model")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
input_doses = torch.linspace(start=0, end=1, steps=11)
inputs = torch.tensor([0.,0.25, 0.5,0.75, 1.] * 10000)
labels = torch.tensor([0.,0.5, 1.,0.5, 0.] * 10000)
inputs = inputs.to(device)
labels = labels.to(device)
dataset = TensorDataset(inputs,labels)
dataloader = DataLoader(dataset)
class BasicLightning(L.LightningModule):
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

class lightning_with_ray_mods(L.LightningModule):
    def __init__(self,config):
        super(lightning_with_ray_mods,self).__init__()
        self.l1_size = config['l1_size']
        self.l2_size = config['l2_size']
        self.s1 = nn.Sequential(
          nn.Linear(1,self.l1_size),
          nn.Tanh(),
          nn.Linear(self.l1_size,self.l2_size),
          nn.Tanh(),
          nn.Linear(self.l2_size,1),
          nn.Tanh()
        )

        



model = BasicLightning()
model = model.to(device)
trainer = L.Trainer(max_epochs=1000,accelerator="auto",default_root_dir="/home/daniel/Documents/Development/Logging",callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0, patience=3, verbose=True, mode="min")])
print(dataset.tensors)
trainer.fit(model,dataloader)
