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
from lightning.pytorch.loggers import TensorBoardLogger
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
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from torch.autograd import grad
from ray.tune import CLIReporter
import functorch    

data_df = pd.read_csv('/home/daniel/Downloads/MSc_data.csv',names=['rho','T','P','U'])

#Preprocessing the data
train_df,test_df = train_test_split(data_df,train_size=0.6)

train_arr = train_df.values
val_arr = test_df.values

#Splitting the preprocessed data into the inputs and targets
train_inputs = torch.tensor(train_arr[:,[0,1]])
train_targets = torch.tensor(train_arr[:,[2]])
val_inputs = torch.tensor(val_arr[:,[0,1]])
val_targets = torch.tensor(val_arr[:,[2]])
print("The Side of the training inputs is {}".format(train_inputs.element_size() * train_inputs.nelement()))