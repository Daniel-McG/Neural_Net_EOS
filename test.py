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

data_df = pd.read_csv('/home/daniel/Documents/Research Project/Neural_Net_EOS/coallated_results.txt',delimiter=" ")
# Preprocessing the data

# The data was not MinMax scaled as the gradient and hessian had to be computed wrt the input e.g. temperature , not scaled temperature.
# It may be possible to write the min max sacaling in PyTorch so that the DAG is retained all the way to the input data but im not sure if
# the TensorDataset and DataLoader would destroy the DAG.
# Since the density is already in ~0-1 scale and the temperature is only on a ~0-10 scale, it will be okay.
# Problems would occur if non-simulated experimental data was used as pressures are typically ~ 100kPa and temperatures ~ 298K,
# very far from the typical 0-1 range we want for training a neural network

train_df,test_df = train_test_split(data_df,train_size=0.7)

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
mu_jt_column = betaT_column + 1
Z_column = mu_jt_column + 1
target_columns = [cv_column,gammaV_column,cp_column,alphaP_column,betaT_column,internal_energy_column,pressure_column,mu_jt_column,Z_column]
import numpy as np

def print_values_close_to_zero(my_array, threshold=1e-1):
    close_to_zero = np.abs(my_array) < threshold
    values_close_to_zero = my_array[close_to_zero]
    
    print(f"Values close to zero: {values_close_to_zero}")
data = train_arr[:,density_column].flatten()*train_arr[:,betaT_column].flatten()
print(train_arr[:,pressure_column].flatten().describe())
sns.displot(data.flatten())
plt.show()