import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
data_df = pd.read_csv('/home/daniel/Documents/Research Project/Neural_Net_EOS/coallated_results_debug.txt',delimiter=" ")
# Preprocessing the data

# The data was not MinMax scaled as the gradient and hessian had to be computed wrt the input e.g. temperature , not scaled temperature.
# It may be possible to write the min max sacaling in PyTorch so that the DAG is retained all the way to the input data but im not sure if
# the TensorDataset and DataLoader would destroy the DAG.
# Since the density is already in ~0-1 scale and the temperature is only on a ~0-10 scale, it will be okay.
# Problems would occur if non-simulated experimental data was used as pressures are typically ~ 100kPa and temperatures ~ 298K,
# very far from the typical 0-1 range we want for training a neural network


data_arr = data_df.values
print(data_arr[:,0] != 2000500.0)
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

def find_closest(array, value, n=100):
    array = np.asarray(array)
    idx = np.argsort(np.abs(array - value))[:n]
    return idx
index_of_points_close_to_temp = find_closest(data_arr[:,temperature_column],0.746975,100)

isotherm = data_arr[index_of_points_close_to_temp,:]




for i in range(20,27,):
    Outlier_model = LocalOutlierFactor(n_neighbors=20)
    y_pred = Outlier_model.fit_predict(data_arr[:,i][~np.isnan(data_arr[:,i])].reshape(-1, 1) )
    
    data_arr[:,i][~np.isnan(data_arr[:,i])] = np.where(y_pred == -1,np.nan,data_arr[:,i][~np.isnan(data_arr[:,i])])
    # sns.kdeplot(cleaned_arr[:,i])
    print(np.sum(y_pred== -1))
    # plt.show()
# print(len(data_arr))
np.savetxt("coallated_results_debug.txt",data_arr)
# for i in range(22,27):
#     Outlier_model = LocalOutlierFactor(n_neighbors=20)
#     y_pred = Outlier_model.fit_predict(data_arr[:,i].reshape(-1, 1) )
    
#     cleaned_arr = np.delete(data_arr,y_pred == -1,0)
#     # sns.kdeplot(cleaned_arr[:,i])
#     print(np.sum(y_pred== -1))
#     # plt.show()
#     data_arr=cleaned_arr
# sns.scatterplot(x = y_pred,y = train_arr[:,pressure_column])
# plt.show()




# train_inputs = torch.tensor(train_arr[:,[density_column,temperature_column]])
# train_targets = torch.tensor(train_arr[:,target_columns])
# val_inputs = torch.tensor(val_arr[:,[density_column,temperature_column]])
# val_targets = torch.tensor(val_arr[:,target_columns])
# train_inputs = train_inputs.float()
# train_targets = train_targets.float()
# val_inputs = val_inputs.float()
# val_targets = val_targets.float()
# mins, _=train_inputs.min(dim=0,keepdim=True)
# train_inputs[:,0]-mins[0,0]
