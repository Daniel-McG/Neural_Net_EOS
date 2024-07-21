import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
data_df = pd.read_csv('collated_results_debug.txt',delimiter=" ")

data_arr = data_df.values
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

for i in range(20,27,):
    Outlier_model = LocalOutlierFactor(n_neighbors=20)
    y_pred = Outlier_model.fit_predict(data_arr[:,i][~np.isnan(data_arr[:,i])].reshape(-1, 1) )
    data_arr[:,i][~np.isnan(data_arr[:,i])] = np.where(y_pred == -1,np.nan,data_arr[:,i][~np.isnan(data_arr[:,i])])
    print(np.sum(y_pred== -1))
np.savetxt("coallated_results.txt",data_arr)

