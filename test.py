import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
data_df = pd.read_csv('/home/daniel/Downloads/MSc_data.csv',names=['rho','T','P','U'])
train_df,test_df = train_test_split(data_df)
scaler = MinMaxScaler(feature_range =(0,1))
train_arr= scaler.fit_transform(train_df)
test_arr = scaler.transform(test_df)
train_inputs = train_arr[:,[0,1]]
train_targets = train_arr[:,[2,3]]
test_inputs = test_arr[:,[0,1]]
test_targets = test_arr[:,[2,3]]

print(train_targets)
# input_df = data_df[['rho','T']]
# target_df = data_df[['P','U']]
# input_values = input_df.values
# target_values= target_df.values
# input_tensor = torch.tensor(input_values)
# target_tensor = torch.tensor(target_values)
# scaler = MinMaxScaler(feature_range =(0,1))
# scaler.fit_transform(input_values)
# print(scaler.fit_transform(input_values))