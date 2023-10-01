import pandas as pd
import torch
data_df = pd.read_csv('/home/daniel/Downloads/MSc_data.csv',names=['rho','T','P','U'])
input_df = data_df[['rho','T']]
target_df = data_df[['P','U']]
input_values = input_df.values
target_values= target_df.values
input_tensor = torch.tensor(input_values)
target_tensor = torch.tensor(target_values)
print(target_tensor)