import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math
# data_df = pd.read_csv('/home/daniel/Downloads/MSc_data.csv',names=['rho','T','P','U'])
# train_df,test_df = train_test_split(data_df)
# scaler = MinMaxScaler(feature_range =(0,1))
# train_arr= scaler.fit_transform(train_df)
# test_arr = scaler.transform(test_df) 
# train_inputs = train_arr[:,[0,1]]
# train_targets = train_arr[:,[2,3]]
# test_inputs = test_arr[:,[0,1]]
# test_targets = test_arr[:,[2,3]]

def num_of_sobol_points(number_of_points:int):
    """
    Rounds the number of points you input to the nearest number required to be input to the sobol sequence
    i.e.Rounds to the nearest 2^n where n is an int
    """
    exponent_of_two_to_get_required_number_of_points = math.log(number_of_points)/math.log(2)
    rounded_exponent = round(exponent_of_two_to_get_required_number_of_points,0)
    number_of_sobol_points = 2**rounded_exponent
    return int(number_of_sobol_points)

dimension = 2
number_of_points = 1000
points_to_generate = num_of_sobol_points(number_of_points)
sobol_sequence = scipy.stats.qmc.Sobol(dimension) 
sobol_values = sobol_sequence.random(points_to_generate)
sns.scatterplot(x = sobol_values[:,0], y = sobol_values[:,1])
plt.show()
# print(train_targets)
# input_df = data_df[['rho','T']]
# target_df = data_df[['P','U']]
# input_values = input_df.values
# target_values= target_df.values
# input_tensor = torch.tensor(input_values)
# target_tensor = torch.tensor(target_values)
# scaler = MinMaxScaler(feature_range =(0,1))
# scaler.fit_transform(input_values)
# print(scaler.fit_transform(input_values))