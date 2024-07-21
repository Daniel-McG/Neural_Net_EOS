import pandas as pd
import numpy as np 

data_df = pd.read_csv('/home/daniel/Documents/Research Project/Neural_Net_EOS/coallated_results_debug.txt',delimiter=" ")
print(data_df.describe())
arr = data_df.values
print(arr.shape)
arr = arr[arr[:, 13] >= 0]
arr = arr[arr[:,0] == 2000500.0]
print(arr.shape)
np.savetxt("coallated_results_debug.txt",arr)
