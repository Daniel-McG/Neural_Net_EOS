import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df = pd.read_csv("/home/daniel/Documents/Research Project/Neural_Net_EOS/coallated_standard_deviations.txt",delimiter=" ")
arr = df.values

# def plot_distribution(arr):
#     fig, axs = plt.subplots(nrows=17, ncols=17, figsize=(10, 10))
#     for i in range(arr.shape[1]-1):
#         axs[i, i].hist(arr[:, i], bins=50)
#         axs[i, i].set_title(f"Distribution of column {i}")
#         axs[i, i].set_xlabel("Value")
#         axs[i, i].set_ylabel("Frequency")
#     plt.show()
df.drop(columns=[df.columns[0],df.columns[5],df.columns[10],df.columns[15]], axis=1, inplace=True)
for columns in df.columns:
    fraction_deviation_from_mean = 0.5
    bounds_mask = (df[columns]<df[columns].mean()+fraction_deviation_from_mean*df[columns].mean())&(df[columns]>df[columns].mean()-fraction_deviation_from_mean*df[columns].mean())
    print(bounds_mask.value_counts().loc[True])

# df.apply(pd.Series.value_counts)

# bounds_mask.apply(pd.Series.value_counts).to_csv("describe.csv", sep='\t',)





