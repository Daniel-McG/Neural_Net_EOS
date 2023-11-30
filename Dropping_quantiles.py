import pandas as pd
import numpy as np

df = pd.read_csv("cleaned_coallated_results.txt",delimiter=" ")
# df.drop(columns=df.columns[0,2,4],inplace=True)
for i in range(20,27):
    index = df[df.columns[i]].le(df[df.columns[i]].quantile(0.05)) | df[df.columns[i]].ge(df[df.columns[i]].quantile(0.95))
    df[df.columns[i]][index]= None

np.savetxt("nan_coallted_results.txt",df.values)
