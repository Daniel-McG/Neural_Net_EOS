import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/home/daniel/Documents/Research Project/Neural_Net_EOS/coallated_results_debug_working_with_nan.txt", delimiter=" ")
arr = df.to_numpy()
density_column = 4
temperature_column = 2
pressure_column = 3
internal_energy_column = 1
cp_column = 20
alphaP_column = cp_column +1
betaT_column = alphaP_column +1
mu_jt_column = betaT_column + 1
Z_column = mu_jt_column+1
cv_column = Z_column+1
gammaV_column = cv_column+1

sns.scatterplot(x= arr[:,density_column],y=arr[:,temperature_column],label = "Training Data")
plt.ylim(0,1)
plt.show()