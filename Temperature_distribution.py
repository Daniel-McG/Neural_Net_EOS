import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
df = pd.read_csv(r"C:\Users\Daniel.000\Documents\New_research_project\Neural_Net_EOS\models\TrainedNaNs_LowLR\training_data_for_current_ANN.txt",delimiter=" ")
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
print(pd.DataFrame(arr[:,cp_column]).describe())
sns.scatterplot(y = arr[:,cp_column],x = arr[:,temperature_column])
plt.show()