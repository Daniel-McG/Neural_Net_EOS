import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("/home/daniel/Documents/Research Project/Neural_Net_EOS/Dan_and_sanjay_data.txt",delimiter=" ")
df_debug = pd.read_csv("/home/daniel/Documents/Research Project/Neural_Net_EOS/coallated_results_debug.txt",delimiter=" ")
arr = df.to_numpy()
debug_arr = df_debug.to_numpy()

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


gammaV_from_data_coallation = arr[:,gammaV_column]
gammaV_from_thermo_relation = arr[:,alphaP_column]/arr[:,betaT_column]

gammaV_from_data_coallation_debug = debug_arr[:,gammaV_column]
gammaV_from_thermo_relation_debug = debug_arr[:,alphaP_column]/debug_arr[:,betaT_column]



cp_from_thermo_relation = arr[:,cv_column] + (arr[:,temperature_column]/arr[:,density_column])*((arr[:,alphaP_column]**2)/arr[:,betaT_column])
cp_from_data_coallation = arr[:,cp_column]
cp_from_thermo_relation_debug = debug_arr[:,cv_column] + (debug_arr[:,temperature_column]/debug_arr[:,density_column])*((debug_arr[:,alphaP_column]**2)/debug_arr[:,betaT_column])
cp_from_data_coallation_debug = debug_arr[:,cp_column]
# mu_jt_predicted = (1/(rho*cp_predicted))*((T*alphaP_predicted)-1)
Z_from_thermo_relation= arr[:,pressure_column]/(arr[:,density_column]*arr[:,temperature_column])
Z_from_data_coallation= arr[:,Z_column]




sns.scatterplot(x=gammaV_from_data_coallation,y=gammaV_from_thermo_relation,color ="r" )
sns.scatterplot(x = gammaV_from_data_coallation_debug,y=gammaV_from_thermo_relation_debug,color="c")
plt.xlabel(r"$\gamma_V$ from MD")
plt.ylabel(r"$\gamma_V$ from $\alpha_P$ / $\beta_T$")
sns.lineplot(x=[-0.001,0.0025],y=[-0.001,0.0025])
plt.xlim(-0.001,0.0025)
plt.ylim(-0.001,0.0025)
plt.show()

sns.scatterplot(x=cp_from_data_coallation,y=cp_from_thermo_relation,color ="r" )
sns.scatterplot(x = cp_from_data_coallation_debug,y=cp_from_thermo_relation_debug,color="c")
plt.xlabel(r"Cp from MD")
plt.ylabel(r"Cp from relation")
range = [0,100]
sns.lineplot(x=range,y=range)
plt.xlim(range)
plt.ylim(range)
plt.show()

sns.scatterplot(x=Z_from_data_coallation,y=Z_from_thermo_relation,color ="r" )
# sns.scatterplot(x = cp_from_data_coallation_debug,y=cp_from_thermo_relation_debug,color="c")
plt.xlabel("Z from MD")
plt.ylabel("Z from relation")
range = [0,10]
sns.lineplot(x=range,y=range)
plt.xlim(range)
plt.ylim(range)
plt.show()
