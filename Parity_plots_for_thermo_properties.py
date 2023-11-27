import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("/home/daniel/Documents/Research Project/Neural_Net_EOS/coallated_results_debug.txt",delimiter=" ")
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
non_nan_alphaP = ~np.isnan(arr[:,alphaP_column])
# print(arr[:,betaT_column])
# arr[:,alphaP_column] = arr[:,alphaP_column]
# # print(debug_arr[:,alphaP_column].flatten()/debug_arr[:,betaT_column].flatten())
# print(pd.DataFrame(arr[:,gammaV_column]).describe())
# print(pd.DataFrame(arr[:,alphaP_column]==).value_counts(ascending=False))
# print(pd.DataFrame(arr[:,betaT_column]).describe())
# print(pd.DataFrame(arr[:,alphaP_column].flatten()/arr[:,betaT_column].flatten()).describe())
# non_zero_betaT_index = np.where(arr[:,betaT_column]!=0)

# gammaV_from_data_coallation = arr[non_zero_betaT_index,gammaV_column].flatten()
# gammaV_from_thermo_relation = arr[non_zero_betaT_index,alphaP_column].flatten()/arr[non_zero_betaT_index,betaT_column].flatten()

# # indices = np.where(np.logical_and(np.isfinite(gammaV_from_thermo_relation), gammaV_from_thermo_relation != np.inf, gammaV_from_thermo_relation != -np.inf))
# # indicies = 1
gammaV_from_data_coallation = debug_arr[non_nan_alphaP,gammaV_column].flatten()
gammaV_from_thermo_relation = debug_arr[non_nan_alphaP,alphaP_column].flatten()/debug_arr[non_nan_alphaP,betaT_column].flatten()



cp_from_thermo_relation = arr[:,cv_column].flatten() + (arr[:,temperature_column].flatten()/arr[:,density_column].flatten())*((arr[:,alphaP_column].flatten()**2)/arr[:,betaT_column].flatten())
# # cp_from_thermo_relation2 = arr[:,cv_column] + (arr[:,temperature_column]/arr[:,density_column])*(arr[:,alphaP_column]*arr[:,gammaV_column])

# # print(pd.DataFrame(np.abs((arr[:,alphaP_column]**2)/arr[:,betaT_column])<1e-20).describe())

cp_from_data_coallation = arr[:,cp_column].flatten()
# # print((cp_from_thermo_relation - cp_from_data_coallation)<1e-5)
# # cp_from_thermo_relation_debug = debug_arr[:,cv_column] + (debug_arr[:,temperature_column]/debug_arr[:,density_column])*((debug_arr[:,alphaP_column]**2)/debug_arr[:,betaT_column])
# # cp_from_data_coallation_debug = debug_arr[:,cp_column]
# # mu_jt_predicted = (1/(rho*cp_predicted))*((T*alphaP_predicted)-1)
# # mu_jt_from_data_coallation = arr[:,mu_jt_column]
# # mu_jt_from_thermo_relation = (1/(arr[:,density_column]*arr[:,cp_column]))*((arr[:,temperature_column]*arr[:,alphaP_column])-1)
# Z_from_thermo_relation= arr[:,pressure_column]/(arr[:,density_column]*arr[:,temperature_column])
# Z_from_data_coallation= arr[:,Z_column]



# # print(pd.DataFrame(np.abs(((arr[:,alphaP_column]/arr[:,betaT_column])- (arr[:,gammaV_column])))<1e-1).value_counts())
# # sns.histplot(data = arr[non_zero_betaT_index,alphaP_column]/arr[non_zero_betaT_index,betaT_column])
# # plt.show()
sns.scatterplot(x=gammaV_from_data_coallation,y=gammaV_from_thermo_relation,color ="r" )
# sns.scatterplot(x = gammaV_from_data_coallation_debug,y=gammaV_from_thermo_relation_debug,color="c")
plt.xlabel(r"$\gamma_V$ from MD")
plt.ylabel(r"$\gamma_V$ from $\alpha_P$ / $\beta_T$")
sns.lineplot(x=[0,7],y=[0,7])
# plt.xlim(-0.001,0.0025)
# plt.ylim(-0.001,0.0025)
plt.show()

sns.scatterplot(x=cp_from_data_coallation,y=cp_from_thermo_relation,color ="r" )
# # sns.scatterplot(x = cp_from_data_coallation_debug,y=cp_from_thermo_relation_debug,color="c")
plt.xlabel(r"Cp from MD")
plt.ylabel(r"Cp from relation")
range = [0,60]
sns.lineplot(x=range,y=range)
plt.xlim(range)
plt.ylim(range)
plt.show()

# sns.scatterplot(x=Z_from_data_coallation,y=Z_from_thermo_relation,color ="r" )
# # sns.scatterplot(x = cp_from_data_coallation_debug,y=cp_from_thermo_relation_debug,color="c")
# plt.xlabel("Z from MD")
# plt.ylabel("Z from relation")
# range = [0,10]
# sns.lineplot(x=range,y=range)
# plt.xlim(range)
# plt.ylim(range)
# plt.show()

# sns.scatterplot(x=mu_jt_from_data_coallation,y=mu_jt_from_thermo_relation,color ="r" )
# # sns.scatterplot(x = cp_from_data_coallation_debug,y=cp_from_thermo_relation_debug,color="c")
# plt.xlabel("mu_jt from MD")
# plt.ylabel("mu_jt from relation")
# range = [0,10]
# sns.lineplot(x=range,y=range)
# plt.xlim(range)
# plt.ylim(range)
# plt.show()
