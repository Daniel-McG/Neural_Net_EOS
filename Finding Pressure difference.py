import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
Kb = 1
def isochoric_heat_capacity(N,E,T):
    """
    Calcualtes the isochoric heat capacity from the results of a NVT ensemble Molecular dynamics simulation

    N: Number of particles
    PE: Instantaneous potential energy
    T: Temperature

    """
    E = E*N
    T = np.mean(T)
    E_squared = E**2
    mean_E = np.mean(E)
    mean_E_squared = mean_E**2
    E_squared_mean = np.mean(E_squared)
    cv = (E_squared_mean-mean_E_squared)/(Kb*(T**2))
    cv = cv/N
    return cv

def isobaric_heat_capacity(E,N,P,V,T):
    '''
    Calculates the isobarc heat capacity from the results of a NPT ensemble molecuar dynamics simulation

    E: Instantaneous total energy
    N: Number of particles
    P: Instantaneous pressure
    V: Instantaneous volume
    T: Instantaneous Temperature
    '''
    E = E*N
    T_average = np.mean(T)
    P_average = np.mean(P)
    V_average = np.mean(V)
    E_PV = E + (P_average*V_average)
    squared_1 = np.mean(E_PV**2)
    squared_2 = np.mean(E_PV)**2
    cp = (1/(Kb*(T_average**2)))*(squared_1 - squared_2)
    cp = cp/N
    return cp

def thermal_expansion_coefficient(E,P,T,V,N):
    '''
    Calculates the thermal expansion coefficient from the results of a NPT ensemble molecular dynamics simulation

    E: Instantaneous total energy
    N: Number of particles
    P: Instantaneous pressure
    V: Instantaneous volume
    T: Instantaneous Temperatur
    '''
    E = E*N
    T_average = np.mean(T)
    P_average = np.mean(P)
    V_average = np.mean(V)
    E_PV = E + (P_average*V_average)
    E_PV_average = np.mean(E_PV)
    alpha_p = (1/(Kb*(T_average**2)*V_average))*(np.mean((V - V_average)*(E_PV-E_PV_average)))
    alpha_p = alpha_p/N
    return alpha_p

def isothermal_compressibility(V,T):
    """
    Calculates the isothermal compressibility from the results of a NPT ensemble molecular dynamics simulation

    V: Instantaneous volume
    T: Instantaneous temperature
    """
    T_average = np.mean(T)
    V_average = np.mean(V)
    squared_1 = np.mean(V**2)
    squared_2 = V_average**2
    return (1/(V_average*Kb*T_average))*(squared_1-squared_2)

def thermal_pressure_coefficient(P,PE,rho,T,N):
    """
    Calculates the thermal pressure coefficent from the results of a NVT ensemble molecular dynamics simulation

    P: Instantaneous pressure
    PE: Instantaneous potential energy
    rho: Instantaneous density
    T: Instantaneous temperature
    N: Number of molecules
    """
    PE = PE*N
    rho_average = np.mean(rho)
    T_average = np.mean(T)
    PE_average = np.mean(PE)
    P_average = np.mean(P)
    part1 = (np.mean((PE-PE_average)*(P-P_average)))/(Kb*(T_average**2))
    part2 = rho_average*Kb
    gamma_v = part1 + part2
    gamma_v = gamma_v/N
    return gamma_v

def compressibility_factor(P,rho,T):
    rho_average = np.mean(rho)
    P_average = np.mean(P)
    T_average = np.mean(T)
    return P_average/(rho_average*T_average)
def inital_property_check(initial_temperature,temperature_vector,initial_density,density_vector,tolerance):
    '''
    Checks if the temperature from the simulation is within a given tolerance of the initial temperature

    initial_temperature: A scaler of the temperature that the simulation was started at
    temperature_vector: A vector of the temperatures from the simulation
    tolerance: A scalar of the allowable difference between the initial temperature
    '''
    def aboslute_difference(inital_property,property_vector):
        mean_property = np.mean(property_vector)
        difference = abs(inital_property-mean_property)
        return difference

    temperature_difference = aboslute_difference(initial_temperature,temperature_vector)
    density_difference = aboslute_difference(initial_density,density_vector)
    if temperature_difference > tolerance:
        print("T {} {}".format(initial_density,initial_temperature))
    if density_difference > tolerance:
        print("D {} {}".format(initial_density,initial_temperature))   

def joule_thompson(rho,T,Cp,alpha_p):
    T = np.mean(T)
    rho = np.mean(rho)
    JT = (1/(rho*Cp))*(T*alpha_p-1)
    return JT





data_df = pd.read_csv("/home/daniel/Documents/Research Project/Neural_Net_EOS/cleaned_coallated_results.txt",delimiter=" ")
arr = data_df.values
pressure_npt_column = 3
pressure_nvt_column = pressure_npt_column+10
density_npt_column = 4
density_nvt_column = density_npt_column+10
temp_npt_column = 2
temp_nvt_column = temp_npt_column+10

density_column = 4
N_column = density_column + 1 
V_columne = 9
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

# diff = (arr[:,pressure_nvt_column].flatten()/(arr[:,density_nvt_column].flatten()*arr[:,temp_nvt_column].flatten()))-arr[:,Z_column]
# diff = (np.array([[1],[2],[3]]).flatten()/(arr[:,density_nvt_column].flatten()*arr[:,temp_nvt_column].flatten()))-arr[:,Z_column]
# diff = arr[:,Z_column]
# print(pd.DataFrame(diff*100).describe())
# sns.histplot(diff,label="Our Z Data")

# diff = arr[:,pressure_npt_column].flatten()/(arr[:,density_npt_column].flatten()*arr[:,temp_npt_column].flatten())-arr[:,Z_column]
# print(pd.DataFrame(diff*100).describe())
# dif_arr = []
# for i in range(len(arr)):
#     diff = isochoric_heat_capacity(arr[i,N_column].flatten(),arr[i,internal_energy_column+10].flatten(),arr[i,temp_nvt_column].flatten()).flatten()-arr[i,cv_column].flatten()
#     dif_arr.append(diff)
# print(np.array(dif_arr).mean())

# dif_arr = []
# for i in range(len(arr)):
#     diff = isobaric_heat_capacity(arr[i,internal_energy_column],arr[i,N_column],arr[i,pressure_npt_column],arr[i,V_columne],arr[i,temp_npt_column])-arr[i,cp_column]
#     dif_arr.append(diff)
# print(np.array(dif_arr).mean())

for i in range(0,11):
    print(np.mean(np.abs(arr[:,i]-arr[:,i+10])))


# print("================CP=================")
# print(pd.DataFrame(diff*100).describe())

# diff = isothermal_compressibility)-arr[:,cp_column]
# print("================CP=================")
# print(pd.DataFrame(diff*100).describe())








