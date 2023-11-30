import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def isochoric_heat_capacity(N,E,T):
    """
    Calcualtes the isochoric heat capacity from the results of a NVT ensemble Molecular dynamics simulation

    N: Number of particles
    PE: Instantaneous potential energy
    T: Temperature

    """
    N_mean = np.mean(N)
    dE = (E-np.mean(E))*N_mean
    T = np.mean(T)
    cv = np.mean(dE**2)/(Kb*(T**2))
    cv /= N_mean
    return cv

def isobaric_heat_capacity(H,N,T):
    '''
    Calculates the isobarc heat capacity from the results of a NPT ensemble molecuar dynamics simulation

    E: Instantaneous total energy
    N: Number of particles
    P: Instantaneous pressure
    V: Instantaneous volume
    T: Instantaneous Temperature
    '''
    N_mean = np.mean(N)
    T_mean = np.mean(T)
    dH = (H-np.mean(H))*N_mean
    cp = np.mean(dH**2) / T_mean**2
    cp /= N_mean
    return cp

def thermal_expansion_coefficient(H,T,V,N):
    '''
    Calculates the thermal expansion coefficient from the results of a NPT ensemble molecular dynamics simulation

    E: Instantaneous total energy
    N: Number of particles
    P: Instantaneous pressure
    V: Instantaneous volume
    T: Instantaneous Temperatur
    '''
    N_mean = np.mean(N)
    dH = (H-np.mean(H))*N_mean
    dV = (V-np.mean(V))
    T_average = np.mean(T)
    V_average = np.mean(V)
    alpha_p = np.mean(dH*dV) / (V_average*T_average**2)
    return alpha_p

def isothermal_compressibility(V,T):
    """
    Calculates the isothermal compressibility from the results of a NPT ensemble molecular dynamics simulation

    V: Instantaneous volume
    T: Instantaneous temperature
    """
    V_average = np.mean(V)
    T_average = np.mean(T)
    dV = V-np.mean(V)
    betaT = np.mean(dV**2) / (V_average*T_average)
    return betaT

def thermal_pressure_coefficient(P,PE,rho,T,N):
    """
    Calculates the thermal pressure coefficent from the results of a NVT ensemble molecular dynamics simulation

    P: Instantaneous pressure
    PE: Instantaneous potential energy
    rho: Instantaneous density
    T: Instantaneous temperature
    N: Number of molecules
    """
    N_mean = np.mean(N)
    T_average = np.mean(T)
    rho_average = np.mean(rho)
    dPE = (PE-np.mean(PE))*N_mean
    dP = P-np.mean(P)
    gamma_v = np.mean(dPE*dP) / T_average**2 + rho_average
    return gamma_v

def compressibility_factor(P,rho,T):
    rho_average = np.mean(rho)
    P_average = np.mean(P)
    T_average = np.mean(T)
    return P_average/(rho_average*T_average)

def joule_thompson(rho,T,Cp,alpha_p):
    T = np.mean(T)
    rho = np.mean(rho)
    JT = (1/(rho*Cp))*(T*alpha_p-1)
    return JT

path_to_results = r"/home/daniel/Downloads/LJ-2d-md-results/tad0.571489_rhoad0.624014/simulation1/nptave.lammps"
data_df = pd.read_csv(path_to_results,
                    skiprows=[0,1],
                    delimiter=" ",
                    names=["TimeStep","v_TENE","v_TEMP","v_PRES","v_DENS","v_nMolecule","v_KENE","v_PENE","v_ENTH","v_VOL"]
                    )

sns.scatterplot(data = data_df, x = "TimeStep", y="v_PRES")
plt.show()

for (root,dirs,files) in os.walk("/home/daniel/Downloads/LJ-2d-md-results/", topdown=True):
            for filename in files:
                if (filename == "nptave.lammps"):
                    TE_list = []
                    IC_list = []
                    path_to_results = os.path.join(root,filename)
                    
                    # Read the data and convert to numpy array
                    data_df = pd.read_csv(path_to_results,
                                        skiprows=[0,1],
                                        delimiter=" ",
                                        names=["TimeStep","v_TENE","v_TEMP","v_PRES","v_DENS","v_nMolecule","v_KENE","v_PENE","v_ENTH","v_VOL"]
                                        )
                    for timestep in range(1*len(data_df)//6,6*len(data_df)//6,10):
                        pressure = data_df["v_PRES"][:timestep]
                        PE = data_df["v_PENE"][:timestep]
                        density = data_df["v_DENS"][:timestep]
                        temperature=data_df["v_TEMP"][:timestep]
                        molecules = data_df["v_nMolecule"][:timestep]
                        enthalpy = data_df["v_ENTH"][:timestep]
                        volume = data_df["v_VOL"][:timestep]
                        TE_list.append(thermal_expansion_coefficient(enthalpy,temperature,volume,molecules))
                        IC_list.append(isothermal_compressibility(volume,temperature))
                    sns.lineplot(data = TE_list,label = "Thermal expansion coefficient")
                    last_value = TE_list[-1]
                    0.
                    plt.ylabel("Thermal expansion coefficient")
                    plt.xlabel("Timesteps/(4/6)")
                    plt.axhline(y=last_value-1e-3, color='r', linestyle='--',label = "+- 1e-3 Bound")
                    plt.axhline(y=last_value+1e-3, color='r', linestyle='--', )
                    plt.legend()
                    plt.show()
                    # print(IC_list)
                    



