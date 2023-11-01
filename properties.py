import numpy as np
import pandas as pd

Na = 6.02*10**23
#Kb = 1.380649*(10**-23)
Kb = 1 # Reduced Units?
# Temperature - Intensive
# Pressure - Intensive
# Density - intensive

# Total energy - since its the sum of kinetic and potential
# these are both extensive thus the sum will be extensive
# Kinetic energy - Extensive
# Potential energy - extensive
# Volume - Extensive
# Enthalpy - extensive
def isochoric_heat_capacity(N,PE,T):
    """
    Calcualtes the isochoric heat capacity from the results of a NVT ensemble Molecular dynamics simulation

    N: Number of particles
    PE: Instantaneous potential energy
    T: Temperature

    """
    PE = PE*N
    T_average = np.mean(T)
    squared_1 = np.mean(PE**2)
    squared_2 = np.mean(PE)**2
    part1 = ((squared_1 - squared_2)/(Kb*T_average**2))
    part2 = (3/2)*(N*Kb)
    cv = part1 + part2
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





    
    