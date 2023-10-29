import numpy as np
import pandas as pd

Na = 6.02*(10**23)
#Kb = 1.380649*(10**-23)
Kb = 1 # Reduced Units?

def isochoric_heat_capacity(N,PE,T):

    squared_1 = np.mean(PE**2)
    squared_2 = np.mean(PE)**2
    part1 = (Na/N)*((squared_1 - squared_2)/(Kb*T**2))
    part2 = (Na/N)*(3/2)*(N*Kb)
    return part1 + part2

def isobaric_heat_capacity(E,N,P,V,T):
    P_average = np.mean(P)
    V_average = np.mean(V)
    E_PV = E + (P_average*V_average)
    squared_1 = np.mean(E_PV**2)
    squared_2 = np.mean(E_PV)**2
    return (Na/N)*(1/(Kb*(T**2)))*(squared_1 - squared_2)

def thermal_expansion_coefficient(E,P,PE,T,V):
    P_average = np.mean(P)
    V_average = np.mean(V)
    E_PV = E + (P_average*V_average)
    E_PV_average = np.mean(E_PV)
    return (1/(Kb*(T**2)*V_average))*(np.mean((V - V_average)*(E_PV-E_PV_average)))

def isothermal_compressibility(V,T):
    V_average = np.mean(V)
    squared_1 = np.mean(V**2)
    squared_2 = V_average**2
    return (1/(V_average*Kb*T))*(squared_1-squared_2)

def thermal_pressure_coefficient(P,PE,rho,T):
    PE_average = np.mean(PE)
    P_average = np.mean(P)
    part1 = (np.mean((PE-PE_average)*(P-P_average)))/(Kb*(T**2))
    part2 = rho*Na*Kb
    return part1 + part2

def compressibility_factor(P,rho,T):
    return P/(rho*T)





    
    