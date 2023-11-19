import os
import pandas as pd
import numpy as np
import re
import sys
path_to_results = r"/rds/general/user/dcm120/home/LJ-2d-md-results"
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
def isochoric_heat_capacity(N,E,T):
    """
    Calcualtes the isochoric heat capacity from the results of a NVT ensemble Molecular dynamics simulation

    N: Number of particles
    PE: Instantaneous potential energy
    T: Temperature

    """
    E_squared = E**2
    mean_E = np.mean(E)
    mean_E_squared = mean_E**2
    E_squared_mean = np.mean(E_squared)
    cv = (E_squared_mean-mean_E_squared)/(Kb*(T**2))
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

def script(path_to_results):
    '''
    Calculates derivative properties for many LAMMPS runs by iterating through the results directory,
    calculating the derivative properties for each run then appending them to the results array. 
    '''
    NVT_results_filename = "nvtave.lammps"
    NPT_results_filename = "nptave.lammps"
    NPT_NVT_convergence_tolerance = 1e-3
    # Creating numpy arrays with the correct dimensions to append the results to
    array_size = (1,27)
    coallated_properties = np.zeros(array_size)
    for (root,dirs,files) in os.walk(path_to_results, topdown=True):
        # print(path_to_results)
        # print(root)
        # print(files)
        derivative_properties = []
        mean_NVT_results = np.zeros((1,1))
        mean_NPT_results = np.zeros((1,1))
        for filename in files:
            if (filename == NVT_results_filename):
                # Separate the folder structure into individual items in a list
                split_root_folder_structure = str.split(root,sep="/")
                # print(split_root_folder_structure)

                # Index the folder structure where the foldername that contains the temperature and density 
                temp_and_density_foldername = split_root_folder_structure[7]

                # Use regex to find the temperature and density from the folder name
                temp_and_density = re.findall("\d+\.\d+", temp_and_density_foldername)

                # Unpack the temperature and density list into the temperature and density variables
                temperature, density = temp_and_density

                # Convert temperature and density stings to floats
                temperature = float(temperature)
                density = float(density)

                # Join the results filename to the path to allow the data to be read
                path_to_results = os.path.join(root,filename)

                # Read the data and convert to numpy array
                data_df = pd.read_csv(path_to_results,
                                      skiprows=[0,1],
                                      delimiter=" ",
                                      )
                data_arr= data_df.to_numpy()
                mean_NVT_results = data_arr.mean(axis=0)
                # Assign columns of data to the respective variable
                total_energy = data_arr[:,1]
                temperature = data_arr[:,2]
                pressure = data_arr[:,3]
                density = data_arr[:,4]
                number_of_particles = 2048 #data_arr[:,5]
                kinetic_energy = data_arr[:,6]
                potential_energy = data_arr[:,7]
                enthalpy = data_arr[:,8]
                volume = data_arr[:,9]


                cv = isochoric_heat_capacity(number_of_particles,total_energy,temperature)
                gamma_v = thermal_pressure_coefficient(pressure,potential_energy,density,temperature,number_of_particles)
                derivative_properties.append(cv)
                derivative_properties.append(gamma_v)

            if filename== NPT_results_filename:
                 # Separate the folder structure into individual items in a list
                split_root_folder_structure = str.split(root,sep="/")
                # Index the folder structure where the foldername that contains the temperature and density 
                temp_and_density_foldername = split_root_folder_structure[7]

                # Use regex to find the temperature and density from the folder name
                temp_and_density = re.findall("\d+\.\d+", temp_and_density_foldername)

                # Unpack the temperature and density list into the temperature and density variables
                temperature, density = temp_and_density

                # Convert temperature and density stings to floats
                initial_temperature = float(temperature)
                initial_density = float(density)

                # Join the results filename to the path to allow the data to be read
                path_to_results = os.path.join(root,filename)

                # Read the data and convert to numpy array
                data_df = pd.read_csv(path_to_results,
                                      skiprows=[0,1],
                                      delimiter=" ",
                                      )
                data_arr= data_df.to_numpy()
                mean_NPT_results = data_arr.mean(axis=0)
                # Assign columns of data to the respective variable
                total_energy = data_arr[:,1]
                temperature = data_arr[:,2]
                pressure = data_arr[:,3]
                density = data_arr[:,4]
                number_of_particles = 2048 #data_arr[:,5]
                kinetic_energy = data_arr[:,6]
                potential_energy = data_arr[:,7]
                enthalpy = data_arr[:,8]
                volume = data_arr[:,9]

                 
                # inital_property_check(initial_temperature,temperature,initial_density,density,0.01) #Deprecated, script now removes runs with invalid densities from the array

                cp = isobaric_heat_capacity(total_energy,number_of_particles,pressure,volume,temperature)
                alpha_p = thermal_expansion_coefficient(total_energy,pressure,temperature,volume,number_of_particles)
                beta_t = isothermal_compressibility(volume,temperature)
                mu_jt = joule_thompson(density,temperature,cp,alpha_p)
                Z = compressibility_factor(pressure,density,temperature)
                derivative_properties.append(cp)
                derivative_properties.append(alpha_p)
                derivative_properties.append(beta_t)
                derivative_properties.append(mu_jt)
                derivative_properties.append(Z)

        # if the derivatives properties array is empty or or doesnt have the correct nmber of proeprties, dont write the data out
        # if the derivative_properties list is less than 7, the NPT or NVT results for a run were not computed thus the run should be skipped
        if (not derivative_properties) or (len(derivative_properties) < 7 ):
            continue
        print(derivative_properties)
        npt_nvt_derivative_results = np.concatenate((mean_NPT_results,mean_NVT_results,derivative_properties))
        coallated_properties=np.append(coallated_properties,[npt_nvt_derivative_results],axis=0)
    density_nvt_column = 4
    density_npt_column = density_nvt_column+10
    temperature_nvt_column = 2
    temperature_npt_column = temperature_nvt_column+10

    density_difference_between_ensembles = coallated_properties[:,density_nvt_column]-coallated_properties[:,density_npt_column]
    absolute_density_difference_between_ensembles = np.abs(density_difference_between_ensembles)
    temperature_difference_between_ensembles = coallated_properties[:,temperature_nvt_column]-coallated_properties[:,temperature_npt_column]
    absolute_temperature_difference_between_ensembles = np.abs(temperature_difference_between_ensembles)

    density_convergence_criteria = absolute_density_difference_between_ensembles>NPT_NVT_convergence_tolerance
    termperature_convergence_criteria = absolute_temperature_difference_between_ensembles>NPT_NVT_convergence_tolerance

    convergence_mask = (density_convergence_criteria) | (termperature_convergence_criteria)

    coallated_properties_with_removed_invalid_densities_and_temperatures = np.delete(coallated_properties, # From this array
                                                                    convergence_mask, # Delete any rows which violate the convergence criteria
                                                                    axis=0
                                                                    )
    strings_to_log = ["Number of runs before NPT NVT convergence check: {}".format(len(coallated_properties)),
                      "Number of runs after NPT NVT convergence check: {}".format(len(coallated_properties_with_removed_invalid_densities_and_temperatures))
                      ]
    
    with open('log.txt', 'w') as f:
        f.write("\n".join(strings_to_log))
        
    np.savetxt("coallated_results.txt",coallated_properties_with_removed_invalid_densities_and_temperatures)

script(path_to_results)