import os
import pandas as pd
import numpy as np
import re
import sys
path_to_results = r"/home/daniel/Downloads/LJ-2d-md-results/"
Kb = 1 
# Temperature - Intensive
# Pressure - Intensive
# Density - intensive

# Total energy - since its the sum of kinetic and potential
# these are both extensive thus the sum will be extensive
# Kinetic energy - Extensive
# Potential energy - extensive
# Volume - Extensive
# Enthalpy - extensive
def difference_in_mean_of_halves(arr):
    length_of_array = len(arr)
    mean_of_first_half = np.mean(arr[:length_of_array//2,:],axis=0)
    mean_of_second_half = np.mean(arr[length_of_array//2:,:],axis=0)
    absolute_difference_in_mean = np.abs(mean_of_first_half-mean_of_second_half)
    return absolute_difference_in_mean

def isochoric_heat_capacity(N,E,T):
    """
    Calcualtes the isochoric heat capacity from the results of a NVT ensemble Molecular dynamics simulation

    N: Number of particles
    E: Instantaneous total energy
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

    H: Instantaneous enthalpy
    N: Number of particles
    V: Instantaneous volume
    '''
    N_mean = np.mean(N)
    dH = (H-np.mean(H))*N_mean
    dV = (V-np.mean(V))*N_mean
    T_average = np.mean(T)
    V_average = np.mean(V)*N_mean
    alpha_p = np.mean(dH*dV) / (V_average*T_average**2)
    return alpha_p/N_mean

def isothermal_compressibility(V,T,N):
    """
    Calculates the isothermal compressibility from the results of a NPT ensemble molecular dynamics simulation

    V: Instantaneous volume
    T: Instantaneous temperature
    """
    N_mean = np.mean(N)
    V_average = np.mean(V)*N_mean
    T_average = np.mean(T)
    dV = (V-np.mean(V))*N_mean
    betaT = np.mean(dV**2) / (V_average*T_average)
    return betaT/N_mean

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
    return gamma_v/N_mean

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
    coallated_standard_deviations=np.zeros((1,20))
    for (root,dirs,files) in os.walk(path_to_results, topdown=True):
            # print(path_to_results)
            # print(root)
            # print(files)
            derivative_properties_dict = {}
            mean_NVT_results = np.zeros((1,1))
            mean_NPT_results = np.zeros((1,1))
            for filename in files:
                if (filename == NVT_results_filename):
                    convergence_criteria = {"timesteps":np.inf,
                                            "total_energy":1e-03,
                                            "temperature":1e-03,
                                            "pressure":1e-03,
                                            "density":1e-03,
                                            "n_molecules":1e-03,
                                            "kinetic_energy":1e-03,
                                            "potential_energy":1e-03,
                                            "enthalpy":1e-02,
                                            "volume":1e-2}
                    # # Separate the folder structure into individual items in a list
                    # split_root_folder_structure = str.split(root,sep="/")
                    # # print(split_root_folder_structure)

                    # # Index the folder structure where the foldername that contains the temperature and density 
                    # temp_and_density_foldername = split_root_folder_structure[5]

                    # # Use regex to find the temperature and density from the folder name
                    # temp_and_density = re.findall("\d+\.\d+", temp_and_density_foldername)

                    # # Unpack the temperature and density list into the temperature and density variables
                    # temperature, density = temp_and_density

                    # # Convert temperature and density stings to floats
                    # temperature = float(temperature)
                    # density = float(density)

                    # Join the results filename to the path to allow the data to be read
                    path_to_results = os.path.join(root,filename)
                    
                    # Read the data and convert to numpy array
                    data_df = pd.read_csv(path_to_results,
                                        skiprows=[0,1],
                                        delimiter=" ",
                                        )
                    

                    data_arr= data_df.to_numpy()
                    difference_in_means = difference_in_mean_of_halves(data_arr)

                    # If the simulatioon has not converged, dont take the results from this file 
                    # if (difference_in_means[1]>convergence_criteria["total_energy"] or
                    #     difference_in_means[2]>convergence_criteria["temperature"] or
                    #     difference_in_means[3]>convergence_criteria["pressure"] or 
                    #     difference_in_means[4]>convergence_criteria["density"] or
                    #     # Skip 5 since its a constant
                    #     difference_in_means[6]>convergence_criteria["kinetic_energy"] or
                    #     difference_in_means[7]>convergence_criteria["potential_energy"] or
                    #     difference_in_means[8]>convergence_criteria["enthalpy"] or
                    #     difference_in_means[9]>convergence_criteria["volume"]):
                        
                    #     continue

                    mean_NVT_results = data_arr.mean(axis=0)
                    std_NVT_results = data_arr.std(axis = 0)
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
                    derivative_properties_dict["cv"]=cv
                    derivative_properties_dict["gamma_v"]=gamma_v

                if filename== NPT_results_filename:
                    convergence_criteria = {"timesteps":np.inf,
                        "total_energy":1e-03,
                        "temperature":1e-03,
                        "pressure":1e-03,
                        "density":1e-03,
                        "n_molecules":1e-03,
                        "kinetic_energy":1e-03,
                        "potential_energy":1e-03,
                        "enthalpy":1e-02,
                        "volume":1e-2}
                    # Separate the folder structure into individual items in a list
                    # split_root_folder_structure = str.split(root,sep="/")
                    # # Index the folder structure where the foldername that contains the temperature and density 
                    # temp_and_density_foldername = split_root_folder_structure[5]

                    # # Use regex to find the temperature and density from the folder name
                    # temp_and_density = re.findall("\d+\.\d+", temp_and_density_foldername)

                    # # Unpack the temperature and density list into the temperature and density variables
                    # temperature, density = temp_and_density

                    # # Convert temperature and density stings to floats
                    # initial_temperature = float(temperature)
                    # initial_density = float(density)

                    # Join the results filename to the path to allow the data to be read
                    path_to_results = os.path.join(root,filename)

                    # Read the data and convert to numpy array
                    data_df = pd.read_csv(path_to_results,
                                        skiprows=[0,1],
                                        delimiter=" ",
                                        )
                    data_arr= data_df.to_numpy()

                    difference_in_means = difference_in_mean_of_halves(data_arr)

                    # If the simulatioon has not converged, dont take the results from this file 
                    # if (difference_in_means[1]>convergence_criteria["total_energy"] or
                    #     difference_in_means[2]>convergence_criteria["temperature"] or
                    #     difference_in_means[3]>convergence_criteria["pressure"] or 
                    #     difference_in_means[4]>convergence_criteria["density"] or
                    #     # Skip 5 since its a constant
                    #     difference_in_means[6]>convergence_criteria["kinetic_energy"] or
                    #     difference_in_means[7]>convergence_criteria["potential_energy"] or
                    #     difference_in_means[8]>convergence_criteria["enthalpy"] or
                    #     difference_in_means[9]>convergence_criteria["volume"]):
                        
                    #     continue
                    mean_NPT_results = data_arr.mean(axis=0)
                    std_NPT_results = data_arr.std(axis = 0)
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

                    cp = isobaric_heat_capacity(enthalpy,number_of_particles,temperature)
                    alpha_p = thermal_expansion_coefficient(enthalpy,temperature,volume,number_of_particles)
                    beta_t = isothermal_compressibility(volume,temperature,number_of_particles)
                    mu_jt = joule_thompson(density,temperature,cp,alpha_p)
                    Z = compressibility_factor(pressure,density,temperature)
                    derivative_properties_dict["cp"]=cp
                    derivative_properties_dict["alphaP"]=alpha_p
                    derivative_properties_dict["beta_t"]=beta_t
                    derivative_properties_dict["mu_jt"]=mu_jt
                    derivative_properties_dict["Z"]=Z

            # if the derivatives properties array is empty or or doesnt have the correct nmber of proeprties, dont write the data out
            # if the derivative_properties list is less than 7, the NPT or NVT results for a run were not computed thus the run should be skipped
            if (not derivative_properties_dict) or (len(derivative_properties_dict) < 7 ):
                continue
            derivative_properties =[derivative_properties_dict["cp"],
                        derivative_properties_dict["alphaP"],
                        derivative_properties_dict["beta_t"],
                        derivative_properties_dict["mu_jt"],
                        derivative_properties_dict["Z"],
                        derivative_properties_dict["cv"],
                        derivative_properties_dict["gamma_v"]]
            # print(std_NVT_results)
            npt_nvt_derivative_results = np.concatenate((mean_NPT_results,mean_NVT_results,derivative_properties))
            coallated_properties=np.append(coallated_properties,[npt_nvt_derivative_results],axis=0)
            standard_deviations = np.concatenate((std_NPT_results,std_NVT_results))
            coallated_standard_deviations = np.append(coallated_standard_deviations,[standard_deviations],axis=0)
            
    
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
        
    np.savetxt("local_coallated_results_debug.txt",coallated_properties_with_removed_invalid_densities_and_temperatures)
    np.savetxt("coallated_standard_deviations.txt",coallated_standard_deviations)

script(path_to_results)