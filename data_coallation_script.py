import os
import pandas as pd
import numpy as np
import re
import sys

#Importing functions to calculate derivative properties from properties.py file
import properties
                        

path_to_results = r"/home/daniel/Downloads/LJ-2d-md-results/"
def script(path_to_results):
    '''
    Calculates derivative properties for many LAMMPS runs by iterating through the results directory,
    calculating the derivative properties for each run then appending them to the results array. 
    '''
    NVT_results_filename = "nvtave.lammps"
    NPT_results_filename = "nptave.lammps"
    # Creating numpy arrays with the correct dimensions to append the results to
    array_size = (1,12)
    coallated_means = np.zeros(array_size)
    coallated_standard_deviations = np.zeros(array_size)
    coallated_ranges = np.zeros(array_size)

    for (root,dirs,files) in os.walk(path_to_results, topdown=True):
        for filename in files:
            if (filename == NVT_results_filename):

                # Separate the folder structure into individual items in a list
                split_root_folder_structure = str.split(root,sep="/")

                # Index the folder structure where the foldername that contains the temperature and density 
                temp_and_density_foldername = split_root_folder_structure[5]

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

                # Test that the data has been read correctly 
                # if (temperature == 0.406597) & (density == 0.760723):
                #     assert total_energy[0] == -2.01527, "Data read failure , Total energy has been read incorrectly as per test case"
                #     assert pressure[0] == -0.0152766, "Data read failure , Pressure has been read incorrectly as per test case"
                #     assert kinetic_energy[0] ==  0.405846, "Data read failure , Kinetic energy has been read incorrectly as per test case"
                #     assert potential_energy[0] == -2.42111, "Data read failure , Potential energy has been read incorrectly as per test case"
                #     assert enthalpy[0] == -2.03535, "Data read failure , Enthalpy has been read incorrectly as per test case"
                #     assert volume[0] == 2692.18, "Data read failure , Volume has been read incorrectly as per test case"

                isochoric_heat_capacity = properties.isochoric_heat_capacity(number_of_particles,potential_energy,temperature)
                thermal_pressure_coefficient = properties.thermal_pressure_coefficient(pressure,potential_energy,density,temperature,number_of_particles)
                print(thermal_pressure_coefficient,temperature,density)
                # mean_values = np.mean(arr,axis=0)
                # means = np.append(mean_values,[temperature,density])
                # standard_deviations = data.std().values
                # standard_deviations = np.append(standard_deviations,[temperature,density])
                # ranges = data.max().values - data.min().values
                # ranges = np.append(ranges,[temperature,density])
    #             # if means[0] == 20000000.0:
    #             #     sns.lineplot(data = data, x = "TimeStep",y = data["v_TENE"],label = "Raw data")
    #             #     std_of_rolling_mean_arr = []
                    
    #             #     sns.lineplot(data = data, x = "TimeStep",y = data["v_TENE"].rolling(100).mean(),label = "Rolling mean, 100K timetseps")
    #             #     sns.lineplot(data = data, x = "TimeStep",y = data["v_TENE"].rolling(4000).mean(), label = "Rolling mean,4M timesteps")
    #             #     testing_range = range(1000,4000,10)
    #             #     for av_period in testing_range:
    #             #         std_of_rolling_mean = data["v_TENE"].rolling(av_period).mean().std()
    #             #         std_of_rolling_mean_arr.append(std_of_rolling_mean)
    #             #     plt.show()
    #             #     sns.lineplot(x=testing_range,y =std_of_rolling_mean_arr)
    #             #     # plt.show()
                # coallated_means = np.append(coallated_means,[means],axis=0)
                # coallated_standard_deviations = np.append(coallated_standard_deviations,[standard_deviations],axis=0)
                # coallated_ranges = np.append(coallated_ranges,[ranges],axis=0)
                
    # means_df = pd.DataFrame(coallated_means,columns=column_names+["Temperature","Density"])
    # np.savetxt("coallated_results.txt",means_df.values)




            
            if filename== NPT_results_filename:
                 # Separate the folder structure into individual items in a list
                split_root_folder_structure = str.split(root,sep="/")

                # Index the folder structure where the foldername that contains the temperature and density 
                temp_and_density_foldername = split_root_folder_structure[5]

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


                isobaric_heat_capacity = properties.isobaric_heat_capacity(total_energy,number_of_particles,pressure,volume,temperature)
                thermal_expansion_coefficient = properties.thermal_expansion_coefficient(total_energy,pressure,temperature,volume,number_of_particles)
                isothermal_compressibility = properties.isothermal_compressibility(volume,temperature)

script(path_to_results)