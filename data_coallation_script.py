import os
import pandas as pd
import numpy as np
import re
import sys  
results_filename = "nvtave.lammps"
column_names=["TimeStep","v_TENE","v_TEMP","v_PRES","v_DENS","v_nMolecule","v_KENE","v_PENE","v_ENTH","v_VOL"]
array_size = (1,12)
coallated_means = np.zeros(array_size)
coallated_standard_deviations = np.zeros(array_size)
coallated_ranges = np.zeros(array_size)
for (root,dirs,files) in os.walk(r"/home/daniel/LJ-2d-md-results", topdown=True):
    for filename in files:
        if filename == results_filename:
            split_root_folder_structure = str.split(root,sep="/")
            temp_and_density_filename = split_root_folder_structure[4]
            temp_and_density = re.findall("\d+\.\d+", temp_and_density_filename)

            temperature, density = temp_and_density
            temperature = float(temperature)
            density = float(density)

            path_to_results = os.path.join(root,filename)
            data = pd.read_csv(path_to_results,skiprows=[0,1],delimiter=" ",names=column_names)

            means = data.mean().values
            means = np.append(means,[temperature,density])
            standard_deviations = data.std().values
            standard_deviations = np.append(standard_deviations,[temperature,density])
            ranges = data.max().values - data.min().values
            ranges = np.append(ranges,[temperature,density])

            coallated_means = np.append(coallated_means,[means],axis=0)
            coallated_standard_deviations = np.append(coallated_standard_deviations,[standard_deviations],axis=0)
            coallated_ranges = np.append(coallated_ranges,[ranges],axis=0)
            
means_df = pd.DataFrame(coallated_means,columns=column_names+["Temperature","Density"])
print(sys.getsizeof(means_df))