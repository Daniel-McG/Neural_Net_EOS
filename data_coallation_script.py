import os
import pandas as pd
import numpy as np
import re
results_filename = "nvtave.lammps"
column_names=["TimeStep","v_TENE","v_TEMP","v_PRES","v_DENS","v_nMolecule","v_KENE","v_PENE","v_ENTH","v_VOL"]
coallated_means = np.zeros((1,12))
coallated_standard_deviations = np.zeros((1,10))
for (root,dirs,files) in os.walk(r"/home/daniel/LJ-2d-md-results", topdown=True):
    for filename in files:
        if filename == results_filename:
            split_root_folder_structure = str.split(root,sep="/")
            temp_and_density_filename = split_root_folder_structure[4]
            temp_and_density = re.findall("\d+\.\d+", temp_and_density_filename)
            temperature, density = temp_and_density
            path_to_results = os.path.join(root,filename)
            data = pd.read_csv(path_to_results,skiprows=[0,1],delimiter=" ",names=column_names)
            coallated_means = np.append(coallated_means,[data.mean().values],axis=0)
            coallated_standard_deviations = np.append(coallated_standard_deviations,[data.std().values],axis=0)
means_df = pd.DataFrame(coallated_means,columns=column_names)
print(means_df)