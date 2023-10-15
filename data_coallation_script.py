import os
import pandas as pd
results_filename = "nvtave.lammps"
for (root,dirs,files) in os.walk(r"/home/daniel/LJ-2d-md-results", topdown=True):
    for filename in files:
        if filename == results_filename:
            path_to_results = os.path.join(root,filename)
            pd.read_csv(path_to_results,skiprows=[0],delimiter=" ")
            # currently_open_file = open(file=path_to_results,mode="r")
            # print(currently_open_file.read())
