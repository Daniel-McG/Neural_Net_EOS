import os

results_filename = "log_nvt.lammps"
for (root,dirs,files) in os.walk(r"/home/daniel/LJ-2d-md-results", topdown=True):
    for filename in files:
        if filename == results_filename:
            path_to_results = os.path.join(root,filename)
            # currently_open_file = open(file=files[0],mode="r")
            print(path_to_results)
