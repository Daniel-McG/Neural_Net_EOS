# Neural_Net_EOS
A repository for the Python code from a 2023 masters research project on neural networks as equations of state.
## TODO
-[ ] Clean up unused package imports in scripts
-[ ] Create usage examples
  
## File descriptions
* VLSE_curve_notebook : Notebook used in getting the VLSE curves from the literature data  and generating the input points for the simulation
* input_points.txt : Input points for the simulation 
* data_collation_and_derivative_property_calc : Script used to collate data and calculate the derivative properties ( U ,Z , Cv, Cp,...)
* collated_results : The results collated from the MD simulations using the data_collation_and_derivative_property_calc.py script
* DataProcessing/Local_outlier : The script used to perform local outlier factor to replace outliers in the derivative properties
* DataProcessing/Removing_negative_pressure : The script used to remove datapoints where the pressure was negative
* neural_net_cpu_helmholtz_residual_NANS : The script used to train the ANN and perform hyperparameter optimisation
* new_eval : The script used to evaluate the perfomance of the ANN and plot the graphs used in the paper
   
