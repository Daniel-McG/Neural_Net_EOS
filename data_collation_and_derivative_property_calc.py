import os
import pandas as pd
import numpy as np
import re
import sys

path_to_results = r"PATH/TO/RESULTS"
Kb = 1


def difference_in_std_by_mean_of_halves(arr):
    length_of_array = len(arr)
    mean_of_first_half = np.mean(arr[: length_of_array // 2, :], axis=0)
    std_of_first_half = np.std(arr[: length_of_array // 2, :], axis=0)
    mean_of_second_half = np.mean(arr[length_of_array // 2 :, :], axis=0)
    std_of_second_half = np.std(arr[length_of_array // 2 :, :], axis=0)
    absolute_difference_in_mean = np.abs(
        (std_of_first_half / mean_of_first_half)
        - (std_of_second_half / mean_of_second_half)
    )
    return absolute_difference_in_mean


def isochoric_heat_capacity(N, E, T):
    """
    Calcualtes the isochoric heat capacity from the results of a NVT ensemble Molecular dynamics simulation

    N: Number of particles
    E: Instantaneous total energy
    T: Temperature

    """
    N_mean = np.mean(N)
    dE = (E - np.mean(E)) * N_mean
    T_mean = np.mean(T)
    cv = np.mean(dE**2) / (Kb * (T_mean**2))
    cv /= N_mean
    return cv


def isobaric_heat_capacity(H, N, T):
    """
    Calculates the isobarc heat capacity from the results of a NPT ensemble molecuar dynamics simulation
    H: Instantaneous Enthalpy
    N: Number of particles
    T: Instantaneous Temperature
    """
    N_mean = np.mean(N)
    T_mean = np.mean(T)
    dH = (H - np.mean(H)) * N_mean
    cp = np.mean(dH**2) / T_mean**2
    cp /= N_mean
    return cp


def thermal_expansion_coefficient(H, T, V, N):
    """
    Calculates the thermal expansion coefficient from the results of a NPT ensemble molecular dynamics simulation

    H: Instantaneous enthalpy
    N: Number of particles
    V: Instantaneous volume
    T: Instantaneous temperature
    """
    N_mean = np.mean(N)
    dH = (H - np.mean(H)) * N_mean
    dV = V - np.mean(V)
    T_average = np.mean(T)
    V_average = np.mean(V)
    alpha_p = np.mean(dH * dV) / (V_average * T_average**2)
    return alpha_p


def isothermal_compressibility(V, T):
    """
    Calculates the isothermal compressibility from the results of a NPT ensemble molecular dynamics simulation

    V: Instantaneous volume
    T: Instantaneous temperature
    """
    V_average = np.mean(V)
    T_average = np.mean(T)
    dV = V - np.mean(V)
    betaT = np.mean(dV**2) / (V_average * T_average)
    return betaT


def thermal_pressure_coefficient(P, PE, rho, T, N):
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
    dPE = (PE - np.mean(PE)) * N_mean
    dP = P - np.mean(P)
    gamma_v = np.mean(dPE * dP) / T_average**2 + rho_average
    return gamma_v


def compressibility_factor(P, rho, T):
    """
    Calculates the compressibility factor from the results of an NPT or NVT ensemble
    P: Instantaneous pressure
    rho: Instantaneous density
    T: Instantaneous temperature
    """
    rho_average = np.mean(rho)
    P_average = np.mean(P)
    T_average = np.mean(T)
    Z = P_average / (rho_average * T_average)
    return Z


def joule_thomson(rho, T, Cp, alpha_p):
    """
    Calculates the joule thompson coefficient from the results of an NPT ensemble molecular dynamics simulation
    rho: instantaneous heat density
    T: Instantaneous temperature
    Cp: Ensemble heat capacity
    Alpha_P: Ensemble thermal expansion coefficient
    """
    T_mean = np.mean(T)
    rho_mean = np.mean(rho)
    JT = (1 / (rho_mean * Cp)) * (T_mean * alpha_p - 1)
    return JT


def script(path_to_results):
    """
    Calculates derivative properties for many LAMMPS runs by iterating through the results directory,
    calculating the derivative properties for each run then appending them to the results dict.
    """
    NVT_results_filename = "nvtave.lammps"
    NPT_results_filename = "nptave.lammps"
    NPT_NVT_convergence_tolerance = 1e-3
    # Creating numpy arrays with the correct dimensions to append the results to
    array_size = (1, 27)
    collated_properties = np.zeros(array_size)
    collated_standard_deviations = np.zeros((1, 20))
    for root, dirs, files in os.walk(path_to_results, topdown=True):
        derivative_properties_dict = {}
        mean_NVT_results = np.zeros((1, 1))
        mean_NPT_results = np.zeros((1, 1))
        for filename in files:
            if filename == NVT_results_filename:
                convergence_criteria = {
                    "timesteps": np.inf,
                    "total_energy": 5e-02,
                    "temperature": 5e-02,
                    "pressure": 5e-02,
                    "density": 5e-02,
                    "n_molecules": 5e-02,
                    "kinetic_energy": 5e-02,
                    "potential_energy": 5e-02,
                    "enthalpy": 5e-02,
                    "volume": 5e-2,
                }

                # Join the results filename to the path to allow the data to be read
                path_to_results = os.path.join(root, filename)

                # Read the data and convert to numpy array
                data_df = pd.read_csv(
                    path_to_results,
                    skiprows=[0, 1],
                    delimiter=" ",
                )
                data_arr = data_df.to_numpy()
                difference_in_means = difference_in_std_by_mean_of_halves(data_arr)

                # If the simulatioon has not converged, dont take the results from this file
                if (
                    difference_in_means[1] > convergence_criteria["total_energy"]
                    or difference_in_means[2] > convergence_criteria["temperature"]
                    or difference_in_means[3] > convergence_criteria["pressure"]
                    or
                    # Skip 5 since its a constant
                    difference_in_means[6] > convergence_criteria["kinetic_energy"]
                    or difference_in_means[7] > convergence_criteria["potential_energy"]
                    or difference_in_means[8] > convergence_criteria["enthalpy"]
                ):

                    continue

                mean_NVT_results = data_arr.mean(axis=0)
                std_NVT_results = data_arr.std(axis=0)

                # Assign columns of data to the respective variable
                total_energy = data_arr[:, 1]
                temperature = data_arr[:, 2]
                pressure = data_arr[:, 3]
                density = data_arr[:, 4]
                number_of_particles = 2048  # data_arr[:,5]
                kinetic_energy = data_arr[:, 6]
                potential_energy = data_arr[:, 7]
                # enthalpy = data_arr[:,8]
                volume = data_arr[:, 9]

                # Redefining enlthalpy
                enthalpy = (
                    total_energy.flatten()
                    + np.mean(pressure.flatten()) / density.flatten()
                )

                # Check if the values of Cv or GammaV are converged in the last 4/6th to 6/6th of the timesteps
                cv_list = []
                gamma_v_list = []
                for timesteps in range(
                    4 * len(temperature) // 6, len(temperature), 100
                ):
                    cv = isochoric_heat_capacity(
                        number_of_particles,
                        total_energy[:timesteps],
                        temperature[:timesteps],
                    )
                    gamma_v = thermal_pressure_coefficient(
                        pressure[:timesteps],
                        potential_energy[:timesteps],
                        density[:timesteps],
                        temperature[:timesteps],
                        number_of_particles,
                    )
                    cv_list.append(cv)
                    gamma_v_list.append(gamma_v)

                mean_cv = np.mean(cv_list)
                std_cv = np.std(cv_list)

                mean_gamma_v = np.mean(gamma_v_list)
                std_gamma_v = np.std(gamma_v_list)

                if np.abs(std_cv / mean_cv) > 0.05:
                    cv = np.nan
                else:
                    cv = cv_list[-1]

                if np.abs(std_gamma_v / mean_gamma_v) > 0.05:
                    gamma_v = np.nan
                else:
                    gamma_v = gamma_v_list[-1]

                derivative_properties_dict["cv"] = cv
                derivative_properties_dict["gamma_v"] = gamma_v

            if filename == NPT_results_filename:
                convergence_criteria = {
                    "timesteps": np.inf,
                    "total_energy": 5e-02,
                    "temperature": 5e-02,
                    "pressure": 5e-02,
                    "density": 5e-02,
                    "n_molecules": 5e-02,
                    "kinetic_energy": 5e-02,
                    "potential_energy": 5e-02,
                    "enthalpy": 5e-02,
                    "volume": 5e-2,
                }

                # Join the results filename to the path to allow the data to be read
                path_to_results = os.path.join(root, filename)

                # Read the data and convert to numpy array
                data_df = pd.read_csv(
                    path_to_results,
                    skiprows=[0, 1],
                    delimiter=" ",
                )
                data_arr = data_df.to_numpy()

                difference_in_means = difference_in_std_by_mean_of_halves(data_arr)

                # If the simulatioon has not converged, dont take the results from this file
                if (
                    difference_in_means[1] > convergence_criteria["total_energy"]
                    or difference_in_means[2] > convergence_criteria["temperature"]
                    or difference_in_means[3] > convergence_criteria["pressure"]
                    or difference_in_means[4] > convergence_criteria["density"]
                    or
                    # Skip 5 since its a constant
                    difference_in_means[6] > convergence_criteria["kinetic_energy"]
                    or difference_in_means[7] > convergence_criteria["potential_energy"]
                    or difference_in_means[8] > convergence_criteria["enthalpy"]
                    or difference_in_means[9] > convergence_criteria["volume"]
                ):

                    continue
                mean_NPT_results = data_arr.mean(axis=0)
                std_NPT_results = data_arr.std(axis=0)
                # Assign columns of data to the respective variable
                total_energy = data_arr[:, 1]
                temperature = data_arr[:, 2]
                pressure = data_arr[:, 3]
                density = data_arr[:, 4]
                number_of_particles = 2048
                kinetic_energy = data_arr[:, 6]
                potential_energy = data_arr[:, 7]
                volume = data_arr[:, 9]

                # Redefining enlthalpy
                enthalpy = (
                    total_energy.flatten()
                    + np.mean(pressure.flatten()) / density.flatten()
                )

                # Check if the values of Cp, alphaP,BetaT,MuJT, and Z are converged in the last 4/6th to 6/6th of the timesteps.
                # If they aren't converged, swap the value for  that property in that run for "nan",
                # the "nan"'s are then caught in the ANN
                cp_list = []
                alpha_p_list = []
                beta_t_list = []
                mu_jt_list = []
                Z_list = []
                for timesteps in range(
                    4 * len(temperature) // 6, len(temperature), 100
                ):
                    cp = isobaric_heat_capacity(
                        enthalpy[:timesteps],
                        number_of_particles,
                        temperature[:timesteps],
                    )
                    alpha_p = thermal_expansion_coefficient(
                        enthalpy[:timesteps],
                        temperature[:timesteps],
                        volume[:timesteps],
                        number_of_particles,
                    )
                    beta_t = isothermal_compressibility(
                        volume[:timesteps], temperature[:timesteps]
                    )
                    mu_jt = joule_thomson(
                        density[:timesteps], temperature[:timesteps], cp, alpha_p
                    )
                    Z = compressibility_factor(
                        pressure[:timesteps],
                        density[:timesteps],
                        temperature[:timesteps],
                    )

                    cp_list.append(cp)
                    alpha_p_list.append(alpha_p)
                    beta_t_list.append(beta_t)
                    mu_jt_list.append(mu_jt)
                    Z_list.append(Z)

                mean_cp = np.mean(cp_list)
                std_cp = np.std(cp_list)

                mean_alpha_p = np.mean(alpha_p_list)
                std_alpha_p = np.std(alpha_p_list)

                mean_beta_t = np.mean(beta_t_list)
                std_beta_t = np.std(beta_t_list)

                mean_mu_jt = np.mean(mu_jt_list)
                std_mu_jt = np.std(mu_jt_list)

                mean_Z = np.mean(Z_list)
                std_Z = np.std(Z_list)

                tolerance = 0.05
                if np.abs(std_cp / mean_cp) > tolerance:
                    cp = np.nan
                else:
                    cp = cp_list[-1]

                if np.abs(std_alpha_p / mean_alpha_p) > tolerance:
                    alpha_p = np.nan
                else:
                    alpha_p = alpha_p_list[-1]

                if np.abs(std_beta_t / mean_beta_t) > tolerance:
                    beta_t = np.nan
                else:
                    beta_t = beta_t_list[-1]

                if np.abs(std_mu_jt / mean_mu_jt) > tolerance:
                    mu_jt = np.nan
                else:
                    mu_jt = mu_jt_list[-1]

                if np.abs(std_Z / mean_Z) > tolerance:
                    Z = np.nan
                else:
                    Z = Z_list[-1]

                derivative_properties_dict["cp"] = cp
                derivative_properties_dict["alphaP"] = alpha_p
                derivative_properties_dict["beta_t"] = beta_t
                derivative_properties_dict["mu_jt"] = mu_jt
                derivative_properties_dict["Z"] = Z

        # if the derivatives properties array is empty or or doesnt have the correct nmber of proeprties, dont write the data out
        # if the derivative_properties list is less than 7, the NPT or NVT results for a run were not computed thus the run should be skipped
        if (not derivative_properties_dict) or (len(derivative_properties_dict) < 7):
            continue
        derivative_properties = [
            derivative_properties_dict["cp"],
            derivative_properties_dict["alphaP"],
            derivative_properties_dict["beta_t"],
            derivative_properties_dict["mu_jt"],
            derivative_properties_dict["Z"],
            derivative_properties_dict["cv"],
            derivative_properties_dict["gamma_v"],
        ]
        # print(std_NVT_results)
        npt_nvt_derivative_results = np.concatenate(
            (mean_NPT_results, mean_NVT_results, derivative_properties)
        )
        collated_properties = np.append(
            collated_properties, [npt_nvt_derivative_results], axis=0
        )
        standard_deviations = np.concatenate((std_NPT_results, std_NVT_results))
        collated_standard_deviations = np.append(
            collated_standard_deviations, [standard_deviations], axis=0
        )

    density_nvt_column = 4
    density_npt_column = density_nvt_column + 10
    temperature_nvt_column = 2
    temperature_npt_column = temperature_nvt_column + 10

    density_difference_between_ensembles = (
        collated_properties[:, density_nvt_column]
        - collated_properties[:, density_npt_column]
    )
    absolute_density_difference_between_ensembles = np.abs(
        density_difference_between_ensembles
    )
    temperature_difference_between_ensembles = (
        collated_properties[:, temperature_nvt_column]
        - collated_properties[:, temperature_npt_column]
    )
    absolute_temperature_difference_between_ensembles = np.abs(
        temperature_difference_between_ensembles
    )

    density_convergence_criteria = (
        absolute_density_difference_between_ensembles > NPT_NVT_convergence_tolerance
    )
    termperature_convergence_criteria = (
        absolute_temperature_difference_between_ensembles
        > NPT_NVT_convergence_tolerance
    )

    convergence_mask = (density_convergence_criteria) | (
        termperature_convergence_criteria
    )

    collated_properties_with_removed_invalid_densities_and_temperatures = np.delete(
        collated_properties,  # From this array
        convergence_mask,  # Delete any rows which violate the convergence criteria
        axis=0,
    )
    strings_to_log = [
        "Number of runs before NPT NVT convergence check: {}".format(
            len(collated_properties)
        ),
        "Number of runs after NPT NVT convergence check: {}".format(
            len(collated_properties_with_removed_invalid_densities_and_temperatures)
        ),
    ]

    with open("log.txt", "w") as f:
        f.write("\n".join(strings_to_log))

    np.savetxt(
        "local_collated_results_debug.txt",
        collated_properties_with_removed_invalid_densities_and_temperatures,
    )
    np.savetxt("collated_standard_deviations.txt", collated_standard_deviations)


script(path_to_results)
