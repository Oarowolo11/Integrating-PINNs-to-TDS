import os
import argparse
import time
import yaml
import numpy as np

import torch

from src.tds_dae_rk_schemes import TDS_simulation

from src.initial_conditions_calculation import compute_initial_conditions

import src.PINN_architecture

from post_processing.trajectories_overview_plot import trajectories_overview
from post_processing.custom_overview_plots import custom_overview_1, custom_overview_2

parser = argparse.ArgumentParser('TDS-Simulation')
parser.add_argument('--system', type=str, choices=['ieee9bus'], default='ieee9bus')
parser.add_argument('--machine', type=int, choices=[1, 2, 3], default=3)
parser.add_argument('--event_type', choices=['p_setpoint', 'w_setpoint'], default='w_setpoint')
parser.add_argument('--event_location', type=int, choices=[1, 2, 3], default=3)
parser.add_argument('--event_magnitude', type=float, default=1e-2)
parser.add_argument('--sim_time', type=float, default=2.)
parser.add_argument('--time_step_size', type=float, default=1e-2)
parser.add_argument('--rk_scheme', choices=['trapezoidal', 'backward_euler'], default='trapezoidal')
parser.add_argument('--compare_pure_RKscheme', action='store_true')
parser.add_argument('--compare_ground_truth', action='store_true', default=True)
parser.add_argument('--plot_selection', choices=[1, 2, 3], default=1)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


if args.compare_ground_truth:
    gt_dir = './gt_simulations/'
    file_name = f'sim{int(args.sim_time)}s_{args.event_type}_{args.event_location}.npy'
    file_npy = os.path.join(gt_dir, file_name)

    if not os.path.isfile(file_npy):
        args.compare_ground_truth = False

def config_file(yaml_file) -> dict:
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def extract_values_dynamic_components(config_file) -> tuple:
    freq     = config_file['freq']
    H        = list(config_file['inertia_H'].values())
    Rs       = list(config_file['Rs'].values())
    Xd_p     = list(config_file['Xd_prime'].values())
    pg_pf    = list(config_file['Pg_setpoints'].values())
    dampings = list(config_file['Damping_D'].values())
    return freq, H, Rs, Xd_p, pg_pf, dampings

def extract_values_static_components(config_file) -> tuple:
    voltages_magnitude = list(config_file['Voltage_magnitude'].values())
    voltages_angles    = list(config_file['Voltage_angle'].values())
    Xd                 = list(config_file['Xd'].values())
    Xq                 = list(config_file['Xq'].values())
    Xq_prime           = list(config_file['Xq_prime'].values())
    voltages_complex   = np.array(voltages_magnitude)*np.exp(1j*np.array(voltages_angles)*np.pi/180)

    return voltages_complex, Xd, Xq, Xq_prime

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def load_pinn_machine(pinn_model):
    loaded_pinn = torch.load(pinn_model, map_location=device)
    norm_range, lb_range                     = loaded_pinn['range_norm']
    unorm_range, lb_u_range                  = loaded_pinn['range_unorm']
    num_neurons, num_layers, inputs, outputs = loaded_pinn['architecture']
    pinn_integrated = src.PINN_architecture.FCN(inputs,outputs,num_neurons,num_layers, norm_range, lb_range, unorm_range, lb_u_range)
    pinn_integrated.load_state_dict(loaded_pinn['state_dict'])
    pinn_integrated.eval()
    return pinn_integrated

def load_pinn_parameters(pinn_model) -> tuple:
    loaded_pinn = torch.load(pinn_model, map_location=device)
    damping_pinn, pg_pinn, H_pinn, Xd_pinn, _ = loaded_pinn['machine_parameters']
    return H_pinn, Xd_pinn, pg_pinn, damping_pinn

def compute_pinn_ops_limits(pinn_model) -> tuple:
    loaded_pinn = torch.load(pinn_model, map_location=device)
    voltage_limits = loaded_pinn['voltage_stats'][0]
    theta_limits   = loaded_pinn['theta_stats'][0]
    delta_limits   = loaded_pinn['init_state'][2]
    omega_limits   = loaded_pinn['init_state'][3]
    return voltage_limits, theta_limits, delta_limits, omega_limits

def apply_contingency(initial_conditions_sys, pg_pf_sys) -> tuple:
    initial_conditions_sim = initial_conditions_sys
    pg_pf_sim = pg_pf_sys
    if args.event_type == 'w_setpoint':
        if args.event_location == 1:
            initial_conditions_sim[3+ 10*(args.event_location-1)] += args.event_magnitude
        elif args.event_location == 2:
            initial_conditions_sim[3+ 10*(args.event_location-1)] += args.event_magnitude
        elif args.event_location == 3:
            initial_conditions_sim[3+ 10*(args.event_location-1)] += args.event_magnitude
    elif args.event_type == 'p_setpoint':
        assert args.event_location != args.machine
        if args.event_location == 1:
            pg_pf_sim[args.event_location-1] += args.event_magnitude
        elif args.event_location == 2:
            pg_pf_sim[args.event_location-1] += args.event_magnitude
        elif args.event_location == 3:
            pg_pf_sim[args.event_location-1] += args.event_magnitude
    return initial_conditions_sim, pg_pf_sim


if __name__ == "__main__":

    pinn_directory = './final_models/'
    pinn_name  = f'model_DAE_machine_{args.machine}.pth'

    pinn_location = os.path.join(pinn_directory, pinn_name)

    simulation_pinn = load_pinn_machine(pinn_location)

    parameters_dc_raw = config_file('./config_files/config_machines_dynamic.yaml')
    freq, H, Rs, Xd_p, pg_pf, dampings = extract_values_dynamic_components(parameters_dc_raw)
    assert len(H)     == 3; assert len(Rs)       == 3; assert len(Xd_p) == 3
    assert len(pg_pf) == 3; assert len(dampings) == 3

    parameters_pinn = load_pinn_parameters(pinn_location)

    H[args.machine-1] = parameters_pinn[0]
    Xd_p[args.machine-1] = parameters_pinn[1]
    pg_pf[args.machine-1] = parameters_pinn[2]
    dampings[args.machine-1] = parameters_pinn[3]

    assert all(damp > 0 for damp in dampings)
    assert all(inertia > 0 for inertia in H)

    pinn_ops_limits = compute_pinn_ops_limits(pinn_location)

    Yadmittance = torch.load('./config_files/network_admittance.pt')

    parameters_initialization = config_file('./config_files/config_machines_static.yaml')
    volt, Xd, Xq, Xq_p = extract_values_static_components(parameters_initialization)
    study_system = compute_initial_conditions(volt, H, Rs, Xd, Xd_p, Xq, Xq_p, np.array(Yadmittance), simplification=True)
    ini_cond = study_system.compute_initial_conditions(volt)
    
    ini_cond_sim, pg_pf_sim = apply_contingency(initial_conditions_sys=torch.tensor(ini_cond, dtype=torch.float64), pg_pf_sys=pg_pf)

    assert args.sim_time > 0.
    assert args.time_step_size > 0.
    t_final_simulations = args.sim_time
    step_size_pure_rk = args.time_step_size
    step_size_hybrid_rk = args.time_step_size

    solver_pure_rk_method = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf_sim, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_pure_rk)
    t_test_pure_rk, states_evo = solver_pure_rk_method.simulation_main_loop(integration_scheme=args.rk_scheme)

    solver_hybrid_pinn      = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf_sim, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_hybrid_rk, 
                                         pinn_boost=args.machine, pinn_weights=simulation_pinn, pinn_limits=pinn_ops_limits)
    
    t_test_pinn, states_evo_pinn = solver_hybrid_pinn.simulation_main_loop(integration_scheme=args.rk_scheme)

    activate_assimulo = args.compare_ground_truth

    if activate_assimulo:
        gt_dir = './gt_simulations/'
        file_name = f'sim{int(args.sim_time)}s_{args.event_type}_{args.event_location}.npy'
        file_npy = os.path.join(gt_dir, file_name)
        states_assimulo = np.load(file_npy)
    else:
        states_assimulo = None

    if args.plot_selection == 1:
        plotting = trajectories_overview(args.sim_time, t_test_pinn, states_evo_pinn, t_test_pure_rk, states_evo, states_assimulo)
        plotting.compute_results(pure_rk_scheme=True, assimulo_states=True, compare_trajectories=True)
        plotting.show_results(save_fig=False)
    
    elif args.plot_selection == 2:
        plotting = custom_overview_1(args.sim_time, t_test_pinn, states_evo_pinn, states_assimulo)
        plotting.run_plot()
        plotting.show_results(save_fig=False)
    
    elif args.plot_selection == 3:
        plotting = custom_overview_2(args.sim_time, t_test_pinn, states_evo_pinn, t_test_pure_rk, states_evo, states_assimulo)
        plotting.run_plot()
        plotting.show_results(save_fig=False)