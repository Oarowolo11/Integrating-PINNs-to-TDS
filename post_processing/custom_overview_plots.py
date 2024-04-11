import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class custom_overview_1:
    def __init__(self, sim_time, t_test_rk_pinn_boost, states_rk_pinn_boost, states_assimulo) -> None:
        assert sim_time == t_test_rk_pinn_boost[-1]
        assert sim_time == states_assimulo[-1, 30]
        self.simulation_range = sim_time
        self.t_test_rk_pinn_boost = t_test_rk_pinn_boost
        self.t_test_assimulo = states_assimulo[:, 30]
        self.states_rk_pinn_boost = states_rk_pinn_boost
        self.states_assimulo = states_assimulo

    def compute_omegas(self, states_sim, no_machine):
        ind_omega = 3+10*(no_machine-1)
        res_trajectory = states_sim[:, ind_omega]
        return res_trajectory

    def run_plot(self):

        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(2, 1)

        ax0 = plt.subplot(gs[0, 0])

        ax0.plot(self.t_test_rk_pinn_boost, self.compute_omegas(self.states_rk_pinn_boost, 1), color = 'red', linestyle ='-', label='Machine 1', linewidth=3.5)
        ax0.plot(self.t_test_rk_pinn_boost, self.compute_omegas(self.states_rk_pinn_boost, 2), color = 'green', linestyle ='-', label='Machine 2', linewidth=3.5)
        ax0.plot(self.t_test_rk_pinn_boost, self.compute_omegas(self.states_rk_pinn_boost, 3), color = 'dodgerblue', linestyle ='-',  label='Machine 3', linewidth=3.5)

        ax0.plot(self.t_test_assimulo, self.compute_omegas(self.states_assimulo, 1), 'k-.', linewidth=3.5)
        ax0.plot(self.t_test_assimulo, self.compute_omegas(self.states_assimulo, 2), 'k-.', linewidth=3.5)
        ax0.plot(self.t_test_assimulo, self.compute_omegas(self.states_assimulo, 3), 'k-.', linewidth=3.5)

        ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labelsize=18)
        ax0.tick_params(axis='y', labelsize=18)
        ax0.set_ylabel('\u03C9 (rad/s)', fontsize=20)
        ax0.legend(fontsize=18)

        ax1 = plt.subplot(gs[1, 0])

        ax1.plot(self.t_test_rk_pinn_boost, self.states_rk_pinn_boost[:,  8], color = 'red', linestyle ='-',  label='Machine 1', linewidth=3.5)
        ax1.plot(self.t_test_rk_pinn_boost, self.states_rk_pinn_boost[:, 18], color = 'green', linestyle ='-',  label='Machine 2', linewidth=3.5)
        ax1.plot(self.t_test_rk_pinn_boost, self.states_rk_pinn_boost[:, 28], color = 'dodgerblue', linestyle ='-',  label='Machine 3', linewidth=3.5)

        ax1.plot(self.t_test_assimulo, self.states_assimulo[:,  8], 'k-.', linewidth=3.5)
        ax1.plot(self.t_test_assimulo, self.states_assimulo[:, 18], 'k-.', linewidth=3.5)
        ax1.plot(self.t_test_assimulo, self.states_assimulo[:, 28], 'k-.', linewidth=3.5)

        ax1.set_ylabel('V (pu)', fontsize=20)
        ax1.set_xlabel('Time (s)', fontsize=20)

        ax1.tick_params(axis='both', labelsize=18)
        ax1.legend(fontsize=18)

        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

    def show_results(self, save_fig=False):
        plt.tight_layout()

        if save_fig:
            plt.savefig('overviewfinal')

        plt.show()

class custom_overview_2:
    def __init__(self, sim_time, t_test_rk_pinn_boost, states_rk_pinn_boost, t_test_trap, states_evo, states_assimulo) -> None:
        assert sim_time == t_test_rk_pinn_boost[-1]
        assert sim_time == t_test_trap[-1]
        assert sim_time == states_assimulo[-1, 30]
        self.simulation_range = sim_time
        self.t_test_rk_pinn_boost = t_test_rk_pinn_boost
        self.states_rk_pinn_boost = states_rk_pinn_boost
        self.t_test_trap = t_test_trap
        self.states_evo = states_evo
        self.t_test_assimulo = states_assimulo[:, 30]
        self.states_assimulo = states_assimulo

    def compute_omegas(self, states_sim, no_machine):
        ind_omega = 3+10*(no_machine-1)
        res_trajectory = states_sim[:, ind_omega]
        return res_trajectory

    def run_plot(self):

        fig = plt.figure(figsize=(14, 5))
        gs = gridspec.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0, 0])

        ax1.plot(self.t_test_rk_pinn_boost, self.compute_omegas(self.states_rk_pinn_boost, 1), color = 'red', linestyle ='-', label='Machine 1', linewidth=3.5)
        ax1.plot(self.t_test_rk_pinn_boost, self.compute_omegas(self.states_rk_pinn_boost, 2), color = 'green', linestyle ='-', label='Machine 2', linewidth=3.5)
        ax1.plot(self.t_test_rk_pinn_boost, self.compute_omegas(self.states_rk_pinn_boost, 3), color = 'dodgerblue', linestyle ='-',  label='Machine 3', linewidth=3.5)

        ax1.plot(self.t_test_trap, self.compute_omegas(self.states_evo, 1), color = 'red', linestyle ='--', linewidth=3.5)
        ax1.plot(self.t_test_trap, self.compute_omegas(self.states_evo, 2), color = 'green', linestyle ='--', linewidth=3.5)
        ax1.plot(self.t_test_trap, self.compute_omegas(self.states_evo, 3), color = 'dodgerblue', linestyle ='--', linewidth=3.5)

        ax1.plot(self.t_test_assimulo, self.compute_omegas(self.states_assimulo, 1), 'k-.', linewidth=3.5)
        ax1.plot(self.t_test_assimulo, self.compute_omegas(self.states_assimulo, 2), 'k-.', linewidth=3.5)
        ax1.plot(self.t_test_assimulo, self.compute_omegas(self.states_assimulo, 3), 'k-.', linewidth=3.5)
        
        ax1.legend()

    def show_results(self, save_fig=False):
        plt.tight_layout()

        if save_fig:
            plt.savefig('overviewfinal')

        plt.show()