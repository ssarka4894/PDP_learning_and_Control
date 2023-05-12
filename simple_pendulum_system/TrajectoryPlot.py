# inspired from,
# @article{jin2019pontryagin,
#   title={Pontryagin Differentiable Programming: An End-to-End Learning and Control Framework},
#   author={Jin, Wanxin and Wang, Zhaoran and Yang, Zhuoran and Mou, Shaoshuai},
#   journal={arXiv preprint arXiv:1912.12970},
#   year={2019}
# }

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


class Trajectories:
    def __init__(self, pend, pop_pendulum, lqr_solver):
        self.system = pend
        self.pop = pop_pendulum
        self.aop = lqr_solver
        
    def plot_trajectories(self, initial_state, horizon, true_weights):
        true_sol = self.pop.popSolver(initial_state=initial_state, horizon=horizon, weights_value=true_weights)
        load_data = sio.loadmat('data_pdp_pendulum/PDP_results_trial_0.mat')
        weights_trace = load_data['results']['weights_trace'][0,0]
        pdp_weights_trace = np.squeeze(weights_trace)
        pdp_sol = self.pop.popSolver(initial_state=initial_state, horizon=horizon, weights_value=pdp_weights_trace[-1, :])

        # --------------------------- plot ----------------------------------------
        params = {'axes.labelsize': 30,
                  'axes.titlesize': 20,
                  'xtick.labelsize':20,
                  'ytick.labelsize':20,
                  'legend.fontsize':20}
        plt.rcParams.update(params)
        
        # Pendulum Controls
        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots()
        line_gt,=ax.plot(true_sol['control_trajectories'], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
        line_pdp,=ax.plot(pdp_sol['control_trajectories'], color ='#A2142F', linewidth=5)
        ax.set_ylabel('Pendulum Torque')
        ax.set_xlabel('Time')
        ax.set_facecolor('#E6E6E6')
        ax.grid()
        fig.suptitle('Pendulum force vs Ground Truth', fontsize=40)
        plt.legend([line_gt, line_pdp], ['Ground truth', 'PDP'], facecolor='white', framealpha=0.5,
                    loc='best')
        plt.show()

        # Pendulum States
        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots()
        line_qt,=ax.plot(true_sol['state_trajectories'][:,0], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
        line_q_pdp,=ax.plot(pdp_sol['state_trajectories'][:,0], color ='#2A596C', linewidth=5)
        line_dqt,=ax.plot(true_sol['state_trajectories'][:,1], color ='#9784A4', linewidth=10, linestyle='dashed', alpha=0.7)
        line_dq_pdp,=ax.plot(pdp_sol['state_trajectories'][:,1], color ='#704684', linewidth=5)
        ax.set_ylabel('States')
        ax.set_xlabel('Time')
        ax.set_ylim(-2,8)
        ax.set_facecolor('#E6E6E6')
        ax.grid()
        fig.suptitle('Pendulum States vs Ground Truth', fontsize=40)
        plt.legend([line_qt, line_q_pdp, line_dqt, line_dq_pdp], 
                    [ 'Ground truth ($ \\theta$)', 'PDP ($ \\theta$)' , 'Ground truth ($ \dot{\\theta}$)', 'PDP ($ \dot{\\theta}$)'], facecolor='white', framealpha=0.5, fontsize=10, loc='best')
        plt.show()

        # Pendulum Weights
        fig = plt.figure(figsize=(14, 8))
        ax = fig.subplots()
        line_q, =ax.plot(np.linspace(1,10000,10000),pdp_weights_trace[:,0], color ='#090ff7', linewidth=5)
        ax.plot(np.linspace(1,10000,10000),10*np.ones(10000),linestyle = '--', linewidth=5)
        ax.text(5000,10,'True Weight Reference Value',ha='center', va='center',rotation='horizontal', backgroundcolor='lightgray')
        ax.set_ylabel('Weighting Parameters ($w_{q}$)')
        ax.set_xlabel('Number of Iterations')
        ax.set_facecolor('#E6E6E6')
        ax.grid()

        fig.suptitle('Cost Function Weights ($w_{q}$)', fontsize=40)
        plt.legend([line_q], ['$w_{q}$'], facecolor='white', framealpha=0.5, loc='best')
        plt.show()

        # Pendulum Weights
        fig = plt.figure(figsize=(14, 8))
        ax = fig.subplots()
        line_dq, =ax.plot(np.linspace(1,10000,10000),pdp_weights_trace[:,1], color ='#38761d', linewidth=5)
        ax.plot(np.linspace(1,10000,10000),np.ones(10000),linestyle = '--', linewidth=5)
        ax.text(5000,1,'True Weight Reference Value',ha='center', va='center',rotation='horizontal', backgroundcolor='lightgray')
        ax.set_ylabel('Weighting Parameters ($w_{dq}$)')
        ax.set_xlabel('Number of Iterations')
        ax.set_facecolor('#E6E6E6')
        ax.grid()

        fig.suptitle('Cost Function Weights ($w_{dq}$)', fontsize=40)
        plt.legend([line_dq], ['$w_{dq}$'], facecolor='white', framealpha=0.5, loc='best')
        plt.show()
