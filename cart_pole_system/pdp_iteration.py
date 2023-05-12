# inspired from,
# @article{jin2019pontryagin,
#   title={Pontryagin Differentiable Programming: An End-to-End Learning and Control Framework},
#   author={Jin, Wanxin and Wang, Zhaoran and Yang, Zhuoran and Mou, Shaoshuai},
#   journal={arXiv preprint arXiv:1912.12970},
#   year={2019}
# }

import scipy.io as sio
import numpy as np
import time

class PDP_learning:
    def __init__(self, cart_pole, pop_cartpole, lqr_traj_solver):
        self.system = cart_pole
        self.pop = pop_cartpole
        self.aop = lqr_traj_solver
        
    def compute_loss_ioc(self, cart_pole, pop_cartpole, lqr_traj_solver, trajectories, true_weights):
        trial = 0  # trial number
        start_time = time.time()
        step_size = 1e-5  # learning rate
        # initialize
        iml_ioc_loss, weights_trace = [], []
        s = 9.9 # sigma value
        initial_weights = true_weights + s * np.random.random(len(true_weights)) - s / 2
        # initial_weights = np.random.random(len(true_weights))
        current_weights = initial_weights
        for epoch in range(int(1e4)): # iteration loop (or epoch loop)
            ioc_loss = 0
            d_w = np.zeros(current_weights.shape)
            # loop for each tryouts trajectory
            n_tryout = trajectories.shape[1]
            for iter in range(n_tryout):
                tryout_state_traj = trajectories[0, iter]['state_trajectories'][0, 0]
                tryout_control_traj = trajectories[0, iter]['control_trajectories'][0, 0]
                tryout_initial_state = tryout_state_traj[0, :]
                tryout_horizon = tryout_control_traj.shape[0]
                traj = pop_cartpole.popSolver(tryout_initial_state, tryout_horizon, current_weights)
                # Auxiliary control system
                aux_sys = pop_cartpole.getAuxSys(state_trajectories=traj['state_trajectories'],
                                               control_trajectories=traj['control_trajectories'],
                                               costate_trajectories=traj['costate_trajectories'],
                                               weights_value=current_weights)
                lqr_traj_solver.setDyn(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'], dynE=aux_sys['dynE'])
                lqr_traj_solver.initialize_stage_cost(Hxx=aux_sys['Hxx'], Huu=aux_sys['Huu'], Hxu=aux_sys['Hxu'], Hux=aux_sys['Hux'],
                                       Hxe=aux_sys['Hxe'], Hue=aux_sys['Hue'])
                lqr_traj_solver.initialize_terminal_cost(hxx=aux_sys['hxx'], hxe=aux_sys['hxe'])
                aux_sol = lqr_traj_solver.lqrSolver(np.zeros((pop_cartpole.n_state, pop_cartpole.n_weights)), tryout_horizon)
                # Solution of the auxiliary control system
                dxd_w_traj = aux_sol['state_trajectories']
                dud_w_traj = aux_sol['control_trajectories']
                # Ioc_loss
                state_traj = traj['state_trajectories']
                control_traj = traj['control_trajectories']
                dldx_traj = state_traj - tryout_state_traj
                dldu_traj = control_traj - tryout_control_traj
                ioc_loss = ioc_loss + np.linalg.norm(dldx_traj) ** 2 + np.linalg.norm(dldu_traj) ** 2
                for t in range(tryout_horizon):
                    d_w = d_w + np.matmul(dldx_traj[t, :], dxd_w_traj[t]) + np.matmul(dldu_traj[t, :], dud_w_traj[t])
                d_w = d_w + np.dot(dldx_traj[-1, :], dxd_w_traj[-1])

            # take the expectation (average)
            d_w = d_w / n_tryout
            ioc_loss = ioc_loss / n_tryout
            # update
            current_weights = current_weights - step_size * d_w
            weights_trace += [current_weights]
            iml_ioc_loss += [ioc_loss]

            # print and terminal check
            if epoch % 1 == 0:
                print('trial #', trial, 'iter: ', epoch,    ' ioc_loss: ', iml_ioc_loss[-1].tolist())

        # save
        save_data = {'trial_no': trial,
                      'initial_weights': initial_weights,
                      'ioc_loss_trace': iml_ioc_loss,
                      'weights_trace': weights_trace,
                      'learning_rate': step_size,
                      'time_passed': time.time() - start_time}
        sio.savemat('./data_pdp_cartpole/PDP_results_trial_' + str(trial) + '.mat', {'results': save_data})

