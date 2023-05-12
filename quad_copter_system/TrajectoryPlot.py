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
    def __init__(self, quad, pop_quadcopter, lqr_traj_solver):
        self.system = quad
        self.pop = pop_quadcopter
        self.aop = lqr_traj_solver
        
    def plot_trajectories(self, initial_state, horizon, true_weights):
        true_sol = self.pop.popSolver(initial_state=initial_state, horizon=horizon, weights_value=true_weights)
        load_data = sio.loadmat('data_pdp_quadcopter/PDP_results_trial_0.mat')
        weights_trace = load_data['results']['weights_trace'][0,0]
        pdp_weights_trace = np.squeeze(weights_trace)
        pdp_sol = self.pop.popSolver(initial_state=initial_state, horizon=horizon, weights_value=pdp_weights_trace[-1, :])
        
        # # --------------------------- plot ----------------------------------------
        params = {'axes.labelsize': 30,
                  'axes.titlesize': 20,
                  'xtick.labelsize':20,
                  'ytick.labelsize':20,
                  'legend.fontsize':20}
        plt.rcParams.update(params)

        # quadrotor control force
        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots()
        line_gt1,=ax.plot(true_sol['control_trajectories'][:,0], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
        line_pdp1,=ax.plot(pdp_sol['control_trajectories'][:,0], color ='#0052BD', linewidth=5)
        line_gt2,=ax.plot(true_sol['control_trajectories'][:,1], color ='#9784A4', linewidth=10, linestyle='dashed', alpha=0.7)
        line_pdp2,=ax.plot(pdp_sol['control_trajectories'][:,1], color ='#704684', linewidth=5)
        line_gt3,=ax.plot(true_sol['control_trajectories'][:,2], color ='#799A80', linewidth=10, linestyle='dashed', alpha=0.7)
        line_pdp3,=ax.plot(pdp_sol['control_trajectories'][:,2], color ='#194747', linewidth=5)
        line_gt4,=ax.plot(true_sol['control_trajectories'][:,3], color ='#EF9D0E', linewidth=10, linestyle='dashed', alpha=0.7)
        line_pdp4,=ax.plot(pdp_sol['control_trajectories'][:,3], color ='#EF2D0E', linewidth=5)
        ax.set_ylabel('Quadrotor Forces')
        ax.set_xlabel('Time')
        ax.set_xlim(-3,60)
        ax.set_facecolor('#E6E6E6')
        ax.grid()

        fig.suptitle('Quadcopter Thrusts vs Ground Truth', fontsize=40)

        plt.legend([line_gt1, line_pdp1,line_gt2, line_pdp2,line_gt3, line_pdp3,line_gt4, line_pdp4], 
                    ['Ground truth($u_1$)', 'PDP($u_1$)', 'Ground truth($u_1$)', 'PDP($u_2$)','Ground truth($u_3$)', 'PDP($u_3$)', 
                    'Ground truth($u_4$)', 'PDP($u_4$)'], facecolor='white', framealpha=0.5,loc='best')
        plt.show()

        # quadrotor states linear position rx, ry, rz
        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots()
        line_rxt,=ax.plot(true_sol['state_trajectories'][:,0], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
        line_rx_pdp,=ax.plot(pdp_sol['state_trajectories'][:,0], color ='#2A596C', linewidth=5)
        line_ryt,=ax.plot(true_sol['state_trajectories'][:,1], color ='#9784A4', linewidth=10, linestyle='dashed', alpha=0.7)
        line_ry_pdp,=ax.plot(pdp_sol['state_trajectories'][:,1], color ='#704684', linewidth=5)
        line_rzt,=ax.plot(true_sol['state_trajectories'][:,2], color ='#799A80', linewidth=10, linestyle='dashed', alpha=0.7)
        line_rz_pdp,=ax.plot(pdp_sol['state_trajectories'][:,2], color ='#194747', linewidth=5)
        ax.set_ylabel('States')
        ax.set_xlabel('Time')
        # ax.set_ylim(-10,2)
        ax.set_xlim(-3,60)
        ax.set_facecolor('#E6E6E6')
        ax.grid()

        fig.suptitle('Quadcopter Position vs Ground Truth', fontsize=40)

        plt.legend([line_rxt, line_rx_pdp, line_ryt, line_ry_pdp, line_rzt, line_rz_pdp], 
                    ['Ground truth ($r_x$)', 'PDP ($r_x$)', 'Ground truth ($r_y$)', 'PDP ($r_y$)' , 'Ground truth ($r_z$)', 'PDP ($ r_z$)'], facecolor='white', framealpha=0.5, fontsize=10, loc='best')

        plt.show()

        # # quadrotor states linear velocity vx, vy, vz
        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots()
        line_vxt,=ax.plot(true_sol['state_trajectories'][:,3], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
        line_vx_pdp,=ax.plot(pdp_sol['state_trajectories'][:,3], color ='#2A596C', linewidth=5)
        line_vyt,=ax.plot(true_sol['state_trajectories'][:,4], color ='#9784A4', linewidth=10, linestyle='dashed', alpha=0.7)
        line_vy_pdp,=ax.plot(pdp_sol['state_trajectories'][:,4], color ='#704684', linewidth=5)
        line_vzt,=ax.plot(true_sol['state_trajectories'][:,5], color ='#799A80', linewidth=10, linestyle='dashed', alpha=0.7)
        line_vz_pdp,=ax.plot(pdp_sol['state_trajectories'][:,5], color ='#194747', linewidth=5)
        ax.set_ylabel('States')
        ax.set_xlabel('Time')
        # ax.set_ylim(-10,2)
        ax.set_xlim(-3,60)
        ax.set_facecolor('#E6E6E6')
        ax.grid()

        fig.suptitle('Quadcopter Velocity vs Ground Truth', fontsize=40)

        plt.legend([line_vxt, line_vx_pdp, line_vyt, line_vy_pdp, line_vzt, line_vz_pdp], 
                    ['Ground truth ($v_x$)', 'PDP ($v_x$)', 'Ground truth ($v_y$)', 'PDP ($v_y$)' , 'Ground truth ($v_z$)', 'PDP ($v_z$)'], facecolor='white', framealpha=0.5, fontsize=10, loc='best')


        plt.show()

        # # quadrotor q quaternion q0, q1, q2, q3
        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots()
        line_q0t,=ax.plot(true_sol['state_trajectories'][:,6], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
        line_q0_pdp,=ax.plot(pdp_sol['state_trajectories'][:,6], color ='#2A596C', linewidth=5)
        line_q1t,=ax.plot(true_sol['state_trajectories'][:,7], color ='#9784A4', linewidth=10, linestyle='dashed', alpha=0.7)
        line_q1_pdp,=ax.plot(pdp_sol['state_trajectories'][:,7], color ='#704684', linewidth=5)
        line_q2t,=ax.plot(true_sol['state_trajectories'][:,8], color ='#799A80', linewidth=10, linestyle='dashed', alpha=0.7)
        line_q2_pdp,=ax.plot(pdp_sol['state_trajectories'][:,8], color ='#194747', linewidth=5)
        line_q3t,=ax.plot(true_sol['state_trajectories'][:,9], color ='#A3546E', linewidth=10, linestyle='dashed', alpha=0.7)
        line_q3_pdp,=ax.plot(pdp_sol['state_trajectories'][:,9], color ='#A2142F', linewidth=5)
        ax.set_ylabel('States')
        ax.set_xlabel('Time')
        # ax.set_ylim(-10,2)
        ax.set_xlim(-3,60)
        ax.set_facecolor('#E6E6E6')
        ax.grid()

        fig.suptitle('Quadcopter Angular Position ($Q$) vs Ground Truth', fontsize=40)

        plt.legend([line_q0t, line_q0_pdp, line_q1t, line_q1_pdp, line_q2t, line_q2_pdp, line_q3t, line_q3_pdp], 
                    ['Ground truth ($q_0$)', 'PDP ($q_0$)', 'Ground truth ($q_1$)', 'PDP ($q_1$)' , 'Ground truth ($q_2$)', 'PDP ($q_2$)', 
                      'Ground truth ($q_3$)', 'PDP ($q_3$)'], facecolor='white', framealpha=0.5, fontsize=10, loc='best')


        plt.show()


        # quadrotor w quaternion wx, wy, wz
        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots()
        line_wxt,=ax.plot(true_sol['state_trajectories'][:,10], color ='#0072BD', linewidth=10, linestyle='dashed', alpha=0.7)
        line_wx_pdp,=ax.plot(pdp_sol['state_trajectories'][:,10], color ='#2A596C', linewidth=5)
        line_wyt,=ax.plot(true_sol['state_trajectories'][:,11], color ='#9784A4', linewidth=10, linestyle='dashed', alpha=0.7)
        line_wy_pdp,=ax.plot(pdp_sol['state_trajectories'][:,11], color ='#704684', linewidth=5)
        line_wzt,=ax.plot(true_sol['state_trajectories'][:,12], color ='#799A80', linewidth=10, linestyle='dashed', alpha=0.7)
        line_wz_pdp,=ax.plot(pdp_sol['state_trajectories'][:,12], color ='#194747', linewidth=5)
        ax.set_ylabel('States')
        ax.set_xlabel('Time')
        # ax.set_ylim(-10,2)
        ax.set_xlim(-3,60)
        ax.set_facecolor('#E6E6E6')
        ax.grid()

        fig.suptitle('Quadcopter Angular Velocity ($w$) vs Ground Truth', fontsize=40)

        plt.legend([line_wxt, line_wx_pdp, line_wyt, line_wy_pdp, line_wzt, line_wz_pdp], ['Ground truth ($w_x$)', 'PDP ($w_x$)',
                'Ground truth ($w_y$)', 'PDP ($w_y$)' , 'Ground truth ($w_z$)', 'PDP ($w_z$)'], facecolor='white', framealpha=0.5,
                    fontsize=10, loc='best')


        plt.show()


        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots()
        line_wr, =ax.plot(np.linspace(1,10000,10000),pdp_weights_trace[:,0], color ='#090ff7', linewidth=5)
        line_wv, =ax.plot(np.linspace(1,10000,10000),pdp_weights_trace[:,1], color ='#A2142F', linewidth=5)
        line_ww, =ax.plot(np.linspace(1,10000,10000),pdp_weights_trace[:,3], color ='#ef2d0e', linewidth=5)
        ax.set_ylabel('Weighting Parameters ($w_r,w_v, w_w$)')
        ax.set_xlabel('Number of Iterations')
        ax.set_facecolor('#E6E6E6')
        # ax.set_ylim(6,6.4)
        ax.grid()

        fig.suptitle('Cost Function Weights ($w_r,w_v,w_w$)', fontsize=40)
        plt.legend([line_wr, line_wv, line_ww], ['$w_r$','$w_v$','$w_w$'], facecolor='white', framealpha=0.5, loc='best')
        plt.show()

        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots()
        line_wq, =ax.plot(np.linspace(1,10000,10000),pdp_weights_trace[:,2], color ='#38761d', linewidth=5)
        ax.set_ylabel('Weighting Parameters ($w_q$)')
        ax.set_xlabel('Number of Iterations')
        ax.set_facecolor('#E6E6E6')
        # ax.set_ylim(6,6.4)
        ax.grid()

        fig.suptitle('Cost Function Weights ($w_q$)', fontsize=40)
        plt.legend([line_wq], ['$w_q$'], facecolor='white', framealpha=0.5, loc='best')
        plt.show()