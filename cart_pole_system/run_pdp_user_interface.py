# inspired from,
# @article{jin2019pontryagin,
#   title={Pontryagin Differentiable Programming: An End-to-End Learning and Control Framework},
#   author={Jin, Wanxin and Wang, Zhaoran and Yang, Zhuoran and Mou, Shaoshuai},
#   journal={arXiv preprint arXiv:1912.12970},
#   year={2019}
# }

from PDP_library import POP, LQR
from CartPoleEnv import CartInvertedPendulumEnv
from animate_cartpole import Animation_CartPole
from pdp_iteration import PDP_learning
from TrajectoryPlot import Trajectories
from LossPlot import Loss_Computation
from casadi import *
import math
import scipy.io as sio
import numpy as np
import time
import matplotlib.pyplot as plt

# Environment Initialization
cart_pole = CartInvertedPendulumEnv(g=10, mass_c=0.5, mass_p=0.5, length=2)
# cart_pole.cost_initialization(w_controls = 0.1)
cart_pole.cost_initialization()


# Primal Optimization Program using initialized PDP
sample_time = 0.1
dyn = cart_pole.X + sample_time * cart_pole.f
pop_cartpole = POP(weights=vertcat(cart_pole.weights), state = cart_pole.X, state_lb=[], state_ub=[], 
                   control = cart_pole.U, control_lb=[], control_ub=[], ode = dyn, stage_cost = cart_pole.stage_cost,
                   terminal_cost = cart_pole.terminal_cost)
lqr_traj_solver = LQR()

# Choose step wise operation to view results (from 1 to 5)
print('Choose any number from {1,2,3,4,5} to compute results on step wise process')
user_choice = 1
# user_choice = input("# choose any number from {1,2,3,4,5} to compute results on step wise process: ") 

# step 1: generate optimal trajectories with respect to assumed quadratic cost 
if user_choice == 1:
    # structure using Primal Optimization Program
    true_weights = [1, 6, 1, 1]

    # generate animations 
    final_time = 100
    trajectories = []
    initial_state = np.zeros(pop_cartpole.n_state)
    theta_initial = [0, -0.5, -0.25, 0.25, math.pi/6]

    animation_cart_pole = Animation_CartPole(pole_length = 2)

    # 3 sets of trajectories for costate, control and state are generated
    for i in range(5):  
        initial_state[1] = theta_initial[i]
        traj_sol = pop_cartpole.popSolver(initial_state=initial_state, horizon=final_time, weights_value=true_weights)
        animation_cart_pole.save_animation(pole_length=2, dt=sample_time, state_trajectory_pop=traj_sol['state_trajectories'],save_option=1)
        trajectories = trajectories + [traj_sol]
    # save
    sio.savemat('data_pdp_cartpole/cartpole_pop.mat', {'trajectories': trajectories,
                                              'dt': sample_time,
                                              'true_parameter': true_weights})

# step 2: pdp iteration (you can skip this step and move on to 3, 4, 5 for viewing obtained results)
if user_choice == 2:
    data_trajectories = sio.loadmat('data_pdp_cartpole/cartpole_pop.mat')
    trajectories = data_trajectories['trajectories']
    true_weights = data_trajectories['true_parameter']
    
    # define object for loss computation
    pdp_compute = PDP_learning(cart_pole, pop_cartpole, lqr_traj_solver)
    # start iteration (please make user choice = 1 first to load trajectories and true weights for computation then make user choice = 2)
    pdp_compute.compute_loss_ioc(cart_pole, pop_cartpole, lqr_traj_solver, trajectories, true_weights)
    
# step 3: plot loss trace vs iteration
if user_choice == 3:
    # define object for loss computation
    loss_object = Loss_Computation(cart_pole, pop_cartpole, lqr_traj_solver)
    loss_object.plot_loss()
    
    
# step 4: plot trajectories vs ground truth, plot weights vs iteration 
if user_choice == 4:
    #set initial state
    initial_state=np.array([0,math.pi/6,0,0])
    final_time = 100
    true_weights = [1, 6, 1, 1]
    system_trajectory = Trajectories(cart_pole, pop_cartpole, lqr_traj_solver)
    system_trajectory.plot_trajectories(initial_state, final_time, true_weights)
    

# step 5: generate animations for comparing primal optimizaton program and auxillary optimization program
if user_choice == 5:
    # structure using Primal Optimization Program
    true_weights = [1, 6, 1, 1]

    #iteration obtained weights for comparison
    load_pdp_results = sio.loadmat('data_pdp_cartpole/PDP_results_trial_0.mat')
    weight_trace = load_pdp_results['results']['weights_trace'][0,0]
    pdp_weight_trace = np.squeeze(weight_trace)
    
    # structure using Auxillary Optimization Program
    iter_trace = 0 #user choice to view nth iteration obtained weights (0<n<10000)
    obtained_weights = pdp_weight_trace[iter_trace,:]
    
    # generate animations for comparison
    final_time = 100
    trajectories = []
    initial_state = np.zeros(pop_cartpole.n_state)
    theta_initial = [0, -0.5, -0.25, 0.25, math.pi/6]
    animation_cart_pole = Animation_CartPole(pole_length = 2)

    # 3 sets of trajectories for costate, control and state are generated
    for i in range(5):  
        initial_state[1] = theta_initial[i]
        traj_sol = pop_cartpole.popSolver(initial_state=initial_state, horizon=final_time, weights_value=true_weights)
        traj_sol2 = pop_cartpole.popSolver(initial_state=initial_state, horizon=final_time, weights_value=obtained_weights)
        animation_cart_pole.save_animation(pole_length=2, dt=sample_time, state_trajectory_pdp=traj_sol2['state_trajectories'],state_trajectory_pop=traj_sol['state_trajectories'],save_option=1)
        trajectories =  trajectories + [traj_sol]
    # save
    sio.savemat('data_pdp_cartpole/cartpole_pop.mat', {'trajectories': trajectories,
                                              'dt': sample_time,
                                              'true_parameter': true_weights})
    
