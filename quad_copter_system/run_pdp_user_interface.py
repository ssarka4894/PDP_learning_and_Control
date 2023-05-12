# inspired from,
# @article{jin2019pontryagin,
#   title={Pontryagin Differentiable Programming: An End-to-End Learning and Control Framework},
#   author={Jin, Wanxin and Wang, Zhaoran and Yang, Zhuoran and Mou, Shaoshuai},
#   journal={arXiv preprint arXiv:1912.12970},
#   year={2019}
# }

from PDP_library import POP, LQR
from QuadcopterEnv import QuadcopterUAVEnv
from animate_quadcopter import Animation_Quadrotor
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
quad = QuadcopterUAVEnv(g=10, Inertia_x=1, Inertia_y=1, Inertia_z=1, mass=1, length=0.4, kappa=0.01)
quad.cost_initialization(w_controls = 0.1)
# quad.cost_initialization(w_controls = 0.1)


# Primal Optimization Program using initialized PDP
sample_time = 0.1
dyn = quad.X + sample_time * quad.f
pop_quadcopter = POP(weights=vertcat(quad.weights), state = quad.X, state_lb=[], state_ub=[], 
                   control = quad.U, control_lb=[], control_ub=[], ode = dyn, stage_cost = quad.stage_cost,
                   terminal_cost = quad.terminal_cost)
lqr_traj_solver = LQR()


# # converter to quaternion from (slope, vector)
def computeQuarternion(slope, vector):
    if type(vector) == list:
        vector = numpy.array(vector)
    vector = vector / numpy.linalg.norm(vector)
    quat = numpy.zeros(4)
    quat[0] = math.cos(slope / 2)
    quat[1:] = math.sin(slope / 2) * vector
    return quat.tolist()

# Choose step wise operation to view results (from 1 to 5)
print('Choose any number from {1,2,3,4,5} to compute results on step wise process')
user_choice = 1
# user_choice = input("# choose any number from {1,2,3,4,5} to compute results on step wise process: ") 

# step 1: generate optimal trajectories with respect to assumed quadratic cost 
if user_choice == 1:
    # structure using Primal Optimization Program
    true_weights = [1, 1, 5, 1]
    

    # generate animations 
    final_time = 60
    trajectories = []
    alternative_position_I=[[-8, -6, 9.],[8, 6, 9.]]
    initial_velocity_I = [0.0, 0.0, 0.0]
    initial_quarternion = computeQuarternion(0,[1,-1,1])
    initial_omega = [0.0, 0.0, 0.0]

    animation_quad = Animation_Quadrotor(wing_length = 1.5)

    # 3 sets of trajectories for costate, control and state are generated
    for i in range(2):  
        initial_state = alternative_position_I[i] + initial_velocity_I + initial_quarternion + initial_omega
        traj_sol = pop_quadcopter.popSolver(initial_state=initial_state, horizon=final_time, weights_value=true_weights)
        animation_quad.save_animation(wing_length=1.5, dt=sample_time, state_trajectory_pop=traj_sol['state_trajectories'],save_option=1)
        trajectories = trajectories + [traj_sol]
    # save
    sio.savemat('data_pdp_quadcopter/quadcopter_pop.mat', {'trajectories': trajectories,
                                              'dt': sample_time,
                                              'true_parameter': true_weights})

# step 2: pdp iteration (you can skip this step and move on to 3, 4, 5 for viewing obtained results)
if user_choice == 2:
    data_trajectories = sio.loadmat('data_pdp_quadcopter/quadcopter_pop.mat')
    trajectories = data_trajectories['trajectories']
    true_weights = data_trajectories['true_parameter']
    
    # define object for loss computation
    pdp_compute = PDP_learning(quad, pop_quadcopter, lqr_traj_solver)
    # start iteration (please make user choice = 1 first to load trajectories and true weights for computation then make user choice = 2)
    pdp_compute.compute_loss_ioc(quad, pop_quadcopter, lqr_traj_solver, trajectories, true_weights)
    
# step 3: plot loss trace vs iteration
if user_choice == 3:
    # define object for loss computation
    loss_object = Loss_Computation(quad, pop_quadcopter, lqr_traj_solver)
    loss_object.plot_loss()
    
    
# step 4: plot trajectories vs ground truth, plot weights vs iteration 
if user_choice == 4:
    # structure using Primal Optimization Program
    true_weights = [1, 1, 5, 1]
    
    # set initial state
    initial_position_i=[8, 6, 9]
    initial_velocity_i = [0.0, 0.0, 0.0]
    initial_quarternion = computeQuarternion(0,[1,0,0])
    initial_omega = [0.0, 0.0, 0.0]
    initial_state = initial_position_i + initial_velocity_i + initial_quarternion + initial_omega
    final_time = 100
    true_weights = [1, 1, 5, 1]
    
    system_trajectory = Trajectories(quad, pop_quadcopter, lqr_traj_solver)
    system_trajectory.plot_trajectories(initial_state, final_time, true_weights)
    

# step 5: generate animations for comparing primal optimizaton program and auxillary optimization program
if user_choice == 5:
    # structure using Primal Optimization Program
    true_weights = [1, 1, 5, 1]

    #iteration obtained weights for comparison
    load_pdp_results = sio.loadmat('data_pdp_quadcopter/PDP_results_trial_0.mat')
    weight_trace = load_pdp_results['results']['weights_trace'][0,0]
    pdp_weight_trace = np.squeeze(weight_trace)
    
    # structure using Auxillary Optimization Program
    iter_trace = 0 #user choice to view nth iteration obtained weights (0<n<10000)
    obtained_weights = pdp_weight_trace[iter_trace,:]
    
    # generate animations 
    final_time = 60
    trajectories = []
    alternative_position_I=[[-8, -6, 9.],[8, 6, 9.]]
    initial_velocity_I = [0.0, 0.0, 0.0]
    initial_quarternion = computeQuarternion(0,[1,-1,1])
    initial_omega = [0.0, 0.0, 0.0]

    animation_quad = Animation_Quadrotor(wing_length = 1.5)

    # 3 sets of trajectories for costate, control and state are generated
    for i in range(2):  
        initial_state = alternative_position_I[i] + initial_velocity_I + initial_quarternion + initial_omega
        traj_sol = pop_quadcopter.popSolver(initial_state=initial_state, horizon=final_time, weights_value=true_weights)
        traj_sol2 = pop_quadcopter.popSolver(initial_state=initial_state, horizon=final_time, weights_value=obtained_weights)
        animation_quad.save_animation(wing_length=1.5, dt=sample_time, state_trajectory_pdp=traj_sol2['state_trajectories'],
                                      state_trajectory_pop=traj_sol['state_trajectories'],save_option=1)
        trajectories =  trajectories + [traj_sol]
    # save
    sio.savemat('data_pdp_quadcopter/quadcopter_pop.mat', {'trajectories': trajectories,
                                              'dt': sample_time,
                                              'true_parameter': true_weights})
    
