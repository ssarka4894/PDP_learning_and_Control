# inspired from,
# @article{jin2019pontryagin,
#   title={Pontryagin Differentiable Programming: An End-to-End Learning and Control Framework},
#   author={Jin, Wanxin and Wang, Zhaoran and Yang, Zhuoran and Mou, Shaoshuai},
#   journal={arXiv preprint arXiv:1912.12970},
#   year={2019}
# }


from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch
import math
import time


# inverted pendulum
class QuadcopterUAVEnv:
    def __init__(self, g=10, Inertia_x=None, Inertia_y=None, Inertia_z=None, mass=None, length=None, kappa=None):
        # set system weighting_parameters
        self.g = g
        self.Inertia_x = Inertia_x
        self.Inertia_y = Inertia_y
        self.Inertia_z = Inertia_z
        self.mass = mass
        self.length = length
        self.kappa = kappa
        self.Inertia_B = diag(vertcat(self.Inertia_x, self.Inertia_y, self.Inertia_z))
        self.gravity_I = vertcat(0, 0, -g)
        
        # set states in world coordinates
        position_x, position_y, position_z = SX.sym('position_x'), SX.sym('position_y'), SX.sym('position_z')
        self.position_I = vertcat(position_x, position_y, position_z)
        velocity_x, velocity_y, velocity_z = SX.sym('velocity_x'), SX.sym('velocity_y'), SX.sym('velocity_z')
        self.velocity_I = vertcat(velocity_x, velocity_y, velocity_z)
        quaternion_0, quaternion_1, quaternion_2, quaternion_3 = SX.sym('quaternion_0'), SX.sym('quaternion_1'), SX.sym('quaternion_2'), SX.sym('quaternion_3')
        self.quarternion = vertcat(quaternion_0, quaternion_1, quaternion_2, quaternion_3)
        omega_x, omega_y, omega_z = SX.sym('omega_x'), SX.sym('omega_y'), SX.sym('omega_z')
        self.omega_B = vertcat(omega_x, omega_y, omega_z)
        
        # set controls
        force1, force2, force3, force4 = SX.sym('force1'), SX.sym('force2'), SX.sym('force3'), SX.sym('force4')
        self.Thrust_B = vertcat(force1, force2, force3, force4)
        force = self.Thrust_B[0] + self.Thrust_B[1] + self.Thrust_B[2] + self.Thrust_B[3]
        self.force_B = vertcat(0, 0, force)
        
        # Moment in world coordinates
        Moment_x = -self.Thrust_B[1] * self.length / 2 + self.Thrust_B[3] * self.length / 2
        Moment_y = -self.Thrust_B[0] * self.length / 2 + self.Thrust_B[2] * self.length / 2
        Moment_z = (self.Thrust_B[0] - self.Thrust_B[1] + self.Thrust_B[2] - self.Thrust_B[3]) * self.kappa
        self.Moment_B = vertcat(Moment_x, Moment_y, Moment_z)
        dqCosine_B_I = self.direction_cosine(self.quarternion)  
        dqCosine_I_B = transpose(dqCosine_B_I)  

        # set states
        dposition_I = self.velocity_I
        dvelocity_I = 1 / self.mass * mtimes(dqCosine_I_B, self.force_B) + self.gravity_I
        dquarternion = 1 / 2 * mtimes(self.gamma_operate(self.omega_B), self.quarternion)
        domega = mtimes(inv(self.Inertia_B), self.Moment_B - mtimes(mtimes(self.cross_product(self.omega_B), self.Inertia_B), self.omega_B))

        self.X = vertcat(self.position_I, self.velocity_I, self.quarternion, self.omega_B)
        self.U = self.Thrust_B
        self.f = vertcat(dposition_I, dvelocity_I, dquarternion, domega)
        
   

    def cost_initialization(self, w_position=None, w_velocity=None, w_quarternion=None, w_omega=None, w_controls=0.1):
        self.w_controls = w_controls
        weighting_parameters = []
        if w_position is None:
            self.w_position = SX.sym('w_position')
            weighting_parameters += [self.w_position]
        else:
            self.w_position = w_position

        if w_velocity is None:
            self.w_velocity = SX.sym('w_velocity')
            weighting_parameters += [self.w_velocity]
        else:
            self.w_velocity = w_velocity

        if w_quarternion is None:
            self.w_quarternion = SX.sym('w_quarternion')
            weighting_parameters += [self.w_quarternion]
        else:
            self.w_quarternion = w_quarternion

        if w_omega is None:
            self.w_omega = SX.sym('w_omega')
            weighting_parameters += [self.w_omega]
        else:
            self.w_omega = w_omega

        self.weights = vcat(weighting_parameters)

        # set goal states X_g
        goal_position_I = np.array([0, 0, 0])
        self.cost_position_I = dot(self.position_I - goal_position_I, self.position_I - goal_position_I)
        goal_velocity_I = np.array([0, 0, 0])
        self.cost_velocity_I = dot(self.velocity_I - goal_velocity_I, self.velocity_I - goal_velocity_I)
        goal_quarternion = computeQuarternion(0, [0, 0, 1])
        # goal_quarternion = [1,0,0,0]
        goal_gamma_B_I = self.direction_cosine(goal_quarternion)
        gamma_B_I = self.direction_cosine(self.quarternion)
        self.cost_quarternion = trace(np.identity(3) - mtimes(transpose(goal_gamma_B_I), gamma_B_I))
        goal_omega_B = np.array([0, 0, 0])
        self.cost_w_B = dot(self.omega_B - goal_omega_B, self.omega_B - goal_omega_B)

        # set trajectory cost
        self.cost_force = dot(self.Thrust_B, self.Thrust_B)

        self.stage_cost = self.w_position * self.cost_position_I + \
                         self.w_velocity * self.cost_velocity_I+ \
                         self.w_omega * self.cost_w_B + \
                         self.w_quarternion * self.cost_quarternion + \
                         w_controls * self.cost_force
        self.terminal_cost = self.w_position * self.cost_position_I + \
                          self.w_velocity * self.cost_velocity_I+ \
                          self.w_omega * self.cost_w_B + \
                          self.w_quarternion * self.cost_quarternion

    
     
     
     
    def direction_cosine(self, quart):
        dqCosine_B_I = vertcat(
            horzcat(1 - 2 * (quart[2] ** 2 + quart[3] ** 2), 2 * (quart[1] * quart[2] + quart[0] * quart[3]), 2 * (quart[1] * quart[3] - quart[0] * quart[2])),
            horzcat(2 * (quart[1] * quart[2] - quart[0] * quart[3]), 1 - 2 * (quart[1] ** 2 + quart[3] ** 2), 2 * (quart[2] * quart[3] + quart[0] * quart[1])),
            horzcat(2 * (quart[1] * quart[3] + quart[0] * quart[2]), 2 * (quart[2] * quart[3] - quart[0] * quart[1]), 1 - 2 * (quart[1] ** 2 + quart[2] ** 2))
        )
        return dqCosine_B_I

    def cross_product(self, value):
        value_cross = vertcat(
            horzcat(0, -value[2], value[1]),
            horzcat(value[2], 0, -value[0]),
            horzcat(-value[1], value[0], 0)
        )
        return value_cross

    def gamma_operate(self, omega):
        angular = vertcat(
            horzcat(0, -omega[0], -omega[1], -omega[2]),
            horzcat(omega[0], 0, omega[2], -omega[1]),
            horzcat(omega[1], -omega[2], 0, omega[0]),
            horzcat(omega[2], omega[1], -omega[0], 0)
        )
        return angular
    
# # converter to quaternion from (slope, vector)
def computeQuarternion(slope, vector):
    if type(vector) == list:
        vector = numpy.array(vector)
    vector = vector / numpy.linalg.norm(vector)
    quat = numpy.zeros(4)
    quat[0] = math.cos(slope / 2)
    quat[1:] = math.sin(slope / 2) * vector
    return quat.tolist()

    # def quaternion_mul(self, value1, q):
    #     return vertcat(value1[0] * value2[0] - value1[1] * value2[1] - value1[2] * value2[2] - value1[3] * value2[3],
    #                    value1[0] * value2[1] + value1[1] * value2[0] + value1[2] * value2[3] - value1[3] * value2[2],
    #                    value1[0] * value2[2] - value1[1] * value2[3] + value1[2] * value2[0] + value1[3] * value2[1],
    #                    value1[0] * value2[3] + value1[1] * value2[2] - value1[2] * value2[1] + value1[3] * value2[0]
    #                    )


    

