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
class SimplePendulumEnv:
    def __init__(self, g=10, length=None, mass=None, damping_b=None):
        # set system weighting_parameterss
        self.g = g
        self.length = length
        self.mass = mass
        self.b = damping_b
        # set states
        self.theta, self.dtheta = SX.sym('theta'), SX.sym('dtheta')
        self.X = vertcat(self.theta, self.dtheta)
        # set controls
        U = SX.sym('u')
        self.U = U
        self.Inertia =  (self.mass * self.length * self.length)/3
        self.f = vertcat(self.dtheta,(self.U - self.mass * g * self.length * sin(self.theta) - self.b * self.dtheta) / self.Inertia) 


    def cost_initialization(self, w_theta=None, w_dtheta=None, w_controls=0.001):
        self.w_controls = w_controls
        weighting_parameters = []
        if w_theta is None:
            self.w_theta = SX.sym('w_theta')
            weighting_parameters += [self.w_theta]
        else:
            self.w_theta = w_theta

        if w_dtheta is None:
            self.w_dtheta = SX.sym('w_dtheta')
            weighting_parameters += [self.w_dtheta]
        else:
            self.w_dtheta = w_dtheta

        self.weights = vcat(weighting_parameters)

        # control goal
        X_g = [math.pi, 0]
        # cost for q
        self.theta_cost = (self.theta - X_g[0]) ** 2
        # cost for dq
        self.dtheta_cost = (self.dtheta - X_g[1]) ** 2
        # cost for u
        self.cost_u = dot(self.U, self.U)

        self.stage_cost = self.w_theta * self.theta_cost + self.w_dtheta * self.dtheta_cost + self.w_controls * self.cost_u
        self.terminal_cost = self.w_theta * self.theta_cost + self.w_dtheta * self.dtheta_cost






