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


# Cart Pole environment
class CartInvertedPendulumEnv:
    def __init__(self, g=10, mass_c=None, mass_p=None, length=None):
        self.g = g
        self.mass_c = mass_c
        self.mass_p = mass_p
        self.length = length
        
        # set states
        self.x, self.theta, self.dx, self.dtheta = SX.sym('x'), SX.sym('theta'), SX.sym('dx'), SX.sym('dtheta')
        self.X = vertcat(self.x, self.theta, self.dx, self.dtheta)
        # set controls
        self.U = SX.sym('u')
        
        # position acceleration
        ddx = (self.U + self.mass_p * sin(self.theta) * (self.length * self.dtheta * self.dtheta + g * cos(self.theta))) / (
                self.mass_c + self.mass_p * sin(self.theta) * sin(self.theta))  
        
        # angular acceleration
        ddtheta = (-self.U * cos(self.theta) - self.mass_p * self.length * self.dtheta * self.dtheta * sin(self.theta) * cos(self.theta) - (
                self.mass_c + self.mass_p) * g * sin(
            self.theta)) / (
                      self.length * self.mass_c + self.length * self.mass_p * sin(self.theta) * sin(self.theta))  # acceleration of theta
        
        # state_dot      
        self.f = vertcat(self.dx, self.dtheta, ddx, ddtheta)  

    def cost_initialization(self, wx=None, w_theta=None, wdx=None, w_dtheta=None, w_controls=0.001):
        weighting_parameters = []
        if wx is None:
            self.wx = SX.sym('wx')
            weighting_parameters += [self.wx]
        else:
            self.wx = wx

        if w_theta is None:
            self.w_theta = SX.sym('w_theta')
            weighting_parameters += [self.w_theta]
        else:
            self.w_theta = w_theta
        
        if wdx is None:
            self.wdx = SX.sym('wdx')
            weighting_parameters += [self.wdx]
        else:
            self.wdx = wdx

        if w_dtheta is None:
            self.w_dtheta = SX.sym('w_dtheta')
            weighting_parameters += [self.w_dtheta]
        else:
            self.w_dtheta = w_dtheta
        self.weights = vcat(weighting_parameters)

        X_g = [0.0, math.pi, 0.0, 0.0]

        self.stage_cost = self.wx * (self.x - X_g[0]) ** 2 + self.w_theta * (self.theta - X_g[1]) ** 2 + self.wdx * (
                self.dx - X_g[2]) ** 2 + self.w_dtheta * (self.dtheta - X_g[3]) ** 2 + w_controls * (self.U * self.U)
        self.terminal_cost = self.wx * (self.x - X_g[0]) ** 2 + self.w_theta * (self.theta - X_g[1]) ** 2 + self.wdx * (
                self.dx - X_g[2]) ** 2 + self.w_dtheta * (self.dtheta - X_g[3]) ** 2  

    



