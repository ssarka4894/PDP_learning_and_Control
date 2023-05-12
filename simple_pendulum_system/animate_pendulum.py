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

class Animation_Pendulum:
    def __init__(self, length = None):
        self.length = length
    
    def get_pendulum_pos_coordinate(self, length, state_trajectory_pop):

        pos_coordinate = np.zeros((state_trajectory_pop.shape[0], 2))
        for i in range(state_trajectory_pop.shape[0]):
            theta = state_trajectory_pop[i, 0]
            abcissa = length * sin(theta)
            oordinate = -length * cos(theta)
            pos_coordinate[i, :] = np.array([abcissa, oordinate])
        return pos_coordinate

    def save_animation(self, length, dt, state_trajectory_pop, state_trajectory_pdp=None, save_option=0):

        # get the pos_coordinate of pendulum
        pos_coordinate = self.get_pendulum_pos_coordinate(length, state_trajectory_pop)
        horizon = pos_coordinate.shape[0]
        if state_trajectory_pdp is not None:
            pos_coordinate_ref = self.get_pendulum_pos_coordinate(length, state_trajectory_pdp)
        else:
            pos_coordinate_ref = np.zeros_like(pos_coordinate)
        assert pos_coordinate.shape[0] == pos_coordinate_ref.shape[0], 'reference trajectory should have the same length'
    

        # set figure
        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-4, 4), ylim=(-4, 4))
        ax.set_aspect('equal')
        ax.grid()
        ax.set_ylabel('Vertical (m)')
        ax.set_xlabel('Horizontal (m)')
        ax.set_title('Pendulum system')
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        # set bars
        bar_ref, = ax.plot([], [], color='grey', marker='o', lw=1)
        bar, = ax.plot([], [], color='blue',marker='o', lw=1)
        ax.legend([bar, bar_ref],['Auxillary System','True System'])

        def initialize():
            bar.set_data([], [])
            bar_ref.set_data([], [])
            time_text.set_text('')
            return bar, bar_ref, time_text

        def animate_pend(i):
            pend_x = [0, pos_coordinate[i, 0]]
            pend_y = [0, pos_coordinate[i, 1]]
            bar.set_data(pend_x, pend_y)

            pend_x_ref = [0, pos_coordinate_ref[i, 0]]
            pend_y_ref = [0, pos_coordinate_ref[i, 1]]
            bar_ref.set_data(pend_x_ref, pend_y_ref)

            time_text.set_text(time_template % (i * dt))

            return bar, bar_ref, time_text

        ani = animation.FuncAnimation(fig, animate_pend, np.size(state_trajectory_pop, 0),
                                      interval=50, init_func=initialize)

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('Pendulum.mp4', writer=writer)
            print('save_success')

        plt.show()
