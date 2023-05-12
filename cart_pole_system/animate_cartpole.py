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

class Animation_CartPole:
    def __init__(self, pole_length = None):
        self.pole_length = pole_length
    
    def get_cartpole_pos_coordinate(self, pole_length, state_trajectory_pop):
        pos_coordinate = np.zeros((state_trajectory_pop.shape[0], 4))
        for i in range(state_trajectory_pop.shape[0]):
            position = state_trajectory_pop[i, 0]
            theta = state_trajectory_pop[i, 1]
            cart_pos_abcissa = position
            cart_pos_oordinate = 0
            pole_pos_abcissa = position + pole_length * sin(theta)
            pole_pos_oordinate = -pole_length * cos(theta)
            pos_coordinate[i, :] = np.array([cart_pos_abcissa, cart_pos_oordinate, pole_pos_abcissa, pole_pos_oordinate])
        return pos_coordinate
    
        
    def save_animation(self, pole_length, dt, state_trajectory_pop, state_trajectory_pdp=None, save_option=0, title='Cart-pole system'):

        # get the pos_coordinate of cart pole
        pos_coordinate = self.get_cartpole_pos_coordinate(pole_length, state_trajectory_pop)
        horizon = pos_coordinate.shape[0]
        if state_trajectory_pdp is not None:
            pos_coordinate_ref = self.get_cartpole_pos_coordinate(pole_length, state_trajectory_pdp)
            cart_height_ref, cart_width_ref = 0.5, 1
        else:
            pos_coordinate_ref = np.zeros_like(pos_coordinate)
            cart_height_ref, cart_width_ref = 0, 0
        assert pos_coordinate.shape[0] == pos_coordinate_ref.shape[0], 'reference trajectory should have the same length'

        # figure params
        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-10, 10), ylim=(-5, 5), )
        ax.set_aspect('equal')
        ax.grid()
        ax.set_ylabel('Vertical (m)')
        ax.set_xlabel('Horizontal (m)')
        ax.set_title(title)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        # set lines
        cart_height, cart_width = 0.5, 1
        bar, = ax.plot([], [], color='blue', lw=3)
        bar_ref, = ax.plot([], [], color='slategray', lw=3, alpha=0.8)
        block = patches.Rectangle((0, 0), cart_width, cart_height, fc='black')
        block_ref = patches.Rectangle((0, 0), cart_width_ref, cart_height_ref, fc='slategray', alpha=0.8)

        if state_trajectory_pdp is not None:
            plt.legend([bar, bar_ref], ['learned', 'real'], ncol=1, loc='best',
                       bbox_to_anchor=(0.4, 0.4, 0.6, 0.6))

        def initialize():
            bar.set_data([], [])
            bar_ref.set_data([], [])
            ax.add_patch(block)
            ax.add_patch(block_ref)
            ax.axhline(lw=2, c='k')
            time_text.set_text('')
            return bar, bar_ref, block, block_ref, time_text

        def animate_cartpole(i):
            seg_x = [pos_coordinate[i, 0], pos_coordinate[i, 2]]
            seg_y = [pos_coordinate[i, 1], pos_coordinate[i, 3]]
            bar.set_data(seg_x, seg_y)

            seg_x_ref = [pos_coordinate_ref[i, 0], pos_coordinate_ref[i, 2]]
            seg_y_ref = [pos_coordinate_ref[i, 1], pos_coordinate_ref[i, 3]]
            bar_ref.set_data(seg_x_ref, seg_y_ref)

            block.set_xy([pos_coordinate[i, 0] - cart_width / 2, pos_coordinate[i, 1] - cart_height / 2])
            block_ref.set_xy([pos_coordinate_ref[i, 0] - cart_width / 2, pos_coordinate_ref[i, 1] - cart_height / 2])

            time_text.set_text(time_template % (i * dt))

            return bar, bar_ref, block, block_ref, time_text

        ani = animation.FuncAnimation(fig, animate_cartpole, np.size(state_trajectory_pop, 0),
                                      interval=50, init_func=initialize)

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save(title+'.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()
