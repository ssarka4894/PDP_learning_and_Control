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

class Animation_Quadrotor:
    def __init__(self, wing_length = None):
        self.wing_length = wing_length

    
    def get_quadrotor_pos_coordinate(self, wing_length, state_trajectory_pop):

        # force_pos_coordinate in body frame
        rotor1 = vertcat(wing_length / 2, 0, 0)
        rotor2 = vertcat(0, -wing_length / 2, 0)
        rotor3 = vertcat(-wing_length / 2, 0, 0)
        rotor4 = vertcat(0, wing_length / 2, 0)

        # horizon
        horizon = np.size(state_trajectory_pop, 0)
        pos_coordinate = np.zeros((horizon, 15))
        for t in range(horizon):
            # pos_coordinate of COM
            com = state_trajectory_pop[t, 0:3]
            # altitude of quaternion
            quart = state_trajectory_pop[t, 6:10]

            # direction cosine matrix from body to inertial
            dqcosine_I_B = np.transpose(self.direction_cosine(quart).full())

            # pos_coordinate of each rotor in inertial frame
            rotor1_pos = com + mtimes(dqcosine_I_B, rotor1).full().flatten()
            rotor2_pos = com + mtimes(dqcosine_I_B, rotor2).full().flatten()
            rotor3_pos = com + mtimes(dqcosine_I_B, rotor3).full().flatten()
            rotor4_pos = com + mtimes(dqcosine_I_B, rotor4).full().flatten()

            # store
            pos_coordinate[t, 0:3] = com
            pos_coordinate[t, 3:6] = rotor1_pos
            pos_coordinate[t, 6:9] = rotor2_pos
            pos_coordinate[t, 9:12] = rotor3_pos
            pos_coordinate[t, 12:15] = rotor4_pos

        return pos_coordinate
    
    def direction_cosine(self, quart):
        dqCosine_B_I = vertcat(
            horzcat(1 - 2 * (quart[2] ** 2 + quart[3] ** 2), 2 * (quart[1] * quart[2] + quart[0] * quart[3]), 2 * (quart[1] * quart[3] - quart[0] * quart[2])),
            horzcat(2 * (quart[1] * quart[2] - quart[0] * quart[3]), 1 - 2 * (quart[1] ** 2 + quart[3] ** 2), 2 * (quart[2] * quart[3] + quart[0] * quart[1])),
            horzcat(2 * (quart[1] * quart[3] + quart[0] * quart[2]), 2 * (quart[2] * quart[3] - quart[0] * quart[1]), 1 - 2 * (quart[1] ** 2 + quart[2] ** 2))
        )
        return dqCosine_B_I
    
    

    def save_animation(self, wing_length, dt, state_trajectory_pop, state_trajectory_pdp=None,  save_option=0, title='Quadcopter System'):
        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_zlabel('Z (m)', fontsize=10, labelpad=5)
        ax.set_zlim(0, 10)
        ax.set_ylim(-8, 8)
        ax.set_xlim(-8, 8)
        ax.set_title(title, pad=20, fontsize=15)

        pos_coordinate = self.get_quadrotor_pos_coordinate(wing_length, state_trajectory_pop)
        horizon= np.size(pos_coordinate, 0)

        if state_trajectory_pdp is None:
            pos_coordinate_ref = self.get_quadrotor_pos_coordinate(0, numpy.zeros_like(pos_coordinate))
        else:
            pos_coordinate_ref = self.get_quadrotor_pos_coordinate(wing_length, state_trajectory_pdp)

        # animation
        bar_traj, = ax.plot(pos_coordinate[:1, 0], pos_coordinate[:1, 1], pos_coordinate[:1, 2])
        coordinate_x, coordinate_y, coordinate_z = pos_coordinate[0, 0:3]
        rotor1_x, rotor1_y, rotor1_z = pos_coordinate[0, 3:6]
        rotor2_x, rotor2_y, rotor2_z = pos_coordinate[0, 6:9]
        rotor3_x, rotor3_y, rotor3_z = pos_coordinate[0, 9:12]
        rotor4_x, rotor4_y, rotor4_z = pos_coordinate[0, 12:15]
        bar_arm1, = ax.plot([coordinate_x, rotor1_x], [coordinate_y, rotor1_y], [coordinate_z, rotor1_z], linewidth=2, color='blue', marker='o', markersize=2)
        bar_arm2, = ax.plot([coordinate_x, rotor2_x], [coordinate_y, rotor2_y], [coordinate_z, rotor2_z], linewidth=2, color='blue', marker='o', markersize=2)
        bar_arm3, = ax.plot([coordinate_x, rotor3_x], [coordinate_y, rotor3_y], [coordinate_z, rotor3_z], linewidth=2, color='blue', marker='o', markersize=2)
        bar_arm4, = ax.plot([coordinate_x, rotor4_x], [coordinate_y, rotor4_y], [coordinate_z, rotor4_z], linewidth=2, color='blue', marker='o', markersize=2)

        bar_traj_ref, = ax.plot(pos_coordinate_ref[:1, 0], pos_coordinate_ref[:1, 1], pos_coordinate_ref[:1, 2], color='gray', alpha=0.5)
        coordinate_x_ref, coordinate_y_ref, coordinate_z_ref = pos_coordinate_ref[0, 0:3]
        rotor1_x_ref, rotor1_y_ref, rotor1_z_ref = pos_coordinate_ref[0, 3:6]
        rotor2_x_ref, rotor2_y_ref, rotor2_z_ref = pos_coordinate_ref[0, 6:9]
        rotor3_x_ref, rotor3_y_ref, rotor3_z_ref = pos_coordinate_ref[0, 9:12]
        rotor4_x_ref, rotor4_y_ref, rotor4_z_ref = pos_coordinate_ref[0, 12:15]
        bar_arm1_ref, = ax.plot([coordinate_x_ref, rotor1_x_ref], [coordinate_y_ref, rotor1_y_ref], [coordinate_z_ref, rotor1_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)
        bar_arm2_ref, = ax.plot([coordinate_x_ref, rotor2_x_ref], [coordinate_y_ref, rotor2_y_ref], [coordinate_z_ref, rotor2_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)
        bar_arm3_ref, = ax.plot([coordinate_x_ref, rotor3_x_ref], [coordinate_y_ref, rotor3_y_ref], [coordinate_z_ref, rotor3_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)
        bar_arm4_ref, = ax.plot([coordinate_x_ref, rotor4_x_ref], [coordinate_y_ref, rotor4_y_ref], [coordinate_z_ref, rotor4_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)

        # time label
        time_template = 'time = %.1fs'
        time_text = ax.text2D(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if state_trajectory_pdp is not None:
            plt.legend([bar_traj, bar_traj_ref], ['learned', 'real'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))

        def improve_trajectory(num):
            # customize
            time_text.set_text(time_template % (num * dt))

            # trajectory
            bar_traj.set_data(pos_coordinate[:num, 0], pos_coordinate[:num, 1])
            bar_traj.set_3d_properties(pos_coordinate[:num, 2])

            # uav
            coordinate_x, coordinate_y, coordinate_z = pos_coordinate[num, 0:3]
            rotor1_x, rotor1_y, rotor1_z = pos_coordinate[num, 3:6]
            rotor2_x, rotor2_y, rotor2_z = pos_coordinate[num, 6:9]
            rotor3_x, rotor3_y, rotor3_z = pos_coordinate[num, 9:12]
            rotor4_x, rotor4_y, rotor4_z = pos_coordinate[num, 12:15]

            bar_arm1.set_data(np.array([[coordinate_x, rotor1_x], [coordinate_y, rotor1_y]]))
            bar_arm1.set_3d_properties([coordinate_z, rotor1_z])

            bar_arm2.set_data(np.array([[coordinate_x, rotor2_x], [coordinate_y, rotor2_y]]))
            bar_arm2.set_3d_properties([coordinate_z, rotor2_z])

            bar_arm3.set_data(np.array([[coordinate_x, rotor3_x], [coordinate_y, rotor3_y]]))
            bar_arm3.set_3d_properties([coordinate_z, rotor3_z])

            bar_arm4.set_data(np.array([[coordinate_x, rotor4_x], [coordinate_y, rotor4_y]]))
            bar_arm4.set_3d_properties([coordinate_z, rotor4_z])


            # trajectory ref
            num=horizon-1
            bar_traj_ref.set_data(pos_coordinate_ref[:num, 0], pos_coordinate_ref[:num, 1])
            bar_traj_ref.set_3d_properties(pos_coordinate_ref[:num, 2])

            # uav ref
            coordinate_x_ref, coordinate_y_ref, coordinate_z_ref = pos_coordinate_ref[num, 0:3]
            rotor1_x_ref, rotor1_y_ref, rotor1_z_ref = pos_coordinate_ref[num, 3:6]
            rotor2_x_ref, rotor2_y_ref, rotor2_z_ref = pos_coordinate_ref[num, 6:9]
            rotor3_x_ref, rotor3_y_ref, rotor3_z_ref = pos_coordinate_ref[num, 9:12]
            rotor4_x_ref, rotor4_y_ref, rotor4_z_ref = pos_coordinate_ref[num, 12:15]

            bar_arm1_ref.set_data(np.array([[coordinate_x_ref, rotor1_x_ref], [coordinate_y_ref, rotor1_y_ref]]))
            bar_arm1_ref.set_3d_properties([coordinate_z_ref, rotor1_z_ref])

            bar_arm2_ref.set_data(np.array([[coordinate_x_ref, rotor2_x_ref], [coordinate_y_ref, rotor2_y_ref]]))
            bar_arm2_ref.set_3d_properties([coordinate_z_ref, rotor2_z_ref])

            bar_arm3_ref.set_data(np.array([[coordinate_x_ref, rotor3_x_ref], [coordinate_y_ref, rotor3_y_ref]]))
            bar_arm3_ref.set_3d_properties([coordinate_z_ref, rotor3_z_ref])

            bar_arm4_ref.set_data(np.array([[coordinate_x_ref, rotor4_x_ref], [coordinate_y_ref, rotor4_y_ref]]))
            bar_arm4_ref.set_3d_properties([coordinate_z_ref, rotor4_z_ref])

            return bar_traj, bar_arm1, bar_arm2, bar_arm3, bar_arm4, \
                   bar_traj_ref, bar_arm1_ref, bar_arm2_ref, bar_arm3_ref, bar_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, improve_trajectory, horizon, interval=100, blit=True)

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save(title + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()
        
        
        
