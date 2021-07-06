import numpy as np


class Reward(object):
    def __init__(self):
        pass

    @staticmethod
    def J1(sensors, uncontrolled_ctrl_node_history):

        ave_disp, ave_vel, ave_accel = 0., 0., 0.
        for key, value in sensors.ctrl_node_history.items():
            if key == "disp":
                ave_disp = np.mean(sensors.ctrl_node_history[key][-sensors.window_size:])
            if key == "vel":
                ave_vel = np.mean(sensors.ctrl_node_history[key][-sensors.window_size:])
            if key == "accel":
                ave_accel = np.mean(sensors.ctrl_node_history[key][-sensors.window_size:])

        # max from uncontrolled
        max_disp = max(np.abs(uncontrolled_ctrl_node_history['disp']))
        max_vel = max(np.abs(uncontrolled_ctrl_node_history['vel']))
        max_accel = max(np.abs(uncontrolled_ctrl_node_history['accel']))

        # k_g = abs(self.analysis.sensors_daq["groundAccel"][0][-1]) / \
        #       max(np.abs(self.analysis.sensors_daq["groundAccel"][0]))  # coefficient 1

        # k_f = abs(force / ctrl_device.max_force)  # coefficient 2

        # print(f"k_g = {k_g}....k_f = {k_f}")

        # rd = abs(1/moving_ave_disp)
        rd = 1 - abs(ave_disp / max_disp)
        rv = 1 - abs(ave_vel / max_vel)
        ra = 1 - abs(ave_accel / max_accel)

        # if (self.analysis.ctrl_node_disp[itime] * self.analysis.ctrl_node_vel[itime]) > 0:
        #     k = 0.5  # Penalty: reverse the motion direction
        #     if (force * self.analysis.ctrl_node_disp[itime]) > 0:
        #         k *= 0.2  # More penalty: reverse the force direction
        # else:
        #     k = 1.  # No extra penalty

        reward = rd + rv + ra

        return reward

    @staticmethod
    def J2(sensors, force, uncontrolled_ctrl_node_history):

        ave_disp, ave_vel, ave_accel = 0., 0., 0.
        for key, value in sensors.ctrl_node_history.items():
            if key == "disp":
                ave_disp = np.mean(sensors.ctrl_node_history[key][-sensors.window_size:])
            if key == "vel":
                ave_vel = np.mean(sensors.ctrl_node_history[key][-sensors.window_size:])
            if key == "accel":
                ave_accel = np.mean(sensors.ctrl_node_history[key][-sensors.window_size:])

        k_gm = abs(np.mean(sensors.sensors_history['groundAccel'][-sensors.window_size:])
                   / np.max(abs(sensors.sensors_history['groundAccel'])))

        # max from uncontrolled
        max_disp = max(np.abs(uncontrolled_ctrl_node_history['disp']))
        max_vel = max(np.abs(uncontrolled_ctrl_node_history['vel']))
        max_accel = max(np.abs(uncontrolled_ctrl_node_history['accel']))

        # r_f = 0.005 * abs(force)

        # print(f"k_g = {k_g}....k_f = {k_f}")

        # rd = abs(1/moving_ave_disp)
        rd = 1 - abs(ave_disp / max_disp)
        rv = 1 - abs(ave_vel / max_vel)
        ra = 1 - abs(ave_accel / max_accel)

        # if (self.analysis.ctrl_node_disp[itime] * self.analysis.ctrl_node_vel[itime]) > 0:
        #     k = 0.5  # Penalty: reverse the motion direction
        #     if (force * self.analysis.ctrl_node_disp[itime]) > 0:
        #         k *= 0.2  # More penalty: reverse the force direction
        # else:
        #     k = 1.  # No extra penalty
        # print(rd, rv, ra, '|', k_gm)

        reward = k_gm * (rd + rv + ra)

        # clip the rewards?
        return reward
