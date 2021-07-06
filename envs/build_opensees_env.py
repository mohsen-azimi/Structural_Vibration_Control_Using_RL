'''
Refrence:
https://opensees.berkeley.edu/wiki/index.php/Dynamic_Analyses_of_1-Story_Moment_Frame_with_Viscous_Dampers
'''
##########################################################################################################################################################################
import openseespy.opensees as ops
import copy
# import openseespy.postprocessing.ops_vis as opsv
from structural_models.ops_damping import Rayliegh
import numpy as np
import matplotlib.pyplot as plt
from structural_models.visualization import Visualization
from termcolor import colored  # for colorful prints
from analyses import UniformExcitation
from control_devices import ActiveControlDevice
from dl_models import NN
from ground_motions import LoadGM
from rl_algorithms import DQNAgent  # , DDQNAgent
from structural_models import ShearFrameVD1Story1Bay, Sensors
import time
import math
from scipy import signal
from ground_motions.read_peer import LoadGM

from gym import spaces, logger
from collections import deque

from structural_models import ShearFrameVD1Story1Bay, Sensors


class ShearFrame(object):
    def __init__(self, env_params):
        self.env_name = env_params['structure']['name']
        self.env_params = env_params

        self.structure = None
        self.sensors = None
        self.ctrl_device = None
        self.gm = None
        self.analysis = None

        self.build_struture()
        self.place_sensors()
        self.place_control_devices()
        self.define_ground_motion()
        self.set_analysis()

        self.uncontrolled_ctrl_node_history = {}
        self.record_uncontrolled_response()

    def build_struture(self):
        if self.env_name == 'Shear_Frame_1Bay1Story':
            self.structure = ShearFrameVD1Story1Bay()
        else:
            print("no structural env created!")
            pass

        self.structure.create_model()
        if self.env_params['structure']['plot']: self.structure.draw2D()
        self.structure.create_damping_matrix()
        self.structure.run_gravity()

    def place_sensors(self):
        self.sensors = Sensors(self.env_params['sensors'])

    def place_control_devices(self):
        # change later for multiple devices! Now, it has only one device (single agent)
        self.ctrl_device = ActiveControlDevice(self.env_params['control_devices']['ActiveControl'])

    def define_ground_motion(self):
        self.gm = LoadGM(self.env_params['ground_motion'])

    def set_analysis(self):
        if self.env_params['analysis'] == 'UniformExcitation':
            self.analysis = UniformExcitation()
        else:
            print("no analysis set!")

    def record_uncontrolled_response(self):

        sensors = copy.deepcopy(self.sensors)
        analysis = copy.deepcopy(self.analysis)
        ctrl_device = copy.deepcopy(self.ctrl_device)

        for i_timer in range(0, self.gm.resampled_npts):
            ctrl_force = 0.
            sensors, ctrl_device = analysis.run_dynamic('1-step', i_timer, ctrl_device, ctrl_force, self.gm,
                                                        sensors)

        self.uncontrolled_ctrl_node_history = sensors.ctrl_node_history

    def reset(self):
        # self.analysis.time_reset()  # reset each episode to avoid long appended time-histories
        self.sensors.time_reset()  # reset each episode to avoid long appended time-histories
        self.ctrl_device.time_reset()  # reset each episode to avoid long appended time-histories
        ops.setTime(0.0)  # - gm.resampled_dt)

    def step(self, itimer, force, normalize):
        self.sensors, self.ctrl_device = self.analysis.run_dynamic("1-step", itimer, self.ctrl_device, force, self.gm,
                                                                   self.sensors)

        next_state = np.array([], dtype=np.float32).reshape(0, self.sensors.window_size)

        for key, value in self.sensors.sensors_history.items():
            if not key == 'time':  # skip the time array
                while value.shape[1] < self.sensors.window_size:  # add extra zeros if the window is still short
                    value = np.hstack((np.zeros((value.shape[0], 1)), value))

                next_state = np.vstack((next_state, value[:, -self.sensors.window_size:]))

        if normalize:
            print(f"normalize_state = {normalize}")
        return next_state
        # pass
