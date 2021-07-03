import gym
import numpy as np


class Sensors(object):
    def __init__(self, sensors_placement, window_size, ctrl_node):
        # Description
        self.sensors_placement = sensors_placement  # The note to be controlled
        self.sensors_history = {}  # data acquisition/logger
        self.window_size = window_size

        self.n_sensors = 0
        for key, value in self.sensors_placement.items():
            self.n_sensors += len(value)

        self.ctrl_node = ctrl_node  # The note to be controlled
        self.ctrl_node_history = {}

        self.time_reset()

    def time_reset(self):
        self.sensors_history['time'] = [0.]
        for key, value in self.sensors_placement.items():
            self.sensors_history[key] = np.zeros((len(value), 1), dtype=np.float64)
            # self.ctrl_node_history[key] = np.zeros((1, 1), dtype=np.float64)


class ActiveControlDevices(object):
    def __init__(self, max_force=100, n_discrete=11):
        self.max_force = max_force
        self.action_space_discrete = gym.spaces.Discrete(n_discrete)  # for discrete DQN
        self.action_space_discrete_array = np.linspace(-self.max_force, self.max_force,
                                                       num=self.action_space_discrete.n)
        self.action_space_continuous = gym.spaces.Discrete(1)

    def calc_device_force(self, action):
        force = self.action_space_discrete_array[action]
        return force
