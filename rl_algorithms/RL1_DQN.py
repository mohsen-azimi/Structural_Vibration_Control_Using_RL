# import os
import tensorflow as tf
import scipy.io
import random
import pandas as pd
from collections import deque
from scipy.signal import hilbert  # for envelop
import matplotlib.pyplot as plt
import openseespy.opensees as ops
from rl_algorithms.RewardFcns import Reward
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
from dl_models import NN


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_counter = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DQNAgent(object):
    def __init__(self, dqn_params, env_params):
        self.dqn_params = dqn_params
        self.nEPISODES = dqn_params['n_episodes']  # EPISODES - number of games we want the agent to play.
        self.memory = deque(maxlen=dqn_params['memory_len'])  # 2000  # memory>train_start
        self.TRAIN_START = dqn_params['train_start']  # <= memory len ((train starts when reach this))
        self.BATCH_SIZE = dqn_params['batch_size']  # <=train start; batch_size - how much memory DQN will use to learn
        self.GAMMA = dqn_params['discount_factor']  # discount rate, to calculate the future discounted reward.
        self.epsilon = dqn_params['epsilon_initial']  # exploration rate at start
        self.EPSILON_DECAY = dqn_params['epsilon_decay']  # decrease the number of explorations as it gets better.
        self.EPSILON_MIN = dqn_params['epsilon_min']  # we want the agent to explore at least this amount.

        self.n_actions = self.dqn_params['dqn_params']['n_actions']
        self.action_space = [i for i in range(self.n_actions)]

        self.state_len = self.dqn_params['dqn_params']['input_shape'][0]

        # self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
        #                            discrete=True)  # if using Phil's method

        self.q_eval = NN.simple_nn(lr=self.dqn_params['dqn_params']['lr'],
                                   n_hidden=self.dqn_params['dqn_params']['n_hidden'],
                                   n_units=self.dqn_params['dqn_params']['n_units'],
                                   input_shape=self.dqn_params['dqn_params']['input_shape'],
                                   n_actions=self.dqn_params['dqn_params']['n_actions'])


        # if self.ctrl_device.device_type == "passive":
        #     self.TRAIN_START = 1e10  # no training!
        #     self.nEPISODES = 1  # 1 episode only

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.random() <= self.epsilon:  # Exploration mode
            return random.randrange(self.n_actions)
        act_value = self.q_eval.predict(state)  # Exploitation mode
        return np.argmax(act_value)  # return the action that has the best q-value

    def learn(self):  # method 2: a little faster, from original
        if len(self.memory) < self.TRAIN_START:
            return
        # Randomly sample minibatch from the memory
        mini_batch = random.sample(self.memory, min(len(self.memory), self.BATCH_SIZE))
        # print(type(minibatch))
        state = np.zeros((self.BATCH_SIZE, self.state_len))  # original
        next_state = state
        action, reward, done = [], [], []  # clear

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for j in range(self.BATCH_SIZE):
            state[j] = mini_batch[j][0]
            action.append(mini_batch[j][1])
            reward.append(mini_batch[j][2])
            next_state[j] = mini_batch[j][3]
            done.append(mini_batch[j][4])

        # do batch prediction to save speed
        target = self.q_eval.predict(state)
        target_next = self.q_eval.predict(next_state)

        for j in range(self.BATCH_SIZE):  # original
            # correction on the Q value for the action used
            if done[j]:
                target[j][action[j]] = reward[j]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')

                target[j][action[j]] = reward[j] + self.GAMMA * (np.amax(target_next[j]))

        # Train the Neural Network with batches

        self.q_eval.fit(state, target, epochs=1, batch_size=self.BATCH_SIZE, verbose=0)
        if self.epsilon > self.EPSILON_MIN:
            self.epsilon *= self.EPSILON_DECAY  # discount the epsilon after each train

    def save(self, name):
        self.q_eval.save_weights(name)

