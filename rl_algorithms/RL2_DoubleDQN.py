# import os
import tensorflow as tf
import scipy.io
import random
import pandas as pd
import numpy as np
from collections import deque
from scipy.signal import hilbert  # for envelop
import matplotlib.pyplot as plt

import openseespy.opensees as ops
from rl_algorithms.RewardFcns import Reward

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")


import os
import random
import gym
import pylab
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
# import torch
from dl_models import NN


# def controller(input_shape, action_space):
#     X_input = Input(input_shape)
#     X = X_input
#
#     # 'Dense' is the basic form of a neural network layer
#     # Input Layer of state size(4) and Hidden Layer with 512 nodes
#     X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X)
#
#     # Hidden layer with 256 nodes
#     X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
#
#     # Hidden layer with 64 nodes
#     X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
#
#     # Output Layer with # of actions: 2 nodes (left, right)
#     X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)
#
#     model = Model(inputs = X_input, outputs = X, name='CartPole DDQN model')
#     model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
#
#     model.summary()
#     return model


class DoubleDQNAgent:
    def __init__(self, double_dqn_params):
        self.dqn_params = double_dqn_params
        self.nEPISODES = double_dqn_params['n_episodes']  # EPISODES - number of games we want the agent to play.
        self.replay_buffer = deque(maxlen=double_dqn_params['replay_buffer_len'])  # 2000  # memory>train_start
        self.TRAIN_START = double_dqn_params['train_start']  # <= memory len ((train starts when reach this))
        self.BATCH_SIZE = double_dqn_params[
            'batch_size']  # <=train start; batch_size - how much memory DQN will use to learn
        self.GAMMA = double_dqn_params['gamma']  # discount rate, to calculate the future discounted reward.
        self.epsilon = double_dqn_params['epsilon_initial']  # exploration rate at start
        self.EPSILON_DECAY = double_dqn_params[
            'epsilon_decay']  # decrease the number of explorations as it gets better.
        self.EPSILON_MIN = double_dqn_params['epsilon_min']  # we want the agent to explore at least this amount.

        self.n_actions = self.dqn_params['dqn_params']['n_actions']
        self.action_space = [i for i in range(self.n_actions)]

        self.state_len = self.dqn_params['dqn_params']['input_shape'][0]

        # # create main model
        self.online_net = NN.simple_nn(lr=self.dqn_params['dqn_params']['lr'],
                                       n_hidden=self.dqn_params['dqn_params']['n_hidden'],
                                       n_units=self.dqn_params['dqn_params']['n_units'],
                                       input_shape=self.dqn_params['dqn_params']['input_shape'],
                                       n_actions=self.dqn_params['dqn_params']['n_actions'])

        self.target_net = NN.simple_nn(lr=self.dqn_params['dqn_params']['lr'],
                                       n_hidden=self.dqn_params['dqn_params']['n_hidden'],
                                       n_units=self.dqn_params['dqn_params']['n_units'],
                                       input_shape=self.dqn_params['dqn_params']['input_shape'],
                                       n_actions=self.dqn_params['dqn_params']['n_actions'])

        self.Soft_Update = double_dqn_params['soft_update']
        self.soft_update_tau = double_dqn_params['soft_update_tau']

        # self.Save_Path = 'Models'
        # self.scores, self.episodes, self.average = [], [], []

        # if self.ddqn:
        #     print("----------Double DQN--------")
        #     self.Model_name = os.path.join(self.Save_Path, "DDQN.h5")
        # else:
        #     print("-------------DQN------------")
        #     self.Model_name = os.path.join(self.Save_Path, "DQN.h5")

    # after some time interval update the target model to be same with model
    def update_target_net(self):
        if not self.Soft_Update:
            self.target_net.set_weights(self.online_net.get_weights())
            return
        if self.Soft_Update:
            q_model_theta = self.online_net.get_weights()
            target_model_theta = self.target_net.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1 - self.soft_update_tau) + q_weight * self.soft_update_tau
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_net.set_weights(target_model_theta)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.random() <= self.epsilon:  # Exploration mode
            return random.randrange(self.n_actions)
        act_value = self.online_net.predict(state)  # Exploitation mode
        return np.argmax(act_value)  # return the action that has the best q-value

    def learn(self):  # method 2: a little faster, from original
        if len(self.replay_buffer) < self.TRAIN_START:
            return
        # Randomly sample minibatch from the memory
        mini_batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), self.BATCH_SIZE))

        # do this before prediction
        # for speedup, this could be done on the tensor level
        states = np.asarray([transition[0] for transition in mini_batch])
        actions = np.asarray([transition[1] for transition in mini_batch])
        rewards = np.asarray([transition[2] for transition in mini_batch])
        next_states = np.asarray([transition[3] for transition in mini_batch])
        dones = np.asarray([transition[4] for transition in mini_batch])


        # do batch prediction to save speed
        # targets = self.target_net.predict(states)
        targets_next = self.online_net.predict(next_states)
        targets_val = self.target_net.predict(next_states)  # shape = (n_samples, 1, n_actions)

        # standard dqn:
        # DQN chooses the max Q value among next actions
        # selection and evaluation of action is on the target Q Network
        # Q_max = max_a' Q_target(s', a')
        # targets = rewards + self.GAMMA * (1 - dones) * (np.amax(target_nexts))

        # Double dqn:
        # current Q Network selects the action
        # a'_max = argmax_a' Q(s', a')
        a_maxreward = np.squeeze(np.argmax(targets_next, axis=2))



        # target Q Network evaluates the action
        # Q_max = Q_target(s', a'_max)
        q_a_maxreward = np.asarray([targets_val[i, 0, val] for i, val in enumerate(a_maxreward)])
        targets = rewards + self.GAMMA * (1 - dones) * q_a_maxreward

        #


        np.squeeze(np.amax(targets_next, axis=2))

        self.online_net.fit(states, targets, epochs=1, batch_size=self.BATCH_SIZE, verbose=0)
        if self.epsilon > self.EPSILON_MIN:
            self.epsilon *= self.EPSILON_DECAY  # discount the epsilon after each train

    def save(self, name):
        self.online_net.save_weights(name)
