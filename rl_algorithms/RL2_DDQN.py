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


class DDQNAgent:
    def __init__(self, structure, sensors, gm, analysis, dl_model, ctrl_device, uncontrolled_ctrl_node_history):
        self.gm = gm
        self.sensors = sensors
        self.structure = structure
        self.analysis = analysis
        self.dl_model = dl_model
        self.ctrl_device = ctrl_device
        self.uncontrolled_ctrl_node_history = uncontrolled_ctrl_node_history

        self.STATE_SIZE = sensors.n_sensors * sensors.window_size

        self.ACTION_SIZE = ctrl_device.action_space_discrete.n
        self.nEPISODES = 1000  # EPISODES - number of games we want the agent to play.
        self.memory = deque(maxlen=4000)  # 2000  # memory>train_start
        self.TRAIN_START = 1000  # <= memory len ((train starts when reach this))

        self.BATCH_SIZE = 1000  # <=train start; batch_size - Determines how much memory DQN will use to learn
        self.GAMMA = 0.95  # discount rate, to calculate the future discounted reward.
        self.epsilon = 1.0  # exploration rate at start
        self.EPSILON_DECAY = 0.999  # we want to decrease the number of explorations as it gets better.
        self.EPSILON_MIN = 0.01  # we want the agent to explore at least this amount.


        # defining model parameters
        self.ddqn = True
        self.Soft_Update = False

        self.TAU = 0.1  # target network soft update hyperparameter

        self.Save_Path = 'Models'
        self.scores, self.episodes, self.average = [], [], []
        
        if self.ddqn:
            print("----------Double DQN--------")
            self.Model_name = os.path.join(self.Save_Path,"DDQN.h5")
        else:
            print("-------------DQN------------")
            self.Model_name = os.path.join(self.Save_Path,"DQN.h5")
        
        # # create main model
        self.model = self.dl_model
        self.target_model = self.dl_model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and self.ddqn:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:  # Exploration mode
            return random.randrange(self.ACTION_SIZE)
        act_value = self.dl_model.predict(state)  # Exploitation mode
        return np.argmax(act_value)  # return the action that has the best q-value

    def replay(self):
        if len(self.memory) < self.TRAIN_START:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.BATCH_SIZE))
        # print(type(minibatch))
        state = np.zeros((self.BATCH_SIZE, self.STATE_SIZE))  # original
        next_state = state
        action, reward, done = [], [], []  # clear

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.BATCH_SIZE):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn: # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.GAMMA * (target_val[i][a])
                else: # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.GAMMA * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.dl_model.fit(state, target, epochs=1, batch_size=self.BATCH_SIZE, verbose=0)
        if self.epsilon > self.EPSILON_MIN:
            self.epsilon *= self.EPSILON_DECAY  # discount the epsilon after each train


    def step(self, itime, force):
        self.sensors, self.ctrl_device = self.analysis.run_dynamic("1-step", itime, self.ctrl_device, force, self.gm,
                                                                   self.sensors)

        next_state = np.array([], dtype=np.float32).reshape(0, self.sensors.window_size)
        for key, value in self.sensors.sensors_history.items():
            if not key == 'time':  # skip the time array
                while value.shape[1] < self.sensors.window_size:  # add extra zeros if the window is still short
                    value = np.hstack((np.zeros((value.shape[0], 1)), value))

                next_state = np.vstack((next_state, value[:, -self.sensors.window_size:]))

        # print(next_state)

        return next_state

    def save(self, name):
        self.dl_model.save_weights(name)

    def plot_dqn(self, episode):
        fig, (ax1, ax3) = plt.subplots(2, 1)  # 1, 2, figsize=(15, 8))
        color = 'tab:green'
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Force [kN]', color=color)

        ax1.plot(self.ctrl_device.time, self.ctrl_device.force_history, label="Force", color=color, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='lower left')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Displacement [mm]', color=color)  # we already handled the x-label with ax1

        ax2.fill_between(self.uncontrolled_ctrl_node_history['time'],
                         -abs(hilbert(self.uncontrolled_ctrl_node_history['disp'])),
                         abs(hilbert(self.uncontrolled_ctrl_node_history['disp'])),
                         label="Uncontrolled_Env", color='blue', alpha=0.15)
        ax2.plot(self.uncontrolled_ctrl_node_history['time'], self.uncontrolled_ctrl_node_history['disp'],
                 label="Uncontrolled", color='blue', alpha=0.85)
        ax2.plot(self.sensors.ctrl_node_history['time'], self.sensors.ctrl_node_history['disp'],
                 label="Controlled", color='black')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.legend(loc='lower right')
        plt.title(f"Time History Response (episode:{episode})", fontsize=16, fontweight='bold')
        color = 'tab:red'
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Total Rewards', color=color)

        ax3.plot(self.episodes, self.aggr_ep_rewards, label="Reward", color=color, alpha=0.3)
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.legend(loc='lower left')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        # plt.savefig(f"Results/Plots/DQN_episode_{episode}.png", facecolor='w', edgecolor='w',
        #             orientation='landscape', format="png", transparent=False,
        #             bbox_inches='tight', pad_inches=0.3, )

    def reset(self):
        # self.analysis.time_reset()  # reset each episode to avoid long appended time-histories
        self.sensors.time_reset()  # reset each episode to avoid long appended time-histories
        self.ctrl_device.time_reset()  # reset each episode to avoid long appended time-histories
        ops.setTime(0.0)  # - gm.resampled_dt)

    def run(self):
        self.aggr_ep_rewards = []
        self.episodes = []

        for episode in range(1, self.nEPISODES + 1):
            self.reset()  # reset ops & analysis memory
            i_time = 0
            done = False
            ep_rewards = []
            force_memory = []
            aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
            state = np.reshape(np.zeros(self.STATE_SIZE), [1, self.STATE_SIZE])

            while not done:
                # self.env.render()
                action = self.act(state)  # action = np.float(action[0][0]) for continuous
                force = self.ctrl_device.calc_device_force(action)
                next_state = self.step(i_time, force).flatten()
                next_state = np.reshape(next_state, [1, self.STATE_SIZE])  # (n,) --> (1,n)

                # reward = self.reward(i_time, force)
                reward = Reward.J1(self.sensors, self.uncontrolled_ctrl_node_history)
                ep_rewards.append(reward)

                self.remember(state, action, reward, next_state, done)
                state = next_state
                i_time += 1
                # print(f"{i_time}/{gm.resampled_npts}")
                done = i_time == self.gm.resampled_npts - 1  # -1 for python indexing system
                # done = bool(done)
                if i_time % (1 / self.gm.resampled_dt) == 0:
                    if i_time % (10 / self.gm.resampled_dt) == 0:
                        print('.', end='')
                    else:
                        print('|', end='')

                if done:
                    self.episodes.append(episode)
                    self.aggr_ep_rewards.append(np.sum(ep_rewards))

                    # every step update target model
                    self.update_target_model()

                    print(f' episode: {episode}/{self.nEPISODES} total_reward:{np.sum(ep_rewards):.1f},'
                          f' epsilon: {self.epsilon}')
                    if episode % 10 == 0:  # plot at each # eposode
                        self.plot_dqn(episode)

                    if episode % 10 == 0:
                        print(f"Saving DQN_episode_{episode}.hdf5...")
                        # self.save(f"Results/Weights/DQN_episode_{episode}.hdf5")
                        #
                        # scipy.io.savemat(f"Results/MATLAB/DQN_episode_{episode}.mat", {
                        #     'Rewards': ep_rewards
                        # })

                        if episode == self.nEPISODES:
                            return

                if i_time % self.BATCH_SIZE == 0:
                    self.replay()  # when to reply/train? when Done? per episode? per (originally, while not done at each step)

    # def test(self):
    #     self.load("cartpole-ddqn.h5")
    #     for e in range(self.EPISODES):
    #         state = self.env.reset()
    #         state = np.reshape(state, [1, self.state_size])
    #         done = False
    #         i = 0
    #         while not done:
    #             self.env.render()
    #             action = np.argmax(self.model.predict(state))
    #             next_state, reward, done, _ = self.env.step(action)
    #             state = np.reshape(next_state, [1, self.state_size])
    #             i += 1
    #             if done:
    #                 print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
    #                 break

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    agent = DQNAgent(env_name)
    agent.run()
    #agent.test()