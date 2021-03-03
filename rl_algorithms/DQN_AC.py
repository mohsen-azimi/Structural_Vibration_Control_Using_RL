import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

import scipy.io
from ops_2DFrame import ops_Env

import random

import numpy as np
from collections import deque
import gym
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# _______________________________
# from tensorflow.python.client import device_lib
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
# _______________________________

def controller(input_shape, action_space):
    input = Input(shape=input_shape)
    output = Dense(8, activation='relu', kernel_initializer='normal')(input)
    output = Dense(8, activation='relu', kernel_initializer='normal')(output)
    output = Dense(4, activation='relu', kernel_initializer='normal')(output)
    # output = Dense(action_space, activation='linear')(output)
    output = Dense(action_space)(output)

    model = Model(inputs = input, outputs = output, name='ops2DFrame_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    # model.compile(loss="mse", optimizer='Adam', metrics=["accuracy"])
    model.summary()
    return model

class DQNAgent:
    def __init__(self):
        self.env = ops_Env()
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = self.env.observation_space.shape[0]+1  # +1 for ground motion at i & i+1
        self.action_size = self.env.action_space.n
        self.method  = '1-step'  # rull 'full' TH or step-by-step ('1-step')

        self.nEpisodes = 100  # EPISODES - number of games we want the agent to play.

        self.train_start = 100  # from this step, we have predict-->fit
        self.memory = deque(maxlen=200)   # memory>train_start

        self.gamma = 0.95    # gamma - decay or discount rate, to calculate the future discounted reward.
        self.epsilon = 1.0  # epsilon - exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
        self.epsilon_min = 0.001  # epsilon_min - we want the agent to explore at least this amount.
        self.epsilon_decay = 0.999  # epsilon_decay - we want to decrease the number of explorations as it gets good at playing games.

        self.batch_size = 50 # batch_size - Determines how much memory DQN will use to learn


        # create the controller model
        self.controller = controller(input_shape=(self.state_size,), action_space = self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        # print('len_mem = {}'.format(len(self.memory)))

        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load_h5(self, name):
        self.controller = load_model(name)
    #
    def save_h5(self, name):
        self.controller.save(name)

    def initiate(self):
        self.env.create_model_2Dframe()
        # self.env.draw2D()
        self.env.run_gravity_2Dframe()

    def action(self, state):
        return self.controller.predict(state)


    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = state
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.controller.predict(state)
        target_next = self.controller.predict(next_state)


        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                ### target[i][action[i]] = reward[i]
                target[i] = reward[i]
                # print('---------------Done=True-------------------')
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                # print(type(target))
                # print(target.shape)
                # print('---------------Done=False-------------------')

                ### target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))
                target[i] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches

        self.controller.fit(state, target, batch_size=self.batch_size, verbose=0)

    # def action(self, state):
    #     # if np.random.random() <= self.epsilon:
    #     #     return random.randrange(self.action_size)
    #     # else:
    #     #     return self.controller.predict(state)
    #     return 0


    def step(self, controlForce, iTime):
        # '1-step' method
        self.env.run_dynamic_2Dframe('1-step', iTime, controlForce)
        if iTime > (len(self.env.GM_dwnSampled)-2):
            iTime = iTime-1  # to avoid error in the last step

        next_state = [self.env.d3[iTime],
                      self.env.v3[iTime],
                      self.env.a3[iTime],
                      self.env.GM_dwnSampled[iTime+1]]
        # print(iTime)
        # print(next_state)
        # print(self.env.GM_GManalysis[0:5])

        return next_state


    def reward(self, t):

        maxDisp = 20
        # maxVel = 100.0
        # maxAccel = 100.0


        regret_disp = abs(self.env.d3[t]/maxDisp)

        # regret_vel = abs(self.env.v3[t]/maxVel)
        # regret_accel = abs(self.env.a3[t]/maxAccel)

        reward = 1-(regret_disp + 0.0 + 0.0)
        return reward



    def run(self):
        self.env.readPEER('H-E12140.AT2', 'H-E12140.dat')
        Rewards = []


        for episode in range(self.nEpisodes+1):
            # reset analysis

            self.initiate()
            state = np.reshape(np.zeros(self.state_size), [1, self.state_size])

            done = False
            iTime = 0
            controlForce = []
            while not done:
                # self.env.render()
                # print('t={:.4} s'.format(i*self.env.dt_analysis))

                action = self.action(state)    # controlForce = self.controller.predict(state)

                action = np.float(action[0][0])

                controlForce.append(action)

                next_state = self.step(action, iTime)

                reward = self.reward(iTime)

                next_state = np.reshape(next_state, [1, self.state_size])

                self.remember(state, action, reward, next_state, done)
                state = next_state
                iTime += 1
                done = iTime >= len(self.env.GM_dwnSampled)\
                            or self.env.d3[-1]>3  # disp>3inch?

                done = bool(done)
                if iTime % 10 == 0:
                    if iTime % 100 == 0:
                        print('▮', end='')
                    else:
                        print('▯', end='')


                if done:
                    print("episode: {}/{},    Reward: {:.10},   Epsilon: {:.3}".format(episode, self.nEpisodes, reward, self.epsilon))
                    Rewards.append(reward)
                    # self.env.plot_TH('1-step')

                    plt.figure()
                    plt.plot(controlForce, label="Force")
                    plt.xlabel("Time [s]")
                    plt.ylabel("Force [kip]")

                    plt.show()

                    if episode%10==0:
                        print("Saving trained model as ops2DFrame_DQN.h5...")
                        self.save_h5(f"Results/ops2DFrame_DQN_episod{episode}.h5")

                        scipy.io.savemat(f"Results/Results_episod{episode}.mat", {
                            'Rewards': Rewards
                        })

                        if episode == self.nEpisodes:
                            return
                self.replay()


    def test(self, episode):
        # episode = 500
        self.controller = load_model(f"Results/ops2DFrame_DQN_episod{episode}.h5")

        # reset analysis
        self.initiate()

        state = np.reshape(np.zeros(self.state_size), [1, self.state_size])

        done = False
        iTime = 0

        while not done:
            action = self.action(state)  # controlForce = self.controller.predict(state)

            action = np.float(action[0][0])

            next_state = self.step(action, iTime)

            reward = self.reward(iTime)

            next_state = np.reshape(next_state, [1, self.state_size])

            # self.remember(state, action, reward, next_state, done)
            state = next_state
            iTime += 1
            done = iTime >= len(self.env.GM_dwnSampled) \
                   or self.env.d3[-1] > 3  # disp>3inch?

            done = bool(done)

            if done:
                print("episode: {}/{},    Reward: {:.10},   Epsilon: {:.3}".format(episode, self.nEpisodes, reward,
                                                                                   self.epsilon))
                break


if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()
    # agent.env.plot_TH('1-step')

    # # # Test
    # agent.env.readPEER('H-E12140.AT2', 'H-E12140.dat')
    #
    # # Uncontrolled
    #
    # agent.env.create_model_2Dframe()
    # # opsEnv.draw2D()
    # agent.env.run_gravity_2Dframe()
    #
    # method = '1-step'   # 'full' or '1-step'
    # for i in range(0, agent.env.n_analysis):
    #     u = 0.
    #     agent.env.run_dynamic_2Dframe(method, i, u)
    #     if method == 'full':
    #         break
    #
    # t_1, d3_1 = agent.env.t, agent.env.d3
    #
    #
    # # Controlled
    # agent.test(500)
    # t_2, d3_2 = agent.env.t, agent.env.d3
    #
    #
    # plt.plot(t_1, d3_1, label="Uncontrolled")
    # plt.plot(t_2, d3_2, label="Controlled")
    #
    # plt.legend(loc='lower right')
    # plt.suptitle("Time History Results", fontsize=16, fontweight='bold')
    #
    # plt.xlabel("Time [s]")
    # plt.ylabel("Roof Displacement [in]")
    #
    # plt.show()
    #




