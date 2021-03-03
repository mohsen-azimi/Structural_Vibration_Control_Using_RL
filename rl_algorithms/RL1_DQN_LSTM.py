import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

import scipy.io
from ops_GM import ops_GM
from ops_2DFrame import ops_Env
import random

import numpy as np
from collections import deque

from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

from scipy.signal import hilbert, chirp  # for envelop
# _______________________________
# from tensorflow.python.client import device_lib
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
# _______________________________

def controller_DNN(input_shape, action_space):
    input = Input(shape=input_shape)
    output = Dense(32, activation='tanh', kernel_initializer='normal', use_bias=True)(input)
    output = Dense(32, activation='tanh', kernel_initializer='normal', use_bias=True)(output)
    output = Dense(32, activation='tanh', kernel_initializer='normal', use_bias=True)(output)

    output = Dense(action_space, activation="linear", kernel_initializer='normal')(output)

    model = Model(inputs = input, outputs = output, name='ops2DFrame_DQN_model')
    # model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.compile(loss="mse", optimizer='adam', metrics=["accuracy"])
    # model.compile(loss="mse", optimizer='Adam', metrics=["accuracy"])
    model.summary()
    return model

def controller_LSTM_univar(input_shape, action_space):
    input = Input(shape=input_shape)  # like Input(shape=(n_steps, n_features=1))

    hidden1 = LSTM(10)(input)
    hidden2 = Dense(10, activation='relu')(hidden1)
    output = Dense(action_space, activation="linear", kernel_initializer='normal')(hidden2)
    model = Model(inputs=input, outputs=output)
    model.compile(loss="mse", optimizer='adam', metrics=["accuracy"])

    model.summary()
    return model




class DQNAgent:
    def __init__(self):
        self.env = ops_Env()
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = self.env.observation_space.shape[0]  #+1  # +1 for ground motion at i & i+1
        self.action_size = self.env.action_space_discrete.n
        self.method  = '1-step'  # rull 'full' TH or step-by-step ('1-step')

        self.nEpisodes = 100  # EPISODES - number of games we want the agent to play.

        self.memory = deque(maxlen=100)   # memory>train_start

        self.gamma = 0.9    # gamma - decay or discount rate, to calculate the future discounted reward.
        self.epsilon = 1.0  # epsilon - exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
        self.epsilon_min = 0.0001  # epsilon_min - we want the agent to explore at least this amount.
        self.epsilon_decay = 0.9999  # epsilon_decay - we want to decrease the number of explorations as it gets good at playing games.

        self.batch_size = 128 # batch_size - Determines how much memory DQN will use to learn
        self.train_start = 2000  # memory>train_start; start after this many steps, we have predict-->fit

        # create the controller model
        self.controller = controller_LSTM_univar(input_shape=(self.state_size,))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    # def load_h5(self, name):
    #     self.controller = load_model(name)
    #
    def save_h5(self, name):
        self.controller.save(name)

    def initiate(self):
        self.env.create_model_2Dframe()
        self.env.run_gravity_2Dframe()

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.controller.predict(state))



    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory (Delete for LSTM!)
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
        q = self.controller.predict(state)
        q_next = self.controller.predict(next_state)


        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                q[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')

                q[i][action[i]] = reward[i] + self.gamma * (np.amax(q_next[i]))
                # target[i] = reward[i] + self.gamma * (np.amax(target_next[i]))   #(by mohsen)

        # Train the Neural Network with batches

        self.controller.fit(state, q, batch_size=self.batch_size, verbose=0)

    def step(self, controlForce, iTime):
        self.env.run_dynamic_2Dframe(GM, '1-step', iTime, controlForce)

        if iTime > (len(GM.resampled_signal)-2):
            iTime = iTime-1  # to avoid error in the last step

        if iTime < 3:
            next_state = np.zeros(6)
        else:
            next_state = [self.env.d3[iTime],
                          self.env.d3[iTime-1],
                          self.env.d3[iTime-2],
                          self.env.v3[iTime],
                          self.env.a3[iTime],
                          GM.resampled_signal[iTime]]
        return next_state

    def reward(self, t, force):

        cost = []
        # cost.append(abs(self.env.d3[t]/np.max(agent_unctrl.env.d3)))  # account global reduction

        cost.append(abs(self.env.d3[t] / np.max(agent_unctrl.env.d3[t])))

        # cost.append(abs(self.env.v3[t] / np.max(agent_unctrl.env.v3)))
        # cost.append(abs(self.env.a3[t] / np.max(agent_unctrl.env.a3)))
        # cost.append(abs(0.001 * force))


        reward = -(np.sum(cost))
        return reward


    def run(self):
        # self.env.readPEER('RSN1086_NORTHR_SYL090.AT2', 'myEQ.dat')
        Rewards = []
        for episode in range(self.nEpisodes+1):
            # reset analysis
            Rewards_episode = []
            self.initiate()
            state = np.reshape(np.zeros(self.state_size), [1, self.state_size])

            done = False
            iTime = 0
            controlForce = []
            while not done:
                # self.env.render()
                # print('t={:.4} s'.format(i*self.env.dt_analysis))

                action = self.act(state)    # action = np.float(action[0][0]) for continuous

                force = self.env.action_space_discrete_array[action]
                # force = 0.0
                controlForce.append(force)

                next_state = self.step(force, iTime)

                reward = self.reward(iTime, force)
                Rewards_episode.append(reward)

                next_state = np.reshape(next_state, [1, self.state_size])

                self.remember(state, action, reward, next_state, done)
                state = next_state
                iTime += 1
                done = iTime >= len(GM.resampled_signal)
                # done = iTime >= len(self.env.GM_dwnSampled)\
                #             or self.env.d3[-1]>3  # disp>3inch?

                done = bool(done)
                if iTime % (1/GM.analysis_dt) == 0:
                    if iTime % (10/GM.analysis_dt) == 0:
                        print('▮', end='')
                    else:
                        print('|', end='')


                if done:
                    mean_reward_episode = np.mean(Rewards_episode)
                    Rewards.append(mean_reward_episode)

                    print("episode: {}/{},    Reward: {:.10},   Epsilon: {:.3}".format(episode, self.nEpisodes, mean_reward_episode, self.epsilon))

                    # self.env.plot_TH('1-step')

                    if episode % 10 == 0:
                        plt.close('all')
                        plt.figure(figsize=(200, 10))
                        fig, ax1 = plt.subplots()
                        color = 'tab:green'
                        ax1.set_xlabel('Time [s]')
                        ax1.set_ylabel('Force [kip]', color=color)
                        ax1.plot(self.env.t, controlForce, label="Force", color=color, alpha=0.3)
                        ax1.tick_params(axis='y', labelcolor=color)
                        plt.legend(loc='lower left')



                        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                        color = 'tab:blue'
                        ax2.set_ylabel('Displacement [in]', color=color)  # we already handled the x-label with ax1
                        ax2.fill_between(agent_unctrl.env.t,-agent_unctrl.env.d3_env, agent_unctrl.env.d3_env, label="Uncontrolled_Env", color='blue', alpha=0.15)
                        ax2.plot(agent_unctrl.env.t, agent_unctrl.env.d3, label="Uncontrolled", color='blue',  alpha=0.85)
                        ax2.plot(self.env.t, self.env.d3, label="Controlled", color='black')

                        ax2.tick_params(axis='y', labelcolor=color)
                        fig.tight_layout()  # otherwise the right y-label is slightly clipped
                        plt.legend(loc='lower right')
                        plt.title(f"Time History Response (episode:{episode})", fontsize=16, fontweight='bold')
                        plt.savefig(f"Results/ops2DFrame_DQN_episod{episode}.png")
                        plt.show()


                    if episode%10==0:
                        print("Saving trained model as ops2DFrame_DQN.h5...")
                        self.save_h5(f"Results/ops2DFrame_DQN_episod{episode}.h5")

                        scipy.io.savemat(f"Results/Results_episod{episode}.mat", {
                            'Rewards_mean': Rewards
                        })

                        if episode == self.nEpisodes:
                            return
                self.replay()


    # def test(self, episode):
    #     # episode = 500
    #     self.controller = load_model(f"Results/ops2DFrame_DQN_episod{episode}.h5")
    #
    #     # reset analysis
    #     self.initiate()
    #
    #     state = np.reshape(np.zeros(self.state_size), [1, self.state_size])
    #
    #     done = False
    #     iTime = 0
    #
    #     while not done:
    #         action = self.act(state)  # controlForce = self.controller.predict(state)
    #
    #
    #         action = np.float(action[0][0])
    #
    #         next_state = self.step(action, iTime)
    #
    #         reward = self.reward(iTime)
    #
    #         next_state = np.reshape(next_state, [1, self.state_size])
    #
    #         # self.remember(state, action, reward, next_state, done)
    #         state = next_state
    #         iTime += 1
    #         done = iTime >= len(self.env.GM_dwnSampled)
    #         # done = iTime >= len(self.env.GM_dwnSampled) \
    #         #        or self.env.d3[-1] > 20000  # disp>3inch?
    #
    #         done = bool(done)
    #
    #         if done:
    #             print("episode: {}/{},    Reward: {:.10},   Epsilon: {:.3}".format(episode, self.nEpisodes, reward,
    #                                                                                self.epsilon))
    #             break


if __name__ == "__main__":
    GM = ops_GM(dt=0.1, g=386., SF=1., dir=1, inputFile='H-E12140.AT2', outputFile='myEQ.dat', plot=False)

    agent_unctrl = DQNAgent()
    agent_unctrl.env.create_model_2Dframe()
    agent_unctrl.env.run_gravity_2Dframe()
    runSteps = '1-step'
    for i in range(0, GM.analysis_npts):
        u = 0.
        agent_unctrl.env.run_dynamic_2Dframe(GM, runSteps, i, u)
        if runSteps == 'full':
            break
    agent_unctrl.env.d3_env = abs(hilbert(agent_unctrl.env.d3))


    # t_2, d3_unctrl = agent_unctrl.env.t, agent_unctrl.env.d3
    # plt.plot(t_2, d3_unctrl, label="Uncontrolled")
    # plt.legend(loc='lower right')
    # plt.suptitle("Time History Results", fontsize=16, fontweight='bold')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Roof Displacement [in]")
    # plt.show()
########################################
    agent = DQNAgent()
    agent.run()



