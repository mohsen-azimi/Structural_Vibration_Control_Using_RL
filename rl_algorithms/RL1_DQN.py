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


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")


class DQNAgent(object):
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
        if self.ctrl_device.device_type == "passive":
            self.TRAIN_START = 1e10  # no training!
            self.nEPISODES = 1  # 1 episode only

        self.BATCH_SIZE = 1000  # <=train start; batch_size - Determines how much memory DQN will use to learn
        self.GAMMA = 0.95  # discount rate, to calculate the future discounted reward.
        self.epsilon = 1.0  # exploration rate at start
        self.EPSILON_DECAY = 0.9999  # we want to decrease the number of explorations as it gets better.
        self.EPSILON_MIN = 0.01  # we want the agent to explore at least this amount.

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:  # Exploration mode
            return random.randrange(self.ACTION_SIZE)
        act_value = self.dl_model.predict(state)  # Exploitation mode
        return np.argmax(act_value)  # return the action that has the best q-value

    # def replay(self):  # method 1: too slow!
    #     if len(self.memory) < self.BATCH_SIZE:
    #         return
    #     # Randomly sample minibatch from the memory
    #     minibatch = random.sample(self.memory, min(len(self.memory), self.BATCH_SIZE))
    #     x_batch, y_batch = [], []
    #     for state, action, reward, next_state, done in minibatch:
    #         y_target = self.model.predict(state)
    #         y_target[0][action] = reward if done else (reward + self.GAMMA * np.amax(self.model.predict(next_state)[0]))           #predicted future model
    #         x_batch.append(state[0])
    #         y_batch.append(y_target[0])
    #
    #     self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=self.BATCH_SIZE, epochs=1, verbose=0)  # only one memory to reply=epoch
    #
    #     if self.epsilon > self.EPSILON_MIN:
    #         self.epsilon *= self.EPSILON_DECAY # discount the epsilon after each train

    def replay(self):  # method 2: a little faster, from original
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
        for j in range(self.BATCH_SIZE):
            state[j] = minibatch[j][0]
            action.append(minibatch[j][1])
            reward.append(minibatch[j][2])
            next_state[j] = minibatch[j][3]
            done.append(minibatch[j][4])

        # do batch prediction to save speed
        target = self.dl_model.predict(state)
        target_next = self.dl_model.predict(next_state)

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

        self.dl_model.fit(state, target, epochs=1, batch_size=self.BATCH_SIZE, verbose=0)
        if self.epsilon > self.EPSILON_MIN:
            self.epsilon *= self.EPSILON_DECAY  # discount the epsilon after each train

    # #

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

    def reward(self, itime, force):
        # @mohsen: make Reward object with J1-J9 methods!
        # Simple Moving Average (SMA)
        ave_disp, ave_vel, ave_accel = 0., 0., 0.
        for key, value in self.sensors.ctrl_node_history.items():
            if key == "disp":
                ave_disp = np.mean(self.sensors.ctrl_node_history[key][-self.sensors.window_size:])
            if key == "vel":
                ave_vel = np.mean(self.sensors.ctrl_node_history[key][-self.sensors.window_size:])
            if key == "accel":
                ave_accel = np.mean(self.sensors.ctrl_node_history[key][-self.sensors.window_size:])

        # max from uncontrolled
        max_disp = max(np.abs(self.uncontrolled_ctrl_node_history['disp']))
        max_vel = max(np.abs(self.uncontrolled_ctrl_node_history['vel']))
        max_accel = max(np.abs(self.uncontrolled_ctrl_node_history['accel']))

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

        return rd + rv + ra

    # def load(self, name):
    #     self.dl_model.load_weights(name)

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

        plt.savefig(f"Results/Plots/DQN_episode_{episode}.png", facecolor='w', edgecolor='w',
                    orientation='landscape', format="png", transparent=False,
                    bbox_inches='tight', pad_inches=0.3, )

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

                reward = self.reward(i_time, force)
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

                    print(
                        f' episode: {episode}/{self.nEPISODES} total_reward:{np.sum(ep_rewards):.1f}, epsilon: {self.epsilon}')
                    if episode % 10 == 0:  # plot at each # eposode
                        self.plot_dqn(episode)

                    if episode % 10 == 0:
                        print(f"Saving DQN_episode_{episode}.hdf5...")
                        self.save(f"Results/Weights/DQN_episode_{episode}.hdf5")

                        scipy.io.savemat(f"Results/MATLAB/DQN_episode_{episode}.mat", {
                            'Rewards': ep_rewards
                        })

                        if episode == self.nEPISODES:
                            return

                    # self.env.plot_TH('1-step')
                if i_time % self.BATCH_SIZE == 0:
                    self.replay()  # when to reply/train? when Done? per episode? per (originally, while not done at each step)
                    # print(iTime*GM.resampled_dt)

# if __name__ == "__main__":
#     # if not os.path.exists('Results'):
#     #     os.makedirs('Results')
#
#     gm = LoadGM(dt=0.01, t_final=40, g=9810, SF=0.5, inputFile='RSN1086_NORTHR_SYL090.AT2', outputFile='myEQ.dat',
#                 plot=False)
#     sensors_loc = {"groundAccel": [1], "disp": [3], "vel": [3], "accel": [3]}  # future: strct = make("structure1")
#     structure = ShearFrameVD1Story1Bay(sensors_loc=sensors_loc, memory_len=1, ctrl_node=3, device_ij_nodes=[1, 3])
#     structure.create_model().draw2D().create_damping_matrix().run_gravity()  # gravity loading is defined part of structure
#
#     analysis = UniformExcitation(structure)
#     ctrl_device = ActiveControl(max_force=200, n_discrete=21)
#     # ctrl_device = SimpleMRD50kN(GM, max_volt=5, max_force=50, n_discrete=6)
#
#     # matTag, K_el, Cd, alpha = 10, 10, 10, 0.25
#     # ops.uniaxialMaterial('ViscousDamper', matTag, K_el, Cd, alpha)
#     # ctrl_device = PassiveTMD(loc_node=7, direction=1, m=10., mat_tag=10)
#     # ctrl_device.place_tmd()
#     # structure.create_damping_matrix()
#     # Create the controller model
#
#     dl_model = NN.simple_nn(n_units=10, n_hidden=5,
#                             input_shape=(structure.STATE_SIZE,),
#                             action_space=ctrl_device.action_space_discrete.n)
#
#     agent_unctrl = DQNAgent()  # to make sure it does not mix with controlled one below
#
#     run_steps = '1-step'  # do not change to 'full'
#     for i_time in range(0, gm.resampled_npts):
#         ctrl_force = 0.
#         agent_unctrl.analysis.run_dynamic(run_steps, i_time, ctrl_force, gm, structure)
#         if run_steps == 'full':
#             break
#
#     agent_unctrl.analysis.ctrl_node_disp_env = abs(hilbert(agent_unctrl.analysis.ctrl_node_disp))
#
#     # agent_unctrl.env.plot_TH()
#     ########################################
#     analysis = UniformExcitation(structure)  # re-initiate to avoid overwriting the uncontrolled response
#     agent_ctrl = DQNAgent()
#     agent_ctrl.run()
#     # ops.wipe()
