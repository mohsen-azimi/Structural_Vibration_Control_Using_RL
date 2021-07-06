import itertools

import numpy as np
from rl_algorithms import DQNAgent, Reward
import utils
import envs

if __name__ == "__main__":
    utils.make_dir({'Results': ['MATLAB', 'Plots', 'Weights']})

    env_params = {'structure': {'name': 'Shear_Frame_1Bay1Story',
                                'plot': False},

                  'sensors': {'sensors_placement': {'groundAccel': [1], 'accel': [3], 'vel': [3], 'disp': [3]},
                              'window_size': 3, 'ctrl_node': 3},

                  'control_devices': {'ActiveControl': {'max_force': 400, 'n_discrete': 51,
                                                        'ctrl_device_ij_nodes': [1, 3]}},

                  'ground_motion': {'desired_dt': 0.01,
                                    't_end': 40,
                                    'g': 9810,
                                    'scale_factor': 2.0,
                                    'inputFile': 'RSN1086_NORTHR_SYL090.AT2',
                                    'plot': False},

                  'analysis': 'UniformExcitation',
                  }
    env = envs.ShearFrame(env_params)

    agent_params = {'n_episodes': 500,
                    'replay_buffer_len': 4000,
                    'train_start': 2000,
                    'batch_size': 1000,
                    'discount_factor': 0.95,
                    'epsilon_initial': 1.0,
                    'epsilon_decay': 0.95,
                    'epsilon_min': 0.1,
                    'dqn_params': {'n_hidden': 3, 'n_units': 64, 'lr': 5e-4,
                                   'input_shape': (env.sensors.n_sensors * env.sensors.window_size,),
                                   'n_actions': env.ctrl_device.action_space_discrete.n}}
    agent = DQNAgent(agent_params)

    aggr_ep_rewards = []
    episodes = []

    for episode in itertools.count():
        env.reset()  # reset ops & analysis memory
        i_timer = 0
        done = False
        ep_rewards = []
        force_memory = []
        # aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
        state = np.reshape(np.zeros(env.sensors.state_len), [1, env.sensors.state_len])

        while not done:
            # self.env.render()
            action = agent.choose_action(state)  # action = np.float(action[0][0]) for continuous
            force = env.ctrl_device.calc_device_force(action)
            next_state = env.step(i_timer, force, normalize=False).flatten()
            next_state = np.reshape(next_state, [1, env.sensors.state_len])  # (n,) --> (1,n)

            # reward = Reward.J1(env.sensors, env.uncontrolled_ctrl_node_history)
            reward = Reward.J2(env.sensors, force, env.uncontrolled_ctrl_node_history)


            ep_rewards.append(reward)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            i_timer += 1
            # print(f"{i_time}/{gm.resampled_npts}")
            done = i_timer == env.gm.resampled_npts - 1  # -1 for python indexing system
            # done = bool(done)
            if i_timer % (1 / env.gm.resampled_dt) == 0:
                if i_timer % (10 / env.gm.resampled_dt) == 0:
                    print('.', end='')
                else:
                    print('|', end='')

            if done:
                episodes.append(episode)
                aggr_ep_rewards.append(np.sum(ep_rewards))

                print(
                    f' episode: {episode}/{agent.nEPISODES} total_reward:{np.sum(ep_rewards):.1f}, epsilon: {agent.epsilon}')
                if episode % 5 == 0:  # plot at each # eposode

                    utils.plot_dqn(episode, episodes, env.ctrl_device, env.uncontrolled_ctrl_node_history, env.sensors,
                                   aggr_ep_rewards)

                # if episode % 10 == 0:
                #     print(f"Saving DQN_episode_{episode}.hdf5...")
                    # self.save(f"Results/Weights/DQN_episode_{episode}.hdf5")
                    #
                    # scipy.io.savemat(f"Results/MATLAB/DQN_episode_{episode}.mat", {
                    #     'Rewards': ep_rewards
                    # })


                # self.env.plot_TH('1-step')
            if i_timer % 4000 == 0:  #  % agent.BATCH_SIZE == 0:
                agent.learn()  # when to learn? when Done? per episode? per (originally, while not done at each step)
                # print(iTime*GM.resampled_dt)


        if episode == agent.nEPISODES:
            break

    # agent = DQNAgent(structure, sensors, gm, analysis, dl_model, ctrl_device,
    #                           uncontrolled_ctrl_node_history=uncontrolled_ctrl_node_history)
    # agent_controlled.run()
    # ops.wipe()

    # sensors = Sensors()

    # analysis = UniformExcitation()
    # dl_model = NN.simple_nn(n_hidden=5, n_units=64,
    #                         input_shape=(sensors.n_sensors * sensors.window_size,),
    #                         action_space=ctrl_device.action_space_discrete.n)
    #
    # run_steps = '1-step'  # do not change to 'full'
    # for i_time in range(0, gm.resampled_npts):
    #     ctrl_force = 0.
    #     sensors, ctrl_device = analysis.run_dynamic(run_steps, i_time, ctrl_device, ctrl_force, gm,
    #                                                 sensors)
    #     if run_steps == 'full':
    #         break
    # uncontrolled_ctrl_node_history = sensors.ctrl_node_history
    #
    # ########################################
    # analysis = UniformExcitation()
    # sensors = Sensors(sensors_placement={"groundAccel": [1], "accel": [3], "vel": [3], "disp": [3]},
    #                   window_size=3, ctrl_node=3)
    #
    # agent_controlled = DQNAgent(structure, sensors, gm, analysis, dl_model, ctrl_device,
    #                           uncontrolled_ctrl_node_history=uncontrolled_ctrl_node_history)
    # agent_controlled.run()
    # # ops.wipe()
