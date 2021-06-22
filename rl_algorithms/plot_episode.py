
from scipy.signal import hilbert  # for envelop
import matplotlib.pyplot as plt


class Plot:
    def __init__(self):
        pass
        # by default, CartPole-v1 has max episode steps = 500

def plot_dqn(self, episode):
    fig, (ax1, ax3) = plt.subplots(2, 1)  # 1, 2, figsize=(15, 8))
    color = 'tab:green'
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Force [kN]', color=color)

    ax1.plot(self.analysis.time, self.analysis.force_memory, label="Force", color=color, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='lower left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Displacement [mm]', color=color)  # we already handled the x-label with ax1

    # ax2.fill_between(self.agent_unctrld.analysis.time,
    #                  -self.agent_unctrld.analysis.ctrl_node_disp_env, self.agent_unctrld.analysis.ctrl_node_disp_env,
    #                  label="Uncontrolled_Env", color='blue', alpha=0.15)
    # ax2.plot(self.agent_unctrld.analysis.time, self.agent_unctrld.analysis.ctrl_node_disp, label="Uncontrolled", color='blue',
    #          alpha=0.85)
    ax2.plot(self.analysis.time, self.analysis.ctrl_node_disp, label="Controlled", color='black')
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




