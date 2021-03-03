'''
Refrence:
https://opensees.berkeley.edu/wiki/index.php/Dynamic_Analyses_of_1-Story_Moment_Frame_with_Viscous_Dampers
'''
##########################################################################################################################################################################
import openseespy.opensees as ops
# import openseespy.postprocessing.ops_vis as opsv

import numpy as np
import matplotlib.pyplot as plt


class PlotResults:
    """
      Description:
      """
    def __init__(self):
        # Description
        self.x = 0
        # print("dynamic loading")

        # self.obs_nodes = structure.obs_nodes  # The note to be observed
        # self.ctrl_node = structure.ctrl_node  # The note to be controlled
        # self.device_ij_nodes = structure.device_ij_nodes  # device location (one device)

    def plot_disp_TH(self, analysis):
        for node in range(len(analysis.obs_nodes)):
            plt.plot(analysis.time, analysis.obs_nodes_disp[node, :], label=f"Disp @ node {dynamic_analysis.obs_nodes[node]}")
        plt.legend()
        plt.show()
    @staticmethod
    def plot_dqn():
        pass




