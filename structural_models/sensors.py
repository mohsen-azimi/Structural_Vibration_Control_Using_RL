'''
Refrence:
https://opensees.berkeley.edu/wiki/index.php/Dynamic_Analyses_of_1-Story_Moment_Frame_with_Viscous_Dampers
'''
##########################################################################################################################################################################
# import openseespy.opensees as ops
# # import openseespy.postprocessing.ops_vis as opsv
# from structural_models.ops_damping import Rayliegh
import numpy as np
# import matplotlib.pyplot as plt
# from structural_models.visualization import Visualization
# import time
# import math
# from scipy import signal
# from ground_motions.read_peer import LoadGM

# from gym import spaces, logger
# from collections import deque


class Sensors(object):
    """
      """
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
        self.units = 'kN-mm'

        self.time_reset()

    def time_reset(self):
        self.time = [0.]
        for key, value in self.sensors_placement.items():
            self.sensors_history[key] = np.zeros((len(value), 1), dtype=np.float64)
            # self.ctrl_node_history[key] = np.zeros((1, 1), dtype=np.float64)






#
# # Test Class
# if __name__ == "__main__":
#     opsEnv = ShearFrameVD5Story1Bay(obs_nodes=[3], ctrl_node=3, device_ij_nodes=[3, 6])
#     GM = LoadGM(dt=.01, t_final=20, g=9810., SF=0.5, inputFile='RSN1086_NORTHR_SYL090.AT2', outputFile='myEQ.dat', plot=False)
#     # MR = SimpleMRD50kN()
#     opsEnv.create_model_2Dframe()
#     opsEnv.draw2D()
#
#     opsEnv.run_gravity_2Dframe()
#
#     runMethod = '1-step'  # do not change to 'full'
#     for i in range(0, GM.resampled_npts):
#         ctrl_force = 0.
#         # MR.volt = 0.
#         opsEnv.run_dynamic_2Dframe(GM, runMethod, i, ctrl_force)
#         if runMethod == 'full':
#             break
#     opsEnv.plot_TH()
#





