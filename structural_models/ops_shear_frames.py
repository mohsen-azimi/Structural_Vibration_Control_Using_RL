'''
Refrence:
https://opensees.berkeley.edu/wiki/index.php/Dynamic_Analyses_of_1-Story_Moment_Frame_with_Viscous_Dampers
'''
##########################################################################################################################################################################
import openseespy.opensees as ops
# import openseespy.postprocessing.ops_vis as opsv
from structural_models.ops_damping import Rayliegh
import numpy as np
import matplotlib.pyplot as plt
from structural_models.visualization import Visualization
from termcolor import colored  # for colorful prints

import time
import math
from scipy import signal
from ground_motions.read_peer import LoadGM

from gym import spaces, logger
from collections import deque


class ShearFrameVD1Story1Bay(object):
    """
      Description:
      Control Node ID = # ()
      """

    def __init__(self):
        # Description
        self.env_name = "ShearFrameVD_5Story1Bay"

        # self.sensors = sensors
        # self.sensor_remember_window_len = sensor_remember_window_len
        # self.ctrl_node = ctrl_node  # The note to be controlled
        # self.ctrl_device_ij_nodes = ctrl_device_ij_nodes  # The node for which the displacement is minimized
        self.units = 'kN-mm'

        # self.STATE_SIZE = 0
        # for key, value in sensors_placement.items():
        #     if isinstance(value, list):
        #         self.STATE_SIZE += len(value)
        # self.STATE_SIZE *= sensor_remember_window_len   # assume a 2d window slides through the records; then flatten to a vector

        # self.STATE_SHAPE = (len(self.sensors_placement), self.sensor_remember_window_len)
        # self.STATE_SIZE = self.STATE_SHAPE[0]*self.STATE_SHAPE[1]
        # print(self.STATE_SHAPE)
        # print(self.STATE_SIZE)

    def draw2D(self):
        Visualization.draw2D()
        return self

    def create_model(self):
        # Reference: https://opensees.berkeley.edu/wiki/index.php/OpenSees_Example_1b._Elastic_Portal_Frame (E1b)
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)

        h = 3000.0  # story height
        w = 5000.0  # bay width
        # mass
        m = 1000 / 9810  # kN/g (per floor)

        # story 0
        ops.node(1, 0.0, 0 * h)
        ops.node(2, w, 0 * h)
        ops.fix(1, 1, 1, 1)
        ops.fix(2, 1, 1, 1)
        # story 1
        ops.node(3, 0.0, 1 * h, '-mass', 0.5 * m, 0.0, 0.0)
        ops.node(4, w, 1 * h, '-mass', 0.5 * m, 0.0, 0.0)

        K = 8.  # stiffness per story (just an assumption)
        E = 200.  # kN/mm2

        Ic = 0.55 * K * (h ** 3) / (24 * E)  # mm^4 (K=24 EI/h^3) two columns
        Ib = 1e11 * Ic  # almost rigid
        A = 1e11  # mm^2  (no axial deformation)

        transfTag = 1
        ops.geomTransf('Linear', transfTag)

        # Elements
        # story 1
        ops.element('elasticBeamColumn', 1, 1, 3, A, E, Ic, transfTag)  # Columns
        ops.element('elasticBeamColumn', 2, 2, 4, A, E, Ic, transfTag)
        ops.element('elasticBeamColumn', 3, 3, 4, A, E, Ib, transfTag)  # Beam

        # # Two Node Link Element (dampers)
        # # Damper properties: https://openseespydoc.readthedocs.io/en/latest/src/ViscousDamper.html?highlight=viscousdamper
        # matTag = 1
        # Kd, Cd, alpha, = 25.0, 20.7452, 0.35
        # ops.uniaxialMaterial('ViscousDamper', matTag, Kd, Cd, alpha)
        # ops.element('twoNodeLink', 4, *[1, 4], '-mat', matTag, '-dir', *[1])
        return self

    def install_TMD_if_any(self, ctrl_device):
        if ctrl_device.device_type is "passiveTMD":
            ctrl_device.place_tmd()
            print(colored('Passive TMD', 'green'), colored('Installed!', 'green'))
        else:
            print(colored('No TMD', 'red'), colored('Installed!', 'red'))
        return self

    def create_damping_matrix(self):
        Rayliegh(xDamp=0.05, alphaM=0.00, betaKcurr=0.0, betaKinit=0.0).create_damping_matrix()
        return self

    def run_gravity(self):
        # apply control force (static), u, at floor1
        ops.wipeAnalysis()
        # defining gravity loads
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        DL = -0.0  # (Dead Load)
        ops.eleLoad('-ele', 3, '-type', '-beamUniform', DL)

        ops.constraints('Plain')
        ops.numberer('Plain')
        ops.system('BandGeneral')
        ops.test('NormDispIncr', 1e-8, 6)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', 0.1)
        ops.analysis('Static')
        ops.analyze(10)

        ops.loadConst('-time', 0.0)
        return self


class ShearFrameVD5Story1Bay(object):
    """
      Description:
      Control Node ID = # ()
      """

    def __init__(self):
        # Description
        self.env_name = "ShearFrameVD_5Story1Bay"
        # self.sensors = sensors  # The note to be controlled
        # self.memory_len = memory_len  # # window width (works like deque with len=window_size)
        # self.ctrl_node = ctrl_node  # The note to be controlled
        # self.ctrl_device_ij_nodes = ctrl_device_ij_nodes  # The node for which the displacement is minimized
        self.units = 'kN-mm'

        # self.STATE_SIZE = 0
        # for key, value in sensors_loc.items():
        #     if isinstance(value, list):
        #         self.STATE_SIZE += len(value)
        # self.STATE_SIZE *= memory_len   # assume a 2d window slides through the records; then flatten to a vector

        # print("--------------")
        # print(self.STATE_SIZE)

        # self.STATE_SIZE = len(obs_nodes) * 5 + 1  # 5 for the list above + GM

    def draw2D(self):
        Visualization.draw2D()
        return self

    def create_model(self):
        # Reference: https://opensees.berkeley.edu/wiki/index.php/OpenSees_Example_1b._Elastic_Portal_Frame (E1b)
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)

        h = 3000.0  # story height
        w = 5000.0  # bay width
        # mass
        m = 1000 / 9810  # kN/g (per floor)

        # story 0
        ops.node(1, 0.0, 0 * h);
        ops.fix(1, 1, 1, 1)
        ops.node(2, w, 0 * h);
        ops.fix(2, 1, 1, 1)
        # story 1
        ops.node(3, 0.0, 1 * h);
        ops.mass(3, 0.5 * m, 0.0, 0.0)
        ops.node(4, w, 1 * h);
        ops.mass(4, 0.5 * m, 0.0, 0.0)
        # story 2
        ops.node(5, 0.0, 2 * h);
        ops.mass(5, 0.5 * m, 0.0, 0.0)
        ops.node(6, w, 2 * h);
        ops.mass(6, 0.5 * m, 0.0, 0.0)
        # story 3
        ops.node(7, 0.0, 3 * h);
        ops.mass(7, 0.5 * m, 0.0, 0.0)
        ops.node(8, w, 3 * h);
        ops.mass(8, 0.5 * m, 0.0, 0.0)
        # story 4
        ops.node(9, 0.0, 4 * h);
        ops.mass(9, 0.5 * m, 0.0, 0.0)
        ops.node(10, w, 4 * h);
        ops.mass(10, 0.5 * m, 0.0, 0.0)
        # story 5
        ops.node(11, 0.0, 5 * h);
        ops.mass(11, 0.5 * m, 0.0, 0.0)
        ops.node(12, w, 5 * h);
        ops.mass(12, 0.5 * m, 0.0, 0.0)

        K = 8  # stiffness per story (just an assumption)
        E = 200  # kN/mm2

        Ic = 0.5 * K * (h ** 3) / (24 * E)  # mm^4 (K=24 EI/h^3) two columns
        Ib = 1e12 * Ic  # almost rigid
        A = 1e12  # mm^2  (no axial deformation)

        transfTag = 1
        ops.geomTransf('Linear', transfTag)

        # Elements
        # story 1
        ops.element('elasticBeamColumn', 1, 1, 3, A, E, Ic, transfTag)  # Columns
        ops.element('elasticBeamColumn', 2, 2, 4, A, E, Ic, transfTag)
        ops.element('elasticBeamColumn', 3, 3, 4, A, E, Ib, transfTag)  # Beam

        # story 2
        ops.element('elasticBeamColumn', 4, 3, 5, A, E, Ic, transfTag)  # Columns
        ops.element('elasticBeamColumn', 5, 4, 6, A, E, Ic, transfTag)
        ops.element('elasticBeamColumn', 6, 5, 6, A, E, Ib, transfTag)  # Beam

        # story 3
        ops.element('elasticBeamColumn', 7, 5, 7, A, E, Ic, transfTag)  # Columns
        ops.element('elasticBeamColumn', 8, 6, 8, A, E, Ic, transfTag)
        ops.element('elasticBeamColumn', 9, 7, 8, A, E, Ib, transfTag)  # Beam

        # story 4
        ops.element('elasticBeamColumn', 10, 7, 9, A, E, Ic, transfTag)  # Columns
        ops.element('elasticBeamColumn', 11, 8, 10, A, E, Ic, transfTag)
        ops.element('elasticBeamColumn', 12, 9, 10, A, E, Ib, transfTag)  # Beam

        # story 5
        ops.element('elasticBeamColumn', 13, 9, 11, A, E, Ic, transfTag)  # Columns
        ops.element('elasticBeamColumn', 14, 10, 12, A, E, Ic, transfTag)
        ops.element('elasticBeamColumn', 15, 11, 12, A, E, Ib, transfTag)  # Beam

        # Two Node Link Element (dampers)
        # Damper prop: https://openseespydoc.readthedocs.io/en/latest/src/ViscousDamper.html?highlight=viscousdamper
        matTag = 1
        Kd, Cd, alpha, = 25.0, 20.7452, 0.35
        ops.uniaxialMaterial('ViscousDamper', matTag, Kd, Cd, alpha)
        ops.element('twoNodeLink', 16, *[1, 4], '-mat', matTag, '-dir', *[1])
        ops.element('twoNodeLink', 17, *[3, 6], '-mat', matTag, '-dir', *[1])
        ops.element('twoNodeLink', 18, *[5, 8], '-mat', matTag, '-dir', *[1])
        ops.element('twoNodeLink', 19, *[7, 10], '-mat', matTag, '-dir', *[1])
        ops.element('twoNodeLink', 20, *[9, 12], '-mat', matTag, '-dir', *[1])
        return self

    def install_TMD_if_any(self, ctrl_device):
        if ctrl_device.device_type is "passiveTMD":
            ctrl_device.place_tmd()
            print(colored('Passive TMD', 'green'), colored('Installed!', 'green'))
        else:
            print(colored('No TMD', 'red'), colored('Installed!', 'red'))
        return self

    def create_damping_matrix(self):
        Rayliegh(xDamp=0.05, alphaM=0.00, betaKcurr=0.0, betaKinit=0.0).create_damping_matrix()
        return self

    def run_gravity(self):
        # apply control force (static), u, at floor1
        ops.wipeAnalysis()
        # defining gravity loads
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        DL = -0.0  # (Dead Load)
        ops.eleLoad('-ele', 3, '-type', '-beamUniform', DL)

        ops.constraints('Plain')
        ops.numberer('Plain')
        ops.system('BandGeneral')
        ops.test('NormDispIncr', 1e-8, 6)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', 0.1)
        ops.analysis('Static')
        ops.analyze(10)

        ops.loadConst('-time', 0.0)
        return self

    # def plot_TH(self):
    #     for node in range(len(self.obs_nodes)):
    #         plt.plot(self.time, self.obs_nodes_disp[node, :], label=f"Disp @ node {self.obs_nodes[node]}")
    #     plt.legend()
    #     plt.show()

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
