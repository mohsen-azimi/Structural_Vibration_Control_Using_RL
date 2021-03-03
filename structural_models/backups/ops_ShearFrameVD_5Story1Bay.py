'''
Refrence:
https://opensees.berkeley.edu/wiki/index.php/Dynamic_Analyses_of_1-Story_Moment_Frame_with_Viscous_Dampers
'''
##########################################################################################################################################################################
import openseespy.opensees as ops
# import openseespy.postprocessing.ops_vis as opsv

import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy import signal
from ground_motions.read_peer import LoadGM
from gym import spaces, logger
from collections import deque
from structural_models.damping import Rayliegh


class ShearFrameVD5Story1Bay:
    """
      Description:
      Control Node ID = # ()
      """

    def __init__(self, obs_nodes, ctrl_node, device_ij_nodes):
        # Description
        self.env_name = "ShearFrameVD_5Story1Bay"
        self.obs_nodes = obs_nodes  # The note to be controlled
        self.ctrl_node = ctrl_node  # The note to be controlled
        self.device_ij_nodes = device_ij_nodes  # The node for which the displacement is minimized
        self.units = 'kN-mm'

        # # observer
        # obs_max = np.array([
        #     np.finfo(np.float32).max,    # Displ @ ctrl_node, t
        #     np.finfo(np.float32).max,    # Displ @ ctrl_node, t-1
        #     np.finfo(np.float32).max,    # Displ @ ctrl_node, t-2
        #     np.finfo(np.float32).max,    # Vel @ ctrl_node
        #     np.finfo(np.float32).max,    # Accel @ ctrl_node
        #     np.finfo(np.float32).max])   # Ground Accel @ base
        # self.observation_space = spaces.Box(low=-obs_max, high=obs_max, dtype=np.float32)
        self.STATE_SIZE = len(obs_nodes) * 5 +1  # 5 for the list above + GM

        # self.action_maxForce = 100
        # self.action_space_discrete = spaces.Discrete(11)  # for discrete DQN
        # self.action_space_discrete_array = np.linspace(-self.action_maxForce, self.action_maxForce, num=self.action_space_discrete.n)
        # self.action_space_continous = spaces.Discrete(1)  # for continous A-C DQN: https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
        #
        # # self.action_space = spaces.Box(np.float32(-self.maxForce),
        # #                                np.float32(self.maxForce),
        # #                                shape=(1,), dtype=np.float32)

    def draw2D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        etags = ops.getEleTags()
        ntags = ops.getNodeTags()
        if etags is None:
            return
        if isinstance(etags, int):
            etags = [etags]

        for e in etags:
            elenodes = ops.eleNodes(e)
            for i in range(0, len(elenodes)):
                [xi, yi] = ops.nodeCoord(elenodes[i - 1])
                [xj, yj] = ops.nodeCoord(elenodes[i])
                ax.plot(np.array([xi, xj]), np.array([yi, yj]), 'k', marker='s')
                ax.text(0.5 * (xi + xj), 0.5 * (yi + yj), str(e), horizontalalignment='center',
                        verticalalignment='center',
                        color='b', fontsize=10)

            for n in ntags:
                coord = ops.nodeCoord(n)
                ax.text(coord[0], coord[1], str(n), horizontalalignment='center', verticalalignment='center',
                        color='r',
                        fontsize=10)

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('2D Model')
        plt.savefig('Model2D.jpeg', dpi=500)
        plt.show()

    def create_model(self):
        # Reference: https://opensees.berkeley.edu/wiki/index.php/OpenSees_Example_1b._Elastic_Portal_Frame (E1b)
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)

        h = 3000.0   # story height
        w = 5000.0   # bay width
        # mass
        m = 1000/9810  # kN/g (per floor)

        # story 0
        ops.node(1, 0.0, 0*h); ops.fix(1, 1, 1, 1)
        ops.node(2, w, 0*h); ops.fix(2, 1, 1, 1)
        # story 1
        ops.node(3, 0.0, 1*h); ops.mass(3, 0.5*m, 0.0, 0.0)
        ops.node(4, w, 1*h); ops.mass(4, 0.5*m, 0.0, 0.0)
        # story 2
        ops.node(5, 0.0, 2 * h); ops.mass(5, 0.5 * m, 0.0, 0.0)
        ops.node(6, w, 2 * h); ops.mass(6, 0.5 * m, 0.0, 0.0)
        # story 3
        ops.node(7, 0.0, 3 * h); ops.mass(7, 0.5 * m, 0.0, 0.0)
        ops.node(8, w, 3 * h); ops.mass(8, 0.5 * m, 0.0, 0.0)
        # story 4
        ops.node(9, 0.0, 4 * h); ops.mass(9, 0.5 * m, 0.0, 0.0)
        ops.node(10, w, 4 * h); ops.mass(10, 0.5 * m, 0.0, 0.0)
        # story 5
        ops.node(11, 0.0, 5 * h); ops.mass(11, 0.5 * m, 0.0, 0.0)
        ops.node(12, w, 5 * h); ops.mass(12, 0.5 * m, 0.0, 0.0)

        K = 8  # stiffness per story (just an assumption)
        E = 200  # kN/mm2

        Ic = 0.5 * K * (h**3) / (24*E)    # mm^4 (K=24 EI/h^3) two columns
        Ib = 1e12 * Ic   # almost rigid
        A = 1e12  #mm^2  (no axial deformation)


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
        # Damper properties: https://openseespydoc.readthedocs.io/en/latest/src/ViscousDamper.html?highlight=viscousdamper
        matTag = 1
        Kd, Cd, alpha, = 25.0, 20.7452, 0.35
        ops.uniaxialMaterial('ViscousDamper', matTag, Kd, Cd, alpha)
        ops.element('twoNodeLink', 16, *[1, 4], '-mat', matTag, '-dir', *[1])
        ops.element('twoNodeLink', 17, *[3, 6], '-mat', matTag, '-dir', *[1])
        ops.element('twoNodeLink', 18, *[5, 8], '-mat', matTag, '-dir', *[1])
        ops.element('twoNodeLink', 19, *[7, 10], '-mat', matTag, '-dir', *[1])
        ops.element('twoNodeLink', 20, *[9, 12], '-mat', matTag, '-dir', *[1])


        return self

    # def create_damping_matrix(self):
    #     Rayliegh(xDamp=0.05, alphaM=0.00, betaKcurr=0.0, betaKinit=0.0).create_damping_matrix()
    #     return self

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


# Test Class
# if __name__ == "__main__":
#     opsEnv = ShearFrameVD5Story1Bay(obs_nodes=[3], ctrl_node=3, device_ij_nodes=[3, 6])
#     GM = LoadGM(dt=.01, t_final=20, g=9810., SF=0.5, inputFile='RSN1086_NORTHR_SYL090.AT2', outputFile='myEQ.dat', plot=False)
#     # MR = SimpleMRD50kN()
#
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






