'''
Refrence:
https://opensees.berkeley.edu/wiki/index.php/Dynamic_Analyses_of_1-Story_Moment_Frame_with_Viscous_Dampers
'''
##########################################################################################################################################################################
import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy import signal
from ops_GM import ops_GM
from gym import spaces, logger
from collections import deque


class ShearFrameVD_1Story1Bay:
    """
      Description:
      Control Node ID = 3 (top-left)
      """

    def __init__(self, obs_node=3, ctrl_node=3):
        # Description
        self.env_name = "ShearFrameVD"
        self.n_story = 1
        self.obs_node = obs_node  # The note to be controlled
        # self.DQ = deque(maxlen=dq_len) # =3 observation window
        self.units = 'kN-mm'

        # observer
        obs_max = np.array([
            np.finfo(np.float32).max,    # Displ @ ctrl_node, t
            np.finfo(np.float32).max,    # Displ @ ctrl_node, t-1
            np.finfo(np.float32).max,    # Displ @ ctrl_node, t-2
            np.finfo(np.float32).max,    # Vel @ ctrl_node
            np.finfo(np.float32).max,    # Accel @ ctrl_node
            np.finfo(np.float32).max])   # Ground Accel @ base
        self.observation_space = spaces.Box(low=-obs_max, high=obs_max, dtype=np.float32)

        self.action_maxForce = 100
        self.action_space_discrete = spaces.Discrete(11)  # for discrete DQN
        self.action_space_discrete_array = np.linspace(-self.action_maxForce, self.action_maxForce, num = self.action_space_discrete.n)
        self.action_space_continous = spaces.Discrete(1) # for continous A-C DQN: https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69

        # self.action_space = spaces.Box(np.float32(-self.maxForce),
        #                                np.float32(self.maxForce),
        #                                shape=(1,), dtype=np.float32)

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

    def create_model_2Dframe(self):
        # Reference: https://opensees.berkeley.edu/wiki/index.php/OpenSees_Example_1b._Elastic_Portal_Frame (E1b)
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)

        h = 3000.0   # story height
        w = 5000.0   # bay width

        ops.node(1, 0.0, 0.0)
        ops.node(2, w, 0.0)
        ops.node(3, 0.0, h)
        ops.node(4, w, h)

        ops.fix(1, 1, 1, 1)
        ops.fix(2, 1, 1, 1)

        # mass
        m = 1000/9810  # kN/g

        ops.mass(3, 0.5*m, 0.0, 0.0)
        ops.mass(4, 0.5*m, 0.0, 0.0)


        Tn = 0.7  # Natural Period
        K = (2*np.pi/Tn)**2 * m
        E = 200  # kN/mm2
        Ic = 0.5 * K * (h**3) / (24*E)    # mm^4 (K=24 EI/h^3) two columns
        Ib = 1e12 * Ic   # almost rigid
        A = 1e12  #mm^2  (no axial deformation)

        ops.geomTransf('Linear', 1)

        # Elements
        # Columns
        ops.element('elasticBeamColumn', 1, 1, 3, A, E, Ic, 1)
        ops.element('elasticBeamColumn', 2, 2, 4, A, E, Ic, 1)
        # Beam
        ops.element('elasticBeamColumn', 3, 3, 4, A, E, Ib, 1)

        # Two Node Link Element
        # Damper properties: https://openseespydoc.readthedocs.io/en/latest/src/ViscousDamper.html?highlight=viscousdamper
        matTag = 1
        Kd, Cd, alpha, = 25.0, 20.7452, 0.35
        ops.uniaxialMaterial('ViscousDamper', matTag, Kd, Cd, alpha)
        ops.element('twoNodeLink', 4, *[1, 4], '-mat', matTag, '-dir', *[1])



        eigen = ops.eigen('-fullGenLapack', 1)

        # Damping: D=α_M∗M + β_K∗K_curr + β_Kinit∗K_init + β_Kcomm∗K_commit
        xDamp = 0.05  # 5% damping ratio
        alphaM = 0.00  # M-prop. damping; D = alphaM*M
        betaKcurr = 0.0  # K-proportional damping;      +beatKcurr*KCurrent
        betaKcomm = float(2 * (xDamp / np.sqrt(eigen)[0]))
        betaKinit = 0.0  # initial-stiffness proportional damping      +beatKinit*Kini

        ops.rayleigh(alphaM, betaKcurr, betaKinit, betaKcomm)  # https://openseespydoc.readthedocs.io/en/latest/src/reyleigh.html?highlight=rayleigh#rayleigh-command
        # https://openseespydoc.readthedocs.io/en/latest/src/exampleRotDSpectra.html?highlight=rayleigh#rotd-spectra-of-ground-motion

    def run_gravity_2Dframe(self):
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

    def run_dynamic_2Dframe(self, GM, runSteps, i, u):
        if runSteps == 'full':
            ops.wipeAnalysis()
            # applying Dynamic Ground motion analysis
            tsTag = 2
            ops.timeSeries('Path', tsTag, '-dt', GM.resampled_dt, '-filePath', GM.name, '-factor', GM.SF)
            ops.pattern('UniformExcitation', 2, GM.dir, '-accel', tsTag)

            ops.wipeAnalysis()
            ops.constraints('Plain')
            ops.numberer('Plain')
            ops.system('UmfPack')
            ops.test('NormDispIncr', 1e-10, 100)
            ops.algorithm('Newton')
            ops.integrator('Newmark', 0.5, 0.25)
            ops.analysis('Transient')

            self.ctrl_node_disp = []
            self.ctrl_node_vel = []
            self.ctrl_node_accel = []
            self.time = []
            for ii in range(0, GM.resampled_npts):
                ops.analyze(1, GM.resampled_dt)

                self.time.append(ops.getTime())
                self.ctrl_node_disp.append(ops.nodeDisp(self.obs_node, 1))
                self.ctrl_node_vel.append(ops.nodeVel(self.obs_node, 1))
                self.ctrl_node_accel.append(ops.nodeAccel(self.obs_node, 1))

                # self.DQ.append((ops.nodeDisp(self.ctrl_node, 1),
                #                 ops.nodeVel(self.ctrl_node, 1),
                #                 ops.nodeAccel(self.ctrl_node, 1)))
        else:
            # apply control force, u, at floor1
            ops.wipeAnalysis()

            # applying Dynamic Ground motion analysis
            F_tsTag, F_patternTag = 9, 9
            EQ_tsTag, EQ_patternTag = 2, 2
            nodeTag = self.obs_node

            ## EQ
            if i == 0:
                self.ctrl_node_disp = []
                self.ctrl_node_vel = []
                self.ctrl_node_accel = []
                self.time = []

                ops.timeSeries('Path', EQ_tsTag, '-dt', GM.resampled_dt, '-filePath', GM.outputFile, '-factor', GM.SF)
                # ops.timeSeries('Path', F_tsTag, '-dt', dt, '-filePath', 'BM68elc_F.acc', '-factor', 1.0)

            ops.remove('loadPattern', EQ_patternTag)
            ops.pattern('UniformExcitation', EQ_patternTag, GM.dir, '-accel', EQ_tsTag)

            ops.remove('timeSeries', F_tsTag)
            ops.timeSeries('Constant', F_tsTag, '-factor', u)

            ops.remove('loadPattern', F_patternTag)
            ops.pattern('Plain', F_patternTag, F_tsTag)
            ops.load(nodeTag, 1, 0., 0.)

            ops.wipeAnalysis()
            ops.constraints('Plain')
            ops.numberer('Plain')
            ops.system('BandGeneral')
            ops.test('NormDispIncr', 1e-8, 10)
            ops.algorithm('Newton')
            ops.integrator('Newmark', 0.5, 0.25)
            ops.analysis('Transient')

            ops.analyze(1, GM.resampled_dt)

            self.time.append(ops.getTime())
            self.ctrl_node_disp.append(ops.nodeDisp(self.obs_node, 1))
            self.ctrl_node_vel.append(ops.nodeVel(self.obs_node, 1))
            self.ctrl_node_accel.append(ops.nodeAccel(self.obs_node, 1))

            # self.DQ.append((ops.nodeDisp(self.ctrl_node, 1),
            #                 ops.nodeVel(self.ctrl_node, 1),
            #                 ops.nodeAccel(self.ctrl_node, 1)))

            # ops.loadConst()
            ops.wipeAnalysis()

    # def render(self, mode='human', SF=1.0):
    #     screen_width = 600
    #     screen_height = 400
    #
    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #         self.viewer = rendering.Viewer(screen_width, screen_height)
    #
    #         self.linewidth = 30
    #         colL = rendering.Line(start=(10, 0), end=(10+SF*self.d3[-1], 300))
    #         colL.set_color(0, 0, 0)
    #         self.viewer.add_geom(colL)
    #
    #         colR = rendering.Line(start=(390, 0), end=(390+SF*self.d3[-1], 300))
    #         colR.set_color(0, 0, 0)
    #         self.viewer.add_geom(colR)
    #
    #
    #         beam = rendering.Line(start=(10+SF*self.d3[-1], 300), end=(390+SF*self.d3[-1], 300))
    #         beam.set_color(0, 0, 0)
    #         self.viewer.add_geom(beam)
    #
    #     return self.viewer.render(return_rgb_array = mode=='rgb_array')





    def plot_TH(self):


        plt.plot(self.time,self.ctrl_node_disp, 'r-')

        plt.legend("accel")
        plt.show()



# Test Class
if __name__ == "__main__":
    opsEnv = ShearFrameVD_1Story1Bay(ctrl_node=3)
    GM = ops_GM(dt=.01, t_final=40, g=9810., SF=0.5, inputFile='RSN1086_NORTHR_SYL090.AT2', outputFile='myEQ.dat', plot=False)

    opsEnv.create_model_2Dframe()
    opsEnv.draw2D()

    opsEnv.run_gravity_2Dframe()

    runMethod = '1-step'  # do not change to 'full'
    for i in range(0, GM.resampled_npts):
        u = 0.
        opsEnv.run_dynamic_2Dframe(GM, runMethod, i, u)
        if runMethod == 'full':
            break
    opsEnv.plot_TH()






