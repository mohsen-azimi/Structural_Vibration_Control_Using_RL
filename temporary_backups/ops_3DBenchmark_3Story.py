#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Refrence:
https://openseespydoc.readthedocs.io/en/latest/src/portal2deq.html
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


class ops_Env():
    """
      Description:
          Cantiliver column vibration

      Observations:
          Type: Box(3)
          Num	Observation               Min           Max
          0	Floor1 Position (in)     -Inf           Inf

      Actions:
          Type: Box(1)
          Num	Action                     Min           Max
          0	Force on Floor1            -.1           .1     (force in kip)

      Reward:
          The reward is calculated each time step and is a negative cost.
          The cost function is the
            (i) First floor x-position/drift


      Starting State:
          Each episode, the system starts from t=0.0 sec (one step, using 'ops_initiateTH')

      Episode Termination:
          Episode ends after 100 timesteps.

      Solved Requirements:
          To be determined by comparison with the ideal controller.
      """

    def __init__(self):
        # Description
        self.env_name="ops_3DBenchmark_3Story"
        self.n_story=3

        # Control
        # self.n_controller = 1
        self.units='kip-in'
        self.viewer = None


        obs_max = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.observation_space = spaces.Box(low=-obs_max, high=obs_max, dtype=np.float32)
        # print(self.observation_space)

        self.action_maxForce = 500
        self.action_space_discrete = spaces.Discrete(51)  # for discrete DQN
        self.action_space_discrete_array = np.linspace(-self.action_maxForce, self.action_maxForce, num = self.action_space_discrete.n)
        self.action_space_continous = spaces.Discrete(1) # for continous A-C DQN: https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69

        # self.action_space = spaces.Box(np.float32(-self.maxForce),
        #                                np.float32(self.maxForce),
        #                                shape=(1,), dtype=np.float32)




    def create_model_3Dframe(self):
        # Reference: https://opensees.berkeley.edu/wiki/index.php/OpenSees_Example_1b._Elastic_Portal_Frame (E1b)
        # Bench Mark: Benchmark Control Problems for Seismically Excited
        # Nonlinear Buildings, Ohtori et al.
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        h = 13*12  # 13'-00"
        w = 30*12  # 30'-00"

$$$$$ , Until Here!
        ops.node(1, 0.0, 0.0)
        ops.node(2, h, 0.0)
        ops.node(3, 0.0, w)
        ops.node(4, h, w)

        ops.fix(1, 1, 1, 1)
        ops.fix(2, 1, 1, 1)
        ops.fix(3, 0, 0, 0)
        ops.fix(4, 0, 0, 0)

        ops.mass(3, 5.18, 0.0, 0.0)
        ops.mass(4, 5.18, 0.0, 0.0)

        ops.geomTransf('Linear', 1)
        A = 3600000000.0
        E = 4227.0
        Iz = 1080000.0

        A1 = 5760000000.0
        Iz1 = 4423680.0
        ops.element('elasticBeamColumn', 1, 1, 3, A, E, Iz, 1)
        ops.element('elasticBeamColumn', 2, 2, 4, A, E, Iz, 1)
        ops.element('elasticBeamColumn', 3, 3, 4, A1, E, Iz1, 1)

        eigen = ops.eigen('-fullGenLapack', 1)

        betaKcomm = float(2 * (0.02 / np.sqrt(eigen)[0]))
        xDamp = 0.02  # 2% damping ratio
        alphaM = 0.0  # M-prop. damping; D = alphaM*M
        betaKcurr = 0.0  # K-proportional damping;      +beatKcurr*KCurrent
        betaKinit = 0.0  # initial-stiffness proportional damping      +beatKinit*Kini

        ops.rayleigh(alphaM, betaKcurr, betaKinit, betaKcomm)  # https://openseespydoc.readthedocs.io/en/latest/src/reyleigh.html?highlight=rayleigh#rayleigh-command

    # In[3]:

    def run_gravity_2Dframe(self):
        # apply control force (static), u, at floor1
        ops.wipeAnalysis()
        # defining gravity loads
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        ops.eleLoad('-ele', 3, '-type', '-beamUniform', -7.94)

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
            ops.timeSeries('Path', tsTag, '-dt', GM.dt, '-filePath', GM.name, '-factor', GM.SF)
            ops.pattern('UniformExcitation', 2, GM.dir, '-accel', tsTag)

            ops.wipeAnalysis()
            ops.constraints('Plain')
            ops.numberer('Plain')
            ops.system('BandGeneral')
            ops.test('NormDispIncr', 1e-5, 10)
            ops.algorithm('Newton')
            ops.integrator('Newmark', 0.5, 0.25)
            ops.analysis('Transient')

            self.d3 = []
            self.v3 = []
            self.a3 = []
            self.t = []
            for ii in range(0, GM.npts):
                ops.analyze(1, GM.analysis_dt)

                self.t.append(ops.getTime())
                self.d3.append(ops.nodeDisp(3, 1))
                self.v3.append(ops.nodeVel(3, 1))
                self.a3.append(ops.nodeAccel(3, 1))

        else:
            # apply control force, u, at floor1
            ops.wipeAnalysis()

            # applying Dynamic Ground motion analysis

            F_tsTag, F_patternTag = 9, 9
            EQ_tsTag, EQ_patternTag = 2, 2
            nodeTag = 3

            ## EQ
            if i == 0:
                self.d3 = []
                self.v3 = []
                self.a3 = []
                self.t = []

                ops.timeSeries('Path', EQ_tsTag, '-dt', GM.dt, '-filePath', GM.name, '-factor', GM.SF)
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

            ops.analyze(1, GM.analysis_dt)

            self.t.append(ops.getTime())
            self.d3.append(ops.nodeDisp(3, 1))
            self.v3.append(ops.nodeVel(3, 1))
            self.a3.append(ops.nodeAccel(3, 1))

            # ops.loadConst()
            ops.wipeAnalysis()

    def render(self, mode='human', SF=1.0):
        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.linewidth = 30
            colL = rendering.Line(start=(10, 0), end=(10+SF*self.d3[-1], 300))
            colL.set_color(0, 0, 0)
            self.viewer.add_geom(colL)

            colR = rendering.Line(start=(390, 0), end=(390+SF*self.d3[-1], 300))
            colR.set_color(0, 0, 0)
            self.viewer.add_geom(colR)


            beam = rendering.Line(start=(10+SF*self.d3[-1], 300), end=(390+SF*self.d3[-1], 300))
            beam.set_color(0, 0, 0)
            self.viewer.add_geom(beam)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')





    def plot_TH(self, method):

        if method == 'full':
            lineColor = 'r-'
        else:
            lineColor = 'b-'

        # plt.figure(figsize=(15,10))
        plt.plot(self.t,self.d3, lineColor)
        plt.legend([method])
        plt.show()



# Test Class
if __name__ == "__main__":
    opsEnv = ops_Env()
    GM = ops_GM(dt=0.01, g=386., SF=1., dir=1, inputFile='H-E12140.AT2', outputFile='myEQ.dat', plot=False)

    opsEnv.create_model_2Dframe()
    opsEnv.draw2D()
    opsEnv.run_gravity_2Dframe()

    runSteps = '1-step'   # 'full' or '1-step'
    for i in range(0, GM.npts):
        u = 0.
        opsEnv.run_dynamic_2Dframe(GM, runSteps, i, u)
        if runSteps == 'full':
            break

    opsEnv.plot_TH(runSteps)

