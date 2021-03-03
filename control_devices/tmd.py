#  Device1: MR Damper

"""
Refrence:
Mohsen Azimi (...)
"""
import gym
import numpy as np
import openseespy.opensees as ops


class PassiveTMD:
    """
    Active control using DQN
    """
    def __init__(self, loc_node, direction, m, mat_tag):
        self.device_type = "passive"
        self.nodes = [loc_node, loc_node+1]  # pre-define the i-j nodes (j node would change below)
        self.nodes_coord = ops.nodeCoord(loc_node)
        self.dir = direction
        self.m = m
        self.mat_tag = mat_tag
        self.max_force = 0.
        self.action_space_discrete = gym.spaces.Discrete(1)  # for discrete DQN
        self.action_space_discrete_array = np.linspace(-self.max_force, self.max_force,
                                                       num=self.action_space_discrete.n)
        self.action_space_continuous = gym.spaces.Discrete(1)

    def place_tmd(self):
        nodetags = ops.getNodeTags()
        eletags = ops.getEleTags()
        if eletags is None:
            print("No element tag is found!")
            return
        if isinstance(eletags, int):
            eletags = [eletags]

        i_node = self.nodes[0]
        j_node = self.nodes[1]

        eletag = 2
        while j_node in nodetags:
            j_node += 1
        while eletag in eletags:
            eletag += 1

        self.nodes = [i_node, j_node]  # save for future use
        # add a j-node
        # print(self.nodes_coord)
        if len(self.nodes_coord) == 2:
            ops.node(j_node, self.nodes_coord[0], self.nodes_coord[1])  # if a 2D model
        else:
            ops.node(j_node, self.nodes_coord[0], self.nodes_coord[1], self.nodes_coord[2])  # if a 3D model

        # add mass at j_node
        if self.dir == 1:
            ops.mass(j_node, self.m, 0.0, 0.0)  # x-dir
        else:
            ops.mass(j_node, 0.0, self.m, 0.0)  # y-dir

        # add element
        ops.element('zeroLength', eletag, i_node, j_node, '-mat', self.mat_tag, '-dir', self.dir)

        # # run rayreigh again!
        # eigen = ops.eigen('-fullGenLapack', 1)
        # # Damping: D=α_M∗M + β_K∗K_curr + β_Kinit∗K_init + β_Kcomm∗K_commit
        # xDamp = 0.05  # 5% damping ratio
        # alphaM = 0.00  # M-prop. damping; D = alphaM*M
        # betaKcurr = 0.0  # K-proportional damping;      +beatKcurr*KCurrent
        # betaKcomm = float(2 * (xDamp / np.sqrt(eigen)[0]))
        # betaKinit = 0.0  # initial-stiffness proportional damping      +beatKinit*Kini
        #
        # ops.rayleigh(alphaM, betaKcurr, betaKinit, betaKcomm)  # https://openseespydoc.readthedocs.io/en/latest/src/reyleigh.html?highlight=rayleigh#rayleigh-command
        # # https://openseespydoc.readthedocs.io/en/latest/src/exampleRotDSpectra.html?highlight=rayleigh#rotd-spectra-of-ground-motion

    def calc_device_force(self, action):
        # print("Force = 0")
        force = 0. * action
        return force

