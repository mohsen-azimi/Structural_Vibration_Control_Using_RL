# import os
import tensorflow as tf
import scipy.io
import random
import pandas as pd
import numpy as np
from collections import deque
from scipy.signal import hilbert  # for envelop
import matplotlib.pyplot as plt

#  project related imports:
import openseespy.opensees as ops

from ground_motions import LoadGM
from structural_models import ShearFrameVD5Story1Bay, ShearFrameVD1Story1Bay
from control_devices import ActiveControl, PassiveTMD
from analyses import UniformExcitation
from dl_models import NN
from rl_algorithms import DQNAgent



if __name__ == "__main__":
    # if not os.path.exists('Results'):
    #     os.makedirs('Results')

    gm = LoadGM(dt=0.01, t_final=40, g=9810, SF=0.5, inputFile='RSN1086_NORTHR_SYL090.AT2', outputFile='myEQ.dat',
                plot=False)
    sensors_loc = {"groundAccel": [1], "disp": [3], "vel": [3], "accel": [3]}  # future: strct = make("structure1")
    structure = ShearFrameVD1Story1Bay(sensors_loc=sensors_loc, memory_len=1, ctrl_node=3, device_ij_nodes=[1, 3])
    structure.create_model().draw2D().create_damping_matrix().run_gravity()  # gravity loading is defined part of structure

    analysis = UniformExcitation(structure)
    ctrl_device = ActiveControl(max_force=200, n_discrete=21)
    # ctrl_device = SimpleMRD50kN(GM, max_volt=5, max_force=50, n_discrete=6)

    # matTag, K_el, Cd, alpha = 10, 10, 10, 0.25
    # ops.uniaxialMaterial('ViscousDamper', matTag, K_el, Cd, alpha)
    # ctrl_device = PassiveTMD(loc_node=7, direction=1, m=10., mat_tag=10)
    # ctrl_device.place_tmd()
    # structure.create_damping_matrix()
    # Create the controller model

    dl_model = NN.simple_nn(n_units=10, n_hidden=5,
                            input_shape=(structure.STATE_SIZE,),
                            action_space=ctrl_device.action_space_discrete.n)

    agent_unctrl = DQNAgent()  # to make sure it does not mix with controlled one below

    run_steps = '1-step'  # do not change to 'full'
    for i_time in range(0, gm.resampled_npts):
        ctrl_force = 0.
        agent_unctrl.analysis.run_dynamic(run_steps, i_time, ctrl_force, gm, structure)
        if run_steps == 'full':
            break

    agent_unctrl.analysis.ctrl_node_disp_env = abs(hilbert(agent_unctrl.analysis.ctrl_node_disp))

    # agent_unctrl.env.plot_TH()
    ########################################
    analysis = UniformExcitation(structure)  # re-initiate to avoid overwriting the uncontrolled response
    agent_ctrl = DQNAgent()
    agent_ctrl.run()
    # ops.wipe()
