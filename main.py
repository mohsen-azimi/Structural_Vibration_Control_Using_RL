import os

import numpy as np
from scipy.signal import hilbert  # for envelop

from analyses import UniformExcitation
from control_devices import ActiveControl
from dl_models import NN
from ground_motions import LoadGM
from rl_algorithms import DQNAgent
from structural_models import ShearFrameVD5Story1Bay, DAQ

if __name__ == "__main__":
    if not os.path.exists('Results/MATLAB'):
        os.makedirs('Results/MATLAB')
    if not os.path.exists('Results/Plots'):
        os.makedirs('Results/Plots')
    if not os.path.exists('Results/Weights'):
        os.makedirs('Results/Weights')

    gm = LoadGM(dt=0.01, t_final=15, g=9810, SF=0.5, inputFile='RSN1086_NORTHR_SYL090.AT2', outputFile='myEQ.dat',
                plot=True)
    sensors = DAQ(sensors_placement={"groundAccel": [1], "disp": [2], "vel": [2], "accel": [2]},
                               window_size=1, ctrl_node=3)
    structure = ShearFrameVD5Story1Bay(sensors, ctrl_device_ij_nodes=[1, 4])
    structure.create_model().draw2D().create_damping_matrix().run_gravity()  # gravity loading is defined part of structure

    analysis = UniformExcitation(structure)
    ctrl_device = ActiveControl(max_force=200, n_discrete=11)
    # ctrl_device = SimpleMRD50kN(GM, max_volt=5, max_force=50, n_discrete=6)

    # matTag, K_el, Cd, alpha = 10, 10, 10, 0.25
    # ops.uniaxialMaterial('ViscousDamper', matTag, K_el, Cd, alpha)
    # ctrl_device = PassiveTMD(loc_node=7, direction=1, m=10., mat_tag=10)
    # ctrl_device.place_tmd()
    # structure.create_damping_matrix()
    # Create the controller model

    dl_model = NN.simple_nn(n_hidden=10, n_units=15,
                            input_shape=(np.size(sensors.state),),
                            action_space=ctrl_device.action_space_discrete.n)

    agent_unctrld = DQNAgent(structure, sensors, gm, analysis, dl_model, ctrl_device)  # to make sure it does not mix with controlled one below

    run_steps = '1-step'  # do not change to 'full'
    for i_time in range(0, gm.resampled_npts):
        ctrl_force = 0.
        agent_unctrld.analysis.run_dynamic(run_steps, i_time, ctrl_force, gm, structure, sensors)
        if run_steps == 'full':
            break

    agent_unctrld.analysis.ctrl_node_disp_env = abs(hilbert(agent_unctrld.analysis.ctrl_node_disp))
    structure.unctrld_analysis = agent_unctrld.analysis
    # agent_unctrld.env.plot_TH()
    ########################################
    analysis = UniformExcitation(structure)  # re-initiate to avoid overwriting the uncontrolled response
    agent_ctrld = DQNAgent(structure, sensors, gm, analysis, dl_model, ctrl_device)
    agent_ctrld.run()
    # ops.wipe()
