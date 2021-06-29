import os

import numpy as np
from scipy.signal import hilbert  # for envelop

from analyses import UniformExcitation
from control_devices import ControlDevice
from dl_models import NN
from ground_motions import LoadGM
from rl_algorithms import DQNAgent
from structural_models import ShearFrameVD1Story1Bay, Sensors

if __name__ == "__main__":

    if not os.path.exists('Results/MATLAB'):
        os.makedirs('Results/MATLAB')
    if not os.path.exists('Results/Plots'):
        os.makedirs('Results/Plots')
    if not os.path.exists('Results/Weights'):
        os.makedirs('Results/Weights')

    gm = LoadGM(dt=0.01, t_final=15, g=9810, SF=0.5, inputFile='RSN1086_NORTHR_SYL090.AT2', outputFile='myEQ.dat',
                plot=True)

    ctrl_device = ControlDevice(max_force=200, n_discrete=11, ctrl_device_ij_nodes=[1, 3])
    # ctrl_device = SimpleMRD50kN(GM, max_volt=5, max_force=50, n_discrete=6)

    # matTag, K_el, Cd, alpha = 10, 10, 10, 0.25
    # ops.uniaxialMaterial('ViscousDamper', matTag, K_el, Cd, alpha)
    # ctrl_device = PassiveTMD(loc_node=7, direction=1, m=10., mat_tag=10)

    structure = ShearFrameVD1Story1Bay()
    structure.create_model().draw2D()
    structure.create_damping_matrix().run_gravity()  # gravity loading is defined part of structure

    sensors = Sensors(sensors_placement={"groundAccel": [1], "accel": [3, 4], "vel": [3], "disp": [3]},
                       window_size=3, ctrl_node=3)

    analysis = UniformExcitation()
    dl_model = NN.simple_nn(n_hidden=10, n_units=15,
                            input_shape=(sensors.n_sensors * sensors.window_size,),
                            action_space=ctrl_device.action_space_discrete.n)

    run_steps = '1-step'  # do not change to 'full'
    for i_time in range(0, gm.resampled_npts):
        ctrl_force = 0.
        sensors, ctrl_device = analysis.run_dynamic(run_steps, i_time, ctrl_device, ctrl_force, gm,
                                                     sensors)
        if run_steps == 'full':
            break
    uncontrolled_ctrl_node_history = sensors.ctrl_node_history

    ########################################
    analysis = UniformExcitation()
    sensors = Sensors(sensors_placement={"groundAccel": [1], "accel": [3, 4], "vel": [3], "disp": [3]},
                      window_size=3, ctrl_node=3)

    dqn_controlled = DQNAgent(structure, sensors, gm, analysis, dl_model, ctrl_device,
                              uncontrolled_ctrl_node_history=uncontrolled_ctrl_node_history)
    dqn_controlled.run()
    # ops.wipe()
