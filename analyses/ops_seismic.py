'''
Refrence:
https://opensees.berkeley.edu/wiki/index.php/Dynamic_Analyses_of_1-Story_Moment_Frame_with_Viscous_Dampers
'''
##########################################################################################################################################################################
import openseespy.opensees as ops
import numpy as np


class UniformExcitation(object):
    """
      """
    def __init__(self):
        self.type = "UniformExcitation"
        # self.time_reset()

    # def time_reset(self):
    #     self.time = [0.]

    def plot_disp_TH(self):
        # PlotResults.plot_disp_TH()
        pass
        # return self

    def run_dynamic(self, run_steps, i_time, ctrl_device, ctrl_force, gm, sensors):
        if run_steps == 'full':
            ops.wipeAnalysis()
            # applying Dynamic Ground motion analysis
            tsTag = 2
            ops.timeSeries('Path', tsTag, '-dt', gm.resampled_dt, '-filePath', gm.name, '-factor', gm.SF)
            ops.pattern('UniformExcitation', 2, gm.dir, '-accel', tsTag)

            ops.wipeAnalysis()
            ops.constraints('Plain')
            ops.numberer('Plain')
            ops.system('UmfPack')
            ops.test('NormDispIncr', 1e-10, 100)
            ops.algorithm('Newton')
            ops.integrator('Newmark', 0.5, 0.25)
            ops.analysis('Transient')

            for ii in range(0, gm.resampled_npts):
                ops.analyze(1, gm.resampled_dt)

                # sensors.time.append(ops.getTime())
                sensors.sensors_history['time'].append(ops.getTime())

                ctrl_device.time.append(ops.getTime())
                ctrl_device.force_history.append(ctrl_force)

                for key, value in sensors.sensors_placement.items():
                    if key == "groundAccel":
                        accel_g = np.zeros((len(sensors.sensors_placement["groundAccel"]), 1), dtype=np.float64)
                        for i, node in enumerate(value):
                            accel_g[i] = gm.resampled_signal[i_time]  # would raise error for i_time!
                        sensors.sensors_history["groundAccel"] = np.hstack((sensors.sensors_history["groundAccel"], accel_g))
                    if key == "disp":
                        disp = np.zeros((len(sensors.sensors_placement["disp"]), 1), dtype=np.float64)
                        for i, node in enumerate(value):
                            disp[i] = ops.nodeDisp(node, 1)
                        sensors.sensors_history["disp"] = np.hstack((sensors.sensors_history["disp"], disp))
                    if key == "vel":
                        vel = np.zeros((len(sensors.sensors_placement["vel"]), 1), dtype=np.float64)
                        for i, node in enumerate(value):
                            vel[i] = ops.nodeVel(node, 1)
                        sensors.sensors_history["vel"] = np.hstack((sensors.sensors_history["vel"], vel))
                    if key == "accel":
                        accel = np.zeros((len(sensors.sensors_placement["accel"]), 1), dtype=np.float64)
                        for i, node in enumerate(value):
                            accel[i] = ops.nodeAccel(node, 1)
                        sensors.sensors_history["accel"] = np.hstack((sensors.sensors_history["accel"], accel))

                # For convenience, save control node history as separate variable
                sensors.ctrl_node_history['time'] = sensors.sensors_history['time']
                for key, value in sensors.sensors_placement.items():
                    if sensors.ctrl_node in value:
                        indx = value.index(sensors.ctrl_node)
                        sensors.ctrl_node_history[key] = sensors.sensors_history[key][indx]

        else:
            # apply control force, u, at floor1
            ops.wipeAnalysis()

            # applying Dynamic Ground motion analysis
            F_tsTag_i, F_patternTag_i = 9, 9
            F_tsTag_j, F_patternTag_j = 99, 99
            EQ_tsTag, EQ_patternTag = 2, 2

            ## EQ
            if i_time == 0:
                ops.remove('timeSeries', EQ_tsTag)
                ops.timeSeries('Path', EQ_tsTag, '-dt', gm.resampled_dt, '-filePath', gm.outputFile, '-factor', gm.SF)

            ops.remove('loadPattern', EQ_patternTag)
            ops.pattern('UniformExcitation', EQ_patternTag, gm.dir, '-accel', EQ_tsTag)

            # apply control forces at i & j nodes
            ops.remove('timeSeries', F_tsTag_i)
            ops.timeSeries('Constant', F_tsTag_i, '-factor', -ctrl_force)
            ops.remove('loadPattern', F_patternTag_i)
            ops.pattern('Plain', F_patternTag_i, F_tsTag_i)
            ops.load(ctrl_device.ctrl_device_ij_nodes[0], 1, 0., 0.)

            ops.remove('timeSeries', F_tsTag_j)
            ops.timeSeries('Constant', F_tsTag_j, '-factor', ctrl_force)
            ops.remove('loadPattern', F_patternTag_j)
            ops.pattern('Plain', F_patternTag_j, F_tsTag_j)
            ops.load(ctrl_device.ctrl_device_ij_nodes[1], 1, 0., 0.)

            ops.wipeAnalysis()
            ops.constraints('Plain')
            ops.numberer('Plain')
            ops.system('BandGeneral')
            ops.test('NormDispIncr', 1e-8, 10)
            ops.algorithm('Newton')
            ops.integrator('Newmark', 0.5, 0.25)
            ops.analysis('Transient')

            ops.analyze(1, gm.resampled_dt)

            # sensors.time.append(ops.getTime())
            sensors.sensors_history['time'].append(ops.getTime())

            ctrl_device.time.append(ops.getTime())
            ctrl_device.force_history.append(ctrl_force)

            for key, value in sensors.sensors_placement.items():
                if key == "groundAccel":
                    accel_g = np.zeros((len(sensors.sensors_placement["groundAccel"]), 1), dtype=np.float64)
                    for i, node in enumerate(value):
                        accel_g[i] = gm.resampled_signal[i_time-1]
                    sensors.sensors_history["groundAccel"] = np.hstack((sensors.sensors_history["groundAccel"], accel_g))
                if key == "disp":
                    disp = np.zeros((len(sensors.sensors_placement["disp"]), 1), dtype=np.float64)
                    for i, node in enumerate(value):
                        disp[i] = ops.nodeDisp(node, 1)
                    sensors.sensors_history["disp"] = np.hstack((sensors.sensors_history["disp"], disp))
                if key == "vel":
                    vel = np.zeros((len(sensors.sensors_placement["vel"]), 1), dtype=np.float64)
                    for i, node in enumerate(value):
                        vel[i] = ops.nodeVel(node, 1)
                    sensors.sensors_history["vel"] = np.hstack((sensors.sensors_history["vel"], vel))
                if key == "accel":
                    accel = np.zeros((len(sensors.sensors_placement["accel"]), 1), dtype=np.float64)
                    for i, node in enumerate(value):
                        accel[i] = ops.nodeAccel(node, 1)
                    sensors.sensors_history["accel"] = np.hstack((sensors.sensors_history["accel"], accel))

            # For convenience, save control node history as separate variable
            sensors.ctrl_node_history['time'] = sensors.sensors_history['time']
            for key, value in sensors.sensors_placement.items():
                if sensors.ctrl_node in value:
                    indx = value.index(sensors.ctrl_node)
                    sensors.ctrl_node_history[key] = sensors.sensors_history[key][indx]

            # ops.loadConst()
            ops.wipeAnalysis()
        return sensors, ctrl_device
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


    # def plot_TH(self):
    #     for node in range(len(structure.obs_nodes)):
    #         plt.plot(self.time, self.obs_nodes_disp[node, :], label=f"Disp @ node {structure.obs_nodes[node]}")
    #     plt.legend()
    #     plt.show()



# # Test Class
# if __name__ == "__main__":
#     opsEnv = ShearFrameVD5Story1Bay(obs_nodes=[3], ctrl_node=3, device_ij_nodes=[3, 6])
#     GM = ops_GM(dt=.01, t_final=20, g=9810., SF=0.5, inputFile='RSN1086_NORTHR_SYL090.AT2', outputFile='myEQ.dat', plot=False)
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






