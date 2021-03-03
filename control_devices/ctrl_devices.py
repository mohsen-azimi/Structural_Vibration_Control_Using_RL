#  Device1: MR Damper

"""
Refrence:
https://www.mdpi.com/2224-2708/9/2/18/htm
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy import signal
from ground_motions.read_peer import LoadGM
# from structural_models.ops_ShearFrameVD_5Story1Bay import ShearFrameVD5Story1Bay
import gym
from collections import deque


class ActiveControl:
    """
    Active control using DQN
    """

    def __init__(self, max_force=100, n_discrete=11):
        self.device_type = "active"
        self.max_force = max_force
        self.action_space_discrete = gym.spaces.Discrete(n_discrete)  # for discrete DQN
        self.action_space_discrete_array = np.linspace(-self.max_force, self.max_force,
                                                       num=self.action_space_discrete.n)
        self.action_space_continuous = gym.spaces.Discrete(1)

    def calc_device_force(self, action):
        force = self.action_space_discrete_array[action]
        return force


class SimpleMRD50kN:
    """
      Description:
        Simple MR-Damper "Amini et al 2015 , 50kN" (N m kg)'
        doi:10.1088/0964-1726/24/10.05002
      """

    def __init__(self, gm, max_volt=5, max_force=50, n_discrete=6):
        # DQN Action Space
        # Max Capacity
        self.max_volt = max_volt  # volt
        self.max_force = max_force  # kN

        # self.action_maxVolt = self.max_volt
        self.action_space_discrete = gym.spaces.Discrete(n_discrete)  # for discrete DQN
        self.action_space_discrete_array = np.linspace(0, self.max_volt,
                                                       num=self.action_space_discrete.n)
        self.action_space_continuous = gym.spaces.Discrete(1)

        # ###################
        self.device_name = "SimpleMRD50kN"
        self.units = 'N-m'

        #  Device Properties (converted to kN-mm)
        self.c0A = 44  # N.s/m
        self.c0B = 440  # N.s/m . V

        self.alphaA = 1087200  # N/m
        self.alphaB = 4691600  # N/m . V

        self.gamma = 300  # m^-2
        self.beta = 300  # m^-2
        self.A = 1.2
        self.n = 1
        self.eta = 50  # s^-1
        self.x0 = 0  # m

        # Pre-allocations (change later to speed-up)
        # self.c0 = np.zeros_like(GM.analysis_time)  + self.c0A
        # self.alpha = np.zeros_like(GM.analysis_time)  + self.alphaA
        # self.dz = np.zeros_like(GM.analysis_time)
        self.z = np.zeros_like(gm.resampled_time)
        self.force = np.zeros_like(gm.resampled_time)
        self.volt = np.zeros_like(gm.resampled_time)
        self.u = np.zeros_like(gm.resampled_time)
        self.disp = np.zeros_like(gm.resampled_time)
        self.vel = np.zeros_like(gm.resampled_time)

    def calc_device_force(self, i_time, dt, action):
        # dt = GM.resampled_dt
        z2 = self.z[i_time]
        volt = action
        v1 = self.vel[i_time - 1] * 0.001  # (unit=m)
        v2 = self.vel[i_time] * 0.001  # (unit=m)
        eta = self.eta

        gamma = self.gamma
        beta = self.beta
        A = self.A
        n = self.n

        u = self.u[i_time - 1]
        uD = -eta * (u - volt)
        u = uD * dt + u

        c0 = self.c0A + self.c0B * u
        alpha = self.alphaA + self.alphaB * u

        # RK4
        # % y(i + 1) = y(i) + dy
        # % dy = 1 / 6(k1 + 2K2 + 2k3 + k4)h
        # ...
        # k1 = f(x(i), y(i))
        # k2 = f(x(i) + h / 2, y(i) + k1 * h / 2)
        # k3 = f(x(i) + h / 2, y(i) + k2 * h / 2)
        # k4 = f(x(i) = h, y(i) + k3 * h)
        # % h = dt

        kz1 = dt * (- gamma * abs(v1) * z2 * abs(z2) ** (n - 1) - beta * v1 * abs(z2) ** n + A * v1)

        kz2 = dt * (- gamma * abs((v1 + v2) / 2) * (z2 + kz1 / 2) * abs(z2 + kz1 / 2) ** (n - 1)
                    - beta * ((v1 + v2) / 2) * abs(z2 + kz1 / 2) ** n + A * ((v1 + v2) / 2))

        kz3 = dt * (- gamma * abs((v1 + v2) / 2) * (z2 + kz2 / 2) * abs(z2 + kz2 / 2) ** (n - 1)
                    - beta * ((v1 + v2) / 2) * abs(z2 + kz2 / 2) ** n + A * ((v1 + v2) / 2))

        kz4 = dt * (- gamma * abs(v2) * z2 * abs(z2) ** (n - 1) - beta * v2 * abs(z2 + kz3) ** n + A * v2)

        # Next value of  variable (z)and variable (y)
        dz = (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6
        z3 = z2 + dz

        # Damper force using Modified Bouc-Wen model
        self.force[i_time] = force = alpha * z3 + c0 * v2
        # self.force[i] = max(min(alpha * z3 + c0 * v2, self.Fmax), -self.Fmax)

        self.u[i_time] = u
        if i_time >= gm.resampled_npts - 1:
            self.z[i_time] = z3
        else:
            self.z[i_time + 1] = z3
        # self.dz[i] = dz
        # self.alpha[i] = alpha
        # self.c0[i] = c0
        return force


#
if __name__ == "__main__":

    gm = LoadGM(dt=.01, t_final=50, g=9810., SF=0.5, inputFile='ground_motions\\assets\\RSN1086_NORTHR_SYL090.AT2',
                outputFile='myEQ.dat', plot=False)

    dt = 0.001  # load from peer
    t = np.arange(0, 2, dt)
    f = 1  # Hz
    # w = 1/(2*math.pi)*f # in rad/s
    w = 9.43
    gm.resampled_time = t
    gm.resampled_dt = dt
    gm.resampled_npts = len(t)

    MR = SimpleMRD50kN(max_volt=5, max_force=50)
    MR.disp = 6 * np.sin(w * t) * 10
    MR.vel = 40 * np.cos(w * t) * 10  # *0.01 (cm -->m)

    # #############################################
    plt.figure()
    for v in (0, 1, 2, 3, 4, 5):

        for i in range(0, gm.resampled_npts):
            MR.volt[i] = v
            MR.calc_device_force()

            if i % (1 / gm.resampled_dt) == 0:
                if i % (10 / gm.resampled_dt) == 0:
                    print('▮', end='')
                else:
                    print('|', end='')

        # plt.plot(MR.force / 1000)
        plt.plot(MR.vel[100:] * 100, MR.force[100:] / 1000)
        # plt.plot(MR.displ[200:]*100, MR.force[200:]/1000)

    plt.show()
