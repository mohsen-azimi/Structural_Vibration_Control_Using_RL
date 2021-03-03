import openseespy.opensees as ops
from analyses.ops_eigen import Eigen

import numpy as np


class Rayliegh(object):
    """
      Description:
      Control Node ID = # ()
      """
    def __init__(self, xDamp=0.05, alphaM=0.00, betaKcurr=0.0, betaKinit=0.0):
        # # Damping: D=α_M∗M + β_K∗K_curr + β_Kinit∗K_init + β_Kcomm∗K_commit
        self.xDamp = xDamp   # ~5% damping ratio
        self.alphaM = alphaM  # M-prop. damping; D = alphaM*M
        self.betaKcurr = betaKcurr  # K-proportional damping;      +beatKcurr*KCurrent
        self.betaKinit = betaKinit  # initial-stiffness proportional damping      +beatKinit*Kini

    def create_damping_matrix(self):
        eigen = Eigen().eig(1)
        betaKcomm = float(2 * (self.xDamp / np.sqrt(eigen)[0]))
        ops.rayleigh(self.alphaM, self.betaKcurr, self.betaKinit, betaKcomm)  # https://openseespydoc.readthedocs.io/en/latest/src/reyleigh.html?highlight=rayleigh#rayleigh-command
        # https://openseespydoc.readthedocs.io/en/latest/src/exampleRotDSpectra.html?highlight=rayleigh#rotd-spectra-of-ground-motion
        return self





