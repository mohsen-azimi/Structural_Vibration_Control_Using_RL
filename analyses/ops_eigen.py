'''
Refrence:
https://opensees.berkeley.edu/wiki/index.php/Dynamic_Analyses_of_1-Story_Moment_Frame_with_Viscous_Dampers
'''
##########################################################################################################################################################################
import openseespy.opensees as ops
import numpy as np


class Eigen(object):
    """
      Description:
      Control Node ID = # ()
      """
    def __init__(self):
        self.n = 3
        self.eigen = None

    def eig(self, n):
        self.n = n
        ops.wipeAnalysis()  # clear before any analysis
        eigen = ops.eigen('-fullGenLapack', self.n)
        print(f"Eigens = {eigen}")
        return eigen





