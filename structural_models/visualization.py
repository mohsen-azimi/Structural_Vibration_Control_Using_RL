"""
Refrence:
"""
##########################################################################################################################################################################
import openseespy.opensees as ops
# import openseespy.postprocessing.ops_vis as opsv

import numpy as np
import matplotlib.pyplot as plt


class Visualization:
    """
      Description:
      """

    def __init__(self):
        # Description
        pass

    @staticmethod
    def draw2D():
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

        plt.show()





