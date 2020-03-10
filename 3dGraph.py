import treecode as tree
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import DataGen_V1
from mpl_toolkits.mplot3d import Axes3D
import anim
import numpy as np
import main
import pickle
import PS as ps
import PSpec
from matplotlib import rc

if __name__ == "__main__":
    file = ".//FinalData//WDM_1.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, gridSpacing, numIter, numParticles, t0, t100, numTrees, avrgM, avrgR) = pickle.load(infile)
    infile.close()

    point = 999
    scale = 1 / (DataGen_V1.MPc)
    xs = arrB[:, point, 0] * scale
    ys = arrB[:, point, 1] * scale
    zs = arrB[:, point, 2] * scale

    nGPts = 100
    dim = 100*50*DataGen_V1.MPc
    dVol = (dim / nGPts)**3
    dRho = DataGen_V1.MPc*mass / dVol
    grid = ps.genGrid(arrB[:, point, :], nGPts, dim)
    col = PSpec.den(arrB, grid, dim, point) * dRho

    mpl.style.use('default')

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scat = ax.scatter(xs*1e-3, ys*1e-3, zs*1e-3, s=0.05, c=np.sqrt(col), cmap="magma")
    cbaxes = fig.add_axes([0.3, .87, 0.5, 0.03])
    cbar = fig.colorbar(scat, cax=cbaxes, orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label("$\sqrt{\\rho}$ / $kg$ $Mpc^{-3}$")

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.w_xaxis.line.set_color("black")
    ax.w_yaxis.line.set_color("black")
    ax.w_zaxis.line.set_color("black")
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.zaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')  # only affects
    ax.tick_params(axis='y', colors='black')  # tick labels
    ax.tick_params(axis='z', colors='black')
    ax.xaxis._axinfo['tick']['color'] = 'k'
    ax.yaxis._axinfo['tick']['color'] = 'k'
    ax.zaxis._axinfo['tick']['color'] = 'k'

    ax.set_xlabel('$x$ / $Gpc$')
    ax.set_ylabel('$y$ / $Gpc$')
    ax.set_zlabel('$z$ / $Gpc$')

    # plt.savefig("..//Diagrams//CDM3D11.png", dpi=100, bbox_inches='tight')

    plt.show()