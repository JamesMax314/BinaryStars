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
    file = "WDM10.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, numIter, numParticles, t0, t100) = pickle.load(infile)
    infile.close()

    point = 100
    scale = 1 / (DataGen_V1.MPc)
    xs = arrB[:, point, 0] * scale
    ys = arrB[:, point, 1] * scale
    zs = arrB[:, point, 2] * scale

    nGPts = 100
    dim = 100*50*DataGen_V1.MPc
    grid = ps.genGrid(arrB[:, point, :], nGPts, dim)
    col = PSpec.den(arrB, grid, dim, point)

    mpl.style.use('default')

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scat = ax.scatter(xs, ys, zs, s=0.05, c=np.sqrt(col), cmap="plasma")
    fig.colorbar(scat)

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

    ax.set_xlabel('$x$ / $M$Pc')
    ax.set_ylabel('$y$ / $M$Pc')
    ax.set_zlabel('$z$ / $M$Pc')

    plt.show()