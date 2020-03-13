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
    file = ".//FinalData//CDM_1.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, gridSpacing, numIter, numParticles, t0, t100, numTrees, avrgM, avrgR) = pickle.load(infile)
    infile.close()

    point = 999
    scale = 1 / (DataGen_V1.MPc)
    xs = arrB[:, point, 0] * scale
    ys = arrB[:, point, 1] * scale
    zs = arrB[:, point, 2] * scale

    nGPts = 100
    dim = 100*50*DataGen_V1.MPc  # in m
    dVol = (dim / nGPts)**3
    dRho = mass / dVol
    grid = ps.genGrid(arrB[:, point, :], nGPts, dim)
    col = PSpec.den(arrB, grid, dim, point) * dRho

    mpl.style.use('default')

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    # matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size': 12,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    # width = 3.28162  # in inches
    # fig = plt.figure(figsize=(width, width * 0.85))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scat = ax.scatter(xs*1e-3, ys*1e-3, zs*1e-3, s=1, c=np.sqrt(col)*10**13, linewidth=0, cmap="magma")
    cbaxes = fig.add_axes([0.3, .82, 0.5, 0.03])
    cbar = fig.colorbar(scat, cax=cbaxes, orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label("$\sqrt{\\rho}$ / $10^{-13}$$k$g$^{1/2}$m$^{-3/2}$")

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

    ax.set_xlabel('$x$ / $G$pc')
    ax.set_ylabel('$y$ / $G$pc')
    ax.set_zlabel('$z$ / $G$pc')

    plt.savefig("..//Diagrams//CDM3D11.png", dpi=300)

    # plt.show()