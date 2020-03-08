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
    file = "WDM11.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, numIter, numParticles, t0, t100, numTrees, avrgM, avrgR) = pickle.load(infile)
    infile.close()
    file = "CDM11.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, numIter, numParticles, t0, t100, numTrees1, avrgM1, avrgR1) = pickle.load(infile)
    infile.close()

    point = 999
    dt = (10 - 100) / numIter
    scale = 1 / (DataGen_V1.MPc)
    xs = arrB[:, point, 0] * scale
    ys = arrB[:, point, 1] * scale
    zs = arrB[:, point, 2] * scale

    mpl.style.use('default')

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    times = np.linspace(t100, t0, 1000)
    ax = plt.subplot(111)
    ax.plot(times/(1e9*3600*24*365), numTrees, "--", c="k")
    ax.plot(times/(1e9*3600*24*365), numTrees1, "-", c="k")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("$t / Gyr$")
    ax.set_ylabel("Number of high density regions")
    plt.show()