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
from matplotlib import rc

from scipy import signal
import ProcessDat as p
import os
import PSpec


def PS(mat):
    fMat = np.fft.fftn(mat)
    fMat2 = np.abs(fMat)**2
    out = np.empty([fMat2.shape[0]])
    for i in range(fMat2.shape[0]):
        out[i] = np.abs(fMat2[i])
    return out

def g(x):
    out = np.sin(x)
    return out

if __name__ == "__main__":
    file = ".//FinalData//CDM_1.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, gridSpacing, numIter, numParticles, t0, t100, numTrees1, avrgM1, avrgR1) = pickle.load(infile)
    infile.close()
    file = ".//FinalData//WDM_1.pkl"
    infile = open(file, 'rb')
    (arrB2, mass, uniDim, gridSpacing, numIter, numParticles, t0, t100, numTrees1, avrgM1, avrgR1) = pickle.load(infile)
    infile.close()

    point = 700
    scale = 1/(DataGen_V1.MPc)
    xs = arrB2[:, point, 0] * scale
    ys = arrB2[:, point, 1] * scale
    zs = arrB2[:, point, 2] * scale

    mpl.style.use('default')

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    ax = plt.subplot(111)
    lower = 2 * np.pi / 10
    upper = 2 * np.pi / (10 / 100)
    kmin = 1e-4
    kmax = 100
    ax.axvline(x=lower, ymin=kmin, ymax=kmax, linestyle="-.", c="k", lw=0.9)
    ax.axvline(x=upper, ymin=kmin, ymax=kmax, linestyle="-.", c="k", lw=0.9)
    ax.set_xlim([1e-1, kmax])
    # ax.set_ylim([1e-18, 1e-10])
    ax.set_xlabel("$k / Mpc^{-1}$")
    ax.set_ylabel("$P / Mpc^{3}$")

    data = np.random.uniform(-100 / 2, 100 / 2, [100000, 3])
    spec, freq = ps.computePS(data, 100, 100)
    spec2, freq = ps.computePS(arrB[:, 999, :], 200*10*DataGen_V1.MPc, 50)
    spec1, freq = ps.computePS(arrB2[:, 999, :], 200*10*DataGen_V1.MPc, 50)
    freq = np.linspace(lower, upper, 50)
    ax.loglog(freq, spec2, "--", c="k")
    ax.loglog(freq, spec1, "-", c="k")
    plt.savefig("..//Diagrams//FinalTPS.png", dpi=300, bbox_inches='tight')
    # plt.show()



