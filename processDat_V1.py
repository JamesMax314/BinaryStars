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
import DataGen_V2

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
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    width = 3.28162  # in inches
    fig = plt.figure(figsize=(width, width * 0.85))

    ax = plt.subplot(111)
    lower = DataGen_V2.a(DataGen_V2.t100) * 2 * np.pi / 10
    upper = DataGen_V2.a(DataGen_V2.t100) * 2 * np.pi / (10 / 100)
    kmin = 1e-4
    kmax = 100
    ax.axvline(x=lower, ymin=kmin, ymax=kmax, linestyle="-.", c="k", lw=0.9)
    ax.axvline(x=upper, ymin=kmin, ymax=kmax, linestyle="-.", c="k", lw=0.9)
    ax.set_xlim([1e-3, 0.2])
    # ax.set_ylim([8, 1.2e4])
    ax.set_xlabel("$k / M$pc$^{-1}$")
    ax.set_ylabel("$P / M$pc$^{3}$")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    data = np.random.uniform(-100 / 2, 100 / 2, [100000, 3])
    spec, freq = ps.computePS(data, 100, 100)
    spec2, freq = ps.computePS(arrB[:, 999, :], 200*10*DataGen_V1.MPc, 100)
    spec1, freq = ps.computePS(arrB2[:, 999, :], 200*10*DataGen_V1.MPc, 100)
    freq = np.linspace(lower, upper, 100)
    ax.loglog(freq[0:80], spec2[0:80]/(100**3), "--", c="k", lw=1)
    ax.loglog(freq[0:80], spec1[0:80]/(100**3), "-", c="k", lw=1)
    plt.savefig("..//Diagrams//FinalTPS.pgf", bbox_inches='tight', pad_inches=.1)
    # plt.show()


