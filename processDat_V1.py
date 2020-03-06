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
    file = "CDM10.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, numIter, numParticles, t0, t100) = pickle.load(infile)
    infile.close()
    file = "WDM10.pkl"
    infile = open(file, 'rb')
    (arrB2, mass, uniDim, numIter, numParticles, t0, t100) = pickle.load(infile)
    infile.close()

    point = 700
    scale = 1/(DataGen_V1.MPc)
    xs = arrB2[:, point, 0] * scale
    ys = arrB2[:, point, 1] * scale
    zs = arrB2[:, point, 2] * scale

    mpl.style.use('default')

    data = np.random.uniform(-100 / 2, 100 / 2, [100000, 3])
    spec, freq = ps.computePS(data, 100, 100)
    spec2, freq = ps.computePS(arrB[:, 999, :], 200*10*DataGen_V1.MPc, 100)
    spec1, freq = ps.computePS(arrB2[:, 999, :], 200*10*DataGen_V1.MPc, 100)
    plt.loglog(spec2)
    plt.loglog(spec1)
    plt.show()

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(xs, ys, zs, s=0.1, c="k")
    #
    # # make the panes transparent
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # # make the grid lines transparent
    # ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.w_xaxis.line.set_color("black")
    # ax.w_yaxis.line.set_color("black")
    # ax.w_zaxis.line.set_color("black")
    # ax.xaxis.label.set_color('black')
    # ax.yaxis.label.set_color('black')
    # ax.zaxis.label.set_color('black')
    # ax.tick_params(axis='x', colors='black')  # only affects
    # ax.tick_params(axis='y', colors='black')  # tick labels
    # ax.tick_params(axis='z', colors='black')
    # ax.xaxis._axinfo['tick']['color'] = 'k'
    # ax.yaxis._axinfo['tick']['color'] = 'k'
    # ax.zaxis._axinfo['tick']['color'] = 'k'
    #
    # ax.set_xlabel('$x$ / $M$Pc')
    # ax.set_ylabel('$y$ / $M$Pc')
    # ax.set_zlabel('$z$ / $M$Pc')
    #
    # plt.show()

    # plt.scatter(arrB[:, point, 0] / (3.086e16 * 1e9), arrB[:, point, 1] / (3.086e16 * 1e9), s=0.1)
    # point = 1
    # plt.scatter(arrB[:, point, 0] / (3.086e16 * 1e9), arrB[:, point, 1] / (3.086e16 * 1e9), s=0.4)
    # # plt.colorbar()
    # plt.show()
    #
    # mesh = p.toMesh(arrB, 100, 1e24, point)
    # flat = np.average(mesh, axis=2)
    # flat = np.average(flat, axis=1)
    # av = np.average(mesh)
    # delta = (flat-av)/av
    # P = PS(delta)
    # point = 250
    # mesh = p.toMesh(arrB, 100, 1e24, point)
    # flat = np.average(mesh, axis=2)
    # flat = np.average(flat, axis=1)
    # av = np.average(mesh)
    # delta = (mesh - av) / av
    # P1 = PS(flat)
    # plt.plot(np.abs(P), label="0")
    # plt.plot(np.abs(P1), label="250")
    # plt.legend()
    # plt.show()

    # x = np.linspace(0, 10000, 1001)
    # noise = g(x)
    # noise = noise-np.random.uniform(-1, 1, 1001)
    # plt.plot(noise)
    # plt.show()
    # f = np.fft.ifft(noise)
    # f = np.fft.fftshift(f)
    # f1, Pxx_den = signal.periodogram(noise, 10)
    # plt.plot(f1, Pxx_den)
    # plt.show()

    # col = p.den(arrB, mesh, 1e27, point)
    # plt.scatter(arrB[:, point, 0] / (3.086e16 * 1e9), arrB
    # [:, point, 1] / (3.086e16 * 1e9), s=1, c=col, cmap="plasma")
    # plt.colorbar(label="Density / particles number per $1.728$ $G$Pc$^3$")
    # plt.xlabel("distance / $G$Pc")
    # plt.ylabel("distance / $G$Pc")
    # plt.savefig("../Diagrams/colUni1.png", bbox="tight", dpi=400)
    # plt.show()

