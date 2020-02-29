import treecode as tree
import matplotlib.pyplot as plt
import anim
import numpy as np
import pickle
import os
import PSpec

# constants
pi = np.pi
H0 = 67
MPc = 3.09e22  # MPc = 3.09e22 m

# Parameters
rho100 = 2.78e-21 # initial DM density in real units / kgm^-3
t100 = 1.88e13  # start of sim at z=100 / s
t0 = 4.354e17  # now time / s
N = int(20000**(1/3))  # Number of particles
uniDim = 100*MPc  # Universe length in m
numPts = 50  # grid points per dimension
gridSpacing = uniDim / numPts  # PM grid spacing
numIter = 500  # Number of iterations
dt = (t0 - t100)/numIter  # time step

if __name__ == "__main__":
    print("Initialising over-density perturbations...")
    pts, masses = PSpec.genMasses(N, rho100, uniDim)
    print("Initialising bodies...")
    _arr_bodies = np.array([])
    for i in range(N**3):
        # print(body.r)
        _arr_bodies = np.append(_arr_bodies,
                                tree.body(masses[i], pts[i], [0] * 3, [0] * 3))

    print("Running simulation...")
    b = tree.TreePareticleMesh(_arr_bodies, gridSpacing, uniDim, rho100 * 100, numIter, dt, t100)

    colours = np.array([0.5, 0.5])
    colours = np.append(colours, np.array([0.5] * (len(_arr_bodies) - 2)))

    # mation = anim.twoD(b, colours, uniDim, 1e-12, 10)
    # mation.animate(9)
    # mation.run("nTest1.mp4")
    # plt.show()

    arrB = np.empty((len(b), np.shape(np.array(b[0].pos))[0],
                              np.shape(np.array(b[0].pos))[1]))
    for i in range(len(b)):
        arrB[i, :, :] = np.array(b[i].pos)

    point = 10
    plt.scatter(arrB[:, point, 0] / (3.086e16 * 1e9), arrB[:, point, 1] / (3.086e16 * 1e9), s=1)
    plt.show()

    file = "CDM1.pkl"
    outfile = open(file, 'wb')
    pickle.dump(arrB, outfile)
    outfile.close()
