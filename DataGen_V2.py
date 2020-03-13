import treecode as tree
import matplotlib.pyplot as plt
# import anim
import numpy as np
import pickle
import os
import PSpec

# constants
pi = np.pi
H0 = 67
MPc = 3.09e22  # MPc = 3.09e22 m

def get_time(z):
    t = 2*2.2e-18 ** (-1) / (1 + (1+z)**2)
    return t

def a(t):
    return aConst*t**(2/3)

def ad(t):
    return aConst*2/3*t**(-1/3)

def softening(R, N):
    softening = 1.1 * N**(-0.28) * R


# Parameters
Rf = [0, 0.1, 0.2]  # 0.2 is too low mass based on Lyman alpha forrest
aConst = 1.066e-12  # scale factor coefficient
rho100 = 2.78e-21  # initial DM density in real units / kgm^-3
t100 = get_time(100)  # 1.88e13  # start of sim at z=100 / s
t0 = get_time(0)  # now time / s
numInitParticles = int((100**3)**(1/3))  # Number of initialisation particles (grid points)
numParticles = 50000  # Number of particles in the simulation
uniDim = 10*MPc  # Universe length in m
numPts = 50  # grid points per dimension for PM method
gridSpacing = uniDim / numPts  # PM grid spacing
numIter = 1000  # Number of iterations
dt = (t0 - t100)/numIter  # time step
mass = rho100 * uniDim**3 / numParticles  # Particle mass

if __name__ == "__main__":
    print("Loading Bodies")
    file = ".//InitBods//CDM.pkl"
    infile = open(file, 'rb')
    (pts, mass) = pickle.load(infile)
    infile.close()

    print("Initializing bodies...")
    _arr_bodies = np.array([])
    for i in range(numParticles):
        # print(body.r)
        _arr_bodies = np.append(_arr_bodies,
                                tree.body(mass, pts[i], [0] * 3, [0] * 3))

    print("Running simulation...")
    b = tree.TreePareticleMesh(_arr_bodies, gridSpacing / a(t100), uniDim / a(t100), rho100 * 50 * (a(t100)**3), numIter, dt, t100)
    # b = tree.particleMesh(_arr_bodies, gridSpacing / a(t100), uniDim / a(t100), numIter, dt)

    numTrees = np.array(b[0].numTrees)
    avrgR = np.array(b[0].avrgR)
    avrgM = np.array(b[0].avrgM)

    colours = np.array([0.5, 0.5])
    colours = np.append(colours, np.array([0.5] * (len(_arr_bodies) - 2)))

    # mation = anim.twoD(b, colours, uniDim, 1e-12, 10)
    # mation.animate(9)
    # mation.run("nTest1.mp4")
    # plt.show()

    arrB = np.empty((len(b), np.shape(np.array(b[0].pos))[0],
                              np.shape(np.array(b[0].pos))[1]))
    for i in range(len(b)):
        arrB[i, :, :] = np.array(b[i].pos) * a(t0)

    # point = 0
    # plt.scatter(arrB[:, point, 0] / (3.086e16 * 1e9), arrB[:, point, 1] / (3.086e16 * 1e9), s=1)
    # plt.colorbar()
    # plt.show()

    file = ".//FinalData//CDM_1_meta.pkl"
    outfile = open(file, 'wb')
    pickle.dump((arrB, mass, uniDim, gridSpacing, numIter, numParticles, t0, t100, numTrees, avrgM, avrgR), outfile)
    outfile.close()
