import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import treecode as tree
import pickle as pkl
import lfEngine
import numba
import anim
import pickle
import time

G = 6.674e-11
kb = 1.38064852e-23

def dm_array_cube(min, max, num, dim, centre=np.array([0] * 3), temp=2.73):
    locations = np.random.uniform(-dim / 2, dim / 2, [num, 3]) + centre * num
    masses = np.random.uniform(min, max, [num])

    velocities = np.zeros([num, 3])

    arrBods = np.array([])
    loc = np.empty([num, 3])
    mass = np.empty([num])
    for i in range(num):
        loc[i] = locations[i]
        mass[i] = masses[i]
        arrBods = np.append(arrBods, tree.body(masses[i], locations[i], velocities[i], [0] * 3))


    return arrBods, loc, mass


def save(bodies, file):
    arrB = np.empty([len(bodies), np.shape(np.array(bodies[0].pos))[0],
                     np.shape(np.array(bodies[0].pos))[1]])
    for i in range(len(bodies)):
        arrB[i, :, :] = np.array(bodies[i].pos)

    outfile = open(file, 'wb')
    pickle.dump(arrB, outfile)
    outfile.close()


if __name__ == "__main__":
    dt = 2e15 #2e15  # 1e8 #1e2 #1e2
    n_iter = 10000

    m_1 = 1e20
    m_2 = 1e20
    r_1_2 = 14.6e9
    dim = 1e27
    N = 20000

    dmDen = 0.268*0.85e-26
    vol = dim**3
    dmMass = dmDen*vol
    dmPointMass = dmMass/N
    dm, loc, mass = dm_array_cube(dmPointMass, dmPointMass, N, dim)
    _arr_bodies = dm

    spacing = dim / 50

    segment = 500

    # tpm = tree.initTPM(_arr_bodies, spacing, dim, dmDen*100, n_iter, dt)
    for i in range(int(n_iter/segment)):
        b = tree.runTPM(_arr_bodies, spacing, dim, dmDen*100, segment, dt)
        file = "../Saves/" + str(i) + ".pkl"
        save(b, file)
        tree.resetTPM(_arr_bodies)