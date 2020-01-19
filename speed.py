import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import treecode as tree
import pickle as pkl
import lfEngine
import numba
import anim
import time

G = 6.674e-11
kb = 1.38064852e-23


def two_body_init(m_1, m_2, r):
    global G
    com = m_2 * r / (m_1 + m_2)
    r_1 = -np.abs(m_2 * r / (m_1 + m_2))
    r_2 = np.abs(m_1 * r / (m_1 + m_2))
    v_1 = np.sqrt((G * m_2 * np.abs(r_1)) / r ** 2)
    v_2 = np.sqrt((G * m_1 * np.abs(r_2)) / r ** 2)
    _arr_bodies = np.array([])
    _arr_bodies = np.append(_arr_bodies, lfEngine.Body([r_1, 0, 0], [0, v_1, 0], m_1))
    _arr_bodies = np.append(_arr_bodies, lfEngine.Body([r_2, 0, 0], [0, -v_2, 0], m_2))
    return _arr_bodies


def polarCartesian(rad, theta, phi):
    rad = rad**(1/3)
    return np.array([rad*np.sin(theta)*np.cos(phi), rad*np.sin(theta)*np.sin(phi), rad*np.cos(theta)])

def dm_array(min, max, num, dim, centre=np.array([0] * 3), temp=2.73):
    phis = np.random.uniform(0, 2*np.pi, [num])
    thetas = np.random.uniform(0, np.pi, [num])
    rads = np.random.uniform(0, np.pi*(dim/2)**3, [num]) # accounts for the uniform distribution
    # over volume i.e. uniform on r^3
    locations = polarCartesian(rads, thetas, phis)
    locations = np.transpose(locations)

    # locations = np.random.uniform(-dim / 2, dim / 2, [num, 3]) + centre * num
    masses = np.random.uniform(min, max, [num])
    velocities = np.random.uniform(-1, 1, [num, 3]) #* np.transpose(np.array([np.sqrt(kb * temp / masses)] * 3))

    # energy = 0
    # for i in range(len(masses)):
    #     f = 0
    #     for j in range(len(masses)):
    #         if (i != j):
    #             f += G*masses[i]*masses[j]*np.abs(locations[i] - locations[j])**(-3) * (-locations[i] + locations[j])
    #     energy += -1/2 * np.dot(f, locations[i])
    #
    # for i in range(len(masses)):
    #     scaling = np.sqrt(energy*2 / masses[i])
    #     velocities[i] = scaling * velocities[i] / np.abs(velocities[i])

    velocities = np.zeros([num, 3])

    arrBods = np.array([])
    loc = np.empty([num, 3])
    mass = np.empty([num])
    for i in range(num):
        loc[i] = locations[i]
        mass[i] = masses[i]
        arrBods = np.append(arrBods, tree.body(masses[i], locations[i], velocities[i], [0] * 3))


    return arrBods, loc, mass

def dm_array_py(min, max, num, dim, centre=np.array([0] * 3), temp=2.73):
    phis = np.random.uniform(0, 2*np.pi, [num])
    thetas = np.random.uniform(0, np.pi, [num])
    rads = np.random.uniform(0, np.pi*(dim/2)**3, [num]) # accounts for the uniform distribution
    # over volume i.e. uniform on r^3
    locations = polarCartesian(rads, thetas, phis)
    locations = np.transpose(locations)

    masses = np.random.uniform(min, max, [num])

    velocities = np.zeros([num, 3])

    arrBods = np.array([])
    loc = np.empty([num, 3])
    mass = np.empty([num])
    for i in range(num):
        loc[i] = locations[i]
        mass[i] = masses[i]
        arrBods = np.append(arrBods, lfEngine.Body(locations[i], velocities[i], masses[i]))


    return arrBods

def periodic(loc, mass, dim):
    arrBods = np.array([])
    for x in range(-2, 3):
        for y in range(-2, 3):
            for z in range(-2, 3):
                displacement = dim * np.array([x, y, x]) / 2
                if np.abs(displacement[0]) >= dim / 2 and \
                        np.abs(displacement[1]) >= dim / 2 and \
                        np.abs(displacement[2]) >= dim / 2:
                    for i in range(len(mass)):
                            arrBods = np.append(arrBods,
                                                tree.body(mass[i], loc[i] + displacement, [0] * 3, [0] * 3))

    return arrBods


if __name__ == "__main__":
    dt = 1e-3
    n_iter = 1000

    m_1 = 1e20
    m_2 = 1e20
    r_1_2 = 14.6e9
    dim = 1e11
    N = 1000

    dmDen = 4
    vol = 4/3*np.pi*(dim/2)**3
    dmMass = dmDen*vol
    dmPointMass = dmMass/N

    timesPy = np.array([])
    timesC = np.array([])
    ns = np.array([])

    for i in range(2, 100):
        ns = np.append(ns, i)
        #C++
        dm, loc, mass = dm_array(dmPointMass, dmPointMass, i, dim)
        _arr_bodies = dm

        arrCent = np.array([0, 0, 0])
        uniDim = np.array([1e15] * 3)
        start = time.time()
        b = tree.basicRun(_arr_bodies, arrCent, uniDim, int(n_iter), dt)
        end = time.time()
        timesC = np.append(timesC, end-start)

        #Python
        # arr_bodies = dm_array_py(dmPointMass, dmPointMass, i, dim)
        # start = time.time()
        # result = lfEngine.execute(arr_bodies, dt, int(n_iter))
        # end = time.time()
        # timesPy = np.append(timesPy, end-start)

    filename = "speedDatC"
    outfile = open(filename, 'wb')
    pkl.dump((ns, timesPy/n_iter, timesC/n_iter), outfile)
    outfile.close()

    plt.plot(ns, timesPy/n_iter)
    plt.show()

