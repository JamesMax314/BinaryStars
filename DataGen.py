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
    n_iter = 2000

    m_1 = 1e20
    m_2 = 1e20
    r_1_2 = 14.6e9
    dim = 1e27
    N = 50

    arr_bodies = two_body_init(m_1, m_2, r_1_2)
    arr_bodies = lfEngine.half_step(arr_bodies, dt)

    dmDen = 0.268*0.85e-26
    vol = dim**3
    dmMass = dmDen*vol
    dmPointMass = dmMass/N
    dm, loc, mass = dm_array_cube(dmPointMass, dmPointMass, N, dim)
    _arr_bodies = dm
    startTime = 1e9*365*24*3600

    # b = tree.basicRun(_arr_bodies, arrCent, uniDim, int(n_iter), dt)
    spacing = dim / 2
    # tree.PMTestForce(_arr_bodies, spacing, dim, 0)
    # b = tree.particleMesh(_arr_bodies, spacing, dim, n_iter, dt)
    b = tree.TreePareticleMesh(_arr_bodies, spacing, dim, dmDen/500000000000, n_iter, dt*1e250, startTime)


    colours = np.array([0.5, 0.5])
    colours = np.append(colours, np.array([0.5] * (len(_arr_bodies) - 2)))

    mation = anim.twoD(b, colours, dim, 1e-12, 10)
    mation.animate(9)
    mation.run("nTest1.mp4")
    plt.show()

    save(b, "nTest1.pkl")