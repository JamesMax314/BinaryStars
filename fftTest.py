import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import treecode as tree
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl
import lfEngine
import numba
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

@numba.jit(nopython=True)
def diff(Frho):
    scale = np.empty_like(Frho)
    numPts = Frho.shape
    diff = np.empty(((numPts[0], numPts[1], numPts[2], 3)), dtype=np.complex64)
    for axis in range(3):
        for i in range(Frho.shape[0]):
            for j in range(Frho.shape[0]):
                for k in range(Frho.shape[0]):
                    kx = i - Frho.shape[0]
                    ky = i - Frho.shape[1]
                    kz = i - Frho.shape[2]
                    kVec = np.array([kx, ky, kz])
                    scale[i, j, k] = 1j * spacing * kVec[axis] * 4 * np.pi * G / (
                                pow(numPts[axis], 2) * numPts[0] * numPts[1] * numPts[2] *
                                (kx * kx / pow(numPts[0], 2) +
                                 ky * ky / pow(numPts[1], 2) +
                                 kz * kz / pow(numPts[2], 2)))
                    diff[:, :, :, axis] = scale * Frho
    return diff

if __name__ == "__main__":
    dt = 1e8  # 1e8 #1e2 #1e2
    n_iter = 10000

    m_1 = 1e20
    m_2 = 1e20
    r_1_2 = 14.6e9
    dim = 1e11
    N = 1000

    arr_bodies = two_body_init(m_1, m_2, r_1_2)
    arr_bodies = lfEngine.half_step(arr_bodies, dt)

    _arr_bodies = np.array([])
    offsets = [[-1e11, 0, 0], [0]*3, [1e11, 0, 0]]
    for body in arr_bodies:
        # print(body.r)
        _arr_bodies = np.append(_arr_bodies, tree.body(np.real(body.m), np.real(body.r), np.real(body.v), [0] * 3))


    dmDen = 4
    vol = dim**3
    dmMass = dmDen*vol
    dmPointMass = dmMass/N
    # dm, loc, mass = dm_array(dmPointMass, dmPointMass, N, dim)
    dm, loc, mass = dm_array_cube(dmPointMass, dmPointMass, N, dim)
    # _arr_bodies = np.append(_arr_bodies, dm)
    # _arr_bodies = dm
    # perimitor s= periodic(loc, mass, dim)
    # particle1 = tree.body(m_1, [7e9, 0, 0], [0, 0, 0], [0] * 3)
    # particle2 = tree.body(m_2, [-7e9, 0, 0], [-0, 0, 0], [0] * 3)
    # _arr_bodies = np.array([])
    # _arr_bodies = np.append(particle1, _arr_bodies)
    # _arr_bodies = np.append(particle2, _arr_bodies)
    # _arr_bodies = np.append(_arr_bodies, dm)

    arrCent = np.array([0, 0, 0])
    uniDim = np.array([1e15] * 3)
    uniDim = np.array([3e15] * 3)
    # b = tree.basicRun(_arr_bodies, arrCent, uniDim, int(n_iter), dt)
    spacing = dim / 20
    # b = tree.particleMesh(_arr_bodies, spacing, dim, n_iter, dt)
    pot = np.array(tree.PMTestPot(_arr_bodies, spacing, dim))
    # indx = np.argmax(pot)
    a = 0

    im = np.average(pot, axis=2)
    print(np.average(pot))
    # plt.imshow(im)
    # plt.show()

    Frho = np.fft.fftn(pot)
    Frho = np.fft.fftshift(Frho) / (Frho.shape[0] * Frho.shape[1] * Frho.shape[2])
    absRho = np.abs(Frho)
    im = np.average(absRho, axis=2)
    # plt.imshow(im)
    # plt.show()

    fx1 = np.array(tree.PMTestForce(_arr_bodies, spacing, dim, 1))
    fx = np.array(tree.PMTestForce(_arr_bodies, spacing, dim, 0))
    fy = np.array(tree.PMTestForce(_arr_bodies, spacing, dim, 1))
    fz = np.array(tree.PMTestForce(_arr_bodies, spacing, dim, 2))
    a = 0
    a = (fx[8, 10, 10])
    a += (fx[8, 10, 11])
    a += (fx[8, 11, 10])
    a += (fx[8, 10, 11])
    a += (fx[9, 10, 10])
    a += (fx[9, 10, 11])
    a += (fx[9, 11, 10])
    a += (fx[9, 10, 11])
    print(a/8)


    im = fx[:, :, 10]
    plt.imshow(im)
    plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # xx, yy, zz = np.mgrid[-dim / 2:dim / 2:spacing, -dim / 2:dim / 2:spacing, -dim / 2:dim / 2:spacing]
    # ax.scatter(xx, yy, zz, c=fx.flatten())
    # plt.show()
    # Diff
    # dif = diff(Frho)
    #
    # force = np.fft.ifftn(dif[:, :, :, 0])
    #
    # absF = np.real(force)
    # im = absF[:, :, int(force.shape[2]/2)] #np.sum(absF, axis=2)
    # plt.imshow(im)
    # plt.show()


