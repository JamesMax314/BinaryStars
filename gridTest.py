import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import treecode as tree
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
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


def density(x, y, z, array):
    array = np.log(array, out=np.zeros_like(array), where=(array != 0))
    ma = np.max(array)
    mi = np.min(array)
    diff = ma-mi
    scale = 1/diff
    a = array[x, y, z]
    den = scale*(a-[mi]*len(a))
    out = cm.Reds(den)
    out[:, 3] = out[:, 3]*(den)
    return out

def qColour(x, y, z, size):
    norm = np.empty(size**3)
    for i in range(0, size):
        for j in range(0, size):
            for k in range(0, size):
                norm[i*size**2 + j*size + k] = np.sqrt(x[i, j, k]**2 + y[i, j, k]**2 + z[i, j, k]**2)

    i = 0
    while i < len(norm):
        if norm[i] < 2e-9:
            norm[i] = 0
        i += 1
    array = np.log(norm, out=np.zeros_like(norm), where=(norm != 0))
    # array = norm
    ma = np.max(array)
    mi = np.min(array)
    diff = ma-mi
    scale = 1/diff
    den = scale*(array-[mi]*len(array))
    out = cm.Reds(den)
    out[:, 3] = out[:, 3]*(den)

    i = 0
    while i < len(out):
        if norm[i] == 0:
            out[i, 3] = 0
        i += 1
    return out

if __name__ == "__main__":
    dt = 1e-5 #1e-11
    n_iter = 10000

    m_1 = 1e20
    m_2 = 1e20
    r_1_2 = 30e9
    dim = 1e11
    N = 30

    arr_bodies = two_body_init(m_1, m_2, r_1_2)
    arr_bodies = lfEngine.half_step(arr_bodies, dt)

    """ Generating bodies """
    _arr_bodies = np.array([])
    for body in arr_bodies:
        # print(body.r)
        _arr_bodies = np.append(_arr_bodies, tree.body(np.real(body.m), np.real(body.r), np.real(body.v), [0] * 3))
    # _arr_bodies = np.append(_arr_bodies, tree.body(np.real(m_1), np.real([0, 0, 0]), np.real([0, 0, 0]), [0] * 3))
    # for body in arr_bodies:
    #     _arr_bodies = np.append(_arr_bodies, tree.body(np.real(body.m), np.real(body.r), np.real(body.v), [0] * 3))

    dmDen = 4
    vol = 4/3*np.pi*(dim/2)**3
    dmMass = dmDen*vol
    dmPointMass = dmMass/N

    # dm, loc, mass = dm_array(dmPointMass, dmPointMass, N, dim)
    dm, loc, mass = dm_array_cube(dmPointMass, dmPointMass, N, dim)
    # _arr_bodies = np.append(_arr_bodies, dm)
    # _arr_bodies = dm
    # perimitor = periodic(loc, mass, dim)
    particle1 = tree.body(m_1, [0, 0, 0], [0, 0, 0], [0] * 3)
    # particle2 = tree.body(m_2, [-7e9, 0, 0], [-0, 0, 0], [0] * 3)
    _arr_bodies = np.array([])
    _arr_bodies = np.append(particle1, _arr_bodies)
    # _arr_bodies = np.append(particle2, _arr_bodies)
    # _arr_bodies = np.append(_arr_bodies, dm)

    arrCent = np.array([0, 0, 0])
    uniDim = np.array([1e15] * 3)
    # b = tree.basicRun(_arr_bodies, arrCent, uniDim, int(n_iter), dt)
    numPts = 10
    spacing = dim / numPts
    b = tree.PMTest(_arr_bodies, spacing, dim, dt)
    b1 = np.array(tree.PMTest1(_arr_bodies, spacing, dim, dt))
    acc = np.empty([len(b1), 3])
    for i in range(len(b1)):
        s = np.array(b1[i].acc)[-1]
        acc[i, :] = np.array(b1[i].acc)[-1]

    """ Cast to numpy array """
    # array = np.array(b, copy=False)
    Fx = np.array(b.getF(0))
    Fy = np.array(b.getF(1))
    Fz = np.array(b.getF(2))

    # xx, yy, zz = np.meshgrid(xs, ys, zs, sparse=True)

    xx, yy, zz = np.mgrid[-dim/2:dim/2:spacing, -dim/2:dim/2:spacing, -dim/2:dim/2:spacing]
    ix, iy, iz = np.mgrid[0:numPts:1, 0:numPts:1, 0:numPts:1]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    # ax.scatter(xx, yy, zz, color=density(ix.flatten(), iy.flatten(), iz.flatten(), array))

    colours = qColour(Fx, Fy, Fz, numPts)
    c1 = np.repeat(colours, 2, axis=0)
    colours = np.concatenate((colours, c1))

    q = ax.quiver(xx, yy, zz, Fx, Fy, Fz, color=colours, length=5e9, normalize=True)

    bods = np.empty([len(_arr_bodies), 3])
    for i in range(len(_arr_bodies)):
        bods[i] = np.array(_arr_bodies[i].pos[0])
    ax.scatter(bods[:, 0], bods[:, 1], bods[:, 2], color=[0, 0, 0, 1])
    interp = ax.quiver(bods[:, 0], bods[:, 1], bods[:, 2], acc[:, 0], acc[:, 1], acc[:, 2], length=5e9, normalize=True)


    plt.show()

