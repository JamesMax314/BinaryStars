import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import treecode as tree
import lfEngine
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


def dm_array(min, max, num, dim, centre=np.array([0] * 3), temp=2.73):
    locations = np.random.uniform(-dim / 2, dim / 2, [num, 3]) + centre * num
    masses = np.random.uniform(min, max, [num])
    velocities = np.random.uniform(0, 1, [num, 3]) * np.transpose(np.array([np.sqrt(kb * temp / masses)] * 3))

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


if __name__ == "__main__":
    dt = 10 * 24 * 3600
    n_iter = 360 * 24 * 3600 / dt

    m_1 = 2e25
    m_2 = 1.6e25
    r_1_2 = 14.6e9
    dim = 1e11
    N = 1000

    arr_bodies = two_body_init(m_1, m_2, r_1_2)
    arr_bodies = lfEngine.half_step(arr_bodies, dt)

    _arr_bodies = np.array([])
    for body in arr_bodies:
        # print(body.r)
        _arr_bodies = np.append(_arr_bodies, tree.body(np.real(body.m), np.real(body.r), np.real(body.v), [0] * 3))

    dmDen = 6e-22
    vol = dim**2
    dmMass = dmDen*vol
    dmPointMass = dmMass/N
    dm, loc, mass = dm_array(dmPointMass, dmPointMass, N, dim)
    # _arr_bodies = np.append(_arr_bodies, dm)
    _arr_bodies = dm
    # perimitor = periodic(loc, mass, dim)
    particle1 = tree.body(m_1, [0, 0, 0], [1000, 0, 0], [0] * 3)
    _arr_bodies = np.append(particle1, _arr_bodies)

    arrCent = np.array([0, 0, 0])
    uniDim = np.array([1e15] * 3)
    # b = tree.fixedBoundary(_arr_bodies, perimitor, arrCent, uniDim, int(n_iter), dt)
    # result = lfEngine.execute(arr_bodies, dt, int(n_iter))
    b = tree.basicRun(_arr_bodies, arrCent, uniDim, int(n_iter), dt)
    # spacing = 1e11 / 100
    # b = tree.particleMesh(_arr_bodies, spacing, 1e11, 1000, dt)

    # print("ok")

    colours = np.array([0, 0.1])
    colours = np.append(colours, np.array([0.5] * (len(_arr_bodies)-2)))

    mation = anim.twoD(b, colours, dim, 1e-12, 10)
    # mation.animate(10)
    mation.run("test.mp4")
    plt.show()


    # labels = ["1", "2", "DM"]
    # colour = ["orange", "purple", "grey"]
    # for i in range(2):
    #     # plt.plot(result["rs"][i, :, 1]/1e12, result["rs"][i, :, 0]/1e12, label=labels[i], color=colour[i])
    #     plt.plot(np.array(b[i].pos)[:, 0] / 1e12, np.array(b[i].pos)[:, 1] / 1e12, label=labels[i], color=colour[i])
    #     plt.scatter(np.array(b[i].pos)[-1, 0] / 1e12, np.array(b[i].pos)[-1, 1] / 1e12, color=colour[i])
    # for i in range(2, len(_arr_bodies)):
    #     plt.plot(np.array(b[i].pos)[:, 0] / 1e12, np.array(b[i].pos)[:, 1] / 1e12, label=labels[2], color=colour[2])
    #     plt.scatter(np.array(b[i].pos)[-1, 0] / 1e12, np.array(b[i].pos)[-1, 1] / 1e12, color=colour[2])
    # # plt.savefig("bin.png")
    # plt.axvline(x=1e-12*dim/2)
    # plt.axvline(x=-1e-12*dim/2)
    # plt.axhline(y=1e-12*dim/2)
    # plt.axhline(y=-1e-12*dim/2)
    # plt.show()
