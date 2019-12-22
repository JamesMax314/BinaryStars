import numpy as np
import matplotlib.pyplot as plt
import treecode as tree
import lfEngine
import time

G = 6.674e-11
kb = 1.38064852e-23

def two_body_init(m_1, m_2, r):
    global G
    com = m_2*r / (m_1 + m_2)
    r_1 = np.abs(-m_2*r / (m_1 + m_2))
    r_2 = np.abs(m_1*r / (m_1 + m_2))
    v_1 = np.sqrt((G * m_2 * r_1) / r**2)
    v_2 = np.sqrt((G * m_1 * r_2) / r**2)
    _arr_bodies = np.array([])
    _arr_bodies = np.append(_arr_bodies, lfEngine.Body([r_1, 0, 0], [0, v_1, 0], m_1))
    _arr_bodies = np.append(_arr_bodies, lfEngine.Body([r_2, 0, 0], [0, -v_2, 0], m_2))
    return _arr_bodies

def dm_array(min, max, num, dim, centre=np.array([0]*3), temp=2.73):
    locations = np.random.uniform(-dim/2, dim/2, [num, 3]) + centre*num
    masses = np.random.uniform(min, max, [num])
    velocities = np.random.uniform(0, 1, [num, 3]) * np.transpose(np.array([np.sqrt(kb*temp/masses)]*3))

    arrBods = np.array([])
    for i in range(num):
        arrBods = np.append(arrBods, tree.body(masses[i], locations[i], velocities[i], [0]*3))
    return arrBods

if __name__ == "__main__":
    dt = 10 * 24 * 3600
    n_iter = 110 * 360 * 24 * 3600 / dt

    m_1 = 2e25
    m_2 = 1.6e25
    r_1_2 = 149.6e9

    arr_bodies = two_body_init(m_1, m_2, r_1_2)
    arr_bodies = lfEngine.half_step(arr_bodies, dt)

    _arr_bodies = np.array([])
    for body in arr_bodies:
        _arr_bodies = np.append(_arr_bodies, tree.body(np.real(body.m), np.real(body.r), np.real(body.v), [0]*3))

    _arr_bodies = np.append(_arr_bodies, dm_array(10e20, 10e21, 100, 1e10))

    arrCent = np.array([0, 0, 0])
    uniDim = np.array([1e15] * 3)
    # result = lfEngine.execute(arr_bodies, dt, int(n_iter))
    b = tree.basicRun(_arr_bodies, arrCent, uniDim, int(n_iter), dt)

    # print("ok")

    labels = ["1", "2", "DM"]
    colour = ["orange", "purple", "grey"]
    for i in range(2):
        # plt.plot(result["rs"][i, :, 1]/1e12, result["rs"][i, :, 0]/1e12, label=labels[i], color=colour[i])
        plt.plot(np.array(b[i].pos)[:, 0] / 1e12, np.array(b[i].pos)[:, 1] / 1e12, label=labels[i], color=colour[i])
        plt.scatter(np.array(b[i].pos)[-1, 0] / 1e12, np.array(b[i].pos)[-1, 1] / 1e12, color=colour[i])
    for i in range(2, len(_arr_bodies)):
        plt.plot(np.array(b[i].pos)[:, 0] / 1e12, np.array(b[i].pos)[:, 1] / 1e12, label=labels[2], color=colour[2])
        plt.scatter(np.array(b[i].pos)[-1, 0] / 1e12, np.array(b[i].pos)[-1, 1] / 1e12, color=colour[2])
    # plt.savefig("bin.png")
    plt.show()

# if __name__ == "__main__":
#     n = 100
#     uniDim = np.array([10, 10, 10])
#     velRan = np.array([1, 1, 1])
#     arrCent = np.array([0, 0, 0])
#     masDim = 1e2
#     arrBods = np.array([])
#
#     dt = 100
#     numSteps = 10000
#
#     # Generate n bodies
#     np.random.seed(0)
#     randMas = np.ones(n)*masDim # np.random.random(n)*masDim
#     np.random.seed(1)
#     randPos = np.random.random([n, 3])*uniDim/2
#     np.random.seed(2)
#     randVel = np.random.random([n, 3])*velRan*0
#     for i in range(n):
#         arrBods = np.append(arrBods, tree.body(10, randPos[i], randVel[i], [0, 0, 0]))
#
#     # arrBods = [tree.body(100, arrCent, [0, 0, 0], [0, 0, 0]),
#     #            tree.body(100, [10, 0, 0], [0, 0, 0], [0, 0, 0])]
#     start = time.time()
#     b = tree.basicRun(arrBods, arrCent, uniDim*10, numSteps, dt)
#     end = time.time()
#     # print(end - start)
#     pos = np.array(b[0].getPos())
#     print(pos[:, 0])
#     times = np.array(range(0, int(numSteps*dt), int(dt)))
#     for i in range(0, n):
#         plt.plot(np.array(b[i].getPos())[:, 0], np.array(b[i].getPos())[:, 1])
#         plt.scatter(np.array(b[i].getPos())[-1, 0], np.array(b[i].getPos())[-1, 1])
#     # plt.plot(np.array(b[1].getPos())[:, 0])
#     #plt.plot(times, np.array(b[1].getPos())[:, 0])
#     plt.show()


