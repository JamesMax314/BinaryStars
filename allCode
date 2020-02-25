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


if __name__ == "__main__":
    dt = 1e8  # 1e8 #1e2 #1e2
    n_iter = 1000

    m_1 = 1e20
    m_2 = 1e20
    r_1_2 = 14.6e9
    dim = 1e11
    N = 1000

    arr_bodies = two_body_init(m_1, m_2, r_1_2)
    arr_bodies = lfEngine.half_step(arr_bodies, dt)

    _arr_bodies = np.array([])
    for body in arr_bodies:
        # print(body.r)
        _arr_bodies = np.append(_arr_bodies, tree.body(np.real(body.m), np.real(body.r), np.real(body.v), [0] * 3))

    dmDen = 4
    vol = 4/3*np.pi*(dim/2)**3
    dmMass = dmDen*vol
    dmPointMass = dmMass/N
    # dm, loc, mass = dm_array(dmPointMass, dmPointMass, N, dim)
    dm, loc, mass = dm_array_cube(dmPointMass, dmPointMass, N, dim)
    # _arr_bodies = np.append(_arr_bodies, dm)
    # _arr_bodies = dm
    # perimitor = periodic(loc, mass, dim)
    particle1 = tree.body(m_1, [7e9, 0, 0], [0, 0, 0], [0] * 3)
    particle2 = tree.body(m_2, [-7e9, 0, 0], [-0, 0, 0], [0] * 3)
    # _arr_bodies = np.array([])
    # _arr_bodies = np.append(particle1, _arr_bodies)
    # _arr_bodies = np.append(particle2, _arr_bodies)
    # _arr_bodies = np.append(_arr_bodies, dm)

    arrCent = np.array([0, 0, 0])
    uniDim = np.array([1e15] * 3)
    # b = tree.basicRun(_arr_bodies, arrCent, uniDim, int(n_iter), dt)
    spacing = dim / 30
    b = tree.particleMesh(_arr_bodies, spacing, dim, n_iter, dt)

    colours = np.array([0, 0.25])
    # colours = np.array([0.5, 0.5])
    colours = np.append(colours, np.array([0.5] * (len(_arr_bodies)-2)))

    forces = np.empty([len(b), len(b[0].acc), 3])

    # for i in range(len(b)):
    #     acc = np.array(b[i].acc)
    #     mass = b[i].mass[0]
    #     forces[i] = acc * mass
    mation = anim.twoD(b, colours, dim, 1e-12, 10)
    # mation.animate(10)
    mation.run("test2.mp4")
    plt.show()

    # plt.plot(forces[1, :, 0])
    # plt.show()


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


#%%

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import treecode as tree
import lfEngine

#%%

def two_body_init(m_1, m_2, r):
    global G
    com = m_2*r / (m_1 + m_2)
    r_1 = np.abs(com)
    r_2 = np.abs(r - com)
    v_1 = np.sqrt((G * m_2 * r_1) / r**2)
    v_2 = np.sqrt((G * m_1 * r_2) / r**2)
    _arr_bodies = np.array([])
    _arr_bodies = np.append(_arr_bodies, lfEngine.Body([r_1, 0, 0], [0, v_1, 0], m_1))
    _arr_bodies = np.append(_arr_bodies, lfEngine.Body([r_2, 0, 0], [0, -v_2, 0], m_2))
    return _arr_bodies

#%%

def add_moon(body, m, r):
    global G
    v = np.sqrt(G * body.m / r)
    arr_v = -np.array([0, v, 0])
    arr_v = -arr_v + body.v
    r = -np.array([r, 0, 0]) + body.r
    out_body = lfEngine.Body(r, arr_v, m)
    return out_body

#%%

##### Initialise system #####
G = 6.674e-11

dt = 10*24*3600
n_iter = 110*360*24*3600/dt

m_sun = 1.989e30
m_jupiter = 1.898e27
m_earth = 5.927e24
r_sun_jupyter = 778.57e9
r_sun_earth = 149.6e9

arr_bodies = two_body_init(m_sun, m_jupiter, r_sun_jupyter) #init_bodies()
#arr_bodies = two_body_init(m_sun, m_earth, r_sun_earth)
arr_bodies = np.append(add_moon(arr_bodies[0], m_earth, r_sun_earth), arr_bodies)
arr_bodies = lfEngine.half_step(arr_bodies, dt)

#%%

##### Initialize tree bodies #####
_arr_bodies = np.array([])
for body in arr_bodies:
    _arr_bodies = np.append(_arr_bodies, tree.body(body.m, body.r, body.v, [0]*3))


arrCent = np.array([0, 0, 0])
uniDim = np.array([1e13]*3)
b = tree.basicRun(_arr_bodies, arrCent, uniDim, int(n_iter), dt)


result = lfEngine.execute(arr_bodies, dt, int(n_iter))

#%%

##### Process Results #####
def get_theta(vec0, vec1, theta0):
    cross = np.cross(vec0, vec1)
    delta_theta = np.arcsin(norm(cross)/(norm(vec0) * norm(vec1)))
    theta = theta0 + delta_theta
    return theta
    
    
def theta_arr(vecs, init):
    thetas = np.empty(vecs.shape[0])
    thetas[0] = init
    for i in range(1, len(thetas)):
        thetas[i] = get_theta(vecs[i-1], vecs[i], thetas[i-1])
    return thetas

#%%

arrB = np.empty([len(arr_bodies), np.shape(np.array(b[1].pos))[0], np.shape(np.array(b[1].pos))[1]])
for i in range(len(arr_bodies)):
    arrB[i, :, :] = np.array(b[i].pos)
np.shape(arrB)
print(np.shape(b))

#%%

labels = ["Earth", "sun", "Jupiter"]
colour = ["blue", "orange", "purple"]
for i in range(len(arr_bodies)):
    #plt.plot(result["rs"][i, :, 1]/1e12, result["rs"][i, :, 0]/1e12, label=labels[i], color=colour[i])
    plt.plot(np.array(b[i].pos)[:, 0]/1e12, np.array(b[i].pos)[:, 1]/1e12, label=labels[i], color=colour[i])
    plt.scatter(np.array(b[i].pos)[-1, 0]/1e12, np.array(b[i].pos)[-1, 1]/1e12, color=colour[i])
#plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x / $T$m")
plt.ylabel("y / $T$m")
plt.legend()
plt.show()

#%%

##### Compute Angles #####
thetas = np.empty([len(arr_bodies), int(n_iter)])
for i in range(len(arr_bodies)):
    thetas[i, :] = theta_arr(result["rs"][i, :, :], np.pi/2)

#%%

labels = ["Earth", "sun", "Jupiter"]
for i in range(len(arr_bodies)):
    plt.plot(result["r_ts"]/(360*24*3600), thetas[i]/(2*np.pi), label=labels[i])
plt.title("Angle")
plt.xlabel("time / year")
plt.ylabel("Angle / $2 \pi$ rad")
plt.legend()
plt.show()

#%%

time_100 = np.interp(100, thetas[0]/(2*np.pi), result["r_ts"]/(360*24*3600))
print("Time for 100 earth orbits:", np.round(time_100), "years")

#%%

##### Compute Radii #####
labels = ["Earth", "sun", "Jupiter"]
for i in range(len(arr_bodies)):
    rads = norm(result["rs"][i, :, :], axis=1)
    plt.plot(result["r_ts"]/(360*24*3600), rads/1e12, label=labels[i])
plt.title("Orbital Radius")
plt.xlabel("time / year")
plt.ylabel("radius / $T$m")
plt.legend()
plt.show()

#%%

##### Total Energy #####
import matplotlib
##### Durham colours #####
dpi = [196.0 / 255.0, 59.0 / 255.0, 142.0 / 255.0]  # Durham pink
dr = [170.0 / 255.0, 43.0 / 255.0, 74.0 / 255.0]  # Durham red
dy = [232.0 / 255.0, 227.0 / 255.0, 145.0 / 255.0]  # Durham yellow
dg = [159.0 / 255.0, 161.0 / 255.0, 97.0 / 255.0]  # Durham green
db = [0, 99.0 / 255.0, 136.0 / 255.0]  # Durham blue
dp = [126.0 / 255.0, 49.0 / 255.0, 123.0 / 255.0]  # Durham purple
dv = [216.0 / 255.0, 172.0 / 255.0, 244.0 / 255.0]  # Durham violet
matplotlib.rcParams.update({'font.size': 22})
Etot = result["pot"]+result["Eks"]
plt.plot(result["r_ts"]/(360*24*3600), 100*(Etot-Etot[0])/Etot[0], color=dp)
#plt.title("Total Energy")
plt.xlabel("time / year", fontname="Times New Roman")
plt.ylabel("Energy Change / %", fontname="Times New Roman")
plt.tight_layout()
plt.savefig('milestoneEnergy.png', dpi=1000)
#plt.show()

#%%

##### Potential Energy #####
plt.plot(result["r_ts"]/(360*24*3600), result["pot"]/10**35)
plt.title("Potential Energy")
plt.xlabel("time / year")
plt.ylabel("Energy / $10^{35}$ J")
plt.show()

#%%

##### Kinetic Energy
plt.plot(result["r_ts"]/(360*24*3600), result["Eks"]/10**35, color="red")
plt.title("Kinetic Energy")
plt.xlabel("time / year")
plt.ylabel("Energy / $10^{35}$ J")
plt.show()

#%%

print("Total energy difference:", str(1e4*100*(Etot[len(Etot)-1]-Etot[0])/Etot[0])[0:5], "e^(-4)%")

#%%



import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

plt.style.use('dark_background')


class twoD():
    def __init__(self, bodies, colours, dim, scale, sampling):
        self.system = bodies
        self.dimension = scale * dim
        self.border = scale * dim / 20
        self.interval = sampling
        self.scale = scale
        self.colours = colours

        lim = self.border + dim * scale
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-lim, lim), ylim=(-lim, lim))
        self.points = self.ax.scatter([], [], cmap="jet", s=2)

        self.arrB = np.empty([len(self.system), np.shape(np.array(self.system[1].pos))[0],
                              np.shape(np.array(self.system[1].pos))[1]])
        for i in range(len(self.system)):
            self.arrB[i, :, :] = np.array(self.system[i].pos) * scale
        self.frames = int(np.shape(self.arrB)[1] / sampling)

    def init(self):
        self.points.set_offsets([])
        return self.points

    def animate(self, i):
        offsets = np.transpose([self.arrB[:, i*self.interval, 0], self.arrB[:, i*self.interval, 1]])
        self.points.set_offsets(offsets)
        self.points.set_array(self.colours)
        return self.points

    def run(self, name):
        self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                            frames=int(self.frames))
        plt.xlabel("x/Tm")
        plt.ylabel("y/Tm")
        self.anim.save(name, fps=30, writer='ffmpeg', dpi=300)
import numpy as np
import matplotlib.pyplot as plt

# global vars
G = 6.674e-11


class Body:
    def __init__(self, r, v, m):
        self.r = np.array(r, dtype=np.complex64)
        self.v = np.array(v, dtype=np.complex64)
        self.m = m


def gen_force_mat(_arr_bodies, epsilon):
    global G

    vec_f = np.zeros([len(_arr_bodies), len(_arr_bodies), len(_arr_bodies[0].r)], dtype=np.float64)
    for i in range(len(_arr_bodies)):
        for j in range(i+1, len(_arr_bodies)):
            double_mass = _arr_bodies[i].m * _arr_bodies[j].m
            vec_sep = np.array(_arr_bodies[i].r - _arr_bodies[j].r)
            norm = np.linalg.norm(vec_sep)
            vec_f[i, j, :] = np.real(double_mass * vec_sep / (norm**2 + epsilon**2)**(3/2))
            vec_f[i, j, :] = - G * vec_f[i, j, :]

    vec_f_trans = np.transpose(vec_f, (1, 0, 2))
    return vec_f - vec_f_trans


def get_potential(_arr_bodies):
    global G

    u = 0
    for i in range(len(_arr_bodies)-1):
        for j in range(i+1, len(_arr_bodies)):
            vec_sep = np.array(_arr_bodies[i].r - _arr_bodies[j].r)
            norm = np.linalg.norm(vec_sep)
            u += G*_arr_bodies[i].m*_arr_bodies[j].m / norm
    return -u


def half_step(arr_bodies_, dt_, softening=0):
    dt_2_ = dt_/2
    mat_force_ = gen_force_mat(arr_bodies_, softening)
    for j in range(len(arr_bodies_)):
        vec_sum_f_ = np.sum(mat_force_[j, :, :], axis=0)  # Net force on ith particle
        arr_bodies_[j].v -= vec_sum_f_ * dt_2_ / arr_bodies_[j].m
    return arr_bodies_


def execute(arr_bodies, dt, n_iter, softening=0):
    dt_2 = dt/2

    v_t = -dt_2
    r_t = 0

    v_ts = np.empty([int(n_iter)])
    r_ts = np.empty([int(n_iter)])
    e_ts = np.empty([int(n_iter)])
    rs = np.empty([len(arr_bodies), int(n_iter), 3])
    vs = np.empty([len(arr_bodies), int(n_iter), 3])
    Eks = np.empty([int(n_iter)])
    pot = np.empty([int(n_iter)])

    for i in range(int(n_iter)):
        v_t += dt
        r_t += dt

        mat_force = gen_force_mat(arr_bodies, softening)

        Ek = 0

        for j in range(len(arr_bodies)):
            vec_sum_f = np.sum(mat_force[j, :, :], axis=0) # Net force on ith particle

            arr_bodies[j].v += vec_sum_f * dt / arr_bodies[j].m
            arr_bodies[j].r += arr_bodies[j].v * dt

            tmp_v = arr_bodies[j].v + vec_sum_f * dt_2 / arr_bodies[j].m
            Ek += (1/2) * arr_bodies[j].m * np.linalg.norm(tmp_v)**2

            rs[j, i, :] = arr_bodies[j].r
            vs[j, i, :] = arr_bodies[j].v

        Eks[i] = Ek
        pot[i] = get_potential(arr_bodies)

        r_ts[i] = r_t
        v_ts[i] = v_t

    #for i in range(len(arr_bodies)):
    #    plt.plot(rs[i, :, 1], rs[i, :, 0])
    out = {"v_ts": v_ts, "r_ts": r_ts, "vs": vs, "rs": rs, "Eks": Eks, "pot": pot}
    return out

#include <iostream>
#include <utility>
#include "bodies.h"

using namespace std;

/// Body constructors
body::body() = default;

body::body(double &m, vector<double> &p, vector<double> &v, vector<double> &a){
    mass.emplace_back(m);
    pos.emplace_back(p);
    vel.emplace_back(v);
    acc.emplace_back(a);
    active.emplace_back(true);
}

void body::setPos(const vector<double>& p){
    pos.emplace_back(p);
}
void body::setAcc(const vector<double>& a){
    acc.emplace_back(a);
}
void body::setVel(const vector<double>& v){
    vel.emplace_back(v);
}
void body::setMass(double m){
    mass.emplace_back(m);
}
void body::setSoftening(double s){
    softening = s;
}


vector<vector<double>> body::getPos(){
    return pos;
}
vector<vector<double>> body::getAcc(){
    return acc;
}
vector<vector<double>> body::getVel(){
    return vel;
}
vector<double> body::getMass(){
    return mass;
}
double body::getSoftening(){
    return softening;
}

#ifndef BODIES
#define BODIES

#include <vector>

using namespace std;

struct body{
    double softening = 0;
	vector<double> mass;
	vector<vector<double>> pos;
	vector<vector<double>> vel;
	vector<vector<double>> acc;
	vector<double> ek;
	vector<double> ep;
	vector<bool> active;

	void setPos(const vector<double>&);
    void setAcc(const vector<double>&);
    void setVel(const vector<double>&);
    void setMass(double);
    void setSoftening(double);
    vector<vector<double>> getPos();
    vector<vector<double>> getAcc();
    vector<vector<double>> getVel();
    vector<double> getMass();
    double getSoftening();

    body();
    body(double& mass, vector<double>& pos, vector<double>& vel, vector<double>& acc);
};

// Macros to retrieve body data; x is a pointer
#define Vel(x) (((body*) (x))->vel) // cast x to type body* then retrieve vel
#define Acc(x) (((body*) (x))->acc)
#define Ek(x) (((body*) (x))->ek)
#define Ep(x) (((body*) (x))->ep)

#endif //BODIES

cmake_minimum_required(VERSION 3.14)
#project(barnesHut)
add_definitions( "-fpic" ) # Linux only
SET(APP_PY treecode)

project(${APP_PY})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")

#SET(APP_EXE barnesHut)
set(PYBIND11_PYTHON_VERSION 3.6)
add_subdirectory(.//Include//pybind11)

find_package(OpenMP REQUIRED)
ADD_LIBRARY(trees STATIC trees.cpp)
ADD_LIBRARY(bodies STATIC bodies.cpp)
ADD_LIBRARY(vecMaths STATIC vecMaths.cpp)
ADD_LIBRARY(leapfrog STATIC leapfrog.cpp)
ADD_LIBRARY(treeShow STATIC treeShow.cpp)
ADD_LIBRARY(poisson STATIC poisson.cpp)
ADD_LIBRARY(tpm STATIC tpm.cpp)
pybind11_add_module(${APP_PY} pyInterface.cpp)
#add_executable(APP_EXE main.cpp)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(.//Include)
#include_directories(C://ProgramData//Anaconda3//include) # Needs changing for each machine
#include_directories(/home/james/anaconda3/include/.) # Needs changing for each machine
#TARGET_LINK_LIBRARIES(${APP_EXE} trees vecMaths leapfrog bodies)
TARGET_LINK_LIBRARIES(${APP_PY} PRIVATE trees vecMaths leapfrog bodies treeShow poisson tpm fftw3)
TARGET_LINK_LIBRARIES(poisson fftw3 OpenMP::OpenMP_CXX)
#include_directories(include)

#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>

#include "bodies.h"
#include "trees.h"
#include "vecMaths.h"
#include "leapfrog.h"

/// Leapfrog functions
void treeMake(barnesHut &hut){
    hut.treeBuild();
}

void treeBreak(barnesHut &hut){
	hut.treeChop(hut.root);
}

void interaction(barnesHut &hut){
    hut.acceleration(hut.root);
}

//void bodiesUpdate(barnesHut &hut, double dt){
//    node* tree = hut.root;
//    vector<body>* bodies = hut.bodies;
//    partialUpdate(tree, bodies, dt);
//}

void bodiesUpdate(vector<body>* bodies, const vector<int>& activeBods, double dt, vector<double> dim){
    for (auto & bIndx : activeBods) {
        auto v = vecAdd((*bodies)[bIndx].vel.back(), scalMult(dt, (*bodies)[bIndx].acc.back()));
        auto p = vecAdd((*bodies)[bIndx].pos.back(), scalMult(dt, (*bodies)[bIndx].vel.back()));
        for (int axis=0; axis<3; axis++){
            if (p[axis] < -dim[axis]/2) {
                p[axis] += dim[axis];
//                v[axis] *= -1;
            }
            if (p[axis] > dim[axis]/2) {
                p[axis] -= dim[axis];
//                v[axis] *= -1;
            }
        }
        (*bodies)[bIndx].vel.emplace_back(v);
        (*bodies)[bIndx].pos.emplace_back(p);
//        cout << body.vel.back()[0] << endl;
    }
}

void PBC(vector<body>* bodies, const vector<int>& activeBods, vector<double> dim){
    for (auto & bIndx : activeBods) {
        auto p = (*bodies)[bIndx].pos.back();
        for (int axis=0; axis<3; axis++){
            if (p[axis] < -dim[axis]/2) {
                p[axis] += dim[axis];
//                v[axis] *= -1;
            }
            if (p[axis] > dim[axis]/2) {
                p[axis] -= dim[axis];
//                v[axis] *= -1;
            }
        }
#ifndef LEAPFROG
#define LEAPFROG

#include "bodies.h"
#include "trees.h"

void treeMake(barnesHut& hut);
void interaction(barnesHut& hut);
void treeBreak(barnesHut& hut);
void bodiesUpdate(vector<body>* bodies, const vector<int>& activeBods, double dt, vector<double> dim);
void bodiesUpdate(vector<body>* bodies, const vector<int>& activeBods, double dt);
void PBC(vector<body>* bodies, const vector<int>& activeBods, vector<double> dim);
void partialUpdate(node* tree, vector<body>* bodies, double dt);
void boundaryInteract(barnesHut& bh, vector<body>& boundary);

#endif //LEAPFROG
#include <fftw3.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>
#include "vecMaths.h"
#include "poisson.h"

grid::grid(double gridSpacing, double dim): dim({dim, dim, dim}), spacing(gridSpacing) {
    for (int axis=0; axis<3; axis++)
        numPts[axis] = (int) (this->dim[axis] / gridSpacing);
//    cout << "numPts: " << numPts << endl;
    realPot = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numPts[0]*numPts[1]*numPts[2]);
    realField1 = (double *) fftw_malloc(sizeof(double) * numPts[0]*numPts[1]*numPts[2]);
    compFFTRho = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * numPts[0]*numPts[1]*numPts[2]);
    compFFTRho1 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * numPts[0]*numPts[1]*numPts[2]);
    keys = new int[numPts[0]*numPts[1]*numPts[2]];
    for (auto & i : realField) {
        i = new double[int(numPts[0]*numPts[1]*numPts[2])];
    }
    for (int i=0; i<3; i++) {
        comp[i] = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * numPts[0]*numPts[1]*numPts[2]);
        cField[i] = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * numPts[0]*numPts[1]*numPts[2]);
//        realField[i] = (double *) fftw_malloc(sizeof(double) * pow(numPts, 3));
    }
    fwrd = fftw_plan_dft_3d(numPts[0], numPts[1], numPts[2], realPot, compFFTRho, FFTW_FORWARD, FFTW_MEASURE);
//    bwrd = fftw_plan_dft_3d(numPts, numPts, numPts, compFFTRho1, realPot, FFTW_BACKWARD, FFTW_MEASURE);
    for (int i = 0; i < 3; i++) {
        bwrd[i] = fftw_plan_dft_3d(numPts[0], numPts[1], numPts[2], comp[i], cField[i], FFTW_BACKWARD, FFTW_MEASURE);
    }
}

grid::grid(double gridSpacing, vector<int> numPs): spacing(gridSpacing) {
    for (int axis=0; axis<3; axis++)
        dim[axis] = (int) (numPs[axis] * gridSpacing);
//    cout << "numPts: " << numPts << endl;
    realPot = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numPts[0]*numPts[1]*numPts[2]);
    realField1 = (double *) fftw_malloc(sizeof(double) * numPts[0]*numPts[1]*numPts[2]);
    compFFTRho = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * numPts[0]*numPts[1]*numPts[2]);
    compFFTRho1 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * numPts[0]*numPts[1]*numPts[2]);
    keys = new int[numPts[0]*numPts[1]*numPts[2]];
    for (auto & i : realField) {
        i = new double[int(numPts[0]*numPts[1]*numPts[2])];
    }
    for (int i=0; i<3; i++) {
        comp[i] = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * numPts[0]*numPts[1]*numPts[2]);
        cField[i] = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * numPts[0]*numPts[1]*numPts[2]);
//        realField[i] = (double *) fftw_malloc(sizeof(double) * pow(numPts, 3));
    }
    fwrd = fftw_plan_dft_3d(numPts[0], numPts[1], numPts[2], realPot, compFFTRho, FFTW_FORWARD, FFTW_MEASURE);
//    bwrd = fftw_plan_dft_3d(numPts, numPts, numPts, compFFTRho1, realPot, FFTW_BACKWARD, FFTW_MEASURE);
    for (int i = 0; i < 3; i++) {
        bwrd[i] = fftw_plan_dft_3d(numPts[0], numPts[1], numPts[2], comp[i], cField[i], FFTW_BACKWARD, FFTW_MEASURE);
    }
}

grid::grid(const grid & g) : numPts(g.numPts) {
    for (int axis=0; axis<3; axis++)
        dim[axis] = g.dim[axis];
    for (auto & i : realField) {
        i = new double[int(numPts[0]*numPts[1]*numPts[2])];
    }
}

void grid::updateGrid(vector<body>* bods){
    /// Row major format
//    realPot[int(i*pow(numPts, 2) + j*numPts + k)] = 0;
//#pragma omp parallel for //default(none) shared(bods)
    for (int i = 0; i < numPts[0]; i++) {
        for (int j = 0; j < numPts[1]; j++) {
            for (int k = 0; k < numPts[2]; k++) {
                realPot[int(i*numPts[2]*numPts[1] + j*numPts[2] + k)][0] = 0;
            }
        }
    }

    for (auto & bIndx : activeBods){
        vector<vector<int>> mPos = meshPos((*bods)[bIndx].pos.back());
        for (auto vec : mPos) {
            realPot[int(vec[0]*numPts[2]*numPts[1] + vec[1]*numPts[2] + vec[2])][0] +=
                   w({vec[0], vec[1], vec[2]}, (*bods)[bIndx]) * (*bods)[bIndx].mass.back() / spacing;
        }
    }
}

void grid::solveField(){
    fftw_execute(fwrd);
//#pragma omp parallel for // default(none) shared(axis)
    for (int axis=0; axis<3; axis++) {
        for (int i = 0; i < numPts[0]; i++) { //floor(numPts/2) + 1
            for (int j = 0; j < numPts[1]; j++) {
                for (int k = 0; k < numPts[2]; k++) {
                    /// Accounting for weird fft structure
                    int kx = (i <= numPts[0] / 2) ? i : i - numPts[0];
                    int ky = (j <= numPts[1] / 2) ? j : j - numPts[1];
                    int kz = (k <= numPts[2] / 2) ? k : k - numPts[2];
                    vector<int> ks = {kx, ky, kz};

                    fftw_complex scale;

                    /// Normalize FFT with 1/N^3 the inde must be scaled to give k;
                    /// The k vectors are given by index /(spacing * N);
                    /// The Greens function for force is -ik-> / k^2;
                    /// The overall scaling is given by: spacing * N / N^3 = spacing / N^2.
                    scale[1] = (i == 0 && j == 0 && k == 0) ? 0 : spacing * ks[axis] / pow(numPts[0], 2) *
                            4 * pi * G /
                            (numPts[0]*numPts[1]*numPts[2] *
                            (kx * kx / pow(numPts[0], 2) +
                            ky * ky / pow(numPts[1], 2) +
                            kz * kz / pow(numPts[2], 2))); //dim*4*pi*G
                    scale[0] = 0;

    //                compMultFFT(compFFTRho[int(i * pow(numPts, 2) + j * numPts + k)], scale,
    //                            compFFTRho1[int(i * pow(numPts, 2) + j * numPts + k)]);
                    compMultFFT(compFFTRho[int(i * numPts[2]*numPts[1] + j * numPts[2] + k)], scale,
                                comp[axis][int(i * numPts[2]*numPts[1] + j * numPts[2] + k)]);
                }
            }
        }
        fftw_execute(bwrd[axis]);
        ctor(cField[axis], realField[axis]);
    }
//    diff(-1);
}

void grid::interpW(vector<body>* bods, bool resetForce){
    for (auto & bIndx : activeBods){
        vector<double> f(3, 0);
        vector<vector<int>> mPos = meshPos((*bods)[bIndx].pos.back());
//#pragma omp parallel for
        for (auto vec : mPos) {
            for (int axis = 0; axis < 3; axis++) {
                f[axis] += w({vec[0], vec[1], vec[2]}, (*bods)[bIndx]) *
                        realField[axis][int(vec[0] * numPts[2]*numPts[1] + vec[1] * numPts[2] + vec[2])] /
                        (*bods)[bIndx].mass.back();
            }
        }
        if (resetForce)
            (*bods)[bIndx].acc.emplace_back(f);
        else
            (*bods)[bIndx].acc.back() = vecAdd((*bods)[bIndx].acc.back(), f);
//        printVec(body.acc.back());
    }
}

/// !!! Conatains legacy code; no actvie bodies list !!!
void grid::interp(vector<body>* bods){
    for (auto & body : (*bods)){
        vector<double> f(3, 0);
        vector<vector<int>> mPos = meshPos(body.pos.back());
//#pragma omp parallel for
        if (mPos.size() == 8) { /// Check inside mesh
            double xd = (body.pos.back()[0] - vecPos(mPos[0][0], mPos[0][1], mPos[0][2])[0]) / spacing;
            double yd = (body.pos.back()[1] - vecPos(mPos[0][0], mPos[0][1], mPos[0][2])[1]) / spacing;
            double zd = (body.pos.back()[2] - vecPos(mPos[0][0], mPos[0][1], mPos[0][2])[2]) / spacing;
            for (int axis = 0; axis < 3; axis++) {
                double C00 = realField[axis][int(mPos[0][0] * numPts[2]*numPts[1] + mPos[0][1] * numPts[2] + mPos[0][2])] *
                        (1 - xd) + xd*realField[axis][int(mPos[4][0] * numPts[2]*numPts[1] + mPos[4][1] * numPts[2] + mPos[4][2])];
                double C01 = realField[axis][int(mPos[1][0] * numPts[2]*numPts[1] + mPos[1][1] * numPts[2] + mPos[1][2])] *
                        (1 - xd) + xd*realField[axis][int(mPos[5][0] * numPts[2]*numPts[1] + mPos[5][1] * numPts[2] + mPos[5][2])];
                double C10 = realField[axis][int(mPos[2][0] * numPts[2]*numPts[1] + mPos[2][1] * numPts[2] + mPos[2][2])] *
                        (1 - xd) + xd*realField[axis][int(mPos[6][0] * numPts[2]*numPts[1] + mPos[6][1] * numPts[2] + mPos[6][2])];
                double C11 = realField[axis][int(mPos[3][0] * numPts[2]*numPts[1] + mPos[3][1] * numPts[2] + mPos[3][2])] *
                        (1 - xd) + xd*realField[axis][int(mPos[7][0] * numPts[2]*numPts[1] + mPos[7][1] * numPts[2] + mPos[7][2])];

                double C0 = C00 * (1-yd) + C10*yd;
                double C1 = C01 * (1-yd) + C11*yd;

                double C = C0 * (1-zd) + C1*zd;

                f[axis] = C / body.mass.back();

            }
        }
        body.acc.emplace_back(f);
//        printVec(body.acc.back());
    }
}

void grid::diff(double scale) {
//#pragma omp parallel for
    for (int i=0; i<numPts[0]; i++){
        for (int j=0; j<numPts[1]; j++){
            for (int k=0; k<numPts[2]; k++){
                if (i != 0 && i != numPts[0]-1 && j != 0 && j != numPts[1]-1 && k != 0 && k != numPts[2]-1) {
                    realField[0][int(i * numPts[2]*numPts[1] + j * numPts[2] + k)] =
                            scale*(realPot[int((i - 1) * numPts[2]*numPts[1] + j * numPts[2] + k)][0] -
                                    realPot[int((i + 1) * numPts[2]*numPts[1] + j * numPts[2] + k)][0]) / (2 * spacing);
                    realField[1][int(i * numPts[2]*numPts[1] + j * numPts[2] + k)] =
                            scale*(realPot[int(i * numPts[2]*numPts[1] + (j - 1) * numPts[2] + k)][0] -
                                    realPot[int(i * numPts[2]*numPts[1] + (j + 1) * numPts[2] + k)][0]) / (2 * spacing);
                    realField[2][int(i * numPts[2]*numPts[1] + j * numPts[2] + k)] =
                            scale*(realPot[int(i * numPts[2]*numPts[1] + j * numPts[2] + k - 1)][0] -
                                    realPot[int(i * numPts[2]*numPts[1] + j * numPts[2] + k + 1)][0]) / (2 * spacing);
                } else {
                    for (auto & axis : realField){
                        axis[int(i * numPts[2]*numPts[1] + j * numPts[2] + k)] = 0;
                    }
                }
            }
        }
    }
}

vector<double> grid::vecPos(int i, int j, int k){
    return {i*spacing - dim[0]/2, j*spacing - dim[1]/2, k*spacing - dim[2]/2};
}

vector<vector<int>> grid::meshPos(vector<double> pos) {
//    pos = {dim/2, dim/2, dim/2};
    vector<vector<int>> meshPoints;
    vector<int> pt(3, 0);
    for (int i = 0; i < 3; i++) {
        pt[i] = floor((pos[i] + dim[i]/2) / spacing);
//        cout <<  "compute: " << (pos[i] + dim) << endl;
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                if (pt[0]+i<numPts[0] && pt[1]+j<numPts[1] && pt[2]+k<numPts[2] && pt[0]+i>=0 && pt[1]+j>=0 && pt[2]+k>=0)
                    meshPoints.emplace_back(vecAdd(pt, {i, j, k}));
            }
        }
    }
    return meshPoints;
}

double grid::w(vector<int> vec, body& bod){
    double out = 1;
//    cout << "bodPos, vecPos: "; printVec(bod.pos.back()) ;cout << " "; printVec(vecPos(vec[0], vec[1], vec[2])); cout << endl;
    for (int i=0; i<3; i++){
        double dist = abs(bod.pos.back()[i] - vecPos(vec[0], vec[1], vec[2])[i]);
//        cout << "dist" << dist << endl;
        if (dist < spacing) {
            out *= abs((spacing - dist)/spacing);
        } else
            return 0;
    }
//    cout << "ok" << endl;
    return out;
}

void grid::ctor(fftw_complex* arr, double* out){
    for (int i=0; i < numPts[0]; i++) {
        for (int j = 0; j < numPts[1]; j++) {
            for (int k = 0; k < numPts[2]; k++) {
                out[int(i * numPts[2]*numPts[1] + j * numPts[2] + k)] =
                        arr[int(i * numPts[2]*numPts[1] + j * numPts[2] + k)][0];
            }
        }
    }
}

void grid::magF(){
    for (int i=0; i<numPts[0]; i++) {
        for (int j = 0; j < numPts[1]; j++) {
            for (int k = 0; k < numPts[2]; k++) {
//                cout << realField1[int(i * pow(numPts, 2) + j * numPts + k)] << endl;
                realField1[int(i * numPts[2]*numPts[1] + j * numPts[2] + k)] =
                        pow(realField[0][int(i * numPts[2]*numPts[1] + j * numPts[2] + k)], 2) +
                        pow(realField[1][int(i * numPts[2]*numPts[1] + j * numPts[2] + k)], 2) +
                        pow(realField[2][int(i * numPts[2]*numPts[1] + j * numPts[2] + k)], 2);
            }
        }
    }
}

vector<vector<vector<double>>> grid::getF(int indx){
    vector<vector<vector<double>>> out;
    for (int i=0; i<numPts[0]; i++) {
        vector<vector<double>> tmp;
        for (int j = 0; j < numPts[1]; j++) {
            vector<double> tmp1;
            for (int k = 0; k < numPts[2]; k++) {
                tmp1.emplace_back(realField[indx][int(i * numPts[2]*numPts[1] + j * numPts[2] + k)] + 1e-9);
            }
            tmp.emplace_back(tmp1);
        }
        out.emplace_back(tmp);
    }
    return out;
}

void grid::initActiveBods(vector<body>* bods){
    vector<int> tmp;
    for (int i=0; i<(*bods).size(); i++){
        tmp.emplace_back(i);
    }
    activeBods = tmp;
}

grid::grid() = default;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void compMultFFT(fftw_complex v1, fftw_complex v2, fftw_complex out) {
    double real = v1[0]*v2[0] - v1[1]*v2[1];
    double imag = v1[1]*v2[0] + v1[0]*v2[1];
    out[0] = real,
    out[1] = imag;
//    if (real){
//        cout << "(" << v1[0] << " + i" << v1[1] << ")(" <<  v2[0] << " + i" << v2[1] << ") = " << out[0] << " + i" << out[1] << endl;
//        cout << "v1[0]*v2[0] = " << v1[0]*v2[0] << endl;}

}




#ifndef POISSON
#define POISSON

#include <bodies.h>
#include <fftw3.h>
#include <cmath>
#include "bodies.h"

using namespace std;

class grid{
    fftw_plan fwrd{};
    fftw_plan bwrd[3]{};
public:
    double spacing{};
    vector<double> dim;
    vector<int> numPts;
    double G = 6.674e-11;
    double pi = M_PI;
    fftw_complex* realPot{};
    double* realField[3]{};
    double* realField1{};
    int* keys{};
    fftw_complex* comp[3]{};
    fftw_complex* cField[3]{};
    fftw_complex* compFFTRho{};
    fftw_complex* compFFTRho1{};
    vector<int> activeBods;

    grid(double gridSpacing, double dim);
    grid(double gridSpacing, vector<int> numPs);
    grid(grid const &g);
    grid();

    void updateGrid(vector<body>* bods);
    void solveField();
    vector<double> vecPos(int i, int j, int k);
    vector<vector<int>> meshPos(vector<double> pos);
    double w(vector<int> vec, body& bod);
    void diff(double scale);
    void interpW(vector<body>* bods, bool resetForce);
    void interp(vector<body>* bods);
    void initActiveBods(vector<body>* bods);

    void ctor(fftw_complex* arr, double* out);
    void magF();
    vector<vector<vector<double>>> getF(int indx);

    /// Pybinding getters ///
    double *data() { return realField1; }
    vector<int> size()  { return numPts; }
};

void compMultFFT(fftw_complex v1, fftw_complex v2, fftw_complex out);



#endif //POISSON

#include <iostream>
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bodies.h"
#include "trees.h"
#include "leapfrog.h"
#include "pyInterface.h"
#include "treeShow.h"
#include "poisson.h"
#include "tpm.h"

namespace py = pybind11;

progbar::progbar(double maximum, double barLength): max(maximum), length(barLength){}

void progbar::update(int current){
//    cout << "\r";
//    for (int i=0; i<length+20; i++)
//        cout << " ";
    cout << "\rProgress: ";
    int bars = int(length*((current+1) / max));
    for (int i=0; i<bars; i++)
        cout << "#";
    for (int i=0; i<length-bars; i++)
        cout << "_";
    cout << " | (" << int(100*((current+1) / max)) << "%)" << "\r";
}

vector<body> basicRun(vector<body>& bodies, vector<double> centre, vector<double> width, int numIter, double dt){
    barnesHut bh = barnesHut(bodies, width, centre);
    bh.initActiveBods();
    progbar prog = progbar(numIter, 20);
    for(int j=0; j<numIter; j++) {
        treeMake(bh);
        interaction(bh);
        bodiesUpdate(bh.bodies, bh.activeBods, dt);
        treeBreak(bh);
        prog.update(j);
    }
    cout << endl;
    return *bh.bodies;
}

vector<body> fixedBoundary(vector<body>& bodies, vector<body>& boundary, vector<double> centre,
        vector<double> width, int numIter, double dt){
    barnesHut bh = barnesHut(bodies, width, centre);
    bh.initActiveBods();
    cout << "bodLen: " << boundary.size() << endl;
    progbar prog = progbar(numIter, 20);
    for(int j=0; j<numIter; j++) {
        treeMake(bh);
        interaction(bh);
        boundaryInteract(bh, boundary);
        bodiesUpdate(bh.bodies, bh.activeBods, dt);
        treeBreak(bh);
        prog.update(j);
    }
    cout << endl;
    return *bh.bodies;
}

vector<body> particleMesh(vector<body>& bodies, double spacing, double width, int numIter, double dt){
    vector<body>* bods = &bodies;
    grid g = grid(spacing, width);
    g.initActiveBods(bods);
    progbar prog = progbar(numIter, 20);
    for(int j=0; j<numIter; j++) {
        g.updateGrid(bods);
        g.solveField();
        g.interpW(bods, true);
        bodiesUpdate(bods, g.activeBods, dt, g.dim);
        prog.update(j);
    }
    cout << endl;
    return bodies;
}

grid PMTest(vector<body>& bodies, double spacing, double width, double dt){
    vector<body>* bods = &bodies;
    grid g = grid(spacing, width);
    g.updateGrid(bods);
    g.solveField();
//    g.ctor(g.realPot);
    g.magF();
//    g.interp(bods);
//    bodiesUpdate(bods, dt);
    return g;
}

vector<body> PMTest1(vector<body>& bodies, double spacing, double width, double dt){
    vector<body>* bods = &bodies;
    grid g = grid(spacing, width);
    g.initActiveBods(bods);
    g.updateGrid(bods);
    g.solveField();
//    g.ctor(g.realPot);
//    g.magF();
    g.interp(bods);
    bodiesUpdate(bods, g.activeBods, dt);
    return bodies;
}

vector<body> TreePareticleMesh(vector<body>& bodies, double spacing, double width,
        double density, int numIter, double dt){
    tree_PM tpm = tree_PM(bodies, spacing, width, density, dt);
    for(int j=0; j<numIter; j++) {
        tpm.genSeeds();
        tpm.genSubGrids();
        tpm.classiftBods();
        tpm.runTrees();
    }
    return bodies;
}

PYBIND11_MODULE(treecode, m) {
    //m.def("test", &test);
    m.def("basicRun", &basicRun);
    m.def("fixedBoundary", &fixedBoundary);
    m.def("particleMesh", &particleMesh);
    m.def("PMTest", &PMTest);
    m.def("PMTest1", &PMTest1);
    m.def("TreePareticleMesh", &TreePareticleMesh);

    py::class_<barnesHut>(m, "barnesHut")
            .def(py::init<vector<body>&, vector<double>&, vector<double>&>());
    py::class_<body>(m, "body")
            .def(py::init<>())
            .def(py::init<double&, vector<double>&,
                    vector<double>&, vector<double>&>())

            .def_property("acc", &body::getAcc, &body::setAcc)
            .def_property("vel", &body::getVel, &body::setVel)
            .def_property("mass", &body::getMass, &body::setMass)
            .def_property("soft", &body::getSoftening, &body::setSoftening)
            .def_property("pos", &body::getPos, &body::setPos);

    py::class_<grid>(m, "grid", py::buffer_protocol())
            .def("getF", &grid::getF)
            .def_buffer([](grid &m) -> py::buffer_info {
                return py::buffer_info(
                        m.data(),                               /* Pointer to buffer */
                        sizeof(double),                          /* Size of one scalar */
                        py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                        3,                                      /* Number of dimensions */
                        { m.size()[0], m.size()[1], m.size()[2] },                 /* Buffer dimensions */
                        { sizeof(double) * m.size()[2]*m.size()[1], sizeof(double) * m.size()[2],             /* Strides (in bytes) for each index */
                          sizeof(double) });
            });
}

#ifndef PYINTERFACE
#define PYINTERFACE

#include "trees.h"

class progbar{
public:
    double max;
    double length;

    progbar(double max, double barLength);
    void update(int current);
};

vector<body> basicRun(vector<body>&, vector<double> centre, vector<double> dim, int numIter, double dt);

#endif //PYINTERFACE

#include <map>
#include <utility>
#include "poisson.h"
#include "vecMaths.h"
#include "bodies.h"
#include "trees.h"
#include "tpm.h"
#include "leapfrog.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
sg_seed::sg_seed(int key, vector<int> & index) : key(key) {
    indices.emplace_back(index);
    den = 0;
}

void sg_seed::maxMin(vector<int> & vec){
    for (int axis=0; axis<3; axis++){
        if (vec[axis] < min[axis])
            min[axis] = vec[axis];
        else if (vec[axis] > max[axis])
            max[axis] = vec[axis];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<int> sub_grid::getSubIndx(vector<int> index) {
    vector<int> out(3, 0);
    for (int axis=0; axis<3; axis++){
        out[axis] = index[axis] - min[axis] + int(numPts[axis]/3);
        if (out[axis] < int(numPts[axis]/3) || out[axis] >= max[axis] - min[axis] + int(numPts[axis]/3))
            out = {-1, -1, -1};
    }
    return out;
}

sub_grid::sub_grid(const grid& g, const sg_seed& seed, vector<int> pts, int timeStep) : grid(g.spacing, std::move(pts)){
    dt = timeStep;
    min = seed.min;
    max = seed.max;
    mainPoints = seed.indices;
}

sub_grid::sub_grid() = default;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

tree_PM::tree_PM(vector<body>& bods, double gridSpacing, double dimension, double density, double timeStep) :
        g(grid(gridSpacing, dimension)), cg(comp_grid(g)) {
    dt = timeStep;
    dim = {dimension, dimension, dimension};
    gridSpace = gridSpacing;
    den = density;
    bodies = &bods;

}

void tree_PM::genSeeds() {
    vector<int> keys;
    vector<vector<int>> indices;
    int count = 0;
    for (int i = 0; i < g.numPts[0]; i++) {
        for (int j = 0; j < g.numPts[1]; j++) {
            for (int k = 0; k < g.numPts[2]; k++) {
                g.keys[i*g.numPts[2]*g.numPts[1] + j*g.numPts[2] + k] = -1; /// Set Key value to -1
                if (g.realPot[int(i*g.numPts[2]*g.numPts[1] + j*g.numPts[2] + k)][0] > den){
                    /// Above density threshold
                    vector<int> index = {i, j, k}; /// Record index in main grid
                    indices.emplace_back(index);
                    keys.emplace_back(count); /// Generate new key
                    count ++;
                }
            }
        }
    }
    for (int i=0; i<keys.size(); i++){
        for (int j=0; j<keys.size(); j++) {
            bool adjacent = true; /// Assume adjacent
            for (int axis=0; axis<3; axis++){
                if (abs(indices[i][axis] - indices[j][axis]) > 1)
                    adjacent = false; /// if not adjacent in one dimension
            }
            if (adjacent){
                /// Set key values equal
                keys[j] = keys[i];
                g.keys[indices[j][0]*g.numPts[2]*g.numPts[1] + indices[j][1]*g.numPts[2] + indices[j][2]] = keys[i];
            }
        }
    }
    vector<sg_seed> seeds;
    /// Add indices (on main grid) to seeds based on key
    for (int i=0; i<keys.size(); i++){
        int seedIndx = -1;
        for (int s=0; s<seeds.size(); s++){
             if (seeds[s].key == keys[i])
                 seedIndx = s;
        }
        if (seedIndx < 0) {
            seeds.emplace_back(sg_seed(keys[i], indices[i]));
            seeds.back().min = indices[i];
            seeds.back().max = indices[i];
            seeds.back().den = g.realPot[int(indices[i][0]*g.numPts[2]*g.numPts[1] +
                                         indices[i][1]*g.numPts[2] +
                                         indices[i][2])][0];
        } else {
            seeds[seedIndx].indices.emplace_back(indices[i]);
            seeds[seedIndx].maxMin(indices[i]);
            if (seeds[seedIndx].den < g.realPot[int(indices[i][0]*g.numPts[2]*g.numPts[1] +
                                                    indices[i][1]*g.numPts[2] +
                                                    indices[i][2])][0])
                seeds[seedIndx].den = g.realPot[int(indices[i][0]*g.numPts[2]*g.numPts[1] +
                                                    indices[i][1]*g.numPts[2] +
                                                    indices[i][2])][0];
        }
    }
    seedVec = seeds;
}

void tree_PM::genSubGrids(){
    map<int, sub_grid> sgs;
    for (auto & seed : seedVec) {
        vector<int> points(3, 0);
        for (int axis = 0; axis < 3; axis++) {
            /// Compute tha buffering required for non periodicity
            points[axis] = 3 * (seed.max[axis] - seed.min[axis]);
        }
        int subTimeStep = dt / (den - seed.den + 1);
        sgs.insert(pair<int, sub_grid>(seed.key, sub_grid(g, seed, points, dt)));
    }
    sgVec = sgs;
}

void tree_PM::classiftBods() {
    for (int i=0; i<(*bodies).size(); i++){
        vector<vector<int>> pos = g.meshPos((*bodies)[i].pos.back());
        int currKey = -1;
        for (auto & po : pos){
            int k = g.keys[po[1]*g.numPts[2]*g.numPts[1] + po[1]*g.numPts[2] + po[1]];
            if (k == -1 || (k != currKey && currKey != -1))
                break;
            else
                currKey = k; /// If body is in a sub grid according to given mes pos
        }
        if (currKey >= 0)
            sgVec[currKey].activeBods.emplace_back(i); /// Adds body index to active bodies list
    }
}

void tree_PM::runTrees() {
    g.solveField();
    /// For efficiency this should be changed to the centre and size of the sub grids
    vector<double> width = g.dim;
    vector<double> centre = {0, 0, 0};
    for (auto & subG : sgVec) {
        barnesHut bh = barnesHut(*bodies, width, centre);
        for(int j=0; j<int(dt / subG.second.dt); j++) {
            subG.second.updateGrid(bodies);
            subG.second.solveField();
            cg.updateCompGrid(subG.second);
            treeMake(bh);
            interaction(bh);
            cg.interpW(bodies, false);
            bodiesUpdate(bodies, subG.second.activeBods, subG.second.dt);
            treeBreak(bh);
        }
        PBC(bodies, subG.second.activeBods, g.dim); /// Put particles into the correct location with PBCs
    }
    /// Update outside particles
    g.updateGrid(bodies);
    g.solveField();
    g.interpW(bodies, true);
    bodiesUpdate(bodies, g.activeBods, dt, g.dim);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

comp_grid::comp_grid(const grid &g) : grid(g), mainG(g) {}

void comp_grid::updateCompGrid(sub_grid & sg) {
    activeBods = sg.activeBods;
    for (int i = 0; i < numPts[0]; i++) {
        for (int j = 0; j < numPts[1]; j++) {
            for (int k = 0; k < numPts[2]; k++) {
                vector<int> sI = sg.getSubIndx({i, j, k});
                for (int axis=0; axis<3; axis++) {
                    if (sI[axis] > 0) {
                        realField[axis][int(i * numPts[2] * numPts[1] + j * numPts[2] + k)] =
                                mainG.realField[axis][int(i * numPts[2] * numPts[1] + j * numPts[2] + k)] -
                                sg.realField[axis][int(sI[0] * numPts[2] * numPts[1] + sI[1] * numPts[2] + sI[2])];
                    } else {
                        realField[axis][int(i * numPts[2] * numPts[1] + j * numPts[2] + k)] =
                                mainG.realField[axis][int(i * numPts[2] * numPts[1] + j * numPts[2] + k)];
                    }
                }
            }
        }
    }
}


#ifndef TPM
#define TPM

#include <fftw3.h>
#include <cmath>
#include <map>
#include "bodies.h"
#include "poisson.h"

using namespace std;

class sg_seed{
public:
    int key;
    double den;
    vector<vector<int>> indices;
    vector<int> min;
    vector<int> max;
    void maxMin(vector<int> & vec);
    sg_seed(int key, vector<int> & index);
};

class sub_grid : public grid{
public:
//    vector<int> padding;
    double dt{};
    vector<int> min;
    vector<int> max;
    vector<int> activeBods;
//    vector<vector<int>> subPoints;
    vector<vector<int>> mainPoints;
    vector<int> getSubIndx(vector<int> index);
    sub_grid(const grid& g, const sg_seed& seed, vector<int> pts, int timeStep);
    sub_grid();
};

class comp_grid : public grid{
public:
    grid mainG;
    void updateCompGrid(sub_grid & sg);
    explicit comp_grid(const grid& g);
};

class tree_PM{
public:
    double dt;
    double den;
    double gridSpace;
    vector<double> dim;
    vector<sg_seed> seedVec;
    map<int, sub_grid> sgVec;
    vector<body>* bodies;

    grid g;
    comp_grid cg;

    void genSeeds();
    void genSubGrids();
    void classiftBods();
    void runTrees();

    tree_PM(vector<body>& bods, double gridSpacing, double dim, double density, double timeStep);

};



#endif //TPM

#include <cmath>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <thread>
#include "vecMaths.h"
#include "trees.h"
#include "treeShow.h"

/// node constructors

node::node(){
    pos = vector<double>(3);
    width = vector<double>(3);
    centre = vector<double>(3);
}

// node constructors
node::node(vector<double> w, vector<double> &c){
    num = 0;
    width = w;
    centre = c;
    parent = nullptr;
}

node::node(node* tree){
    num = 0;
    parent = tree;
    pos = vector<double>(3);
    width = vector<double>(3);
    centre = {0, 0, 0};
    for(int i=0; i<3; i++){
        width[i] = tree->width[i]/2;
    }
}

node::node(node* tree, int chldIndx){
    num = 0;
    childIndx = chldIndx;
    parent = tree;
    pos = vector<double>(3);
    width = vector<double>(3);
    centre = {0, 0, 0};
    for(int i=0; i<3; i++){
        width[i] = tree->width[i]/2;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// barnesHut constructor

barnesHut::barnesHut(vector<body> &bods, vector<double> dim): bodies(&bods) {
    width = dim;
    centre = {dim[0]/2, dim[1]/2, dim[2]/2};
    root = new node(width, centre);
}

barnesHut::barnesHut(vector<body> &bods, vector<double> &dim, vector<double> &cent): bodies(&bods){
    width = dim;
    centre = cent;
    root = new node(width, centre);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// barnesHut functions

void barnesHut::initActiveBods(){
    vector<int> tmp;
    for (int i=0; i<(*bodies).size(); i++){
        tmp.emplace_back(i);
    }
    activeBods = tmp;
}

bool barnesHut::inNode(const vector<double>& pos, node* nod){
    vector<double> displacement = vecAdd(pos, scalMult(-1, nod->centre));
    for (int j=0; j<3; j++) {
        /// Check in node for each coordinate
        if (abs(displacement[j]) > nod->width[j]/2) {
            return false;
        } else if (abs(displacement[j]) == nod->width[j]){
            vector<double> nPos = vecAdd(pos, scalMult(nod->width[j]/10, pos));
            return inNode(nPos, nod);
        }
    }
	return true;
}

node* barnesHut::whichChild(node* tree, int i){
    node* child = nullptr;
    for(int j=0; j<8; j++) {
        /// For each child in parent
        child = tree->children[j];
        if (inNode((*bodies)[activeBods[i]].pos.back(), child)) {
            /// If in child j
            break;
        }
    }
    if (!child)
        cout << "not in child" << endl;
    return child;
}


/// Tree building functions
void barnesHut::treeBuild(){
    if(!root){
        root = new node(width, centre);
    }
//    vector<vector<int>> segments = segment(root, (*bodies));
//#pragma omp parallel for
    for(int i=0; i<int(activeBods.size()); i++){
        if(inNode((*bodies)[activeBods[i]].pos.back(), root) && (*bodies)[activeBods[i]].active.back()){
            treeInsert(root, i);
        }
    }
//    updateRoot();
}

/// Initialise children for parent node
void barnesHut::addChildren(node* tree){
    tree->children = vector<node*>(8);
    for(int i=0; i<8; i++){
        tree->children[i] = new node(tree, i);
    }
    int child = 0;
    vector<double> w = tree->width;
    for(int i=-1; i<=1; i=i+2){
        for(int j=-1; j<=1; j=j+2){
            for(int k=-1; k<=1; k=k+2){
                /// Setup centre of child
                vector<double> shift{i*w[0]/4, j*w[1]/4, k*w[2]/4};
                tree->children[child]->centre = vecAdd(tree->centre, shift);
                child += 1;
            }
        }
    }
}

/// Segment bodies according to octant !!! Contains legacy code; no activeBods !!!
vector<vector<int>> barnesHut::segment(node* nod, vector<body> bods){
    vector<vector<int>> out = {{}, {}, {}, {}, {}, {}, {}, {}};
    if (!bods.empty()) {
        if (nod->liveChildren.empty()) {
            addChildren(nod);
        }
        for (int i = 0; i < bods.size(); i++) {
            for (int j=0; j < nod->children.size(); j++){
                /// for each body check each child node
                if (inNode(bods[i].pos.back(), nod->children[j]) && bods[i].active[0])
                    out[j].emplace_back(i);
            }
        }
    }
/*
    cout << "segments: {";
    for (int i=0; i<nod->children.size(); i++){
        cout << "{";
        for (int j=0; j<out[i].size(); j++){
            cout << out[i][j];
            if (j<out[i].size()-1)
                cout << ", ";
        }
        cout << "}";
        if (i<nod->children.size()-1)
            cout << ", ";
    }
    cout << "}" << endl;
*/
    return out;
}

/// Recursive insertion of bodies into tree
void barnesHut::treeInsert(node* tree, int i){
    node* current = tree;
    while (current->num !=0){
        /// Update current pos and COM
        current->num += 1;
        current->pos = vecAdd(scalMult(current->mass, current->pos),
                              scalMult((*bodies)[activeBods[i]].mass.back(), (*bodies)[activeBods[i]].pos.back()));
        current->mass += (*bodies)[activeBods[i]].mass.back();
        current->pos = scalMult(1 / current->mass, current->pos);
        if (current->num == 2){
            /// Add children to saturated node
            addChildren(current);

            /// Move original body
            node* child = whichChild(current, current->bodyindx);
            current->liveChildren.emplace_back(child->childIndx);
            child->num += 1;
            child->bodyindx = current->bodyindx;
            child->pos = (*bodies)[current->bodyindx].pos.back();
    		child->mass = (*bodies)[current->bodyindx].mass.back();

    		/// Update current
            current = whichChild(current, i);
        } else if (current->num > 2) {
            /// Children already exist so update current
            current = whichChild(current, i);
        }
    }

    /// Fill in leaf data
    current->num += 1;
    current->bodyindx = activeBods[i];
    current->pos = (*bodies)[activeBods[i]].pos.back();
    current->mass = (*bodies)[activeBods[i]].mass.back();
    if (current->parent != nullptr) {
        current->parent->liveChildren.emplace_back(current->childIndx);
    }
}

/// Fill in root node
void barnesHut::updateRoot(){
    vector<double> cm = {0.,0.,0.};
    for (auto i : root->liveChildren){
        root->num += root->children[i]->num;
        root->mass += root->children[i]->mass;
        cm = vecAdd(cm, scalMult(root->children[i]->mass, root->children[i]->pos));
    }
    root->pos = scalMult(1/root->mass, cm);
}

/// Clear heap for next iteration
void barnesHut::treeChop(node* tree){
    if (tree->liveChildren.size() != 0){
        vector<int> children = tree->liveChildren;
        /// Deallocate memory
        for(int i : children){
            treeChop(tree->children[i]);
        }
    }
    if(tree->parent != nullptr){
        delete tree;
    } else{
        tree->liveChildren = {};
        tree->num = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Kinematic functions

void barnesHut::acceleration(node* tree){
    for(int i=0; i<int(activeBods.size()); i++){
        (*bodies)[activeBods[i]].acc.emplace_back(treeAcc(tree, i));
    }
}

/// Tree traversal for acceleration
vector<double> barnesHut::treeAcc(node* tree, int i){
	vector<double> f(3, 0);
	if (tree->num == 1 && tree->bodyindx != i){
	    /// Particle-particle interaction
		f = ngl((*bodies)[activeBods[i]].pos.back(), tree->pos, tree->mass,
		        (*bodies)[activeBods[i]].softening + (*bodies)[tree->bodyindx].softening);
	} else{
		double distance = m_modulus(vecAdd(scalMult(-1, (*bodies)[activeBods[i]].pos.back()), tree->pos), false);
		vector<double> w = tree->width;
		double minWidth = *min_element(w.begin(), w.end());
        /// Checking approximation...
		if (minWidth/distance < theta){
		    /// Bulk node computation
			f = ngl((*bodies)[activeBods[i]].pos.back(), tree->pos, tree->mass,
			        (*bodies)[activeBods[i]].softening + (*bodies)[tree->bodyindx].softening);
		} else{
		    vector<double> fs = {0, 0, 0};
            for(int j=0; j<tree->liveChildren.size(); j++){
                /// For each sub node
				f = vecAdd(f, treeAcc(tree->children[tree->liveChildren[j]], i));
			}
		}
	}
    return f;
}

/// Newtons Gravitational Law
vector<double> barnesHut::ngl(vector<double> &r1, vector<double> &r2, double mass, double softening){
	vector<double> out;
	vector<double> delta = vecAdd(scalMult(-1, r1), r2);
	out = scalMult(mass*G/pow(m_modulus(delta, false) + softening, 3), delta);
	return out;
}

#ifndef TREES
#define TREES

#include <vector>
#include <tuple>
#include "bodies.h"

using namespace std;

// node could be a group of particles or a single particle
struct node{
	double mass{};
	vector<double> pos;

	vector<double> width;
	vector<double> centre;

	int num{}; // number of particles in tree
	int childIndx{};
	vector<int> liveChildren;
	vector<node*> children;
	node* parent{};

	int bodyindx{};

	node(vector<double> width, vector<double>& centre);
	node(node* tree, int chldIndx);
	explicit node(node* root);
	node();
//	~node();
};

// Macros to retrieve node data; x is a pointer
#define Mass(x) (((node*) (x))->mass)
#define pos(x) (((node*) (x))->pos)

class barnesHut{
    node* whichChild(node* tree, int i);
    void addChildren(node*);
    bool inNode(const vector<double>&, node*);
    void treeInsert(node*, int);
public:
    vector<body>* bodies;
    vector<int> activeBods;
    node* root;
    double theta = 0.9;
    double G = 6.674e-11;
    vector<double> width;
    vector<double> centre;

    // Tree building functions
    void treeBuild();
    void treeChop(node*);
    vector<vector<int>> segment(node* root, vector<body> bodies);
    void updateRoot();

    // Kinematic functions
    void acceleration(node*);
    vector<double> treeAcc(node*, int);
    vector<double> ngl(vector<double>& r1, vector<double>& r2, double mass, double softening);
    void initActiveBods();

    explicit barnesHut(vector<body>& bods, vector<double> dim);
    explicit barnesHut(vector<body>& bods, vector<double>& dim, vector<double>& cent);
};

// Function definitions



#endif //TREES
        (*bodies)[bIndx].pos.back() = p;
//        cout << body.vel.back()[0] << endl;
    }
}

void bodiesUpdate(vector<body>* bodies, const vector<int>& activeBods, double dt){
    for (auto & bIndx : activeBods) {
        (*bodies)[bIndx].vel.emplace_back(vecAdd((*bodies)[bIndx].vel.back(),
                scalMult(dt, (*bodies)[bIndx].acc.back())));
        (*bodies)[bIndx].pos.emplace_back(vecAdd((*bodies)[bIndx].pos.back(),
                scalMult(dt, (*bodies)[bIndx].vel.back())));
    }
}

//void partialUpdate(node* tree, vector<body>* bodies, double dt){
//    if (tree->liveChildren.size() != 0){
//        for(auto i: tree->liveChildren){
//            partialUpdate(tree->children[i], bodies, dt);
//        }
//    } else{
//        auto i = tree->bodyindx;
//        (*bodies)[i].vel.emplace_back(vecAdd((*bodies)[i].vel.back(), scalMult(dt, (*bodies)[i].acc.back())));
//        (*bodies)[i].pos.emplace_back(vecAdd((*bodies)[i].pos.back(), scalMult(dt, (*bodies)[i].vel.back())));
//    }
//}

void boundaryInteract(barnesHut& bh, vector<body>& boundary){
    vector<body>* bodies = bh.bodies;
    for(auto body : *bodies){
        for(auto boundBod : boundary){
            vector<double> dist = vecAdd(body.pos.back(), scalMult(-1, boundBod.pos.back()));
            if (abs(dist[0]) > bh.width[0] && abs(dist[1]) > bh.width[1] && abs(dist[2]) > bh.width[2]) {
                body.acc.back() = vecAdd(body.acc.back(),
                                         bh.ngl(body.pos.back(),
                                                 boundBod.pos.back(), boundBod.mass.back(), 0));
            }
        }
    }
}

#include <iostream>
#include <algorithm>
//#include <windows.h>

#include "treeShow.h"
#include "trees.h"
#include "vecMaths.h"

using namespace std;

void addSpace(int space){
    for (int j=COUNT; j<(space); j++)
        cout << " ";
    cout << "|";
}

void printTree(node* nod, int space){
    // Increase distance between levels
    space += COUNT;

    if (nod->parent == nullptr){
//        if (nod->num==0)
//            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 8);
//        if (nod->num==1)
//            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 11);
//        if (nod->num>1)
//            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 13);
        addSpace(0); cout << INDENT << "root:" << endl;
        addSpace(0); cout << INDENTCLEAR << "~ num: " << nod->num << endl;
        addSpace(0); cout << INDENTCLEAR << "~ numChildren: " << nod->liveChildren.size() << endl;
        addSpace(0); cout << INDENTCLEAR << "~ width: "; printVec(nod->width); cout << endl;
        addSpace(0); cout << INDENTCLEAR << "~ centre: "; printVec(nod->centre); cout << endl;
        addSpace(0); cout << INDENTCLEAR << "~ COM: "; printVec(nod->pos); cout << endl;
    } else{
        vector<int> v = nod->parent->liveChildren;
        if (find(v.begin(), v.end(), nod->childIndx) != v.end()) {
//            if (nod->num==1)
//                SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 11);
//            if (nod->num!=1)
//                SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 13);
            addSpace(space); cout << INDENT << nod->childIndx << ": num: " << nod->num << endl;
            addSpace(space); cout << INDENTCLEAR << "~ body: " << nod->bodyindx << endl;
            addSpace(space); cout << INDENTCLEAR << "~ numChildren: " << nod->liveChildren.size() << endl;
            addSpace(space); cout << INDENTCLEAR << "~ width: "; printVec(nod->width); cout << endl;
            addSpace(space); cout << INDENTCLEAR << "~ centre: "; printVec(nod->centre); cout << endl;
            addSpace(space); cout << INDENTCLEAR << "~ COM: "; printVec(nod->pos); cout << endl;
        }
    }

    if (nod->num) {
        for (int i = 0; i < nod->children.size(); i++) {
            vector<int> v = nod->liveChildren;
            if (find(v.begin(), v.end(), i) != v.end()) {
                printTree(nod->children[i], space);
            } else {
//                SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 8);
                addSpace(space + COUNT);
                cout << endl;
                addSpace(space + COUNT);
                cout << INDENT << i << ": #" << endl;
            }
        }
    }
//    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 15);

}

#ifndef TREESHOW
#define TREESHOW

#include "trees.h"

#define COUNT 10
#define INDENT "--- "
#define INDENTCLEAR "    "

void printTree(node* nod, int space);

#endif //TREESHOW

#include <cmath>
#include <vector>
#include <iostream>

using namespace std;

double dot(vector<double>& v1, vector<double>& v2, bool square){
	double out = 0;
	for(int i=0; i<3; i++){
		out += v1[i]*v2[i];
	}
	if(!square){
		return pow(out, 0.5);
	} else{
		return out;
	}
}

double m_modulus(vector<double> vec, bool square){
    double out = dot(vec, vec, square);
    return out;
}

vector<double> vecAdd(vector<double> v1, vector<double> v2){
	vector<double> out(3);
	for(int i=0; i<3; i++){
		out[i] = v1[i] + v2[i];
	}
	return out;
}

vector<int> vecAdd(vector<int> v1, vector<int> v2){
	vector<int> out(3);
	for(int i=0; i<3; i++){
		out[i] = v1[i] + v2[i];
	}
	return out;
}

vector<double> scalMult(double scal, vector<double> v){
	vector<double> out(30);
	for(int i=0; i<3; i++){
		out[i] = scal*v[i];
	}
	return out;
}

vector<double> compMult(vector<double> v1, vector<double> v2) {
    double real = v1[0]*v2[0] - v1[1]*v2[1];
    double imag = v1[1]*v2[0] + v1[0]*v2[1];
    return {real, imag};
}


void printVec(vector<double> vec){
    cout << "{";
    for(int i=0; i<3; i++){
        cout << vec[i];
        if(i != 2) {cout << ", ";}
    }
    cout << "}";
}

#ifndef VECMATHS
#define VECMATHS

#include <vector>

using namespace std;

double m_modulus(vector<double>, bool);
double dot(vector<double>&, vector<double>&, bool);
vector<double> vecAdd(vector<double>, vector<double>);
vector<int> vecAdd(vector<int>, vector<int>);
vector<double> scalMult(double, vector<double>);
vector<double> compMult(vector<double>, vector<double>);
void printVec(vector<double> vec);

#endif //VECMATHS
