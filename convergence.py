import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
from matplotlib import rc
from scipy.optimize import curve_fit
import matplotlib.animation as animation
import treecode as tree
import scipy.stats as st
import scipy.signal as ss
import scipy.optimize as opt
import pickle as pkl
import lfEngine
import numba
import anim
import time

G = 6.674e-11
kb = 1.38064852e-23


class models:
    """Containes model functions and params (with initialiser)"""

    def __init__(self):
        self.float_Chi2 = 0

    ##########Define models here###########
    def polynomial(self, arr_X, arr_Vals):
        '''Polynumial model input x and values'''
        float_Y = 0.0
        for i in range(self.int_N):
            float_Y += arr_Vals[i] * arr_X ** i
        return float_Y

    def exp(self, arr_X, arr_Vals):
        '''Polynumial model input x and values'''
        float_Y = arr_Vals[0]*np.exp(-arr_Vals[1]*arr_X) + arr_Vals[2]
        return float_Y

    def harmonic(self, arr_X, arr_Vals):
        '''a*sin(b*x)+d*cos(e*x) model input x and values'''
        s = arr_Vals[0] * np.sin(arr_Vals[1] * arr_X)
        c = arr_Vals[2] * np.cos(arr_Vals[3] * arr_X)
        return s + c

    #######################################

    def initVals(self, int_N):
        '''Initialiser for the parameters to be optimised for the model'''
        self.arr_Vals = np.random.rand(int_N)
        self.float_UnVal = np.random.rand(int_N)

    def setMod(self, mod, params):
        '''Initialises the users chosem model type'''
        if mod == 0:
            # Polynomial model with params[0] number of parameters
            self.initVals(self, params[0])
            self.int_N = params[0]  # Degree of polynomial
            self.f = self.polynomial
        if mod == 1:
            self.initVals(self, 4)
            self.f = self.harmonic
        if mod == 2:
            self.initVals(self, 3)
            self.f = self.exp


# %%

def Chi2(arr_Vals, model, data):
    '''Chi^2 calculation'''
    float_Top = data[:, 1] - model.f(model, data[:, 0], arr_Vals)
    float_Chi = np.sum(np.power((float_Top / data[:, 2]), 2))
    return float_Chi


def Chi2a1(float_Val, model, data, fixIndx):
    '''Chi^2 modification function allowing all but one variable
    to be manipulated by the optimiser used for Step 1 of algorithm
    from lab session 2'''
    # Fills in the arr_Valls with the fixed parameter and all the free ones
    arr_Vals = model.arr_Vals
    arr_Vals[fixIndx] = float_Val  # Replaces the value in arr_vals with the fixed value
    Chi = Chi2(arr_Vals, model, data)
    # Is optimised when Chi^2 calculated here approaches minimised Chi^2 + 1
    return abs(Chi - model.float_Chi2 - 1)


def Chi3(arr_Val, model, data, freeIndx):
    '''Chi^2 modification function allowing only one variable
    to be manipulated by the optimiser used for Step 2 of algorithm
    from lab session 2'''
    arr_Vals = model.arr_Vals
    arr_Vals[freeIndx] = arr_Val  # Replaces the value in arr_vals with the free values
    Chi = Chi2(arr_Vals, model, data)
    # Is optimised when Chi^2 is minimised
    return Chi


# %%

def UncIndv(model, data, fixIndx):
    '''Performs iterative chi^2 uncertainty calculation'''
    arr_Vals = np.copy(model.arr_Vals)  # initialise value array
    # Free index = [1,2,...,n-1,n+1,...m] where fixIndex = n
    freeIndx = np.delete(np.arange(0, len(arr_Vals), 1), fixIndx)
    for i in range(10):
        # Step 1
        # Optimisation function
        dict_Unc1 = opt.minimize(Chi2a1,
                                 arr_Vals[fixIndx],
                                 method='nelder-mead',
                                 args=(model, data, fixIndx))

        # gets output from the optimisation
        arr_Vals[fixIndx] = dict_Unc1.x
        # Step 2
        # Optimisation function
        dict_Unc2 = opt.minimize(Chi3,
                                 arr_Vals[freeIndx],
                                 method='nelder-mead',
                                 args=(model, data, freeIndx))

        # gets output from the optimisation
        arr_Vals[freeIndx] = dict_Unc1.x

    # Perform Step 1 again to end on odd number
    # Optimisation function
    dict_Unc1 = opt.minimize(Chi2a1,
                             arr_Vals[fixIndx],
                             method='nelder-mead',
                             args=(model, data, fixIndx))
    return dict_Unc1.x


# %%

def Uncertainty(model, data):
    '''Uncertainty is calculated (iteratively0 using the method from lab session 2'''
    uncert = np.copy(model.arr_Vals)  # Prevents overriding the memory
    for i in range(len(uncert)):
        uncert[i] -= UncIndv(model, data, i)
    return abs(uncert)


# %%

def Resid(arr_Vals, model, data):
    '''Calculates the residuals given a model with model params and data'''
    resid = np.ones(len(data))
    resid = (data[:, 1] - model.f(model, data[:, 0], arr_Vals)) / data[:, 2]
    return resid


# %%

def linResid(arr_Vals, data):
    resid = np.ones(len(data[0]))
    resid = (data[1] - [arr_Vals[0]] * len(data[0]) - arr_Vals[1] * data[0])
    return resid


# %%

def DW(residuals):
    '''Calculates the Durbin_Watson statistic given an array of residuals'''
    num = 0
    for i in range(1, len(residuals)):
        num += (residuals[i] - residuals[i - 1]) ** 2
    den = np.sum(residuals ** 2)
    return num / den


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


def getRad(x, y):
    return np.sqrt(x**2 + y**2)


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


if __name__ == "__main__":
    nIter = 100
    NOrbits = 0.1
    time = NOrbits * 9.59e10  # Orbital period in s
    dt = time / nIter #15  # 1e8 #1e2 #1e2
    n_iter = 100000

    m_1 = 1e20
    m_2 = 1e20
    r_1_2 = 14.6e9
    dim = 1e27
    N = 1000

    mpl.style.use('default')
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    arr_bodies = two_body_init(m_1, m_2, r_1_2)
    arr_bodies = lfEngine.half_step(arr_bodies, dt)

    _arr_bodies = np.array([])
    for body in arr_bodies:
        # print(body.r)
        _arr_bodies = np.append(_arr_bodies,
                                tree.body(np.real(body.m), np.real(body.r), np.real(body.v), [0] * 3))

    arrCent = np.array([0, 0, 0])

    nIter = np.linspace(200, 1000, 100)
    NOrbits = 10
    time = np.array([NOrbits * 9.59e10]*nIter.shape[0])  # Orbital period in s
    dt = time / nIter

    average = []

    for i in range(nIter.shape[0]):
        _arr_bodies = np.array([])
        for body in arr_bodies:
            # print(body.r)
            _arr_bodies = np.append(_arr_bodies,
                                    tree.body(np.real(body.m), np.real(body.r), np.real(body.v), [0] * 3))
        b = tree.basicRun(_arr_bodies, arrCent, [4e10]*3, int(nIter[i]), dt[i])

        arrB = np.empty((len(b), np.shape(np.array(b[0].pos))[0],
                         np.shape(np.array(b[0].pos))[1]))
        rads = np.empty((np.shape(np.array(b[0].pos))[0]))
        times = np.linspace(0, time, nIter[i]+1)
        for i in range(len(b)):
            arrB[i, :, :] = np.array(b[i].pos)
        rads = np.vectorize(getRad)(arrB[0, :, 0], arrB[i, :, 1])#
        avrgSqr = np.sqrt(np.mean((rads - r_1_2/2)**2))
        average.append((avrgSqr))

        # plt.plot(arrB[0, :, 0], arrB[0, :, 1])
        # plt.plot(arrB[1, :, 0], arrB[1, :, 1])
        # plt.plot(times, rads)
    # plt.plot(nIter, np.array(average))
    # popt, pcov = curve_fit(func, nIter, np.array(average)/1e9)

    _arr_bodies = np.array([])
    nIter1 = 1000
    NOrbits = 0.4
    time = NOrbits * 9.59e10  # Orbital period in s
    dt = time / nIter1
    for body in arr_bodies:
        # print(body.r)
        _arr_bodies = np.append(_arr_bodies,
                                tree.body(np.real(body.m), np.real(body.r), np.real(body.v), [0] * 3))
    b = tree.basicRun(_arr_bodies, arrCent, [4e10] * 3, int(nIter1), dt)

    arrB = np.empty((len(b), np.shape(np.array(b[0].pos))[0],
                     np.shape(np.array(b[0].pos))[1]))
    for i in range(len(b)):
        arrB[i, :, :] = np.array(b[i].pos)
    # obj_Model = models
    # obj_Model.setMod(obj_Model, 2, [])
    # data = np.array([nIter/1000, np.array(average)/1e9, 0.0001*np.ones_like(nIter)])
    # data = np.reshape(data, (100, 3))
    # # Optimise model
    # dict_Result = opt.minimize(Chi2,
    #                            obj_Model.arr_Vals,
    #                            method='nelder-mead',
    #                            args=(obj_Model, data))
    #
    # # Storing model data
    # obj_Model.arr_Vals = dict_Result.x
    # obj_Model.float_Chi2 = dict_Result.fun
    # obj_Model.float_UnVal = Uncertainty(obj_Model, data)
    #
    # # Output results
    # print("Chi^2: ", obj_Model.float_Chi2)
    # print("DOF: ", len(data) - len(obj_Model.arr_Vals))
    # print("Reduced Chi^2: ", obj_Model.float_Chi2 / (len(data) - len(obj_Model.arr_Vals)))
    # print("parameters:")
    # for i in range(len(obj_Model.arr_Vals)):
    #     print("    ", obj_Model.arr_Vals[i], "+-", obj_Model.float_UnVal[i])

    ax = plt.subplot(111)
    ax1_inset = ax.inset_axes([0.5, 0.5, 0.5, 0.5])
    ax.scatter(nIter, np.array(average)/1e9, c="k", s=1)

    ax1_inset.plot(arrB[0, :, 0]*1e-9, arrB[0, :, 1]*1e-9, "-", c="k")
    ax1_inset.plot(arrB[1, :, 0]*1e-9, arrB[1, :, 1]*1e-9, "--", c="k")
    ax1_inset.scatter(arrB[0, -1, 0]*1e-9, arrB[0, -1, 1]*1e-9, c="k")
    ax1_inset.scatter(arrB[1, -1, 0]*1e-9, arrB[1, -1, 1]*1e-9, c="k")

    # ax.plot(nIter/1000, obj_Model.f(obj_Model, nIter, obj_Model.arr_Vals), c="k")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("RMS Error $/ Gm$")
    ax1_inset.set_xlabel("$x / Gm$")
    ax1_inset.set_ylabel("$y/ Gm$")
    # plt.show()
    plt.savefig("..//Diagrams//Conv.png", dpi=300, bbox_inches='tight')
