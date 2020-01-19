# %%

# imports
import os
import numpy as np
import scipy.stats as st
import scipy.signal as ss
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

# %%

# Test data
txt_Data = "sample.txt"
xv = np.linspace(0, 2 * np.pi, 500)
yv = 5 * np.sin(xv)
ye = np.ones(500) / 100
z = np.matrix([xv, yv, ye]).T
np.savetxt(txt_Data, z, delimiter=',')


# %%

def loadDat(txt_Data):
    '''Read data from .dat file [time, volume]'''
    # Parameters
    mass_error = 0.01  # g
    p = 998.2336  # Density of water
    p_error = 0.0001  # Error in density

    # load data file
    dat = np.loadtxt(txt_Data)

    error = np.ones(len(dat))  # Generate array of error values
    data = np.zeros([len(dat), 3])  # Initialise output data array
    # Set up data array [t, v=m/p, v uncertainty]
    for i in range(len(dat)):
        error = (dat[i, 1] / (p * 10 ** 3)) * np.sqrt((mass_error / dat[i, 1]) ** 2 + (p_error / p) ** 2)
        data[i, 0] = dat[i, 0] / 1000
        data[i, 1] = dat[i, 1] / (p * 10 ** 3)
        data[i, 2] = error
    return data


# %%

# defines model object
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

    def harmonic(self, arr_X, arr_Vals):
        '''a*sin(b*x)+d*cos(e*x) model input x and values'''
        s = arr_Vals[0] * np.sin(arr_Vals[1] * arr_X)
        c = arr_Vals[2] * np.cos(arr_Vals[3] * arr_X)
        return s + c

    def nlogn(selfself, arr_x, arr_Vals):
        out = arr_Vals[0] + arr_x*np.log(arr_x)
        return out

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
            self.initVals(self, 2)
            self.f = self.nlogn



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