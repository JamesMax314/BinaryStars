import treecode as tree
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import DataGen_V1
from mpl_toolkits.mplot3d import Axes3D
import anim
import numpy as np
import main
import pickle
import PS as ps
import PSpec
from matplotlib import rc
from scipy.optimize import curve_fit

def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B



if __name__ == "__main__":
    file = ".//FinalData//CDM_1_meta.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, gridSpacing, numIter, numParticles, t0, t100, numTrees, avrgM, avrgR) = pickle.load(infile)
    infile.close()

    print(1)


