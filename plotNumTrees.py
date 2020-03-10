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
    file = ".//FinalData//CDM_1.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, gridSpacing, numIter, numParticles, t0, t100, numTrees, avrgM, avrgR) = pickle.load(infile)
    infile.close()
    file = ".//FinalData//WDM_1.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, gridSpacing, numIter, numParticles, t0, t100, numTrees1, avrgM1, avrgR1) = pickle.load(infile)
    infile.close()

    times = np.linspace(t100, t0, numTrees.shape[0])

    numSamples = 20
    avrgs1 = np.empty([int(numTrees1.shape[0]//numSamples)])
    stds1 = np.empty([int(numTrees1.shape[0]//numSamples)])
    avrgsT1 = np.empty([int(numTrees1.shape[0] // numSamples)])

    for i in range(0, int(numTrees.shape[0]//numSamples)):
        subArrT1 = times[i * numSamples:i * numSamples + numSamples]
        # std = np.std(subArr)
        avrT1 = np.mean(subArrT1)
        avrgsT1[i] = avrT1
        subArr1 = numTrees1[i*numSamples:i*numSamples+numSamples]
        popt1, pcov1 = curve_fit(f, subArrT1, subArr1)
        std1 = (f(subArrT1, popt1[0], popt1[1]) - subArr1)**2  # np.std(subArr)
        std1 = np.sqrt(np.mean(std1))
        avr = np.mean(subArr1)
        avrgs1[i] = avr
        stds1[i] = std1 / np.sqrt(numSamples)

    avrgs = np.empty([int(numTrees.shape[0] // numSamples)])
    stds = np.empty([int(numTrees.shape[0] // numSamples)])
    avrgsT = np.empty([int(numTrees.shape[0] // numSamples)])

    for i in range(0, int(numTrees.shape[0] // numSamples)):
        subArrT = times[i * numSamples:i * numSamples + numSamples]
        # std = np.std(subArr)
        avrT = np.mean(subArrT)
        avrgsT[i] = avrT
        subArr = numTrees[i * numSamples:i * numSamples + numSamples]
        popt, pcov = curve_fit(f, subArrT, subArr)
        std = (f(subArrT, popt[0], popt[1]) - subArr) ** 2  # np.std(subArr)
        std = np.sqrt(np.mean(std))
        avr = np.mean(subArr)
        avrgs[i] = avr
        stds[i] = std / np.sqrt(numSamples)

    point = 99
    dt = (10 - 100) / numIter
    scale = 1 / (DataGen_V1.MPc)
    xs = arrB[:, point, 0] * scale
    ys = arrB[:, point, 1] * scale
    zs = arrB[:, point, 2] * scale

    mpl.style.use('default')

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    avrgR = 6.18e21*(avrgR+np.array([1.]*len(avrgR)))/uniDim
    avrgR1 = 6.18e21*(avrgR1+np.array([1.]*len(avrgR1)))/uniDim


    ax = plt.subplot(111)
    ax.errorbar(avrgsT/(1e9*3600*24*365), avrgs, stds*20, linestyle='None', marker="o", ms=3, c="k")
    ax.errorbar(avrgsT/(1e9*3600*24*365), avrgs1, stds*20, linestyle='None', fmt='o', mfc='white', ms=3, c="k")
    # ax.plot(times/(1e9*3600*24*365), numTrees1, c="k")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([3, np.max(times/(1e9*3600*24*365))])
    ax.set_xlabel("$t / Gyr$")
    ax.set_ylabel("Number of high density regions")
    # plt.show()
    plt.savefig("..//Diagrams//numTrees.png", dpi=300, bbox_inches='tight')