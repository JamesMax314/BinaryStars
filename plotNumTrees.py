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
    file = ".//FinalData//CDM_1_50.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, gridSpacing, numIter, numParticles, t0, t100, numTrees3, avrgM3, avrgR3) = pickle.load(infile)
    infile.close()
    file = ".//FinalData//WDM_1.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, gridSpacing, numIter, numParticles, t0, t100, numTrees1, avrgM1, avrgR1) = pickle.load(infile)
    infile.close()
    file = ".//FinalData//WDM_1_smallClusters.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, gridSpacing, numIter, numParticles, t0, t100, numTrees2, avrgM2, avrgR2) = pickle.load(infile)
    infile.close()

    # Subtract off high density nnumber:
    numTrees1 = numTrees1 - numTrees2  # numTrees1 at >30; numTrees2 at >50
    numTrees = numTrees - numTrees3  # numTrees3 at >50; numTrees at >30

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

    avrgs2 = np.empty([int(numTrees2.shape[0] // numSamples)])
    stds2 = np.empty([int(numTrees2.shape[0] // numSamples)])
    avrgsT2 = np.empty([int(numTrees2.shape[0] // numSamples)])

    for i in range(0, int(numTrees2.shape[0] // numSamples)):
        subArrT2 = times[i * numSamples:i * numSamples + numSamples]
        # std = np.std(subArr)
        avrT2 = np.mean(subArrT2)
        avrgsT2[i] = avrT2
        subArr2 = numTrees2[i * numSamples:i * numSamples + numSamples]
        popt2, pcov2 = curve_fit(f, subArrT2, subArr2)
        std2 = (f(subArrT2, popt2[0], popt2[1]) - subArr2) ** 2  # np.std(subArr)
        std2 = np.sqrt(np.mean(std2))
        avr2 = np.mean(subArr2)
        avrgs2[i] = avr2
        stds2[i] = std2 / np.sqrt(numSamples)

    avrgs3 = np.empty([int(numTrees3.shape[0] // numSamples)])
    stds3 = np.empty([int(numTrees3.shape[0] // numSamples)])
    avrgsT3 = np.empty([int(numTrees3.shape[0] // numSamples)])

    for i in range(0, int(numTrees3.shape[0] // numSamples)):
        subArrT3 = times[i * numSamples:i * numSamples + numSamples]
        # std = np.std(subArr)
        avrT3 = np.mean(subArrT3)
        avrgsT3[i] = avrT3
        subArr3 = numTrees3[i * numSamples:i * numSamples + numSamples]
        popt3, pcov3 = curve_fit(f, subArrT3, subArr3)
        std3 = (f(subArrT3, popt3[0], popt3[1]) - subArr3) ** 2  # np.std(subArr)
        std3 = np.sqrt(np.mean(std3))
        avr3 = np.mean(subArr3)
        avrgs3[i] = avr3
        stds3[i] = std3 / np.sqrt(numSamples)

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
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size': 8,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    width = 3.28162  # in inches
    fig = plt.figure(figsize=(width, width*0.85))

    avrgR = 6.18e21*(avrgR+np.array([1.]*len(avrgR)))/uniDim
    avrgR1 = 6.18e21*(avrgR1+np.array([1.]*len(avrgR1)))/uniDim


    ax = plt.subplot(111)
    ax.errorbar(avrgsT/(1e9*3600*24*365), avrgs, stds*20, linestyle='None', marker="v", ms=3, c="k")
    ax.errorbar(avrgsT/(1e9*3600*24*365), avrgs3, stds3*20, linestyle='None', marker="^", ms=3, c="k")
    ax.errorbar(avrgsT/(1e9*3600*24*365), avrgs2, stds2*20, linestyle='None', marker="^", mfc='white', ms=3, c="k")
    ax.errorbar(avrgsT/(1e9*3600*24*365), avrgs1, stds1*20, linestyle='None', fmt='v', mfc='white', ms=3, c="k")
    # ax.plot(times/(1e9*3600*24*365), numTrees3, c="k")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([3, np.max(times/(1e9*3600*24*365))])
    ax.set_xlabel("$t / G$yr")
    ax.set_ylabel("Number of high density regions")
    # plt.show()
    plt.savefig("..//Diagrams//numTrees50.pgf", bbox_inches='tight', pad_inches=.1)