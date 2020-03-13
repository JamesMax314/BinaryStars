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
from sklearn.cluster import KMeans
from matplotlib import rc
import scipy

if __name__ == "__main__":
    file = ".//FinalData//CDM_1.pkl"
    infile = open(file, 'rb')
    (arrB, mass, uniDim, gridSpacing, numIter, numParticles, t0, t100, numTrees, avrgM, avrgR) = pickle.load(infile)
    infile.close()

    point = 999

    pts = arrB[:, point, :]

    nGPts = 100
    dim = 100*50*DataGen_V1.MPc  # in m
    dVol = (dim / nGPts)**3
    dRho = (DataGen_V1.MPc**3) * mass / dVol
    grid = ps.genGrid(arrB[:, point, :], nGPts, dim)
    col = PSpec.den(arrB, grid, dim, point) * dRho
    numClust = 60
    results = KMeans(n_clusters=numClust).fit(arrB[:, -1, :])
    clusters = results.cluster_centers_
    labels = results.labels_
    unique, counts = np.unique(labels, return_counts=True)

    scale = 1 / (DataGen_V1.MPc)
    xs = clusters[:, 0] * scale
    ys = clusters[:, 1] * scale
    zs = clusters[:, 2] * scale

    avd = np.mean(col)

    rads = np.empty(numClust)
    newCounts = np.empty(numClust)
    for i in range(numClust):
        indices = np.argwhere(labels == i)
        densities = col[indices]
        densities = np.reshape(densities, (densities.shape[0]))
        points = pts[indices]
        points = np.reshape(points, (points.shape[0], 3))
        maxDen = np.max(densities)
        inFWHM = np.argwhere(densities >= maxDen/2)
        inFWHM = np.reshape(inFWHM, (inFWHM.shape[0]))
        dists = scipy.spatial.distance.cdist(points[inFWHM], points[inFWHM], metric='euclidean')
        diam = np.max(dists)
        rad = diam/2
        newCounts[i] = inFWHM.shape[0]
        rads[i] = rad
    avRad = np.mean(rads)
    error = np.std(rads) / np.sqrt(numClust)
    print("radius:", avRad, "+-", error)
    avrgMass = np.mean(newCounts) * mass
    errorMass = np.std(newCounts) * mass / np.sqrt(numClust)

    print(avrgMass, "+-", errorMass)

    mpl.style.use('default')

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scat = ax.scatter(xs*1e-3, ys*1e-3, zs*1e-3, s=1)
    # cbaxes = fig.add_axes([0.3, .87, 0.5, 0.03])
    # cbar = fig.colorbar(scat, cax=cbaxes, orientation='horizontal')
    # cbar.ax.xaxis.set_ticks_position('top')
    # cbar.ax.xaxis.set_label_position('top')
    # cbar.set_label("$\sqrt{\\rho}$ / $k$gm$^{-3}$")
    #
    # # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.w_xaxis.line.set_color("black")
    ax.w_yaxis.line.set_color("black")
    ax.w_zaxis.line.set_color("black")
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.zaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')  # only affects
    ax.tick_params(axis='y', colors='black')  # tick labels
    ax.tick_params(axis='z', colors='black')
    ax.xaxis._axinfo['tick']['color'] = 'k'
    ax.yaxis._axinfo['tick']['color'] = 'k'
    ax.zaxis._axinfo['tick']['color'] = 'k'

    ax.set_xlabel('$x$ / $Gpc$')
    ax.set_ylabel('$y$ / $Gpc$')
    ax.set_zlabel('$z$ / $Gpc$')

    # plt.savefig("..//Diagrams//CDM3D11.png", dpi=100, bbox_inches='tight')

    plt.show()