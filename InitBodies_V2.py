import treecode as tree
import matplotlib.pyplot as plt
# import anim
import numpy as np
import pickle
import os
import PSpec

# constants
pi = np.pi
H0 = 67
MPc = 3.09e22  # MPc = 3.09e22 m

def get_time(z):
    t = 2*2.2e-18 ** (-1) / (1 + (1+z)**2)
    return t

def a(t):
    return aConst*t**(2/3)

def ad(t):
    return aConst*2/3*t**(-1/3)

def softening(R, N):
    softening = 1.1 * N**(-0.28) * R


# Parameters
Rf = [0, 0.1, 0.2]  # 0.2 is too low mass based on Lyman alpha forrest
aConst = 1.066e-12  # scale factor coefficient
rho100 = 2.78e-21  # initial DM density in real units / kgm^-3
t100 = get_time(100)  # 1.88e13  # start of sim at z=100 / s
t0 = get_time(0)  # now time / s
numInitParticles = int((100**3)**(1/3))  # Number of initialisation particles (grid points)
numParticles = 50000  # Number of particles in the simulation
uniDim = 10*MPc  # Universe length in m
numPts = 50  # grid points per dimension for PM method
gridSpacing = uniDim / numPts  # PM grid spacing
numIter = 1000  # Number of iterations
dt = (t0 - t100)/numIter  # time step

if __name__ == "__main__":
    print("Initialising over-density perturbations...")
    pts, masses = PSpec.genMasses(numInitParticles, numParticles, rho100, uniDim, Rf[1])
    mass = rho100 * uniDim**3 / numParticles
    pts = pts / a(t100)  # convert to CC

    file = ".//InitBods//WDM1.pkl"
    outfile = open(file, 'wb')
    pickle.dump((pts, mass), outfile)
    outfile.close()
