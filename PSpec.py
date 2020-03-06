""" Power Spectrum generator for dark matter simulation
 1: Generate equal mass points on a cartesian grid;
 2: Displace positions and velocities according to the Zeldovich approximation:
    x = q + D(t)psi(q); v = a dD/dt psi"""

import treecode as tree
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import numpy as np
import os
import camb
from camb import model, initialpower
from matplotlib import rc


# print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

MPc = 3.09e22

class progBar:
    def __init__(self, num, numPoints):
        self.num = num
        self.numPts = numPoints

    def update(self, n):
        frac = n/self.num
        nHash = int(self.numPts*frac)
        nBar = self.numPts-nHash
        print(" " * (100 + self.numPts), end="\r")
        print("Progress: " + nHash*"#" + nBar*"_" + "|", str(int(frac*100)), "%", end="\r")

    def close(self):
        print("")


def multPs(pts, spacing, PS, Rf, z):
    out = np.empty_like(pts)
    kxs = np.fft.fftfreq(pts.shape[0], spacing)
    kys = np.fft.fftfreq(pts.shape[1], spacing)
    kzs = np.fft.fftfreq(pts.shape[2], spacing)
    for i in range(pts.shape[0]):
        for j in range(pts.shape[1]):
            for k in range(pts.shape[2]):
                kx = kxs[i]
                ky = kys[j]
                kz = kzs[k]
                kt = np.sqrt(kx**2 + ky**2 + kz**2)*2*np.pi
                if kt == 0:
                    p = 0
                else:
                    p = np.sqrt(PS.P(z, kt * MPc)) * np.exp(-kt*Rf*MPc/2 - (kt*Rf*MPc)**2/2)
                out[i, j, k] = np.sqrt(p)*pts[i, j, k]
    return out


def genPts(PS, N, spacing, rhoBar, Rf, z):
    """Takes an input power spectrum and computes the real space over density field on a grid"""
    # noise = np.random.normal(0, 1, N**3)
    noise_3D = np.random.normal(0, 1, size=[N, N, N])
    # noise_3D = np.ones([N, N, N])
    F_noise_3D = np.fft.fftn(noise_3D)
    F_delta = multPs(F_noise_3D, spacing, PS, Rf, z)
    delta = np.abs(np.fft.ifftn(F_delta))
    mass = delta*spacing**3*rhoBar + rhoBar

    avrg = np.mean(mass)
    overDen = (mass - avrg) / (avrg)
    fGrid = np.fft.fftn(overDen)
    freq = np.fft.fftfreq(100, d=10 / 100)
    fGrid_shift = np.abs(np.fft.fftshift(fGrid) * np.conj(np.fft.fftshift(fGrid)))
    x, y, z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
    R = np.sqrt((x - N / 2) ** 2 + (y - N / 2) ** 2 + (z - N / 2) ** 2)

    f = lambda r: fGrid_shift[(R >= r - 0.5) & (R < r + 0.5)].mean()
    r = np.linspace(1, N, N)
    mean = np.vectorize(f)(r)
    freq = np.linspace(2*np.pi/(N*spacing), 2*np.pi/spacing, mean.shape[0])
    plt.loglog(freq * MPc, mean)
    plt.show()

    # plt.plot(noise)
    # plt.show()
    return mass

########################################################################################################################
# Particle in a cell interpolation based monte-carlo
def vecPos(i, j, k, spacing, dim):
    return np.array([i*spacing - dim/2, j*spacing - dim/2, k*spacing - dim/2])


def gmeshPos(pos, spacing, dim, numPts):
    meshPoints = []
    pt = np.zeros(3)
    for i in range(3):
        pt[i] = (pos[i] + dim/2) // spacing
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if pt[0]+i<numPts and pt[1]+j<numPts and pt[2]+k<numPts and pt[0]+i>=0 and pt[1]+j>=0 and pt[2]+k>=0:
                    meshPoints.append(pt + np.array([i, j, k]))
    return np.array(meshPoints, dtype=int)


def toMesh(bodies, mSize, dim, point):
    mesh = np.zeros([mSize, mSize, mSize])
    spacing = dim / mSize
    meshPos = np.empty(3, dtype=int)
    for i in range(bodies.shape[0]):
        mPts = gmeshPos(bodies[i, point, :], spacing, dim, mSize)
        # for axis in range(3):
        #     meshPos[axis] = int((bodies[i, point, axis] + dim / 2) / spacing)
        for j in range(len(mPts)):
            mesh[mPts[j, 0], mPts[j, 1], mPts[j, 2]] += w(bodies[i, point, :], mPts[j], spacing, dim)
    return mesh


def w(body, pos, spacing, dim):
    out_ = 1
    for i in range(3):
        dist = abs(body[i] - vecPos(pos[0], pos[1], pos[2], spacing, dim)[i])
        if dist < spacing:
            out_ = out_ * abs((spacing - dist) / spacing)
        else:
            return 0
    return out_


def den(bodies, mesh, dim, point):
    colours = np.empty(bodies.shape[0])
    spacing = dim / mesh.shape[0]
    meshPos = np.empty(3, dtype=int)
    for i in range(bodies.shape[0]):
        for axis in range(3):
            meshPos[axis] = (bodies[i, point, axis] + dim / 2) // spacing
        colours[i] = mesh[meshPos[0], meshPos[1], meshPos[2]] #/np.max(mesh)
    return colours


def monteCarlo(dim, spacing, numPts, gridMass):
    """Samples the density field using monte-carlo simulation"""
    points = np.empty((numPts, 3))
    massPts = np.empty(numPts)
    count = 0
    N = int(gridMass.shape[0])
    # gridMass = np.random.normal(0, 1, [int(gridMass.shape[0])])
    normalize = np.max(gridMass) - np.min(gridMass)
    prog = progBar(numPts, 20)
    while count < numPts:
        rx = np.random.uniform(-dim/2, dim/2)
        ry = np.random.uniform(-dim/2, dim/2)
        rz = np.random.uniform(-dim/2, dim/2)
        pt = np.array([rx, ry, rz])
        meshPoints = gmeshPos(pt, spacing, dim, N)  # array of array of indices
        den = 0
        for i in range(len(meshPoints)):
            p = meshPoints[i]
            den += w(pt, meshPoints[i], spacing, dim)*gridMass[p[0], p[1], p[2]]
        randP = np.random.uniform(0, 1)
        prob = (den-np.min(gridMass))/normalize
        if (prob > randP):
            prog.update(count)
            points[count] = pt
            massPts[count] = den
            count += 1
    prog.close()
    return points, massPts
########################################################################################################################


def genMasses(N, numParticles, dmDen, uniDim, Rf, z=100):
    # For calculating large-scale structure and lensing results yourself, get a power spectrum
    # interpolation object. In this example we calculate the CMB lensing potential power
    # spectrum using the Limber approximation, using PK=camb.get_matter_power_interpolator() function.
    # calling PK(z, k) will then get power spectrum at any k and redshift z in range.

    nz = 100  # number of steps to use for the radial/redshift integration
    kmax = 1e2  # kmax to use
    # First set up parameters as usual
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)

    # For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
    # so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
    results = camb.get_background(pars)
    chistar = results.conformal_time(0) - results.tau_maxvis
    chis = np.linspace(0, chistar, nz)
    zs = results.redshift_at_comoving_radial_distance(chis)
    # Calculate array of delta_chi, and drop first and last points where things go singular
    zs = zs[1:-1]

    # Get the matter power spectrum interpolation object (based on RectBivariateSpline).
    # Here for lensing we want the power spectrum of the Weyl potential.
    PK = camb.get_matter_power_interpolator(pars, nonlinear=True,
                                            hubble_units=False, k_hunit=False, kmax=kmax,
                                            var1=model.Transfer_Weyl, var2=model.Transfer_Weyl, zmax=zs[-1])

    # Have a look at interpolated power spectrum results for a range of redshifts
    # Expect linear potentials to decay a bit when Lambda becomes important, and change from non-linear growth
    k = np.exp(np.log(10) * np.linspace(-4, 2, 200))

    spacing = uniDim / (N)
    # spec = PK.P(z, k)
    masses = genPts(PK, N, spacing, dmDen, Rf, z)
    # masses = np.empty([N, N, N])
    # N1 = masses.shape[0]
    # for i in range(0, N1):
    #     for j in range(0, N1):
    #         for k1 in range(0, N1):
    #             pos = ((i-N1/2)**2 + (j-N1/2)**2 + (k1-N1/2)**2)**1/2
    #             masses[i, j, k1] = np.exp(-1/2*(pos/400)**2)
    # masses = np.reshape(masses, N**3)
    # tmpPts = np.linspace(-uniDim/2, uniDim/2, spacing)
    # mps = np.meshgrid(tmpPts, tmpPts, tmpPts)
    # pts = np.empty((N**3, 3))
    # for i in range(N):
    #     for j in range(N):
    #         for k in range(N):
    #             pts[N**2*i + N*j + k] = np.array([spacing*i, spacing*j, spacing*k]) - uniDim/2

    points, massPts = monteCarlo(uniDim, spacing, numParticles, masses)
    return points, massPts

if __name__ == "__main__":
    # out = genMasses(20, 10000, 1, 1000)
    # For calculating large-scale structure and lensing results yourself, get a power spectrum
    # interpolation object. In this example we calculate the CMB lensing potential power
    # spectrum using the Limber approximation, using PK=camb.get_matter_power_interpolator() function.
    # calling PK(z, k) will then get power spectrum at any k and redshift z in range.

    nz = 100  # number of steps to use for the radial/redshift integration
    kmax = 100  # kmax to use
    # First set up parameters as usual
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)

    # For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
    # so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
    results = camb.get_background(pars)
    chistar = results.conformal_time(0) - results.tau_maxvis
    chis = np.linspace(0, chistar, nz)
    zs = results.redshift_at_comoving_radial_distance(chis)
    # Calculate array of delta_chi, and drop first and last points where things go singular
    dchis = (chis[2:] - chis[:-2]) / 2
    chis = chis[1:-1]
    zs = zs[1:-1]

    # Get the matter power spectrum interpolation object (based on RectBivariateSpline).
    # Here for lensing we want the power spectrum of the Weyl potential.
    PK = camb.get_matter_power_interpolator(pars, nonlinear=True,
                                            hubble_units=False, k_hunit=False, kmax=kmax,
                                            var1=model.Transfer_Weyl, var2=model.Transfer_Weyl, zmax=zs[-1])

    # Have a look at interpolated power spectrum results for a range of redshifts
    # Expect linear potentials to decay a bit when Lambda becomes important, and change from non-linear growth
    # plt.figure(figsize=(8, 5))
    k = np.exp(np.log(10) * np.linspace(-8, 3, 300))
    zplot = [100]
    Rf = 0.2
    mpl.style.use('default')

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)

    ax = plt.subplot(111)

    # for z in zplot:
    ax.loglog(k, PK.P(zplot[0], k) * np.exp(-k * Rf / 2 - ((k * Rf) ** 2) / 2), "--", c="k")
    ax.loglog(k, PK.P(zplot[0], k), "-", c="k")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    lower = 2*np.pi/10
    upper = 2*np.pi/(10/100)
    kmin = 1e-4
    ax.axvline(x=lower, ymin=kmin, ymax=kmax, linestyle="-.", c="k", lw=0.9)
    ax.axvline(x=upper, ymin=kmin, ymax=kmax, linestyle="-.", c="k", lw=0.9)


    ax.set_xlim([kmin, kmax])
    ax.set_ylim([1e-18, 1e-10])
    ax.set_xlabel("$k / Mpc^{-1}$")
    ax.set_ylabel("$P / Mpc^{-3}$")
    # plt.legend(['z=%s' % z for z in zplot])
    # plt.show()
    plt.savefig("..//Diagrams//InitSpec.png", dpi=300, bbox_inches='tight')

    # spec = PK.P(100, k)
    # genPts(spec, 20, 2, 1000, 0, 100)