""" Power Spectrum generator for dark matter simulation
 1: Generate equal mass points on a cartesian grid;
 2: Displace positions and velocities according to the Zeldovich approximation:
    x = q + D(t)psi(q); v = a dD/dt psi"""

import treecode as tree
import matplotlib.pyplot as plt
import numpy as np
import os
import camb
from camb import model, initialpower
# print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

def multPs(pts, spacing, PS, pSpacing):
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
                kt = np.sqrt(kx**2 + ky**2 + kz**2)
                p = PS[int(kt/pSpacing)]
                out[i, j, k] = np.sqrt(p)*pts[i, j, k]
    return out


def genPts(PS, N, spacing, rhoBar):
    """Takes an input power spectrum and computes the real space over density field on a grid"""
    noise = np.random.normal(0, 1, N**3)
    noise_3D = np.reshape(noise, (N, N, N))
    F_noise_3D = np.fft.fftn(noise_3D)
    F_delta = multPs(F_noise_3D, 10, PS, 10)
    delta = np.abs(np.fft.ifftn(F_delta))
    mass = delta*spacing**3*rhoBar + rhoBar
    # plt.plot(noise)
    # plt.show()
    return mass

def genMasses(N, dmDen, uniDim, z=100):
    # For calculating large-scale structure and lensing results yourself, get a power spectrum
    # interpolation object. In this example we calculate the CMB lensing potential power
    # spectrum using the Limber approximation, using PK=camb.get_matter_power_interpolator() function.
    # calling PK(z, k) will then get power spectrum at any k and redshift z in range.

    nz = 100  # number of steps to use for the radial/redshift integration
    kmax = 10  # kmax to use
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

    spacing = uniDim / (N**3)
    spec = PK.P(z, k)
    masses = genPts(spec, N, spacing, dmDen)
    masses = np.reshape(masses, N**3)
    # tmpPts = np.linspace(-uniDim/2, uniDim/2, spacing)
    # mps = np.meshgrid(tmpPts, tmpPts, tmpPts)
    pts = np.empty((N**3, 3))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                pts[N**2*i + N*j + k] = np.array([spacing*i, spacing*j, spacing*k]) - uniDim/2
    return pts, masses

if __name__ == "__main__":
    out = genMasses(20, 1, 1000)
    # For calculating large-scale structure and lensing results yourself, get a power spectrum
    # interpolation object. In this example we calculate the CMB lensing potential power
    # spectrum using the Limber approximation, using PK=camb.get_matter_power_interpolator() function.
    # calling PK(z, k) will then get power spectrum at any k and redshift z in range.

    nz = 100  # number of steps to use for the radial/redshift integration
    kmax = 10  # kmax to use
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
    k = np.exp(np.log(10) * np.linspace(-4, 2, 200))
    zplot = [100]
    for z in zplot:
        plt.loglog(k, PK.P(z, k))
    plt.xlim([1e-4, kmax])
    plt.xlabel('k Mpc')
    plt.ylabel('$P_\Psi\, Mpc^{-3}$')
    plt.legend(['z=%s' % z for z in zplot])
    plt.show()

    spec = PK.P(100, k)
    genPts(spec, 20, 2, 1000)