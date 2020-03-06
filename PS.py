import numpy as np
import numba
import PSpec

def genGrid(pos, nPts, dim):
    spacing = dim / nPts
    grid = np.zeros((nPts, nPts, nPts))
    for i in range(len(pos)):
        meshPt = PSpec.gmeshPos(pos[i], spacing, dim, nPts)
        for j in range(meshPt.shape[0]):
            grid[meshPt[j][0], meshPt[j][1], meshPt[j][2]] += PSpec.w(pos[i], meshPt[j], spacing, dim)
    return grid


def computePS(pos, dim, nGPts):
    #grid = np.random.normal(0, 1, size=[100, 100, 100])
    grid = genGrid(pos, nGPts, dim)
    avrg = np.mean(grid)
    overDen = (grid - avrg) / (avrg)
    fGrid = np.fft.fftn(overDen)
    freq = np.fft.fftfreq(nGPts, d=dim/nGPts)
    fGrid_shift = np.abs(np.fft.fftshift(fGrid)*np.conj(np.fft.fftshift(fGrid)))
    x, y, z = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), np.arange(grid.shape[2]))
    R = np.sqrt((x-nGPts/2)**2 + (y-nGPts/2)**2 + (z-nGPts/2)**2)

    f = lambda r : fGrid_shift[(R >= r-0.5) & (R < r+0.5)].mean()
    r = np.linspace(1, nGPts, nGPts)
    mean = np.vectorize(f)(r)

    return mean, freq


if __name__ == "__main__":
    print(1)
