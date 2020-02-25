import pickle
import numpy as np
import matplotlib.pyplot as plt

def load(file):
    infile = open(file, 'rb')
    out = pickle.load(infile)
    infile.close()
    return out

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
    mesh = np.empty([mSize, mSize, mSize])
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
    out = 1
    for i in range(3):
        dist = abs(body[i] - vecPos(pos[0], pos[1], pos[2], spacing, dim)[i])
    if (dist < spacing):
        out = out * abs((spacing - dist) / spacing)
    else:
        return 0
    return out


def den(bodies, mesh, dim, point):
    colours = np.empty(bodies.shape[0])
    spacing = dim / mesh.shape[0]
    meshPos = np.empty(3, dtype=int)
    for i in range(bodies.shape[0]):
        for axis in range(3):
            meshPos[axis] = (bodies[i, point, axis] + dim / 2) // spacing
        colours[i] = mesh[meshPos[0], meshPos[1], meshPos[2]] #/np.max(mesh)
    return colours


if __name__ == "__main__":
    data = load("nTest.pkl")
    point = 2000
    mesh = toMesh(data, 25, 1e27, point)
    fft = np.fft.fftn(mesh)
    ffts = np.fft.fftshift(fft)
    avr = np.average(ffts, axis=1)
    # lin = np.average(avr, axis=0)
    # plt.plot(np.abs(lin))
    # plt.show()
    # plt.imshow(np.abs(avr))
    # plt.show()

    col = den(data, mesh, 1e27, point)

    plt.scatter(data[:, point, 0]/(3.086e16*1e9), data
    [:, point, 1]/(3.086e16*1e9), s=1, c=col, cmap="plasma")
    plt.colorbar(label="Density / particles number per $1.728$ $G$Pc$^3$")
    plt.xlabel("distance / $G$Pc")
    plt.ylabel("distance / $G$Pc")
    plt.savefig("../Diagrams/colUni.png", bbox="tight", dpi=400)
    plt.show()
