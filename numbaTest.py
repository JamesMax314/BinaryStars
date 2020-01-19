import numba
import numpy as np

@numba.jit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(len(arr[0,:])):
        for j in range(N):
            result += arr[i,j]
    return result

arr = np.ones([100, 100])
print(sum2d(arr))

