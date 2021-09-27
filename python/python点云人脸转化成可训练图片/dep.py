import math
import numpy as np

def rret(data,size_x,size_y):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    x= np.array(x)

    max_x = max(x)
    min_x = min(x)
    max_y = max(y)
    min_y = min(y)
    min_z = min(z)
    max_z = max(z)
    range_x = max_x - min_x
    range_y = max_y - min_y
    ret = np.zeros((size_x, size_y))
    ret[:,:] = min_z
    len =x.shape[0]
    for i in range(len) :
        X = math.floor((x[i] - min_x) / range_x * (size_x - 1))
        Y = math.floor((y[i] - min_y) / range_y * (size_y - 1))
        ret[X,Y] = max(ret[X, Y], z[i])
        ret[X,Y]=max(ret[X,Y],min_z)
        min_z = min(min_z, ret[X, Y])

    range_z = max_z - min_z
    # ret = max(ret, min_z)

    ret = np.round((ret - min_z) / range_z * 255)
    return ret