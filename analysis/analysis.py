from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm, eig
from numpy import dot

def read_csv():
    data = np.genfromtxt('data/front.csv', delimiter='\t')
    return data

def plot3d(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def plot2d(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = np.arange(len(x))
    ax.plot(t, x, label="X")
    ax.plot(t, y, label="Y")
    ax.plot(t, z, label="Z")
    plt.legend()
    plt.show()

def plot2dx(data):
    x = data[:, 0]
    y = data[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()

def pct(data):
    data = data - np.mean(data, axis=0)
    V = np.cov(data.T)
    values, vectors = eig(V)
    idx = values.argsort()[::-1]
    values = values[idx]
    vectors = vectors[:, idx]
    result = vectors.T.dot(data.T).T
    return result


# plot3d(read_csv())
# plot2d(read_csv())
# plot3d(pct(read_csv()))
# plot2d(pct(read_csv()))
plot2dx(pct(read_csv()))
