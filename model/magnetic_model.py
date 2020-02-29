from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from numpy import dot

def read_csv():
    data = np.genfromtxt('data/front.csv', delimiter='\t')
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    return x, y, z


def plot(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def magnetic_field(m, r):
    """m: magnetic dipole moment; r: relative position of sensor"""
    if any(r):
        # print(m, r)
        mu0 = 1.2566e-6
        coefficient = mu0 / (4*np.pi)
        vector = (3 * r * dot(m, r) - m * (norm(r) ** 2)) / (norm(r) ** 5)
        return coefficient * vector
    else:
        return np.array([np.nan, np.nan, np.nan])


def synthesize():
    magnet_position = np.array([0, -1, 0])
    magnetic_moment = np.array([0, -1, 0]) * 0.1  # Field pointing front
    rotation_axis   = np.array([1, 0, 0])  # Rotating about left/right axis
    sensor_position = np.array([0, -3, 0]) # Sensor in front of magnet

    angle = np.linspace(0, 2*np.pi)
    field = np.zeros([len(angle), 3])

    print(magnetic_field(magnetic_moment, sensor_position-magnet_position))

    for i, a in enumerate(angle):
        rot = Rotation.from_rotvec(rotation_axis * a)

        moment = rot.apply(magnetic_moment)
        position = sensor_position - rot.apply(magnet_position)
        field[i, :] = magnetic_field(moment, position)

    x = field[:, 0]
    y = field[:, 1]
    z = field[:, 2]
    return x, y, z



# plot(*read_csv())
# print(synthesize())
plot(*synthesize())

# n = 51
# x = np.linspace(-2, 2, n)
# z = np.linspace(-2, 2, n)
#
# moment = np.array([0, 0, 1]) * 1
#
# field_x = np.zeros([len(x), len(z)])
# field_z = np.zeros([len(x), len(z)])
# for i in range(len(x)):
#     for j in range(len(z)):
#         pos = np.array([x[i], 0, z[j]])
#         field = magnetic_field(moment, pos)
#         field_x[i, j] = field[0]
#         field_z[i, j] = field[2]
#
# plt.quiver(x, z, field_x, field_z)
# plt.streamplot(x, z, field_x, field_z)
# plt.show()
