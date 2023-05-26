import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate a set of rays through the origin
k = np.linspace(0, 10, num=1000)
x = k*np.random.rand(1000)
y = k*np.random.rand(1000)
z = k*np.random.rand(1000)

# Plot the rays in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=0.1)

# Generate a set of planes passing through the origin
normal1 = np.random.rand(3)
normal2 = np.random.rand(3)
d = 0
xx, yy = np.meshgrid(range(-10, 10), range(-10, 10))
z1 = (-normal1[0] * xx - normal1[1] * yy) * 1. / normal1[2]
z2 = (-normal2[0] * xx - normal2[1] * yy) * 1. / normal2[2]

# Plot the planes in 3D space
ax.plot_surface(xx, yy, z1, color='blue', alpha=0.2)
ax.plot_surface(xx, yy, z2, color='green', alpha=0.2)

# Set axis labels and show the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()