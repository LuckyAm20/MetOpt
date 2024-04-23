import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-3,3,100)
y = np.linspace(-3,3,100)
z = np.linspace(-3,3,100)

X,Y = np.meshgrid(x,y)

eq1 = X**2 + Y**2 -2
eq2 = Y**2 - 2
eq3 = X + Y - 1/(2*np.sqrt(2)) - 1/np.sqrt(2)

Z_eq1 = np.sqrt(2 - X**2 - Y**2)
Z_eq2 = np.sqrt(3 - Y**2)
Z_eq3 = eq3

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z_eq1, cmap='Blues_r', alpha=0.5)
ax.plot_surface(X, Y, -Z_eq1, cmap='plasma', alpha=0.5)
ax.plot_surface(X, Y, Z_eq2, cmap='inferno', alpha=0.5)
ax.plot_surface(X, Y, Z_eq3, cmap='magma', alpha=0.5)

ax.scatter(-1/(2*np.sqrt(2)), -1/(2*np.sqrt(2)), 0, color='red')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.view_init(elev=20, azim=45)
plt.show()








