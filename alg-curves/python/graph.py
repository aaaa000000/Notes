import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.computed_zorder = True

theta = np.linspace(0, 2*np.pi, 60)
r = np.linspace(0, 4, 40)
R, Theta = np.meshgrid(r, theta)

X = R * np.cos(Theta)
Y = R * np.sin(Theta)
Z = R
ax.plot_surface(X, Y, Z, color='blue', alpha=0.4)
ax.plot_surface(X, Y, -Z, color='blue', alpha=0.4)
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.set_zlim(-4,4)
y = np.linspace(-4, 4, 30)
z = np.linspace(-4, 4, 30)
Xp, Zp = np.meshgrid(y, z)
Yp = np.ones_like(Xp)*1.0
ax.plot_surface(Xp, Yp, Zp, alpha=0.2)
ax.set_box_aspect((8,8,8))
ax.view_init(elev=20, azim=45)
ax.set_axis_off()

x_curve = np.linspace(-4, 4, 600)
z_curve = np.sqrt(1 + x_curve**2)
y_curve = np.ones_like(x_curve) * 1.0
ax.plot(x_curve, y_curve, z_curve, color='red', linewidth=2, alpha=1.0)
ax.plot(x_curve, y_curve, -z_curve, color='red', linewidth=2, alpha=1.0)
plt.tight_layout()
ax.text(1.0, -1.0, 0.0, r"$U_3:Z=1$", fontsize=14)
plt.savefig("圓錐截線_雙曲線")
