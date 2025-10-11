import numpy as np
import matplotlib.pyplot as plt
import integrate as I   # your module that now includes yoshida4_step
import pdb

mu = 1.0
h = 0.001
N = 1000000

# initial conditions
x0 = np.array([1.0, 1.0, 1.0])
v0 = np.array([0.0, 1.0, 0.0])

# initialize lists
verlet_pos = [x0.copy()]
verlet_vel = [v0.copy()]
rk4_pos    = [x0.copy()]
rk4_vel    = [v0.copy()]
yosh_pos   = [x0.copy()]
yosh_vel   = [v0.copy()]

for n in range(N):
    # Verlet
    x_prev, v_prev = verlet_pos[-1], verlet_vel[-1]
    x_new, v_new = I.verlet_step(x_prev, v_prev, h, mu)
    verlet_pos.append(x_new)
    verlet_vel.append(v_new)

    # RK4
    x_prev, v_prev = rk4_pos[-1], rk4_vel[-1]
    x_new, v_new = I.rk4_step(x_prev, v_prev, h, mu)
    rk4_pos.append(x_new)
    rk4_vel.append(v_new)

    # Yoshida 4
    x_prev, v_prev = yosh_pos[-1], yosh_vel[-1]
    x_new, v_new = I.yoshida_step(x_prev, v_prev, h, mu)
    yosh_pos.append(x_new)
    yosh_vel.append(v_new)

# convert to arrays for plotting
verlet_pos = np.array(verlet_pos)
rk4_pos = np.array(rk4_pos)
yosh_pos = np.array(yosh_pos)
verlet_vel = np.array(verlet_vel)
rk4_vel = np.array(rk4_vel)
yosh_vel = np.array(yosh_vel)

# pdb.set_trace()
# plot
plt.plot(verlet_pos[:,0], verlet_pos[:,1], label="Verlet", alpha=0.8)
plt.plot(rk4_pos[:,0], rk4_pos[:,1], '--', label="RK4", alpha=0.8)
plt.plot(yosh_pos[:,0], yosh_pos[:,1], ':', label="Yoshida4", alpha=0.8)
plt.axis('equal')
plt.legend()
plt.title("Numerical Integration Comparison")
plt.xlabel('x position')
plt.ylabel('y position')
plt.show()

# plot
plt.plot(verlet_pos[:,0], verlet_vel[:,0], label="Verlet", alpha=0.8)
plt.plot(rk4_pos[:,0], rk4_vel[:,0], '--', label="RK4", alpha=0.8)
plt.plot(yosh_pos[:,0], yosh_vel[:,0], ':', label="Yoshida4", alpha=0.8)
plt.axis('equal')
plt.legend()
plt.title("Numerical Integration Comparison")
plt.xlabel('x position')
plt.ylabel('x velocity')
plt.show()

# plot
verlet_r = (verlet_pos[:,0]**2 + verlet_pos[:,1]**2 + verlet_pos[:,2]**2)**(0.5)
verlet_E = 0.5 * ( verlet_vel[:,0]**2 + verlet_vel[:,1]**2 + verlet_vel[:,2]**2 )**(0.5)

rk4_r = (rk4_pos[:,0]**2 + rk4_pos[:,1]**2 + rk4_pos[:,2]**2)**(0.5)
rk4_E = 0.5 * ( rk4_vel[:,0]**2 + rk4_vel[:,1]**2 + rk4_vel[:,2]**2 )**(0.5)

yosh_r = (yosh_pos[:,0]**2 + yosh_pos[:,1]**2 + yosh_pos[:,2]**2)**(0.5)
yosh_E = 0.5 * ( yosh_vel[:,0]**2 + yosh_vel[:,1]**2 + yosh_vel[:,2]**2 )**(0.5)

# pdb.set_trace()
plt.plot(verlet_r-1, verlet_E-0.5, label="Verlet", alpha=0.8)
plt.plot(rk4_r-1, rk4_E-0.5, '--', label="RK4", alpha=0.8)
plt.plot(yosh_r-1, yosh_E-0.5, ':', label="Yoshida4", alpha=0.8)
plt.axis('equal')
plt.legend()
plt.title("Numerical Integration Comparison")
plt.xlabel('position')
plt.ylabel('energy')
plt.show()
