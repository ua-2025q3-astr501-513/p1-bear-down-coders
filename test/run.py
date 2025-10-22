import sys
import os
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src','project'))
sys.path.append(module_path)

from orbit_integrator import orbit_integrator
import numpy as np

mu = 1.0
h = 0.01
N = 100000
x0 = [1.0, 0.0, 0.0]
v0 = [0.0, 1.0, 1.0]

methods = ["leapfrog", "rk4", "yoshida4"]
for m in methods:
    sim = orbit_integrator(mu=mu, h=h, method=m)
    sim.integrate(x0, v0, N)
    sim.plot_orbit(show_energy=False)


