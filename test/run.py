from orbit_integrator import orbit_integrator

mu = 1.0
h = 0.01
N = 100000
x0 = [1.0, 1.0, 1.0]
v0 = [0.0, 1.0, 0.0]

methods = ["leapfrog", "rk4", "yoshida4"]
for m in methods:
    sim = orbit_integrator(mu=mu, h=h, method=m)
    sim.integrate(x0, v0, N)
    sim.plot_orbit(show_energy=True)
