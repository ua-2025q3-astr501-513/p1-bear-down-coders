import numpy as np
import matplotlib.pyplot as plt

class orbit_integrator:
    """
    Integrator for a particle moving in a central 1/r^2 potential.
    Supports Verlet, RK4, and Yoshida 4th-order symplectic schemes.
    """

    def __init__(self, mu=1.0, h=0.01, method="verlet"):
        """
        Parameters
        ----------
        mu : float
            Gravitational parameter (G*M).
        h : float
            Time step size.
        method : str
            Integration method: 'verlet', 'rk4', or 'yoshida4'.
        """
        self.mu = mu
        self.h = h
        self.method = method.lower()
        if self.method not in ["verlet", "rk4", "yoshida4"]:
            raise ValueError("method must be 'verlet', 'rk4', or 'yoshida4'")

        # storage
        self.positions = []
        self.velocities = []
        self.energies = []

    # -------------------------
    # Core acceleration function
    # -------------------------
    def accel(self, x):
        r2 = np.dot(x, x)
        r = np.sqrt(r2)
        if r == 0:
            raise ValueError("Singular position: r = 0")
        return -self.mu * x / (r2 * r)

    # -------------------------
    # Integration methods
    # -------------------------
    def verlet_step(self, x, v):
        a = self.accel(x)
        v_half = v + 0.5 * self.h * a
        x_new = x + self.h * v_half
        a_new = self.accel(x_new)
        v_new = v_half + 0.5 * self.h * a_new
        return x_new, v_new

    def rk4_step(self, x, v):
        h = self.h
        mu = self.mu

        k1x = v
        k1v = self.accel(x)

        k2x = v + 0.5 * h * k1v
        k2v = self.accel(x + 0.5 * h * k1x)

        k3x = v + 0.5 * h * k2v
        k3v = self.accel(x + 0.5 * h * k2x)

        k4x = v + h * k3v
        k4v = self.accel(x + h * k3x)

        x_new = x + (h / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        v_new = v + (h / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        return x_new, v_new

    def yoshida4_step(self, x, v):
        h = self.h
        mu = self.mu

        w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
        w0 = - (2.0**(1.0/3.0)) / (2.0 - 2.0**(1.0/3.0))
        c = np.zeros(5)
        d = c
        c[0] = w1/2
        c[1] = (w0+w1)/2
        c[2] = (w0+w1)/2
        c[3] = w1/2
        d[0] = w1
        d[1] = w0
        d[2] = w1
        d[3] = 0.0

        for i in range(4):
            v = v + d[i] * h * self.accel(x)
            x = x + c[i] * h * v
        return x, v

    # -------------------------
    # Energy computation
    # -------------------------
    def energy(self, x, v):
        return 0.5 * np.dot(v, v) + self.mu / np.dot(x, x)

    # -------------------------
    # Main integration routine
    # -------------------------
    def integrate(self, x0, v0, Nsteps):
        """
        Integrate for Nsteps starting from x0, v0.
        """
        x = np.array(x0, dtype=float)
        v = np.array(v0, dtype=float)

        self.positions = [x.copy()]
        self.velocities = [v.copy()]
        self.energies = [self.energy(x, v)]

        for _ in range(Nsteps):
            if self.method == "verlet":
                x, v = self.verlet_step(x, v)
            elif self.method == "rk4":
                x, v = self.rk4_step(x, v)
            elif self.method == "yoshida4":
                x, v = self.yoshida4_step(x, v)

            self.positions.append(x.copy())
            self.velocities.append(v.copy())
            self.energies.append(self.energy(x, v))

        self.positions = np.array(self.positions)
        self.velocities = np.array(self.velocities)
        self.energies = np.array(self.energies)
        return self.positions, self.velocities

    # -------------------------
    # Plotting helpers
    # -------------------------
    def plot_orbit(self, show_energy=False):
        pos = np.array(self.positions)
        plt.plot(pos[:,0], pos[:,1], label=self.method)
        plt.axis('equal')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title(f"Orbit using {self.method.upper()} Integrator")
        plt.show()

        if show_energy:
            plt.plot(self.energies)
            plt.xlabel("Step")
            plt.ylabel("Energy")
            plt.title(f"Energy Conservation: {self.method.upper()}")
            plt.show()
# Example script
if __name__ == "__main__":
    mu = 1.0
    h = 0.01
    N = 10000
    x0 = [1.0, 0.0, 0.0]
    v0 = [0.0, 1.0, 0.0]

    methods = ["verlet", "rk4", "yoshida4"]
    for m in methods:
        sim = orbit_integrator(mu=mu, h=h, method=m)
        sim.integrate(x0, v0, N)
        sim.plot_orbit(show_energy=True)