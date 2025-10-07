import numpy as np

def accel(x, mu):
    r2 = np.dot(x, x)
    r = np.sqrt(r2)
    if r == 0:
        raise ValueError("singular position r=0")
    return -mu * x / (r2 * r)   # -mu * x / r^3

def verlet_step(x, v, h, mu=1.0):

    a = accel(x, mu)
    v_half = v + 0.5 * h * a
    x_new = x + h * v_half

    a_new = accel(x_new, mu)
    v_new = v_half + 0.5 * h * a_new

    return x_new, v_new

def rk4_step(x, v, h, mu=1.0):
    # k1
    k1x = v
    k1v = accel(x, mu)

    # k2
    k2x = v + 0.5*h*k1v
    k2v = accel(x + 0.5*h*k1x, mu)

    # k3
    k3x = v + 0.5*h*k2v
    k3v = accel(x + 0.5*h*k2x, mu)

    # k4
    k4x = v + h*k3v
    k4v = accel(x + h*k3x, mu)

    # Update
    x_new = x + (h/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    v_new = v + (h/6.0)*(k1v + 2*k2v + 2*k3v + k4v)

    return x_new, v_new

def yoshida_step(x, v, h, mu=1.0):
    # Yoshida coefficients
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
        # kick
        v = v + d[i] * h * accel(x, mu)
        # drift
        x = x + c[i] * h * v
    return x, v
