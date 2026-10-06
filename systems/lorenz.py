"""
Lorenz system — smooth 3-D flow, contract v2 example.

    x' = sigma (y - x)
    y' = x (rho - z) - y
    z' = x y - beta z

Sweep rho for the classic route to chaos (onset near rho ~ 24.74 at the
default sigma, beta); record z maxima (record_component=2) for the Lorenz map.
"""

from numba import cuda, float64

DIM = 3
N_PARAMS = 3

PARAMS = [
    dict(name="rho", default=28.0, min=0.0, max=250.0, sweepable=True, sweep=(20.0, 200.0)),
    dict(name="sigma", default=10.0, min=0.1, max=30.0, sweepable=True),
    dict(name="beta", default=8.0 / 3.0, min=0.1, max=10.0, sweepable=True),
]
STATE = ["x", "y", "z"]
RUN_DEFAULTS = dict(record_component=2)


@cuda.jit(device=True)
def rhs(t, y, p, dy):
    rho = p[0]
    sigma = p[1]
    beta = p[2]
    dy[0] = sigma * (y[1] - y[0])
    dy[1] = y[0] * (rho - y[2]) - y[1]
    dy[2] = y[0] * y[1] - beta * y[2]


y0 = [1.0, 1.0, 1.0]
