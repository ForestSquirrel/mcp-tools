"""
Sprott case B (J. C. Sprott, "Some simple chaotic flows", Phys. Rev. E 50, 1994),
with the two constants made parameters:

    x' = y z
    y' = x - b y
    z' = a - x y

a = b = 1 is exactly case B.

Scaling: (x, y, z, t) -> (alpha x, beta y, gamma z, tau t) maps the system onto
itself with (a, b) -> (tau^3 a, tau b), so the dynamics depends only on a / b^3.
Sweeping a or b explores the same family; in an (a, b) period map the bands
follow curves a ~ b^3.
"""

from numba import cuda

DIM = 3
N_PARAMS = 2

PARAMS = [
    dict(name="a", default=1.0, min=0.01, max=5.0, sweepable=True, sweep=(0.2, 3.0)),
    dict(name="b", default=1.0, min=0.2, max=3.0, sweepable=True, sweep=(0.6, 1.6)),
]
STATE = ["x", "y", "z"]
RUN_DEFAULTS = dict(dt=0.01, t_end=600.0, transient=300.0, record_component=0)


@cuda.jit(device=True)
def rhs(t, y, p, dy):
    a = p[0]
    b = p[1]
    dy[0] = y[1] * y[2]
    dy[1] = y[0] - b * y[1]
    dy[2] = a - y[0] * y[1]


y0 = [0.05, 0.05, 0.05]
