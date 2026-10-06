from numba import cuda, float64

DIM = 3
N_PARAMS = 3

PARAMS = [
    dict(name="c", default=5.7, min=1.0, max=20.0, sweepable=True, sweep=(2.0, 8.0)),
    dict(name="a", default=0.2, min=0.0, max=0.5, sweepable=True),
    dict(name="b", default=0.2, min=0.0, max=2.0, sweepable=True),
]
STATE = ["x", "y", "z"]

@cuda.jit(device=True)
def rhs(t, y, p, dy):
    c = p[0]
    a = p[1]
    b = p[2]
    x = y[0]
    yy = y[1]
    z = y[2]
    dy[0] = -yy - z
    dy[1] = x + a * yy
    dy[2] = b + z * (x - c)

y0 = [0.1, 0.0, 0.0]
