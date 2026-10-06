"""
Izhikevich spiking neuron — the reference reset-based (hybrid) model.

    v' = 0.04 v^2 + 5 v + 140 - u + I
    u' = a (b v - u)
    if v >= 30 mV:  v <- c,  u <- u + d

Time is in ms, so use dt ~ 0.25-0.5 with integrator='euler' and
t_end/transient in the thousands.

Parameters:  p[0] = I (input current)
             p[1] = c (reset voltage,   ~ -65 regular spiking, ~ -50 chattering)
             p[2] = d (reset increment of u, ~ 2 chattering, ~ 8 regular spiking)

Good record modes: 'isi' (inter-spike intervals — the cluster count is then the
number of distinct intervals in the burst pattern) or 'spike' with
record_component=1 (value of u at each spike).
"""

from numba import cuda, float64

DIM = 2
N_PARAMS = 3
HAS_RESET = True

PARAMS = [
    dict(name="I", default=10.0, min=0.0, max=40.0, sweepable=True, label="I (input)"),
    dict(name="c", default=-65.0, min=-80.0, max=-40.0, sweepable=True, label="c (reset v)"),
    dict(name="d", default=8.0, min=0.0, max=10.0, sweepable=True, label="d (reset Δu)"),
]
STATE = ["v", "u"]
RUN_DEFAULTS = dict(dt=0.25, t_end=3000.0, transient=1500.0,
                    integrator="euler", record_mode="isi")

A = 0.02
B = 0.2
V_PEAK = 30.0


@cuda.jit(device=True)
def rhs(t, y, p, dy):
    I = p[0]
    v = y[0]
    u = y[1]
    dy[0] = 0.04 * v * v + 5.0 * v + 140.0 - u + I
    dy[1] = A * (B * v - u)


@cuda.jit(device=True)
def reset(t, y, p):
    if y[0] >= V_PEAK:
        y[0] = p[1]          # c
        y[1] = y[1] + p[2]   # d
        return 1
    return 0


y0 = [-65.0, -13.0]
