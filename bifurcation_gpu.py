"""
GPU bifurcation diagram generator — importable version with progress callback.

Supports
  * 1-D sweeps  (run_bifurcation)    — one p[i] varied, classic bifurcation diagram
  * 2-D sweeps  (run_bifurcation_2d) — two parameters varied over
                                       a grid, for a 2-parameter (isoperiodic)
                                       map
  * smooth flows (rhs only) and reset-based / hybrid models such as
    Izhikevich, leaky integrate-and-fire, adaptive exponential I&F, which
    additionally define a `reset(t, y, p)` device function.
"""

import hashlib
import importlib.util
import math
import threading
from collections import OrderedDict

import numpy as np
from numba import cuda, float64

# A trajectory whose recorded component exceeds this (or goes NaN) is treated as
# diverged: recording stops and the cell reports zero points. Reset models that
# are mis-specified (threshold never reached) blow up quickly, so this keeps a
# bad corner of the parameter plane from poisoning the whole run.
BLOWUP = 1e12

class Cancelled(Exception):
    """Raised between GPU chunks when the caller's should_cancel() says so."""


RECORD_MODES = ("maxima", "tail", "isi", "spike")
INTEGRATORS = ("rk4", "euler")


def load_rhs_module(path):
    spec = importlib.util.spec_from_file_location("user_rhs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for r in ["DIM", "N_PARAMS", "rhs", "y0"]:
        if not hasattr(mod, r):
            raise AttributeError(f"RHS file must define '{r}'")
    return mod


# Modules and built kernels keyed by the RHS file's content hash, so a warm
# process re-running the same system skips the import and the CUDA JIT.
_CACHE_SIZE = 8
_modules: "OrderedDict[str, object]" = OrderedDict()
_kernels: "OrderedDict[tuple, object]" = OrderedDict()
_cache_lock = threading.Lock()


def _remember(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_rhs_cached(path):
    """load_rhs_module, reusing the module while the file's content is unchanged."""
    key = file_hash(path)
    with _cache_lock:
        mod = _modules.get(key)
        if mod is not None:
            _modules.move_to_end(key)
            return mod, key
    mod = load_rhs_module(path)
    with _cache_lock:
        _remember(_modules, key, mod)
    return mod, key


def get_reset(mod):
    """
    Return the optional reset device function of an RHS module, or None.

    A model is treated as reset-based iff it defines `reset`. HAS_RESET is
    accepted as an explicit opt-out (HAS_RESET = False disables an otherwise
    present reset function) but is not required.
    """
    reset = getattr(mod, "reset", None)
    if reset is None:
        return None
    if not getattr(mod, "HAS_RESET", True):
        return None
    return reset


def build_kernel(rhs_device, DIM, N_PARAMS, record_mode,
                 integrator="rk4", reset_device=None):
    if record_mode not in RECORD_MODES:
        raise ValueError(
            f"record_mode must be one of {RECORD_MODES}, got {record_mode!r}"
        )
    if integrator not in INTEGRATORS:
        raise ValueError(
            f"integrator must be one of {INTEGRATORS}, got {integrator!r}"
        )

    is_maxima = (record_mode == "maxima")
    is_tail   = (record_mode == "tail")
    is_isi    = (record_mode == "isi")
    is_spike  = (record_mode == "spike")
    use_rk4   = (integrator == "rk4")
    has_reset = reset_device is not None

    if (is_isi or is_spike) and not has_reset:
        raise ValueError(
            f"record_mode='{record_mode}' needs a reset-based model: the RHS "
            "file must define reset(t, y, p). Use 'maxima' or 'tail' for "
            "smooth flows."
        )

    if has_reset:
        reset_fn = reset_device
    else:
        @cuda.jit(device=True)
        def _no_reset(t, y, p):
            return 0
        reset_fn = _no_reset

    @cuda.jit
    def integrate_kernel(params, y0_arr, dt, n_steps, n_transient,
                         record_component, record_n, out, out_count):
        idx = cuda.grid(1)
        if idx >= params.shape[0]:
            return

        y   = cuda.local.array(DIM,      dtype=float64)
        k1  = cuda.local.array(DIM,      dtype=float64)
        k2  = cuda.local.array(DIM,      dtype=float64)
        k3  = cuda.local.array(DIM,      dtype=float64)
        k4  = cuda.local.array(DIM,      dtype=float64)
        tmp = cuda.local.array(DIM,      dtype=float64)
        p   = cuda.local.array(N_PARAMS, dtype=float64)

        for j in range(DIM):
            y[j] = y0_arr[j]
        for j in range(N_PARAMS):
            p[j] = params[idx, j]

        t = 0.0
        rec_i = 0
        prev_val = 0.0
        prev_prev_val = 0.0
        have_prev = False
        have_prev_prev = False
        last_spike_t = 0.0
        have_last_spike = False
        diverged = False

        for step in range(n_steps):
            if use_rk4:
                rhs_device(t, y, p, k1)
                for j in range(DIM):
                    tmp[j] = y[j] + 0.5 * dt * k1[j]
                rhs_device(t + 0.5 * dt, tmp, p, k2)
                for j in range(DIM):
                    tmp[j] = y[j] + 0.5 * dt * k2[j]
                rhs_device(t + 0.5 * dt, tmp, p, k3)
                for j in range(DIM):
                    tmp[j] = y[j] + dt * k3[j]
                rhs_device(t + dt, tmp, p, k4)
                for j in range(DIM):
                    y[j] += (dt / 6.0) * (k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j])
            else:
                rhs_device(t, y, p, k1)
                for j in range(DIM):
                    y[j] += dt * k1[j]
            t += dt

            # Reset/threshold is applied once per completed step, so the
            # recorded pre-reset value is the state at the end of the step
            # in which the threshold was crossed.
            spiked = 0
            pre_val = y[record_component]
            if has_reset:
                spiked = reset_fn(t, y, p)

            val = y[record_component]
            if val != val or val > BLOWUP or val < -BLOWUP:
                diverged = True
                break

            if step >= n_transient:
                if is_tail:
                    out[idx, rec_i % record_n] = val
                    rec_i += 1
                elif is_maxima:
                    if has_reset and spiked == 1:
                        # A reset is a jump, not a smooth turning point —
                        # drop the history so it cannot fake a local maximum.
                        have_prev = False
                        have_prev_prev = False
                    else:
                        if have_prev_prev and have_prev:
                            if prev_val > prev_prev_val and prev_val > val:
                                if rec_i < record_n:
                                    out[idx, rec_i] = prev_val
                                    rec_i += 1
                        prev_prev_val = prev_val
                        prev_val = val
                        have_prev_prev = have_prev
                        have_prev = True
                elif is_isi:
                    if spiked == 1:
                        if have_last_spike and rec_i < record_n:
                            out[idx, rec_i] = t - last_spike_t
                            rec_i += 1
                        last_spike_t = t
                        have_last_spike = True
                elif is_spike:
                    if spiked == 1 and rec_i < record_n:
                        out[idx, rec_i] = pre_val
                        rec_i += 1

        if diverged:
            out_count[idx] = 0
        else:
            out_count[idx] = min(rec_i, record_n)

    return integrate_kernel


def _integrate(rhs_path, params_host, dt, t_end, transient,
               record_component, record_mode, record_n, integrator,
               threads_per_block, _prog, prog_lo=5, prog_hi=90,
               should_cancel=None):
    """
    Shared core: run one batch of parameter sets (rows of params_host) on the
    GPU. Returns (values, counts, module).

    The batch is launched in chunks so progress is real and should_cancel()
    (checked between chunks) can stop it by raising Cancelled.
    """
    n_cells = params_host.shape[0]

    def prog(frac, msg):
        _prog(int(prog_lo + (prog_hi - prog_lo) * frac), msg)

    mod, rhs_key = load_rhs_cached(rhs_path)
    DIM = mod.DIM
    N_PARAMS = mod.N_PARAMS
    rhs_device = mod.rhs
    reset_device = get_reset(mod)
    y0 = np.array(mod.y0, dtype=np.float64)

    if params_host.shape[1] != N_PARAMS:
        raise ValueError(
            f"parameter array has {params_host.shape[1]} columns but the RHS "
            f"file declares N_PARAMS = {N_PARAMS}"
        )
    if not (0 <= record_component < DIM):
        raise ValueError(
            f"record_component must be in [0, {DIM}) for this system, "
            f"got {record_component}"
        )

    n_steps     = int(t_end / dt)
    n_transient = int(transient / dt)
    if n_transient >= n_steps:
        raise ValueError(
            f"transient ({transient}) must be smaller than t_end ({t_end})"
        )

    nbytes = n_cells * record_n * 8
    if nbytes > 2 << 30:
        raise ValueError(
            f"requested output is {nbytes / 2**30:.1f} GiB "
            f"({n_cells} parameter points x record_n={record_n}). "
            "Reduce the grid resolution or record_n."
        )

    prog(0.0, "Transferring data to GPU...")
    d_params    = cuda.to_device(params_host)
    d_y0        = cuda.to_device(y0)
    d_out       = cuda.device_array((n_cells, record_n), dtype=np.float64)
    d_out_count = cuda.device_array(n_cells, dtype=np.int32)

    kind = "reset-based" if reset_device is not None else "smooth"
    kernel_key = (rhs_key, record_mode, integrator)
    with _cache_lock:
        kernel = _kernels.get(kernel_key)
    if kernel is None:
        prog(0.06, f"Compiling CUDA kernel ({kind}, {integrator})...")
        kernel = build_kernel(rhs_device, DIM, N_PARAMS, record_mode,
                              integrator=integrator, reset_device=reset_device)
        with _cache_lock:
            _remember(_kernels, kernel_key, kernel)

    # At least 8192 cells per launch keeps the GPU busy; at most ~16 launches.
    chunk = max(8192, math.ceil(n_cells / 16))
    chunk = math.ceil(chunk / threads_per_block) * threads_per_block
    for start in range(0, n_cells, chunk):
        if should_cancel is not None and should_cancel():
            raise Cancelled()
        stop = min(start + chunk, n_cells)
        prog(0.12 + 0.76 * start / n_cells,
             f"Running GPU integration ({start}/{n_cells} parameter points)...")
        kernel[math.ceil((stop - start) / threads_per_block), threads_per_block](
            d_params[start:stop], d_y0, dt, n_steps, n_transient,
            record_component, record_n, d_out[start:stop], d_out_count[start:stop]
        )
        cuda.synchronize()
    if should_cancel is not None and should_cancel():
        raise Cancelled()

    prog(0.94, "Copying results to host...")
    values = d_out.copy_to_host()
    counts = d_out_count.copy_to_host()

    prog(1.0, "Integration done.")
    return values, counts, mod


def run_bifurcation(
    rhs_path,
    param_min, param_max, param_steps,
    dt=0.001, t_end=200.0, transient=100.0,
    record_component=0, record_mode="maxima", record_n=500,
    integrator="rk4",
    fixed_params=None,
    threads_per_block=128,
    progress_callback=None,   # callable(percent: int, message: str)
    param_index=0,
    should_cancel=None,       # callable() -> bool, checked between GPU chunks
):
    """
    Run a 1-D GPU bifurcation diagram: p[param_index] (default p[0]) is swept
    from param_min to param_max.

    fixed_params : optional list of length N_PARAMS giving the values of the
                   non-swept parameters p[1:]; p[0] is always overwritten by
                   the sweep. Defaults to the PARAMS defaults (zeros without PARAMS).

    Returns dict with keys: params, values, counts, param_min, param_max,
                            record_mode, record_component, integrator,
                            has_reset, kind='1d'.
    progress_callback(pct, msg) is called at key milestones (0-100).
    """
    def _prog(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _prog(0, "Loading RHS module...")
    mod, _ = load_rhs_cached(rhs_path)
    N_PARAMS = mod.N_PARAMS
    if not (0 <= param_index < N_PARAMS):
        raise ValueError(f"param_index must be in [0, {N_PARAMS}), got {param_index}")

    params_host = _base_params(N_PARAMS, param_steps, fixed_params, mod)
    sweep_vals = np.linspace(param_min, param_max, param_steps)
    params_host[:, param_index] = sweep_vals

    values, counts, mod = _integrate(
        rhs_path, params_host, dt, t_end, transient,
        record_component, record_mode, record_n, integrator,
        threads_per_block, _prog, prog_lo=5, prog_hi=90,
        should_cancel=should_cancel,
    )

    return dict(
        kind="1d",
        params=sweep_vals,
        param_index=param_index,
        values=values,
        counts=counts,
        param_min=param_min,
        param_max=param_max,
        record_mode=record_mode,
        record_component=record_component,
        integrator=integrator,
        has_reset=get_reset(mod) is not None,
    )


def run_bifurcation_2d(
    rhs_path,
    param_min, param_max, param_steps,
    param2_min, param2_max, param2_steps,
    param2_index=1,
    dt=0.001, t_end=200.0, transient=100.0,
    record_component=0, record_mode="maxima", record_n=64,
    integrator="rk4",
    fixed_params=None,
    threads_per_block=128,
    progress_callback=None,
    param_index=0,
    should_cancel=None,
):
    """
    Run a 2-parameter GPU sweep: p[param_index] (default p[0]) on the x axis,
    p[param2_index] on the y axis. Every (x, y) cell integrates one trajectory and records up to
    record_n values exactly like the 1-D case; clustering those values is what
    turns the result into a 2-D map (see clustering.cluster_grid).

    Returns dict with keys: params (x axis), params2 (y axis), values, counts,
    shape (n_y, n_x), plus metadata. `values`/`counts` are flat, row-major with
    row index = j * n_x + i, so `counts.reshape(shape)` lines up with
    imshow(origin='lower').
    """
    def _prog(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _prog(0, "Loading RHS module...")
    mod, _ = load_rhs_cached(rhs_path)
    N_PARAMS = mod.N_PARAMS

    if N_PARAMS < 2:
        raise ValueError(
            "A 2-parameter sweep needs N_PARAMS >= 2 in the RHS file "
            f"(got {N_PARAMS}). Add the second parameter as p[1]."
        )
    if not (0 <= param2_index < N_PARAMS):
        raise ValueError(
            f"param2_index must be in [0, {N_PARAMS}), got {param2_index}"
        )
    if not (0 <= param_index < N_PARAMS):
        raise ValueError(f"param_index must be in [0, {N_PARAMS}), got {param_index}")
    if param2_index == param_index:
        raise ValueError(
            f"param2_index must differ from param_index ({param_index}): "
            "the two axes must sweep different parameters."
        )

    n_x, n_y = int(param_steps), int(param2_steps)
    x_vals = np.linspace(param_min, param_max, n_x)
    y_vals = np.linspace(param2_min, param2_max, n_y)

    params_host = _base_params(N_PARAMS, n_x * n_y, fixed_params, mod)
    # Row-major grid: index = j * n_x + i, i indexes x, j indexes y.
    xx, yy = np.meshgrid(x_vals, y_vals)
    params_host[:, param_index] = xx.ravel()
    params_host[:, param2_index] = yy.ravel()

    values, counts, mod = _integrate(
        rhs_path, params_host, dt, t_end, transient,
        record_component, record_mode, record_n, integrator,
        threads_per_block, _prog, prog_lo=5, prog_hi=70,
        should_cancel=should_cancel,
    )

    return dict(
        kind="2d",
        params=x_vals,
        params2=y_vals,
        values=values,
        counts=counts,
        shape=(n_y, n_x),
        param_min=param_min, param_max=param_max,
        param2_min=param2_min, param2_max=param2_max,
        param_index=param_index,
        param2_index=param2_index,
        record_mode=record_mode,
        record_component=record_component,
        integrator=integrator,
        has_reset=get_reset(mod) is not None,
    )


def _base_params(n_params, n_rows, fixed_params, mod=None):
    """
    Build the (n_rows, n_params) parameter block from the fixed values.
    Without fixed_params, uses the RHS file's PARAMS defaults (contract v2)
    when it declares them, else zeros.
    """
    arr = np.zeros((n_rows, n_params), dtype=np.float64)
    declared = getattr(mod, "PARAMS", None)
    if fixed_params is None and declared is not None and len(declared) == n_params:
        arr[:, :] = np.array([q["default"] for q in declared], dtype=np.float64)
    if fixed_params is not None:
        fp = np.asarray(fixed_params, dtype=np.float64).ravel()
        if fp.size != n_params:
            raise ValueError(
                f"fixed_params must have exactly N_PARAMS={n_params} entries "
                f"(got {fp.size}). Entries for swept parameters are ignored."
            )
        arr[:, :] = fp[None, :]
    return arr
