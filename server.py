"""
FastMCP server — GPU bifurcation diagrams and period maps (Bifurcation Studio).

Runs are driven by the /bif pane (the bif-studio Claude Code mod); results are
files on disk, and every tool returns paths and short summaries only.

Tools:
  get_rhs_template()      → the RHS file contract (read before writing a system)
  describe(path)          → validate an RHS file, JSON of its params/state/run options
  save_system(name, src)  → save a draft system and check it
  keep_system(name)       → make a draft permanent (systems/)
  list_systems()          → drafts + permanent systems, name → file
  submit_run(...)         → {jobId, runId}       (the pane)
  job_status / cancel_job / list_runs / raster / cleanup   (the pane)

Run analysis for Claude lives in runinspect.py (summary / slice / compare).
"""

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import contract
import studio
import systems

HERE = Path(__file__).parent

RHS_TEMPLATE_DOC = '''
RHS FILE CONTRACT (v2) — one system per .py file:

  DIM        : int            — number of state variables
  N_PARAMS   : int            — number of parameters in p[] (== len(PARAMS))
  PARAMS     : list of dicts, one per p[] slot, in p[] order
                 dict(name="c", default=5.7, min=1.0, max=20.0,
                      sweepable=True,        # may be an axis of a sweep
                      sweep=(2.0, 8.0),      # optional default axis range
                      label="c")             # optional display name
               at least one parameter must be sweepable
  STATE      : list of DIM names for y[0..DIM-1]            (optional)
  RUN_DEFAULTS : dict of run-option defaults for this system   (optional)
               e.g. dict(dt=0.25, integrator="euler", record_mode="isi")
               keys: dt, t_end, transient, integrator (rk4|euler),
               record_mode (maxima|tail|isi|spike), record_component,
               record_n, param_steps, param2_steps, eps, eps_scale,
               min_samples, min_points, max_clusters, mode
  rhs(t,y,p,dy) : @cuda.jit(device=True) function
               — writes the derivative into dy[] IN PLACE. y, p, dy are numba
                 cuda.local.array: plain indexed float64 access only (y[0],
                 p[0], dy[0] ...). No numpy, no Python lists/objects, no
                 exceptions, no print(). Scalar arithmetic and math.* only.
  y0         : list[float]    — initial condition, length DIM

and MAY define, for reset-based (hybrid) models only:

  HAS_RESET  : bool           — False switches an existing reset off
  reset(t,y,p) : @cuda.jit(device=True) function
               — modifies y IN PLACE and RETURNS 1 if a reset fired on this
                 step, else 0

EXAMPLE 1 — smooth flow, Rössler system (DIM=3, N_PARAMS=3):

    from numba import cuda

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
        c = p[0]; a = p[1]; b = p[2]
        dy[0] = -y[1] - y[2]
        dy[1] = y[0] + a * y[1]
        dy[2] = b + y[2] * (y[0] - c)

    y0 = [0.1, 0.0, 0.0]


=============================================================================
RESET-BASED / HYBRID MODELS  (Izhikevich, LIF, adaptive exponential I&F, ...)
=============================================================================

These need NO special server support beyond one extra device function: the
integrator already applies `reset` once after each completed step, so any
"integrate the ODE, then snap the state when it crosses a threshold" model
works. Define `reset` and set HAS_RESET = True; leave both out for smooth flows.

EXAMPLE 2 — Izhikevich neuron (DIM=2, N_PARAMS=3, p[0]=I, p[1]=c, p[2]=d):

    from numba import cuda

    DIM = 2
    N_PARAMS = 3
    HAS_RESET = True

    PARAMS = [
        dict(name="I", default=10.0, min=0.0, max=40.0, sweepable=True),
        dict(name="c", default=-65.0, min=-80.0, max=-40.0, sweepable=True),
        dict(name="d", default=8.0, min=0.0, max=10.0, sweepable=True),
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
        v = y[0]; u = y[1]
        dy[0] = 0.04*v*v + 5.0*v + 140.0 - u + I
        dy[1] = A * (B*v - u)

    @cuda.jit(device=True)
    def reset(t, y, p):
        if y[0] >= V_PEAK:
            y[0] = p[1]          # v <- c
            y[1] = y[1] + p[2]   # u <- u + d
            return 1
        return 0

    y0 = [-65.0, -13.0]

  (This exact model ships as systems/izhikevich.py.)

GUIDELINES FOR RESET MODELS

  1. `reset` MUST return an int on every path: 1 when it fired, 0 otherwise.
     That return value is what drives the 'isi' and 'spike' record modes.
  2. It is applied ONCE per completed step, so the effective threshold-crossing
     resolution is dt. Keep dt small enough that the state does not overshoot
     the threshold badly (for Izhikevich the v-nullcline is steep near the peak,
     which is exactly why the threshold is set at a finite v = 30 rather than
     at infinity).
  3. Use integrator="euler" for the standard neuron models. Izhikevich, LIF and
     friends are conventionally integrated with forward Euler at dt = 0.25-0.5
     (time in ms); RK4 evaluates the RHS across the spike upstroke, where the
     quadratic term is enormous, and buys accuracy you do not want to pay for.
     Use the default integrator="rk4" for smooth flows.
  4. Time units are the model's own. For Izhikevich-style ms units, use
     t_end ~ 2000-5000 and transient ~ half of t_end — NOT the t_end=200 that
     suits Rössler.
  5. Choose the record mode to match the model:
       record_mode="isi"    — inter-spike intervals (time between resets).
                              The natural observable for spiking models: the
                              period is then the length of the repeating
                              interval pattern, i.e. 1 = tonic regular
                              spiking, n = n-spike bursting, none = chaotic.
       record_mode="spike"  — value of state variable `record_component` at the
                              instant of each reset, sampled just BEFORE the
                              reset is applied (e.g. record_component=1 gives u
                              at each spike for Izhikevich). This is the
                              stroboscopic map of the model.
       record_mode="maxima" — still available; local maxima of the recorded
                              component. Steps on which a reset fires are
                              excluded so the jump cannot fake a maximum.
     'isi' and 'spike' REQUIRE a reset function and error out without one.
  6. Cells where the neuron never spikes (or where the trajectory diverges)
     record nothing and are drawn as "none" in the period map. A large
     no-data region is usually correct — it is the sub-threshold, quiescent
     part of the parameter plane, not a bug.

=============================================================================
SAVING A SYSTEM
=============================================================================

Ask the user to name the system first (lowercase, digits, underscore). Then
save_system(name, source) stores a draft and returns the contract check
(errors with line numbers); fix and save again until it passes. The user
opens it with /bif <name>. keep_system(name) makes it permanent when the
user wants; examples: list_systems() → systems/rossler.py, izhikevich.py,
lorenz.py, sprott_b.py.
'''


INSTRUCTIONS = """\
Bifurcation Studio: GPU bifurcation diagrams and 2-D period maps of dynamical
systems (ODEs, spiking/reset neuron models such as Izhikevich, LIF, AdEx).
Runs are started and shown by the user's /bif pane in Claude Code; submit_run,
job_status, cancel_job, raster and cleanup are the pane's, not Claude's.
Claude's part: write systems and answer questions about runs.
- New system: read get_rhs_template() first, ask the user to name it, then
  save_system(name, source) until it passes; the user opens it with /bif <name>.
  keep_system(name) only when the user wants it kept permanently.
- Questions about runs: list_runs() (no session = latest pane session) gives
  paths and an inspect command; analyse with runinspect.py summary/slice/compare,
  never print raw arrays, view a preview image only if numbers can't answer.
"""

mcp = FastMCP("bifurcation-studio", instructions=INSTRUCTIONS)


@mcp.tool()
def get_rhs_template() -> str:
    """
    The contract for an RHS file (a dynamical system the /bif pane can run):
    required fields, PARAMS metadata, worked examples, and how to write
    reset-based neuron models (Izhikevich, LIF, AdEx): reset(), integrator,
    record mode and time scale. Read it before writing a new system.
    """
    return RHS_TEMPLATE_DOC


@mcp.tool()
def describe(path: str) -> dict:
    """
    Validate an RHS file against the contract (see get_rhs_template) and
    return its description: system name/kind/hash, PARAMS (names, defaults,
    ranges, sweepable), STATE, and the run options with their defaults.
    Never raises: a broken file gives {"ok": false, "errors": [...]} with
    line numbers.
    """
    return contract.describe(path)


# ── Bifurcation Studio tools (path-only; used by the /bif mod) ────────────────
@mcp.tool()
def submit_run(rhs_path: str, session: str, sweep: dict,
               params: dict | None = None, run_options: dict | None = None) -> dict:
    """
    Start a run of an RHS file (contract v2) into the session folder
    ~/.cache/bif-studio/<session>/. Returns {"jobId", "runId"}.

    sweep:       {"x": {"param": "c", "min": 2, "max": 8}}; add "y" for
                 run_options.mode="period-map". min/max default to the
                 parameter's sweep range.
    params:      {name: value} for parameters not swept; default PARAMS defaults.
    run_options: names and defaults as describe() lists them in runOptions.
    """
    return studio.submit_run(rhs_path, session, params, sweep, run_options)


@mcp.tool()
def job_status(job_id: str) -> dict:
    """
    State of a studio run: {"state": running|done|error|cancelled, "progress",
    "message", "result"?: {"paths": {png, preview, chart, npz, json},
    "summary"}}. Never returns image bytes or arrays.
    """
    return studio.job_status(job_id)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """Cancel a running studio run (stops at the next GPU chunk)."""
    return studio.cancel_job(job_id)


@mcp.tool()
def list_runs(session: str = "", last: int = 0) -> dict:
    """
    Runs of a Bifurcation Studio session (the /bif pane), oldest first, as
    {"session", "runs": [...], "inspect": "<python> <path>/runinspect.py"}:
    id, state, params, sweep, summary, and paths (npz data, preview png).
    Leave session empty for the most recently used one. Analyse a run with
    the inspect command (summary <npz> | slice <npz> p=v | compare <a> <b>);
    view a preview image only if the numbers are not enough.
    """
    session = session or studio.latest_session()
    # Wrapped in a dict: FastMCP sends a bare list as one content block per item.
    # "inspect" is the command line for runinspect.py (summary/slice/compare).
    return {"session": session, "runs": studio.list_runs(session, last),
            "inspect": f"{sys.executable} {HERE / 'runinspect.py'}"}


@mcp.tool()
def save_system(name: str, source: str) -> dict:
    """
    Save an RHS file (contract v2, see get_rhs_template) as a draft system
    named `name` and check it. Ask the user for the name first. Returns the
    path, the contract result (errors with line numbers, or params/state),
    and what to do next. Saving the same name again replaces the draft.
    Drafts last ~24 h untouched; keep_system makes one permanent.
    """
    return systems.save_system(name, source)


@mcp.tool()
def keep_system(name: str, overwrite: bool = False) -> dict:
    """
    Make a passing draft system permanent (moves it into the systems folder).
    Only when the user wants it kept. overwrite=true replaces an existing
    permanent system of that name: ask the user before passing it.
    """
    return systems.keep_system(name, overwrite)


@mcp.tool()
def list_systems() -> dict:
    """
    Systems available to /bif: drafts and permanent ones, each with its
    contract status and parameters, plus "resolve" (name -> the file
    /bif <name> opens; a draft wins over a permanent one of the same name).
    """
    return systems.list_systems()


@mcp.tool()
def raster(png_path: str, columns: int, rows: int) -> dict:
    """
    For the /bif pane, not for Claude: a run image (inside the studio folder)
    as packed half-block terminal cells {columns, rows, cells(base64)}.
    """
    return studio.raster(png_path, columns, rows)


@mcp.tool()
def cleanup(older_than_hours: float = 24.0, keep_session: str = "") -> dict:
    """Delete studio session folders and legacy job folders untouched for that long."""
    return studio.cleanup(older_than_hours, keep_session)


if __name__ == "__main__":
    mcp.run()
