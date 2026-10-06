---
name: bif-studio
description: Bifurcation Studio (the bifurcation-studio MCP server and the /bif pane of the bif-studio mod): GPU bifurcation diagrams and 2-D period maps of dynamical systems. Use whenever a request has the shape of a nonlinear-dynamics task, even if the user never names the tool: how a system's long-term behavior changes as a parameter varies, period-doubling or route-to-chaos questions, Rössler/Lorenz-style attractors, spiking or reset neuron models (Izhikevich, LIF, AdEx, integrate-and-fire), ISI analysis, Poincaré maps, sweeping one or two parameters of an ODE system, "period map", "cluster map" or "isoperiodic diagram". Also when writing or fixing an RHS file, when the user asks about a run made in the pane (a message starting "[Bifurcation Studio]"), or when analysing run-NNNN.npz results. Not for chaos theory with no computation, or plotting data the user already has.
---

# Bifurcation Studio

The user explores systems in the **/bif pane**: they pick sweep axes, nudge parameters and press
Run themselves, and see the preview there. Don't ask permission before using the server's
tools (they compute or read, no side effects) except where this skill says to ask (a system's
name; keeping or overwriting a permanent system): say briefly what you're doing and proceed.
Your part is small and specific:

1. **Write the system** (an RHS file, contract below), once per system, saved under a name
   the user chose.
2. **Answer questions about runs**, from compact numbers, not pictures.

You do not start, poll or display runs: the pane does (the server's `submit_run`,
`job_status`, `cancel_job` and `raster` tools are the pane's, not yours).

## 1. Writing a system

**Before saving, ask the user to name the system.** Use AskUserQuestion with two or three
suggested names (lowercase, digits, underscore, e.g. `chua`, `lif_adaptive`, `rossler_3p`);
they can type their own. Don't invent a name silently. The name is how they open it
(`/bif <name>`) and how it is stored.

**Save it with `mcp__bifurcation-studio__save_system(name, source)`**, not with Write: it works
from any client, stores a draft (`~/.cache/bif-studio/drafts/<name>.py`, kept ~24 h untouched,
survives MCP reconnects) and returns the contract check: `ok`, or `errors` with line numbers.
Fix and call `save_system` again with the same name until it passes, then tell the user:
`/bif <name>`. If the pane already shows that system, it reloads by itself.

**Keep it permanently only when the user wants to:** `keep_system(name)` moves the draft to
`~/mcp-tools/systems/<name>.py`. If a permanent system of that name exists, the call refuses;
pass `overwrite=true` only after the user agreed to replace it, otherwise ask for another name.
`list_systems` shows drafts and permanent systems with their status.

### The contract

**Call `get_rhs_template()` before writing your first system in a session.** It is the full
contract (v2) with a smooth-flow and a reset-model example and the `RUN_DEFAULTS` keys. In short:
one system per `.py` file with `DIM`, `N_PARAMS`, `PARAMS` (name, default, min, max, `sweepable`,
optional `sweep=(lo, hi)`), optional `STATE` / `RUN_DEFAULTS`, a `@cuda.jit(device=True)`
`rhs(t, y, p, dy)` writing into `dy` in place (scalar math and `math.*` only), and `y0`.
Reset/spiking models add `reset(t, y, p)` returning 1 if it fired, else 0 on every path.
Working examples: `rossler`, `lorenz`, `izhikevich`, `sprott_b` (`list_systems` gives paths).

GPU compile errors (a typo inside `rhs`) only appear when the user presses Run; they show in the
pane and in the run's `error`. (In Claude Code a Write/Edit of an RHS file is contract-checked
too, a `bif-studio:` line on the tool result, but prefer `save_system`.)

## 2. Answering questions about runs

A question from the pane arrives as a message starting `[Bifurcation Studio] Question about
run-NNNN ...`, with the run's npz path and the analysis command. Otherwise list the runs with
`mcp__bifurcation-studio__list_runs` with no session (the most recent pane session; `last` limits it): it
gives id, sweep, parameters, summary, `npz` / `preview` / `chart` paths and the `inspect` command.

Analyse with `runinspect.py` via Bash, using the exact `inspect` command line you were given:

```
<inspect> summary  run-0003.npz          # what was run + period regions along the sweep
<inspect> slice    run-0003.npz c=5.7    # one parameter value: period and recorded values
<inspect> compare  run-0003.npz run-0004.npz   # where two runs' periods differ
```

Add `--json` for structured output. Output is a few lines; work from it.

- **Never print raw arrays** (`np.load(...)["values"]` is up to millions of numbers). If you
  need something runinspect doesn't give, compute it in a short script and print only the
  result.
- **Look at the preview image only when the numbers can't answer the question**
  (e.g. "what does the attractor region look like"). Prefer `preview` (small) over `png`/`chart`.
- "Period" = the smallest k for which the recorded sequence repeats, |m[n+k] − m[n]| ≤ tol
  for all n (k ≤ 32); `chaotic` = no such k; `none` = too few values (quiescent or diverged).
  1-D summaries and period maps use the same rule (map summary `rule: "sequence"`). Record
  mode `tail` is not in time order and falls back to DBSCAN cluster counts (`rule: "dbscan"`).
- **"Chaotic" right next to a bifurcation is often an orbit that hasn't converged yet**
  (the repeat test is strict). Before reporting a chaotic sliver inside a periodic region,
  suggest re-running that range with a longer `transient` (e.g. Rössler c≈2.9–3.1 is
  period 2 with transient 1200, "chaotic" with 200).
- `tol` is `period_tol` (default 1e-3) times the spread of all recorded values, so a narrow
  sweep has a tighter tolerance than a wide one.
- Maps made before this rule (summary without `rule`) counted DBSCAN clusters, which reads a
  chaotic attractor with few bands as a low period; don't compare them with new maps.

Answer in plain language with parameter values ("period-doubling at c≈3.04 and ≈3.85, chaos
from c≈4.43 with a period-3 window at c≈5.18–5.39"), and suggest a next run when it would
settle the question (narrower sweep, longer `t_end`, a slice in the other parameter).

## 3. Suggesting a run

Suggest runs in pane terms: mode (bifurcation / period map), sweep axes and ranges, and the run
options that matter (record mode, integrator, `t_end` / `transient`, resolution).

- **Resolution:** `param_steps` up to 1e6 for 1-D sweeps, `param_steps × param2_steps` around
  1e3 × 1e3 for period maps, unless the user wants it faster. Lower `record_n` for very large
  1-D sweeps: a run whose output buffer (`param_steps × record_n × 8` bytes) exceeds 2 GiB is
  refused.
- **GPU memory:** if the user wants much more than that (a 2-D run costs `param_steps ×
  param2_steps` trajectories), warn first that it can exhaust GPU memory and offer to scale
  back or split the range.
