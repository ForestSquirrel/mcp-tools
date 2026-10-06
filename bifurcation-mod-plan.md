# Plan: Bifurcation Studio — moving the GPU MCP server behind a Claude Code mod

> **Historical plan.** Since 2026-10-06 the server is registered as `bifurcation-studio` (was
> `bifurcation-wizard`), the mod has no `results` tool (use `list_runs`), and the
> `bifurcation-wizard-trigger` skill is merged into the plugin skill `bif-studio`.

This document is a brief for Claude Code. It describes how to restructure the existing
stdio MCP server (GPU bifurcation diagrams, job/poll design) into a **Claude Code mod**
with its own interactive UI, so that Claude writes the model once and the user explores
it interactively without spending tokens.

> **Before implementing:** read the mods docs and, most importantly, the TypeScript
> declarations Claude Code writes for the installed build (see "Get the types for your build"
> in the docs). Several API details below are marked **VERIFY**: they come from the docs
> overview pages, and exact signatures must be checked against the types.
> Mods require Claude Code **v2.1.287+** (`claude --version`).

---

## 1. Why

Current pain points with the MCP-only design:

- **Polling burns tokens.** Every status check is a full tool-call round trip that re-processes
  the whole context. Running the same Python directly via Bash is ~2x+ cheaper.
- **Base64 images and raw data in tool results** bloat context and are re-read every later turn.
- **Tool schemas** for many tools sit in context all session.

Target: Claude's job shrinks to (a) writing the RHS contract, and (b) analyzing results on demand.
Everything else (parameter tweaking, running jobs, polling, displaying plots) happens in the mod
UI and Python, with zero tokens spent.

---

## 2. Target architecture

```
                    writes rhs.py (once per system)
   Claude  ─────────────────────────────────────────────►  rhs.py (contract)
     ▲                                                          │
     │ path + question (only when user asks)                    │ describe → JSON schema
     │                                                          ▼
 ┌───┴──────────────────── Mod (JS/TS, in Claude Code) ─────────────────────┐
 │  pane UI: controls generated from schema, plot preview, actions          │
 │  timers: poll jobs                                                        │
 │  hooks: detect new rhs.py, register small tools for Claude               │
 └───┬──────────────────────────────────────────────────────────────────────┘
     │ $.mcp.call (warm process, GPU context stays alive)
     ▼
  Python MCP server (existing compute core, slimmed tool surface)
     │
     ▼
  Session results folder: run-XXXX.npz, run-XXXX.png (full res), run-XXXX.preview.png, index.json
```

Key decisions:

1. **Python parses the contract, JS never parses Python.** A `describe` step validates the RHS
   and emits JSON; the mod builds UI from that JSON.
2. **The mod calls the existing long-lived MCP server** (`$.mcp.call`) rather than spawning a new
   Python process per run, so CUDA context / compiled kernels stay warm.
   `$.process.run` is the fallback (30 s default timeout, 10 min max).
3. **Results live on disk**; Claude only ever receives paths + compact metadata.

---

## 3. Components

### 3.1 Compute core (existing Python)

Keep as is. Refactor only where needed so that the RHS is injected from a file
(`rhs.py`) rather than hardcoded or passed as a string through MCP.

### 3.2 RHS contract (`rhs.py`)

Claude writes this file. Rules must be strict enough for `describe` to introspect it.
Proposed shape (adapt to the existing core's conventions):

```python
# rhs.py — contract v1
from core.contract import system, param, state

@system(name="rossler", kind="ode")          # kind: "ode" | "map"
class Model:
    # parameters: name, default, range, sweepable
    a = param(0.2, min=0.0, max=0.5)
    b = param(0.2, min=0.0, max=2.0)
    c = param(5.7, min=1.0, max=20.0, sweepable=True)

    # state variables with default initial conditions
    x = state(1.0)
    y = state(1.0)
    z = state(1.0)

    @staticmethod
    def rhs(t, s, p):
        x, y, z = s
        return (-y - z, x + p.a * y, p.b + z * (x - p.c))
```

Contract rules to enforce in `core.contract` (and document for Claude in a skill/CLAUDE.md):

- Only the decorated class is read; one system per file.
- Every `param` has `default`, `min`, `max`; `sweepable` marks params eligible for the x-axis.
- `rhs` must be GPU-compilable per the core's requirements (no Python objects, no I/O, etc.).
- Optional: `observables` (which state var / Poincaré section to record).

### 3.3 `describe` step

Exposed both as an MCP tool (for the mod) and a CLI (`python -m core describe rhs.py`) for
debugging. Output JSON (stable schema, versioned):

```json
{
  "contractVersion": 1,
  "ok": true,
  "system": { "name": "rossler", "kind": "ode", "rhsHash": "sha256:..." },
  "params": [
    { "name": "a", "default": 0.2, "min": 0.0, "max": 0.5, "sweepable": false },
    { "name": "c", "default": 5.7, "min": 1.0, "max": 20.0, "sweepable": true }
  ],
  "state": [ { "name": "x", "default": 1.0 } ],
  "runOptions": [
    { "name": "resolution", "type": "int", "default": 2000, "min": 100, "max": 20000 },
    { "name": "transient", "type": "int", "default": 1000 },
    { "name": "mode", "type": "enum", "options": ["bifurcation", "period-map"] }
  ]
}
```

On failure: `{ "ok": false, "errors": ["line 12: param 'b' missing max", ...] }`.

### 3.4 MCP server changes

- **New/changed tools** (consumed by the mod, not by Claude):
  - `describe(path)` → JSON above.
  - `start_job(rhs_path, params, run_options, out_dir)` → `{ jobId }`.
  - `job_status(jobId)` → `{ state, progress, result? }` where `result` is
    `{ npz, png, preview, summary }` (paths + tiny summary, **never base64, never raw arrays**).
  - `cancel_job(jobId)`.
  - `cleanup(older_than_hours)` → deletes stale session folders.
- **Remove from Claude's view** the old polling/data/base64 tools. Options: drop them, or keep them
  for back-compat and hide them via the mod (see 3.6).
- Every run writes to the session folder (3.5). Full-res PNG is rendered **directly from the
  density histogram** (PIL/imageio), not via matplotlib, so it's pixel-exact and fast.

### 3.5 Results storage

- Folder per session: `~/.cache/bif-studio/<session-id>/` (session id from `$.session.id()` — VERIFY).
  Keep it outside the repo.
- Per run:
  - `run-0007.npz` — raw data **plus metadata arrays/attrs**: params, run options, rhsHash,
    axis meanings, timestamp.
  - `run-0007.png` — full resolution, for the external viewer.
  - `run-0007.preview.png` — small (< 2 MiB, ideally much smaller), for the in-pane `Image`
    and for Claude to view if needed.
- `index.json` — list of runs with id, params, paths, short summary.
- **Cleanup is done by Python**, not the mod: `$.fs` has no delete method, and all `session.end`
  hooks together get only 1.5 s. On `session.start`, the mod calls `cleanup(24)`.

### 3.6 The mod (plugin)

Layout:

```
bif-studio/
├── .claude-plugin/
│   └── plugin.json          # manifest; also lists the MCP server so $.mcp.connect can reach it (VERIFY)
├── hooks/
│   ├── hooks.json           # { "modules": ["./register.ts"] }
│   └── register.ts          # hooks module
├── types/
│   └── index.d.ts           # PluginState declarations if $.state is used
├── server/                  # the Python MCP server + core (or a path to it)
└── skills/
    └── bif-studio/SKILL.md  # instructions for Claude: contract rules, helper module, result paths
```

Hooks to implement:

| Hook | Purpose |
| :- | :- |
| `session.start` | Create session folder, call `cleanup`, register `/bif` command and Claude-facing tools, load saved UI prefs from `$.store`. Register commands **last** or in try/catch (a throw skips the rest of the hook). |
| `command.run` `{ command: 'bif' }` | Open the pane: `$.ui.open({ id: 'bif', title: 'Bifurcation', focus: true })`. |
| `tool.call` on Claude's Write/Edit | After `next(e)` resolves, if the edited path is the RHS file, call `describe` and rebuild controls. If `ok: false`, send errors back to Claude with `$.prompt.submit`. (Check exact event fields — VERIFY.) |
| `tool.call` on old MCP tools (optional) | `{ deny: 'Use the bif-studio pane / list_results tool.' }` to stop Claude polling. Alternative: `tool.describe` with `isDeferred: true` or a shorter description. |
| `ui.render` `{ component: 'Pane' }` | Draw the UI (3.7). Check `e.requestId === 'bif'`; otherwise `return next(e)`. |
| Timer from `session.start` | `$.clock.every(1000, poll)` while a job is active: `job_status` via `$.mcp.call`, update progress, on completion append to the run list and redraw. Do **not** poll inside a single hook (10 s own-time budget; `$.clock.sleep` counts). |

Claude-facing tools (registered with `$.tool.register`, seen by Claude as `mcp__bif-studio__<name>`):

- `list_results` → `[{ id, params, npz, preview, summary }]` for this session. Small schema, no args or
  an optional `last: n`.
- (optional) `current_contract` → the describe JSON, so Claude knows what's loaded.

### 3.7 UI (pane)

Built from the describe JSON each render — controls are data-driven:

- **Header:** system name, rhsHash short, contract status (ok / errors).
- **Parameter controls**, one per `params` entry:
  - number → stepper "slider": `−` button, bar drawn from block chars sized to `e.props.bodyColumns`,
    value, `+` button. Step = (max − min) / N; optional `Input` for exact value.
  - `sweepable` params: a `Select` to choose the sweep param, and two inputs for the sweep range.
- **Run options:** enum → `Select`, int → stepper or `Input`.
- **Actions:** `Run` (explicit; no auto-run on every step, or debounce with `$.clock.after` and cancel
  stale timers), `Cancel`.
- **Progress line** while a job runs.
- **Result area:**
  - Terminal: `Image` with the preview PNG (file path is accepted).
  - Desktop app: no `Image`; `Svg` only (max 131,072 chars). Dense diagrams won't fit as vectors —
    test embedding the preview as a raster inside SVG, or show a text fallback. Branch on `e.surface`.
- **Per-run actions:**
  - `Open full-size` (hotkey `o`): launch external viewer **detached**:
    `$.process.run(['setsid', '-f', 'feh', '--auto-zoom', pngPath])` in try/catch.
    Plain `feh` would block `$.process.run` until timeout. Hide the button if neither
    `DISPLAY` nor `WAYLAND_DISPLAY` is set (`$.env.get`). macOS: `['open', path]`.
  - `Ask Claude` + `Input` for a question: `$.prompt.submit({ text })` with the question, the npz
    path, params, and the helper module hint. Don't `await` it while Claude is working
    (it resolves when the turn starts).
  - `Re-run with these params`.
  - Run history list (from `index.json`), each row selectable.

UI constraints to respect:

- Every control needs a unique, **stable** `key` based on param name / run id, not list position.
- Hotkeys: single digit or lowercase letter only; Tab/arrows/Enter/Esc are reserved. With many
  params, don't assign hotkeys per param — rely on Tab + Enter.
- Redraws capped at 30/s (visible pane); call `$.ui.invalidate('ui.render')` after state changes,
  or use `$.state` (reactive, survives module reloads; declare values in `types/index.d.ts`).
- `autoFocus` accepts only `true`; omit otherwise. `closeOnEscape` accepts only `true`.
- Invalid trees are silently replaced by Claude Code's own drawing — check the transcript line
  (with `--plugin-dir`) or the debug log.

### 3.8 Claude's side (skill + helper module)

- `core.inspect` helper module so Claude gets compact, meaningful output via Bash/venv python:
  `summary(npz)`, `period_regions(npz)`, `slice(npz, param=value)`, `compare(npz_a, npz_b)`.
- `skills/bif-studio/SKILL.md`: contract rules (3.2) with an example, where results live,
  how to use `core.inspect`, "never print full arrays", "view `*.preview.png` only if needed".

---

## 4. Implementation phases

1. ✅ (2026-10-06) **Read the repo** and summarize the current MCP server: tools, job model, how RHS is supplied,
   output formats. Confirm with the user before refactoring.
2. ✅ (2026-10-06) **Contract + describe:** implement `core.contract`, `describe` (CLI + MCP tool), tests with
   2–3 example systems (logistic map, Rössler, Izhikevich).
3. ✅ (2026-10-06) **Server slimming:** new job tools returning paths only; full-res + preview PNG writers;
   session folder + `index.json`; `cleanup`.
4. ✅ (2026-10-06) **Minimal mod:** plugin skeleton, `/bif` opens a pane showing the describe JSON as text.
   Run with `claude --plugin-dir ./bif-studio`; validate with `claude plugin validate ./bif-studio`.
5. ✅ (2026-10-06) **Controls + run loop:** generated controls, Run/Cancel, timer polling, preview `Image`.
6. ✅ (2026-10-06) **Claude integration:** RHS-change detection, error feedback via `$.prompt.submit`,
   `list_results` tool, Ask Claude button, skill + `core.inspect`.
7. ✅ (2026-10-06, desktop fallback parked) **Polish:** external viewer button, run history, hiding old tools, `$.store` prefs,
   Desktop-app fallback, tests with `claude plugin test`.

---

## 5. Items to VERIFY against the types for the installed build

- `$.mcp.call` argument shape and result shape; whether `$.mcp.connect` is needed for a server
  listed in the plugin manifest.
- `$.session.id()` / exact accessor for the session id.
- `tool.call` event fields for Write/Edit (file path field name) and the shape of the resolved
  result of `next(e)`.
- `Image` props for a file path; size limits (docs: PNG/RGBA up to 2 MiB, or a file path).
- `$.process.spawn` semantics (lifetime of spawned processes) — currently using `setsid -f` instead.
- `Client` element (custom pointer-driven widgets, e.g. draggable sliders): docs only say it's a
  region drawn by a second module that posts data to hooks via `ui.message`. Not needed for v1.

## 6. Decisions (2026-10-06)

- **Surface: terminal first.** Desktop `Svg` fallback is out of scope for v1.
- **External viewer: `xdg-open`** (verified on this WSL2/WSLg box: opens an ImageMagick window
  on the host). Launch detached anyway — `$.process.run(['setsid', '-f', 'xdg-open', png])` —
  since xdg-open's generic fallback can run the viewer in the foreground and would hold the call.
- **Old MCP tools: deprecated now, removed once the terminal setup works.** Deprecation =
  `tool.call` hook returns `{ deny: 'Deprecated: use the /bif pane or list_results.' }`, plus
  a "DEPRECATED" prefix on their docstrings.
- **Run button only.** No auto-run on parameter change.
- **RHS contract: extend the existing one, don't replace it** (see 3.2 revision below).

### 3.2 revision — contract v2 = existing contract + metadata

The class-based sketch in 3.2 is not numba-cuda compilable (`return` tuple, `p.a` attribute
access). The existing contract (`DIM`, `N_PARAMS`, `@cuda.jit(device=True) rhs(t, y, p, dy)`,
`y0`, optional `reset(t, y, p)` / `HAS_RESET`) already works with the GPU core, so v2 only adds
declarative metadata that `describe` reads:

```python
PARAMS = [   # order = index into p[]; N_PARAMS must equal len(PARAMS)
    dict(name="c", default=5.7, min=1.0, max=20.0, sweepable=True),
]
STATE = ["x", "y", "z"]          # optional names for y[0..DIM-1]; len must equal DIM
```

- `kind` is derived: `"hybrid"` if `reset` is present, else `"ode"`. (`"map"` is not supported
  by the core today.)
- Files without `PARAMS` still run (back-compat); `describe` reports them as `p[0]..p[N-1]`
  with no ranges and a warning.
- **Sweep index:** the core sweeps `p[0]` on x (and `p[param2_index]` on y, never 0). To let the
  pane pick any sweepable param for x, add a `param_index` argument to `run_bifurcation` /
  `run_bifurcation_2d` (small change in `_base_params` / grid fill) rather than reordering `p[]`.

### Phase 1 findings — current server (server.py, bifurcation_gpu.py, clustering.py)

- **Tools (10):** `get_rhs_template`, `start_job`, `start_job_2d`, `check_status`,
  `get_job_paths`, `get_result` (ImageContent base64), `get_data` (JSON arrays),
  `get_result_b64`, `save_chart`, `get_rhs_source`.
- **Job model:** in-memory `jobs` dict + daemon thread per job; progress 0–100 + message.
  No cancel. Job state is lost on server restart (files remain).
- **RHS supply:** built-in by name (`rossler`, `izhikevich`), a file path (`rhs_path`), or
  `custom_rhs=True` with body strings that `_prepare_rhs` templates into `custom_rhs.py`.
- **Outputs:** per job in `$TMPDIR/bifurcation_jobs/<uuid>/`: `chart.png` (matplotlib, 150 dpi),
  `data.npz`, `meta.json`, optionally `custom_rhs.py`. Nothing ever deletes these
  (the `get_job_paths` docstring claims they're cleared on exit — they aren't).
- **Already path-friendly:** `check_status` returns paths once done, so the new
  `job_status` is mostly a rename + `preview` + `summary` from the existing `meta`.
- **Two run kinds:** 1-D (scatter) and 2-D (DBSCAN cluster-count map, `clustering.py`) →
  `runOptions.mode` = `"bifurcation"` | `"period-map"`; 2-D adds `eps`, `eps_scale`,
  `min_samples`, `min_points`, `max_clusters`, `record_n`.
- **Shared run options:** `dt`, `t_end`, `transient`, `record_component`,
  `record_mode` (`maxima|tail|isi|spike`; last two need reset), `record_n`, `integrator`
  (`rk4|euler`), `fixed_params`.
- **Warm process caveat:** each job re-execs `bifurcation_gpu.py` and the RHS module, so the
  kernel is JIT-compiled per job. The warm MCP process still keeps the CUDA context, but to get
  real reuse, cache built kernels by `(rhsHash, record_mode, integrator)`.
- **Existing skill:** `skills/bifurcation-wizard-trigger/SKILL.md` — update it rather than adding
  a second skill.

### Phase 2 notes (done)

- `contract.py` (flat, not `core/contract`): `describe(path)` + CLI
  `python contract.py describe <file> [--pretty]` (exit 1 when not ok). MCP tool `describe` in
  server.py. Pass **absolute** paths from the mod; relative ones resolve against the server's cwd.
- Contract v2 adds `PARAMS` (name/default/min/max/sweepable, optional `sweep` range and `label`),
  `STATE`, `RUN_DEFAULTS`. runOptions entries may carry `modes` (only shown in those modes) and
  `modeDefaults` (per-mode default).
- `describe` checks structure and signatures only; it does not JIT-compile `rhs`. Compile errors
  still surface at job start (phase 3: report them as `job_status` errors).
- `_base_params` now fills unswept `p[]` slots from `PARAMS` defaults when no `fixed_params`
  are given (was zeros — which silently gave Izhikevich c = d = 0).
- Rössler is now 3 params (`c`, `a`, `b`, same order as before for `c` = p[0]); Lorenz added.
- Tests: `.venv/bin/python -m unittest discover tests` (17, incl. 2 GPU smoke runs).

### Phase 3 notes (done)

- **`studio.py`** holds the run API; `server.py` wraps it as MCP tools `submit_run`, `job_status`,
  `cancel_job`, `list_runs`, `cleanup`, `raster`. Names differ from §3.4 (`start_job` was taken
  by the legacy tool). Chart code moved to **`charts.py`** (shared by both).
- `submit_run(rhs_path, session, sweep, params, run_options)`: `sweep = {x: {param, min?, max?},
  y?: {...}}`; any sweepable param on either axis (core gained `param_index`). Everything is
  validated against `describe()` before a job starts.
- Session folder `~/.cache/bif-studio/<session>/` (override: `BIF_STUDIO_ROOT`); session id must be
  `[A-Za-z0-9_-]`. Per run: `run-NNNN.{json,npz,png,preview.png,chart.png}` + `index.json`. The
  `.png` is pixel-exact (1-D: log density, 1200 px high, ≤4096 wide; 2-D: period map, nearest
  upscale); `.chart.png` is the labelled matplotlib version (the one to open in the viewer).
- GPU core: launches in ≤16 chunks → real progress + cancel between chunks; modules and built
  kernels cached by file hash (warm re-run of the same system ≈ 0.6 s vs 1.2 s cold).
  `server.py` now imports the core once instead of re-exec'ing it per job.
- Compile errors come back as `job_status` `state: "error"` with numba's message (cut to 2000 chars).
- Legacy tools are marked DEPRECATED in their descriptions only (not denied yet: the pane can't
  run jobs until phase 5).

### Phase 4 notes (done) — and the display finding

- Mod at `~/.claude/dev-mods/<session>/bif-studio/` (move it next to the server before sharing).
  `/bif [path]` → `describe` via `$.mcp.call('bifurcation-wizard', ...)` → pane with params,
  state, run options, contract errors/warnings, `r` = re-describe. Last path kept in `$.store`.
  `session.start` calls `cleanup(24, keep_session=<session id>)`.
- Checks: `claude plugin validate`, tsc (strict), `claude plugin test` (4 tests, fake server).
- **The dev-mods folder is per session ID.** When the session ID changed mid-conversation, hot
  reload started watching `~/.claude/dev-mods/<new id>/` and `/bif` vanished (no error line).
  Fix was copying the mod over. For durability, move it next to the server and load it with
  `CLAUDE_CODE_PLUGIN_DIRS` (settings `env`) or `claude --plugin-dir`.
- **Auto mode blocks the mod's `$.mcp.call`**: the classifier gives "no verdict" for a call no
  prompt asked for. Fix (done 2026-10-06): allow rules in `~/.claude/settings.json` for the 7 studio
  tools (`mcp__bifurcation-wizard__{describe,submit_run,job_status,cancel_job,list_runs,cleanup,raster}`).
  Anyone installing the mod needs the same rules (document in the README / skill).
- Gotchas: `Text` drops `key` (wrap keyed rows in `Box`); test hooks for `$` calls answer
  `{ value }`; a test must stand in for every engine call the mod makes (`ui.open`, `store.*`, ...).
- **`Image` only draws with the kitty graphics protocol (kitty, Ghostty)**; elsewhere its `alt`.
  Windows Terminal (likely here, WSL2) has no kitty protocol, so the pane preview uses **`Raster`**
  instead: the `raster` MCP tool turns a run PNG into half-block cells (2 px per cell vertically,
  ≤512×256 cells, 32 colours). Phase 5: `Raster` by default, `Image` when the terminal is
  kitty/Ghostty.

### Phase 5 notes (done)

- Pane: Mode `Select`; sweep axis `Select` + min/max `Input` (y appears in period-map mode);
  per-parameter stepper `−` bar `+` (step = (max−min)/50) plus an `Input` for exact values;
  swept params show "swept on x/y"; `a` toggles run options (enum → `Select`, numbers → `Input`,
  only options for the current mode). Invalid input is refused with a toast, never sent.
- Hotkeys: `r` Run, `x` Cancel, `d` Re-describe, `a` run options.
- Run → `submit_run` (session = `$.session.id()`), `$.clock.every(500)` polls `job_status` until
  not running, then `list_runs` and selects the new run. A reload restarts polling from
  `session.start` if `$.state.job` is still running.
- Preview: `raster` tool → `Raster` (Image when TERM/TERM_PROGRAM says kitty/ghostty or
  KITTY_WINDOW_ID is set); cached per (png, columns, rows). Runs `Select` switches between runs.
- Form values are kept across re-describe of the same file (clamped to new ranges), reset for a
  different file. Mode switch re-applies `modeDefaults`.
- **FastMCP gotcha:** a tool returning a bare list is sent as one content block per item →
  `list_runs` now returns `{"runs": [...]}`. Any future list-returning tool: wrap it.
- Tests: 12 mod tests (fake server + `mock.clock`/`mock.env`/`mock.store`), 28 Python tests.
  Note the test engine validates `Raster` cells (columns×rows×12 bytes) like the real one.
- Desktop/VS Code surfaces show a one-line note only (terminal first, per §6).

### Preview quality (2026-10-06)

- Preview = full image whenever its PNG fits `PREVIEW_MAX_BYTES` (2 MiB); otherwise shrunk only
  as far as needed (iterative, by byte size). In practice all diagrams fit: 16–391 KiB
  (1-D 200…20000 steps, 2-D 300²/1000²). Low-res 1-D diagrams are widened to ≥1600 px by
  repeating columns (were 200×1200 strips).
- `raster` picks its shrink from the run's mode (`run-NNNN.json`): period map → nearest
  (no blended false colours), keeps aspect; 1-D → fills the box, block alpha = presence
  (0.45) + sqrt(mean density) (0.55), so thin branches and chaotic shading both survive a ~50×
  shrink. ~30 ms for 4096×1200 → 100×30 cells, ~200 colour pairs.
- Cost note: Claude viewing a big preview is capped by the API's own downscale (~1.15 MP),
  so a 4096×1200 preview costs about the same tokens as a 2000×1200 one.

### Phase 6 notes (done)

- **RHS edit check:** `tool.call` hooks on Write/Edit (with `.catch` → pass-through). The
  loaded file, or any Write whose content has `@cuda.jit … def rhs(`, is described after the
  write and a `bif-studio:` line goes into the tool result's `context` — Claude reads it in the
  same turn. Chosen over `$.prompt.submit` (§3.6): no extra turn, no loop risk.
- **`mcp__bif-studio__results`** (registered in `session.start`, served with a RegExp matcher):
  compact runs (id, sweep, fixed, summary, npz/preview/chart) + the `inspect` command line
  (server's `list_runs` now returns it, so no path is hardcoded in the mod).
- **Ask Claude:** `Input` under a finished run → `$.prompt.submit` with the question, run,
  paths and the runinspect command (framed as the plugin's message).
- **`runinspect.py`** (not `core.inspect`; flat repo): `summary`, `slice`, `compare`, `--json`.
  1-D periods use `cluster_grid(..., max_noise_frac=0.3)`.
- **Clustering finding:** plain cluster counting labels a chaotic column "period 1" when one
  accidental cluster forms (Rössler c=7: 34 maxima, 1 cluster + 31 noise). New opt-in
  `max_noise_frac` in `clustering.cluster_grid`. **Maps switched (user's call, 2026-10-06):**
  run option `max_noise_frac` (period-map only, default 0.3, 1 = old plain count), also on the
  legacy `start_job_2d`. Rössler 300² map: chaotic 8.3 % → 34.0 %, 25.6 % of cells relabelled,
  2.0 s → 2.6 s.
- `param_steps` max raised 20000 → 1,000,000 (trigger skill recommends 1e6 for 1-D).
- Skill `skills/bif-studio/SKILL.md` ships inside the mod; the repo's trigger skill now points
  to the studio workflow (the copy installed in claude.ai must be re-uploaded to match).
- Tests: 17 mod, 37 Python.

## 7. References

- Mods overview: https://code.claude.com/docs/en/plugins/mods/overview
- Draw in the interface: https://code.claude.com/docs/en/plugins/mods/interface
- Mods API: https://code.claude.com/docs/en/plugins/mods/api
- Mods reference (events, methods, elements, limits): https://code.claude.com/docs/en/plugins/mods/reference
- Interface gallery: https://code.claude.com/docs/en/plugins/mods/gallery
- Type declarations: https://github.com/anthropics/claude-code/blob/main/mods/types/claude-code.d.ts


### Systems folder + named systems (2026-10-06)

- Built-ins moved to **`~/mcp-tools/systems/`** (`rossler`, `izhikevich`, `lorenz`).
- New server tools (`systems.py`): `save_system(name, source)` → draft in
  `~/.cache/bif-studio/drafts/<name>.py` + contract check; `keep_system(name, overwrite=False)` →
  moves a passing draft to `systems/` (refuses to replace unless `overwrite`); `list_systems()` →
  both + `resolve` (draft wins). Drafts survive `/mcp` reconnects (server restarts) and expire
  per file after 24 h untouched (`cleanup`). Works for any MCP client — no file access needed.
- Skill rule: **ask the user to name the system before saving** (AskUserQuestion with
  suggestions); save via `save_system`, not Write; `keep_system` only on the user's wish,
  `overwrite` only with consent.
- Mod: `/bif <name>` resolves via `list_systems` (allow rule added — the mod calls it); follows
  `save_system`/`keep_system` of the loaded system. `list_runs(session="")` → latest session,
  the fallback for Claude when the mod's tool is unavailable.
- **`mcp__bif-studio__results` unavailable (503 CLIENT_HTTP_NOT_IMPLEMENTED):** `$.tool.register`
  tools are served by the engine itself over an in-process HTTP MCP endpoint (hence "built-in",
  HTTP, `127.0.0.1:<port>` owned by the session process). Since 14:12 this session runs
  daemon-hosted (`--fork-session --resume`, `bg-pty-host`), and that endpoint answers 503 there.
  Not fixable in the mod; re-test in a fresh terminal session. `list_runs` is the fallback.
- Tests: 20 mod, 44 Python.

### Sequence period rule (2026-10-06, user's call)

- DBSCAN cluster counting labels a chaotic attractor whose maxima fall into a few narrow bands
  as a low period (Sprott B at a=b=1, known chaotic: "period 4/8"). New
  `clustering.period_grid`: period = smallest k with |m[n+k] − m[n]| ≤ tol for all n
  (tol = `period_tol` × robust spread, default 1e-3; k ≤ `max_period`, default 32; needs
  ≥ 2k+2 values), else aperiodic. Vectorised in row chunks: 90k cells in 0.6 s.
- Used for 1-D (`runinspect`) and period maps (studio + legacy `start_job_2d`) for record
  modes maxima/isi/spike; `tail` (ring buffer, not time-ordered) keeps DBSCAN
  (+ `max_noise_frac`). Map summary carries `rule`. Chart legend/footer name the rule.
- Results: Rössler 1-D shows 1→2→4→8→16→chaos (c≈4.2), period-3 window at 5.18 doubling to
  6, 12. Sprott B (a,b) map: windows lie on a ∝ b³ curves, as the scaling predicts.
- Caveat: strict test → unconverged orbits near bifurcations read "chaotic"; longer
  transient fixes it (Rössler c≈2.9–3.1: transient 200 → chaotic, 1200 → period 2).
  Documented in the skill.
- `sprott_b` (ẋ=yz, ẏ=x−by, ż=a−xy; a=b=1 is case B; dynamics depends on a/b³ only) kept in
  `systems/`. Tests: 47 Python.

### Phase 7 — polish (2026-10-06)

- **Old job tools removed** (user's call): `start_job`, `start_job_2d`, `check_status`,
  `get_job_paths`, `get_result`, `get_data`, `get_result_b64`, `save_chart`, `get_rhs_source`
  and their worker/templates. `server.py` 951 → 305 lines, 11 tools. `get_rhs_template` rewritten
  for contract v2 + `save_system`. Backup: `~/.cache/server.py.before-removal`. Claude Desktop
  chat can't run anything until the desktop question is settled.
- **Mod moved** to `~/mcp-tools/bif-studio/`, loaded via `CLAUDE_CODE_PLUGIN_DIRS` in
  `~/.claude/settings.json` `env` (from the next session). This session: the dev-mods entry is a
  symlink to it. Old session's dev-mods copy deleted. README added (setup, allow rules, keys).
- **Viewer**: `o` = labelled chart, `p` = full-res PNG, via `setsid -f xdg-open` (hidden without
  DISPLAY/WAYLAND_DISPLAY). Untested live: in auto mode `$.process.run` may be gated like
  `$.mcp.call` was; a failure shows as a toast.
- **Remembered settings** (`$.store`): run-options toggle; per-file form (`form:<path>`), so a
  system reopens with its last parameters/sweeps/mode, also across sessions.
- Run history = the Runs select (phase 5). Desktop fallback: parked (see §6).
- Skills: trigger skill rewritten for the studio-only workflow (re-upload to claude.ai);
  bif-studio skill no longer mentions the removed tools.
- Tests: 23 mod, 47 Python.

### Beyond the terminal: what's possible (research, 2026-10-06)

Sources: mods docs (overview, interface, reference; v2.1.289), types for 2.1.291, Desktop-WSL
docs, MCP Apps docs/examples (modelcontextprotocol/ext-apps).

| Where | Mod runs? | Draws? | Our pane |
| :- | :- | :- | :- |
| Terminal `claude` (incl. editor terminals) | yes | everything | works today |
| Desktop app, Code tab, Windows-native session | yes (v2.1.286+) | Pane, Input/Select, **Svg**, Client — **no Raster/Image** | needs a desktop branch; server reached via `wsl.exe` |
| Desktop app, Code tab, **WSL session** | **no** ("plugins aren't available in WSL sessions") | no | not possible |
| VS Code extension chat panel | hooks yes | **no** | not possible |
| Desktop **chat** / claude.ai | no mods at all | — | **MCP App** instead |

- Desktop `Svg`: ≤131,072 chars, drawn as an image (or a script-less sandboxed frame with
  `isInteractive`); whether `<image href="data:image/png;base64,…">` survives the scrub is
  undocumented → test. Fallback: vector (RLE rects of a downsampled map, binned 1-D density).
- `Client` (terminal + desktop): local state, pointer/keys, frame timer, but text elements only
  (no pixels); good for a draggable slider, not for the plot.
- **MCP Apps** (Claude Desktop + claude.ai, VS Code, ChatGPT, …): tool `_meta.ui.resourceUri` →
  `ui://` HTML resource (`text/html;profile=mcp-app`) rendered in a sandboxed iframe; the view
  can call server tools (incl. app-only tools the model never sees), send chat messages, update
  model context; inline/fullscreen/pip; images via data URIs/canvas; network only to declared
  CSP domains. Our `mcp` 1.28 FastMCP supports `tool(meta=)` and `resource(mime_type=)`.
- Windows Desktop ↔ WSL server: `claude_desktop_config.json` command `wsl.exe` running the venv
  python (to verify).