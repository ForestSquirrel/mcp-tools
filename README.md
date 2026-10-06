# mcp-tools: Bifurcation Studio

GPU bifurcation diagrams and 2-D period maps of dynamical systems, explored from inside
Claude Code.

You open a system in the `/bif` pane, pick sweep axes, parameters and run options, and press
Run. The sweep runs on your GPU (numba-cuda), one trajectory per parameter value, and the
preview appears in the pane. Claude has a small, specific part: it writes new systems for you
and answers questions about runs ("where does chaos start?") from compact summaries, never
from raw arrays or pictures. A run costs Claude no tokens.

It has two parts:

| Part | What it is | Where |
| :- | :- | :- |
| **`bifurcation-studio`** MCP server | Python (FastMCP, stdio): integrates the systems on the GPU, stores runs, validates RHS files | `server.py` and the modules beside it |
| **`bif-studio`** Claude Code plugin | The `/bif` pane (a hooks mod) and the skill Claude reads | `bif-studio/` |

The pane calls the server, so both are needed. The two names differ on purpose: Claude Code
reserves the plugin's name, and a server registered under it gets shut down.

## Requirements

- An NVIDIA GPU and a CUDA toolkit numba can find (developed on an RTX 5070 Ti with CUDA 13.2
  under WSL2).
- Python 3.11+ (developed on 3.13).
- Claude Code in a terminal. The pane docks beside the transcript in the fullscreen renderer
  (the default; `/tui fullscreen` if yours is off) from 110 columns, and opens inline above
  the prompt otherwise. kitty and Ghostty show the preview as an image; other terminals draw it
  in coloured half-block characters.

## Install

1. **Clone and set up the server**

   ```
   git clone https://github.com/ForestSquirrel/mcp-tools.git ~/mcp-tools
   cd ~/mcp-tools
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. **Register the server** with Claude Code (user scope):

   ```
   claude mcp add -s user bifurcation-studio -- ~/mcp-tools/.venv/bin/python ~/mcp-tools/server.py
   ```

3. **Install the plugin**, at the prompt of a Claude Code session:

   ```
   /plugin install bif-studio --marketplace ForestSquirrel/mcp-tools
   ```

   Answer `y` to add the marketplace, then pick a scope (user is the usual one).

4. **Allow the pane's tools (auto mode).** The pane calls server tools on its own, which auto
   mode's classifier can't tie to a request of yours and refuses. Add these to
   `permissions.allow` in `~/.claude/settings.json`:

   ```json
   "mcp__bifurcation-studio__describe", "mcp__bifurcation-studio__submit_run",
   "mcp__bifurcation-studio__job_status", "mcp__bifurcation-studio__cancel_job",
   "mcp__bifurcation-studio__list_runs", "mcp__bifurcation-studio__cleanup",
   "mcp__bifurcation-studio__raster", "mcp__bifurcation-studio__list_systems"
   ```

   `save_system` and `keep_system` write files and are Claude's: leave them to the normal
   permission check.

Restart Claude Code and run `/bif rossler`.

## Use

- `/bif <name>` opens a system: `rossler`, `lorenz`, `izhikevich`, `sprott_b`, or one you
  saved. `/bif path/to/file.py` opens a file.
- **New system:** describe it to Claude ("a Chua circuit with α and β sweepable"). Claude asks
  you to name it, saves a draft and checks it against the contract, then tells you to
  `/bif <name>`. Drafts live in `~/.cache/bif-studio/drafts/` for about 24 h; say "keep it" to
  make one permanent in `systems/`.
- **Questions:** type a question in the pane's **Ask Claude** field below a finished run, or
  just ask in the chat. Claude analyses the run with `runinspect.py` (summary, slice,
  compare) and answers with parameter values.

Pane keys:

| Key | Does |
| :- | :- |
| `r` | Run |
| `x` | Cancel the running job |
| `d` | Re-describe the file (after you edited it) |
| `a` | Show/hide run options |
| `o` / `p` | Open the labelled chart / the full-resolution image in the desktop viewer (needs a display) |
| Tab / arrows / Enter | Move between and use the controls |

Each system's parameter, sweep and option settings are remembered across sessions. Runs are
stored in `~/.cache/bif-studio/<session>/` (`run-NNNN.npz`, `.png`, `.preview.png`,
`.chart.png`); folders untouched for 24 h are deleted when a session starts.

## Writing a system by hand

One system per `.py` file in `systems/`: plain numba-cuda device code, scalar math only.

```python
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
```

Spiking and reset models (Izhikevich, LIF, AdEx) add a `reset(t, y, p)` device function that
returns 1 when it fired. The full contract, with a reset-model example and the run-option
defaults a system can set, is `RHS_TEMPLATE_DOC` in `server.py` (the `get_rhs_template` tool).
`systems/izhikevich.py` is a working reset model.

## Development

```
.venv/bin/python -m unittest discover tests   # server
claude plugin test bif-studio                 # pane
```

To run the plugin from this working copy instead of an installed copy, so edits hot-reload,
skip step 3 and add the folder to `env` in `~/.claude/settings.json`:

```json
"env": { "CLAUDE_CODE_PLUGIN_DIRS": "/home/<you>/mcp-tools/bif-studio" }
```

(or for one session: `claude --plugin-dir ~/mcp-tools/bif-studio`).

`bif-studio/README.md` describes the plugin's own files.
