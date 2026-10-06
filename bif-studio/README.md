# Bifurcation Studio (Claude Code mod)

A `/bif` pane in Claude Code for GPU bifurcation diagrams and period maps. You pick sweep
axes, parameters and run options and press Run; Claude writes systems and answers questions
about runs from compact summaries, never from raw arrays.

The mod is the UI. The compute is the Python MCP server in the parent folder
(`../server.py`), registered as `bifurcation-studio` (not `bif-studio`: that name is the plugin's, and a server of the same name gets shut down). The plugin's skill (`skills/bif-studio`) is the
only skill; the server's `get_rhs_template` holds the RHS contract.

## Setup

1. **Server**: register it once (user scope):

   ```
   claude mcp add -s user bifurcation-studio -- ~/mcp-tools/.venv/bin/python ~/mcp-tools/server.py
   ```

2. **Mod**: load this folder in every session by adding it to `~/.claude/settings.json`:

   ```json
   "env": { "CLAUDE_CODE_PLUGIN_DIRS": "/home/<you>/mcp-tools/bif-studio" }
   ```

   (or for one session: `claude --plugin-dir ~/mcp-tools/bif-studio`). It hot-reloads while
   you edit it.

3. **Permissions (auto mode)**: the pane calls server tools on its own, which auto mode's
   classifier cannot tie to a request and refuses. Allow the pane's tools in
   `~/.claude/settings.json` → `permissions.allow`:

   ```
   mcp__bifurcation-studio__describe        mcp__bifurcation-studio__submit_run
   mcp__bifurcation-studio__job_status      mcp__bifurcation-studio__cancel_job
   mcp__bifurcation-studio__list_runs       mcp__bifurcation-studio__cleanup
   mcp__bifurcation-studio__raster          mcp__bifurcation-studio__list_systems
   ```

   `save_system` / `keep_system` write files and are Claude's: leave them to the normal check.

## Use

- `/bif <name>` opens a system (`rossler`, `izhikevich`, `lorenz`, `sprott_b`, or any you
  saved); `/bif path/to/file.py` opens a file.
- Ask Claude for a new system: it asks you for a name, saves a draft (kept ~24 h), and says
  when it passes the contract. "Keep it" makes it permanent in `../systems/`.

| Key | Does |
| :- | :- |
| `r` | Run |
| `x` | Cancel the running job |
| `d` | Re-describe the file (after you edited it) |
| `a` | Show/hide run options (remembered) |
| `o` / `p` | Open the labelled chart / the full-resolution image in the desktop viewer (`xdg-open`; only with a display) |
| Tab / arrows / Enter | Move between and use the controls |

Below a finished run: the preview, **Ask Claude** (a question about this run, sent with its
data paths), and **Runs** (switch between this session's runs). Each system's parameter,
sweep and option settings are remembered across sessions.

The preview is an `Image` on kitty/Ghostty and coloured half-block cells elsewhere.

## Files

- `hooks/register.tsx`: the hooks module (pane, `/bif`, polling, RHS-edit checks, Ask Claude)
- `types/index.d.ts`: the `$.state` contract
- `skills/bif-studio/SKILL.md`: what Claude reads (contract, workflow, analysis)
- `tests/pane.test.ts`: `claude plugin test .`

Results live in `~/.cache/bif-studio/<session>/` (`run-NNNN.{json,npz,png,preview.png,chart.png}`);
folders untouched for 24 h are deleted when a session starts.

## Known limits

- Terminal only for now; other surfaces show a one-line note.
