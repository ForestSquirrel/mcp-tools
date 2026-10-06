"""
RHS contract v2 — validation and introspection of an RHS file.

v2 is the v1 contract (DIM, N_PARAMS, rhs, y0, optional reset/HAS_RESET; see
server.RHS_TEMPLATE_DOC) plus declarative metadata the Bifurcation Studio pane
builds its controls from:

  PARAMS = [dict(name="c", default=5.7, min=1.0, max=20.0,
                 sweepable=True, sweep=(2.0, 8.0), label="c")]   # p[i] order
  STATE  = ["x", "y", "z"]                                       # optional
  RUN_DEFAULTS = dict(dt=0.25, integrator="euler")               # optional

describe(path) never raises: it returns {"ok": False, "errors": [...]} for a
broken file, and files without PARAMS still describe (with a warning).

CLI:  python contract.py describe systems/rossler.py [--pretty]   (exit 1 if not ok)
"""

import ast
import hashlib
import importlib.util
import json
import keyword
import math
import sys
import traceback
import uuid
from pathlib import Path

CONTRACT_VERSION = 2

MODES = ("bifurcation", "period-map")
RECORD_MODES = ("maxima", "tail", "isi", "spike")
RESET_ONLY_RECORD_MODES = ("isi", "spike")
INTEGRATORS = ("rk4", "euler")

PARAM_KEYS = {"name", "default", "min", "max", "sweepable", "sweep", "label"}


def _run_options(dim: int, hybrid: bool) -> list[dict]:
    """
    Options a run takes besides parameter values, with their server-side
    names. `modes` limits an option to some run modes; `modeDefaults`
    overrides `default` per mode.
    """
    record_modes = list(RECORD_MODES if hybrid else
                        [m for m in RECORD_MODES if m not in RESET_ONLY_RECORD_MODES])
    p2d = ["period-map"]
    return [
        {"name": "mode", "type": "enum", "options": list(MODES),
         "default": "bifurcation"},
        # Up to 1e6 (the trigger skill's recommended 1-D resolution); the core
        # refuses runs whose output buffer exceeds 2 GiB (lower record_n).
        {"name": "param_steps", "type": "int", "default": 2000,
         "modeDefaults": {"period-map": 300}, "min": 10, "max": 1_000_000},
        {"name": "param2_steps", "type": "int", "default": 300,
         "min": 10, "max": 2000, "modes": p2d},
        {"name": "dt", "type": "float", "default": 0.25 if hybrid else 0.005,
         "min": 1e-6, "max": 10.0},
        {"name": "t_end", "type": "float", "default": 3000.0 if hybrid else 200.0,
         "min": 0.0, "max": 1e7},
        {"name": "transient", "type": "float",
         "default": 1500.0 if hybrid else 100.0, "min": 0.0, "max": 1e7},
        {"name": "integrator", "type": "enum", "options": list(INTEGRATORS),
         "default": "euler" if hybrid else "rk4"},
        {"name": "record_mode", "type": "enum", "options": record_modes,
         "default": "isi" if hybrid else "maxima"},
        {"name": "record_component", "type": "int", "default": 0,
         "min": 0, "max": dim - 1},
        {"name": "record_n", "type": "int", "default": 500,
         "modeDefaults": {"period-map": 64}, "min": 1, "max": 4096},
        {"name": "eps", "type": "float", "default": 0.0, "min": 0.0,
         "max": 1e9, "modes": p2d},
        {"name": "eps_scale", "type": "float", "default": 0.01,
         "min": 1e-4, "max": 1.0, "modes": p2d},
        {"name": "min_samples", "type": "int", "default": 3,
         "min": 1, "max": 100, "modes": p2d},
        {"name": "min_points", "type": "int", "default": 4,
         "min": 1, "max": 1000, "modes": p2d},
        {"name": "max_clusters", "type": "int", "default": 6,
         "min": 1, "max": 20, "modes": p2d},
        # Period test (record modes maxima/isi/spike): the smallest k with
        # |m[n+k] - m[n]| <= period_tol * spread for all n; none up to
        # max_period = aperiodic. See clustering.period_grid.
        {"name": "period_tol", "type": "float", "default": 1e-3,
         "min": 1e-7, "max": 0.05, "modes": p2d},
        {"name": "max_period", "type": "int", "default": 32,
         "min": 1, "max": 64, "modes": p2d},
        # DBSCAN (record mode tail only, whose values are not in time order):
        # a cell whose noise share exceeds this is chaotic; 1 = plain count.
        {"name": "max_noise_frac", "type": "float", "default": 0.3,
         "min": 0.0, "max": 1.0, "modes": p2d},
    ]


def _is_num(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def _top_level_lines(tree: ast.Module) -> dict:
    """Name -> (assignment line, [line of each list/tuple element])."""
    out = {}
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        elif isinstance(node, ast.FunctionDef):
            targets = [node.name]
        value = getattr(node, "value", None)
        elts = ([e.lineno for e in value.elts]
                if isinstance(value, (ast.List, ast.Tuple)) else [])
        for name in targets:
            out[name] = (node.lineno, elts)
    return out


def _exec_module(path: Path):
    """Import the file under a throwaway name, never cached in sys.modules."""
    spec = importlib.util.spec_from_file_location(
        f"_rhs_describe_{uuid.uuid4().hex}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _exec_error(path: Path, exc: BaseException) -> str:
    """'line N: Type: msg' using the innermost frame inside the RHS file."""
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if Path(frame.filename).resolve() == path:
            line = frame.lineno
    where = f"line {line}: " if line else ""
    return f"{where}{type(exc).__name__}: {exc}"


def describe(path: str) -> dict:
    """Validate an RHS file and return its JSON description (see module doc)."""
    errors: list[str] = []
    warnings: list[str] = []

    def fail():
        return {"contractVersion": CONTRACT_VERSION, "ok": False,
                "path": str(p), "errors": errors, "warnings": warnings}

    p = Path(path).expanduser().resolve()
    try:
        raw = p.read_bytes()
    except OSError as exc:
        errors.append(f"cannot read file: {exc}")
        return fail()

    rhs_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
    try:
        tree = ast.parse(raw, filename=str(p))
    except SyntaxError as exc:
        errors.append(f"line {exc.lineno}: SyntaxError: {exc.msg}")
        return fail()
    lines = _top_level_lines(tree)

    def at(name, i=None):
        line, elts = lines.get(name, (None, []))
        if i is not None and i < len(elts):
            line = elts[i]
        return f"line {line}: " if line else ""

    try:
        mod = _exec_module(p)
    except BaseException as exc:  # user code: SystemExit, anything
        errors.append(_exec_error(p, exc))
        return fail()

    # ── v1 core ──────────────────────────────────────────────────────────────
    dim = getattr(mod, "DIM", None)
    n_params = getattr(mod, "N_PARAMS", None)
    if not (isinstance(dim, int) and not isinstance(dim, bool) and dim > 0):
        errors.append(f"{at('DIM')}DIM must be a positive int (got {dim!r})")
        dim = None
    if not (isinstance(n_params, int) and not isinstance(n_params, bool)
            and n_params > 0):
        errors.append(f"{at('N_PARAMS')}N_PARAMS must be a positive int "
                      f"(got {n_params!r})")
        n_params = None

    for fn in ("rhs", "reset"):
        f = getattr(mod, fn, None)
        if f is None:
            if fn == "rhs":
                errors.append("rhs is missing: define "
                              "@cuda.jit(device=True) def rhs(t, y, p, dy)")
            continue
        if not hasattr(f, "py_func"):
            errors.append(f"{at(fn)}{fn} must be decorated with "
                          f"@cuda.jit(device=True)")
            continue
        want = 4 if fn == "rhs" else 3
        got = f.py_func.__code__.co_argcount
        if got != want:
            sig = "(t, y, p, dy)" if fn == "rhs" else "(t, y, p)"
            errors.append(f"{at(fn)}{fn} must take {want} arguments {sig}, "
                          f"takes {got}")

    hybrid = getattr(mod, "reset", None) is not None and \
        bool(getattr(mod, "HAS_RESET", True))

    y0 = getattr(mod, "y0", None)
    if y0 is None:
        errors.append("y0 is missing: a list of DIM initial values")
    elif not (isinstance(y0, (list, tuple)) and all(_is_num(v) for v in y0)):
        errors.append(f"{at('y0')}y0 must be a list of finite numbers")
        y0 = None
    elif dim is not None and len(y0) != dim:
        errors.append(f"{at('y0')}y0 has {len(y0)} values, DIM is {dim}")

    # ── PARAMS ───────────────────────────────────────────────────────────────
    params: list[dict] = []
    raw_params = getattr(mod, "PARAMS", None)
    if raw_params is None:
        warnings.append("PARAMS is missing: parameters are shown as p[0].. "
                        "with no names or ranges (contract v1 file)")
        for i in range(n_params or 0):
            params.append({"index": i, "name": f"p{i}", "label": f"p[{i}]",
                           "default": None, "min": None, "max": None,
                           "sweepable": True, "sweep": None})
    elif not isinstance(raw_params, (list, tuple)):
        errors.append(f"{at('PARAMS')}PARAMS must be a list of dicts")
    else:
        if n_params is not None and len(raw_params) != n_params:
            errors.append(f"{at('PARAMS')}PARAMS has {len(raw_params)} entries, "
                          f"N_PARAMS is {n_params}")
        seen = set()
        for i, entry in enumerate(raw_params):
            where = f"{at('PARAMS', i)}PARAMS[{i}]"
            if not isinstance(entry, dict):
                errors.append(f"{where} must be a dict")
                continue
            name = entry.get("name")
            if isinstance(name, str):
                where += f" ({name!r})"
            if not (isinstance(name, str) and name.isidentifier()
                    and not keyword.iskeyword(name)):
                errors.append(f"{where}: 'name' must be an identifier string")
            elif name in seen:
                errors.append(f"{where}: duplicate name")
            seen.add(name)
            extra = set(entry) - PARAM_KEYS
            if extra:
                errors.append(f"{where}: unknown keys {sorted(extra)}")
            missing = [k for k in ("default", "min", "max") if k not in entry]
            if missing:
                errors.append(f"{where}: missing {', '.join(map(repr, missing))}")
                continue
            lo, hi, d = entry["min"], entry["max"], entry["default"]
            if not all(_is_num(v) for v in (lo, hi, d)):
                errors.append(f"{where}: default, min and max must be "
                              f"finite numbers")
                continue
            if not lo < hi:
                errors.append(f"{where}: min ({lo}) must be < max ({hi})")
            elif not lo <= d <= hi:
                errors.append(f"{where}: default {d} is outside [{lo}, {hi}]")
            sweepable = entry.get("sweepable", False)
            if not isinstance(sweepable, bool):
                errors.append(f"{where}: 'sweepable' must be True or False")
            sweep = entry.get("sweep")
            if sweep is not None:
                if not (isinstance(sweep, (list, tuple)) and len(sweep) == 2
                        and all(_is_num(v) for v in sweep)
                        and lo <= sweep[0] < sweep[1] <= hi):
                    errors.append(f"{where}: 'sweep' must be (lo, hi) with "
                                  f"min <= lo < hi <= max")
                    sweep = None
                else:
                    sweep = [float(sweep[0]), float(sweep[1])]
            label = entry.get("label", name)
            params.append({"index": i, "name": name,
                           "label": label if isinstance(label, str) else name,
                           "default": float(d), "min": float(lo),
                           "max": float(hi), "sweepable": sweepable is True,
                           "sweep": sweep or [float(lo), float(hi)]})
        if not errors and not any(q["sweepable"] for q in params):
            errors.append(f"{at('PARAMS')}no parameter is sweepable: mark at "
                          f"least one with sweepable=True")

    # ── STATE ────────────────────────────────────────────────────────────────
    names = getattr(mod, "STATE", None)
    if names is not None and not (
            isinstance(names, (list, tuple))
            and all(isinstance(n, str) and n for n in names)
            and (dim is None or len(names) == dim)):
        errors.append(f"{at('STATE')}STATE must be a list of DIM non-empty "
                      f"names")
        names = None
    state = [{"index": i,
              "name": names[i] if names else f"y{i}",
              "default": float(y0[i]) if y0 and i < len(y0) else None}
             for i in range(dim or 0)]

    # ── run options + RUN_DEFAULTS ───────────────────────────────────────────
    run_options = _run_options(dim or 1, hybrid)
    overrides = getattr(mod, "RUN_DEFAULTS", None)
    if overrides is not None:
        if not isinstance(overrides, dict):
            errors.append(f"{at('RUN_DEFAULTS')}RUN_DEFAULTS must be a dict")
        else:
            by_name = {o["name"]: o for o in run_options}
            for key, value in overrides.items():
                where = f"{at('RUN_DEFAULTS')}RUN_DEFAULTS[{key!r}]"
                opt = by_name.get(key)
                if opt is None:
                    errors.append(f"{where}: unknown run option "
                                  f"(known: {', '.join(by_name)})")
                    continue
                if opt["type"] == "enum":
                    ok = value in opt["options"]
                elif opt["type"] == "int":
                    ok = (isinstance(value, int) and not isinstance(value, bool)
                          and opt["min"] <= value <= opt["max"])
                else:
                    ok = _is_num(value) and opt["min"] <= value <= opt["max"]
                if not ok:
                    allowed = (opt["options"] if opt["type"] == "enum"
                               else f"{opt['type']} in [{opt['min']}, {opt['max']}]")
                    errors.append(f"{where}: {value!r} is not allowed "
                                  f"({allowed})")
                    continue
                opt["default"] = float(value) if opt["type"] == "float" else value
                opt.pop("modeDefaults", None)

    if errors:
        return fail()

    return {
        "contractVersion": CONTRACT_VERSION,
        "ok": True,
        "path": str(p),
        "system": {"name": p.stem, "kind": "hybrid" if hybrid else "ode",
                   "rhsHash": rhs_hash, "dim": dim, "nParams": n_params},
        "params": params,
        "state": state,
        "runOptions": run_options,
        "errors": [],
        "warnings": warnings,
    }


def _main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "describe":
        print("usage: python contract.py describe <rhs.py> [--pretty]",
              file=sys.stderr)
        return 2
    out = describe(argv[1])
    print(json.dumps(out, indent=2 if "--pretty" in argv else None))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
