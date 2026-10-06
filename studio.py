"""
Bifurcation Studio runs — the path-only job API behind the Claude Code mod.

A run lives in a session folder,  STUDIO_ROOT/<session>/ :

  run-0007.json         metadata + status (what job_status and index.json carry)
  run-0007.npz          raw arrays, plus `meta` (the JSON above, as a string)
  run-0007.png          full resolution, pixel-exact (density / period map)
  run-0007.preview.png  small, for the pane's Image and for Claude to view
  run-0007.chart.png    labelled matplotlib chart (axes, legend)
  index.json            [run summaries], newest last

Nothing here returns image bytes or arrays: callers get paths and a short
summary. Python owns cleanup (the mod cannot delete files).
"""

import io
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

import bifurcation_gpu as bif
import charts
import clustering as cl
import contract

STUDIO_ROOT = Path(os.environ.get("BIF_STUDIO_ROOT",
                                  Path.home() / ".cache" / "bif-studio"))
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
DRAFTS = "drafts"   # STUDIO_ROOT/drafts: systems.py's draft RHS files, not a session

FULL_1D_HEIGHT = 1200
FULL_1D_MIN_WIDTH = 1600   # fewer sweep steps: each column repeated, still exact
FULL_1D_MAX_WIDTH = 4096   # more: columns binned
FULL_2D_MIN_SIDE = 1200
# The preview is the full image whenever its PNG fits this budget (the inline
# limit of the pane's Image), and shrinks only as far as needed to fit.
PREVIEW_MAX_BYTES = 2 << 20

# Same palette as the matplotlib charts (server.py).
SURFACE = (0x0f, 0x0f, 0x13)
POINTS = (0x7c, 0x6a, 0xf7)
INK = (0xe0, 0xe0, 0xe8)


@dataclass
class Job:
    job_id: str
    run_id: str
    session_dir: Path
    state: str = "running"          # running | done | error | cancelled
    progress: int = 0
    message: str = "Queued"
    error: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


# ── sessions ──────────────────────────────────────────────────────────────────
def session_dir(session: str) -> Path:
    if not SESSION_RE.match(session or "") or session == DRAFTS:
        raise ValueError(f"session must match {SESSION_RE.pattern}, got {session!r}")
    d = STUDIO_ROOT / session
    d.mkdir(parents=True, exist_ok=True)
    return d


def _reserve_run(d: Path) -> str:
    """Claim the next run-NNNN by creating its json exclusively."""
    with _lock:
        taken = [int(m.group(1)) for f in d.glob("run-*.json")
                 if (m := re.match(r"run-(\d+)\.json$", f.name))]
        n = max(taken, default=0) + 1
        while True:
            run_id = f"run-{n:04d}"
            try:
                with open(d / f"{run_id}.json", "x") as f:
                    f.write("{}")
                return run_id
            except FileExistsError:
                n += 1


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _rebuild_index(d: Path) -> list[dict]:
    runs = []
    for f in sorted(d.glob("run-*.json")):
        try:
            meta = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if meta:
            runs.append({k: meta.get(k) for k in
                         ("id", "state", "system", "mode", "params", "sweep",
                          "summary", "paths", "created", "error")})
    with _lock:
        _write_json(d / "index.json", runs)
    return runs


def latest_session() -> str:
    """The session folder written most recently (for callers without an id)."""
    dirs = [d for d in STUDIO_ROOT.glob("*") if d.is_dir() and d.name != DRAFTS
            and SESSION_RE.match(d.name)]
    if not dirs:
        raise ValueError("no Bifurcation Studio runs yet")
    return max(dirs, key=lambda d: max((f.stat().st_mtime for f in d.iterdir()),
                                       default=d.stat().st_mtime)).name


def list_runs(session: str, last: int = 0) -> list[dict]:
    runs = _rebuild_index(session_dir(session))
    return runs[-last:] if last > 0 else runs


# ── request resolution ────────────────────────────────────────────────────────
def _resolve(rhs_path: str, params: dict | None, sweep: dict,
             run_options: dict | None):
    """Validate a pane request against describe(); return (desc, args)."""
    desc = contract.describe(rhs_path)
    if not desc["ok"]:
        raise ValueError("RHS file is invalid:\n" + "\n".join(desc["errors"]))
    by_name = {q["name"]: q for q in desc["params"]}

    values = {}
    for name, q in by_name.items():
        v = (params or {}).get(name, q["default"])
        if v is None:
            raise ValueError(f"parameter {name!r} has no default (v1 file): "
                             f"pass a value in params")
        values[name] = float(v)
    unknown = set(params or {}) - set(by_name)
    if unknown:
        raise ValueError(f"unknown parameters {sorted(unknown)}; "
                         f"known: {sorted(by_name)}")

    opts_spec = {o["name"]: o for o in desc["runOptions"]}
    unknown = set(run_options or {}) - set(opts_spec)
    if unknown:
        raise ValueError(f"unknown run options {sorted(unknown)}")
    mode = (run_options or {}).get("mode", opts_spec["mode"]["default"])
    if mode not in opts_spec["mode"]["options"]:
        raise ValueError(f"mode must be one of {opts_spec['mode']['options']}")
    opts = {}
    for name, o in opts_spec.items():
        if name == "mode" or ("modes" in o and mode not in o["modes"]):
            continue
        v = (run_options or {}).get(
            name, o.get("modeDefaults", {}).get(mode, o["default"]))
        if o["type"] == "enum":
            if v not in o["options"]:
                raise ValueError(f"{name} must be one of {o['options']}, got {v!r}")
        else:
            v = int(v) if o["type"] == "int" else float(v)
            if not o["min"] <= v <= o["max"]:
                raise ValueError(f"{name}={v} outside [{o['min']}, {o['max']}]")
        opts[name] = v

    axes = ["x", "y"] if mode == "period-map" else ["x"]
    extra = set(sweep or {}) - set(axes)
    if extra:
        raise ValueError(f"sweep axes for mode {mode!r} are {axes}, "
                         f"got extra {sorted(extra)}")
    resolved = {}
    for ax in axes:
        s = (sweep or {}).get(ax)
        if not s or s.get("param") not in by_name:
            raise ValueError(f"sweep.{ax}.param must name one of {sorted(by_name)}")
        q = by_name[s["param"]]
        if not q["sweepable"]:
            raise ValueError(f"parameter {q['name']!r} is not sweepable")
        lo = float(s.get("min", (q["sweep"] or [None, None])[0]))
        hi = float(s.get("max", (q["sweep"] or [None, None])[1]))
        if not lo < hi:
            raise ValueError(f"sweep.{ax}: min ({lo}) must be < max ({hi})")
        resolved[ax] = {"param": q["name"], "index": q["index"],
                        "label": q["label"], "min": lo, "max": hi}
    if mode == "period-map" and resolved["x"]["param"] == resolved["y"]["param"]:
        raise ValueError("sweep.x and sweep.y must be different parameters")

    fixed = [values[q["name"]] for q in desc["params"]]
    return desc, {"mode": mode, "params": values, "sweep": resolved,
                  "options": opts, "fixed": fixed}


# ── images ────────────────────────────────────────────────────────────────────
def _density_1d(params, values, counts) -> np.ndarray:
    """(H, W, 3) uint8: per-column histogram of recorded values, log-scaled."""
    n = len(params)
    cols = np.repeat(np.arange(n), counts.astype(np.int64))
    mask = np.arange(values.shape[1])[None, :] < counts[:, None]
    vals = values[mask]
    width = min(n, FULL_1D_MAX_WIDTH)
    repeat = -(-FULL_1D_MIN_WIDTH // width) if width < FULL_1D_MIN_WIDTH else 1
    img = np.empty((FULL_1D_HEIGHT, width * repeat, 3), np.uint8)
    img[:] = SURFACE
    if vals.size == 0:
        return img
    lo, hi = np.percentile(vals, [0.5, 99.5])
    pad = 0.03 * (hi - lo or abs(hi) or 1.0)
    lo, hi = lo - pad, hi + pad
    hist, _, _ = np.histogram2d(
        vals, cols * (width / n), bins=[FULL_1D_HEIGHT, width],
        range=[[lo, hi], [0, width]])
    hist = hist[::-1]                         # high values at the top
    # Any hit is clearly visible; density adds brightness on a log scale.
    a = np.log1p(hist) / np.log1p(hist.max())
    a = np.where(hist > 0, 0.55 + 0.45 * np.sqrt(a), 0.0)[..., None]
    img = (np.array(SURFACE) * (1 - a) + np.array(POINTS) * a).astype(np.uint8)
    return img.repeat(repeat, axis=1) if repeat > 1 else img


def _period_map_2d(grid_codes: np.ndarray, max_clusters: int) -> np.ndarray:
    from matplotlib.colors import to_rgb
    colors = np.array([to_rgb(c) for c in charts.cluster_colors(max_clusters)])
    img = (colors[grid_codes[::-1]] * 255).round().astype(np.uint8)
    n_y, n_x = grid_codes.shape
    k = max(1, -(-FULL_2D_MIN_SIDE // max(n_x, n_y)))
    return img.repeat(k, axis=0).repeat(k, axis=1)


def _png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _shrink(full: Image.Image, scale: float, nearest: bool) -> Image.Image:
    size = (max(1, round(full.width * scale)), max(1, round(full.height * scale)))
    if nearest:
        return full.resize(size, Image.Resampling.NEAREST)
    # Max-pool before averaging so one-pixel branches survive the downscale.
    from PIL import ImageFilter
    k = int(1 / scale)
    src = full.filter(ImageFilter.MaxFilter(k if k % 2 else k + 1)) if k >= 2 else full
    return src.resize(size, Image.Resampling.BOX)


def _write_images(base: Path, img: np.ndarray, nearest: bool) -> dict:
    """
    Full image, plus a preview at the largest size whose PNG fits
    PREVIEW_MAX_BYTES: small diagrams keep every pixel, big ones lose only
    what the budget forces. Never upscaled.
    """
    full = Image.fromarray(img, "RGB")
    data = _png_bytes(full)
    base.with_suffix(".png").write_bytes(data)
    prev, scale = full, 1.0
    while len(data) > PREVIEW_MAX_BYTES and min(prev.size) > 16:
        # PNG size grows roughly with pixel count; aim a bit under the budget.
        scale *= max(0.5, 0.92 * math.sqrt(PREVIEW_MAX_BYTES / len(data)))
        prev = _shrink(full, scale, nearest)
        data = _png_bytes(prev)
    base.with_suffix(".preview.png").write_bytes(data)
    return {"png": str(base.with_suffix(".png")),
            "preview": str(base.with_suffix(".preview.png")),
            "previewSize": list(prev.size), "previewBytes": len(data)}


# ── worker ────────────────────────────────────────────────────────────────────
def _worker(job: Job, desc: dict, rhs_path: str, args: dict, meta: dict):

    def progress(pct, msg):
        job.progress, job.message = int(pct), msg

    d, base = job.session_dir, job.session_dir / job.run_id
    o, sx = args["options"], args["sweep"]["x"]
    common = dict(dt=o["dt"], t_end=o["t_end"], transient=o["transient"],
                  record_component=o["record_component"],
                  record_mode=o["record_mode"], record_n=o["record_n"],
                  integrator=o["integrator"], fixed_params=args["fixed"],
                  progress_callback=progress, param_index=sx["index"],
                  should_cancel=job.cancel.is_set)
    started = time.time()
    try:
        if args["mode"] == "period-map":
            sy = args["sweep"]["y"]
            data = bif.run_bifurcation_2d(
                rhs_path, sx["min"], sx["max"], o["param_steps"],
                sy["min"], sy["max"], o["param2_steps"],
                param2_index=sy["index"], **common)
            if job.cancel.is_set():
                raise bif.Cancelled()
            if o["record_mode"] == "tail":
                # Ring buffer, not in time order: count clusters instead.
                progress(72, "Clustering recorded values (DBSCAN)…")
                clusters = cl.cluster_grid(
                    data["values"], data["counts"],
                    eps=o["eps"] or None, eps_scale=o["eps_scale"],
                    min_samples=o["min_samples"], min_points=o["min_points"],
                    max_noise_frac=o["max_noise_frac"] if o["max_noise_frac"] < 1 else None,
                    progress_callback=lambda f: progress(72 + int(14 * f), "Clustering…"))
                clusters["rule"] = "dbscan"
            else:
                progress(72, "Finding orbit periods…")
                clusters = cl.period_grid(
                    data["values"], data["counts"], rel_tol=o["period_tol"],
                    max_period=o["max_period"], min_points=o["min_points"],
                    progress_callback=lambda f: progress(72 + int(14 * f), "Finding orbit periods…"))
            progress(88, "Rendering…")
            mc = o["max_clusters"]
            nc = clusters["n_clusters"]
            codes = charts.cluster_codes(nc.reshape(data["shape"]), mc)
            paths = _write_images(base, _period_map_2d(codes, mc), nearest=True)
            hist = np.bincount(codes.ravel(), minlength=mc + 2)
            named = {("none" if k == 0 else f">={mc + 1}_or_chaotic"
                      if k == mc + 1 else str(k)): int(v)
                     for k, v in enumerate(hist) if v}
            tol_key = "tol" if clusters["rule"] == "sequence" else "eps"
            summary = {"cells": int(nc.size), "periods": named, "rule": clusters["rule"],
                       tol_key: round(float(clusters["eps"]), 6)}
            if clusters["rule"] == "dbscan":
                summary["maxNoiseFrac"] = o["max_noise_frac"]
            arrays = dict(params=data["params"], params2=data["params2"],
                          values=data["values"], counts=data["counts"],
                          n_clusters=nc, n_points=clusters["n_points"])
            chart_args = (data, clusters, mc)
        else:
            data = bif.run_bifurcation(rhs_path, sx["min"], sx["max"],
                                       o["param_steps"], **common)
            progress(90, "Rendering…")
            paths = _write_images(
                base, _density_1d(data["params"], data["values"], data["counts"]),
                nearest=False)
            counts = data["counts"]
            summary = {"points": int(counts.sum()),
                       "emptyColumns": int((counts == 0).sum()),
                       "columns": int(counts.size)}
            arrays = dict(params=data["params"], values=data["values"],
                          counts=counts)
            chart_args = (data,)

        progress(94, "Rendering labelled chart…")
        paths["chart"] = str(base.with_suffix(".chart.png"))
        charts.write_chart(paths["chart"], args, desc, *chart_args)

        meta.update(state="done", summary=summary, seconds=round(time.time() - started, 2))
        meta["paths"].update(paths)
        np.savez_compressed(base.with_suffix(".npz"), meta=json.dumps(meta), **arrays)
        job.state = "done"
        progress(100, "Done")
    except bif.Cancelled:
        meta.update(state="cancelled")
        meta["paths"].pop("npz", None)
        job.state, job.message = "cancelled", "Cancelled"
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        # Numba compile errors are long; keep the head, which names the line.
        if len(err) > 2000:
            err = err[:2000] + " …"
        meta.update(state="error", error=err)
        meta["paths"].pop("npz", None)
        job.state, job.error, job.message = "error", err, "Failed"
        traceback.print_exc()
    _write_json(base.with_suffix(".json"), meta)
    _rebuild_index(d)


# ── public API (wrapped as MCP tools in server.py) ────────────────────────────
def _run_mode(png: Path) -> str:
    """The mode of the run a picture belongs to, from its run-NNNN.json."""
    meta = png.with_name(png.name.split(".")[0] + ".json")
    try:
        return json.loads(meta.read_text()).get("mode", "bifurcation")
    except (OSError, json.JSONDecodeError):
        return "bifurcation"


def _to_cells_size(img: Image.Image, w: int, h: int, is_map: bool) -> Image.Image:
    """
    Shrink a run image to (w, h) for terminal cells, keeping what matters:
    a period map by nearest sampling, so no blended in-between colour reads
    as another period; a density diagram by block max-pooling first, so
    one-pixel branches survive a 50x shrink, then quantised to 32 colours
    for the Raster's colour-pair budget.
    """
    if is_map:
        return img.resize((w, h), Image.Resampling.NEAREST)
    # Work on the density image's own alpha (0 = SURFACE, 1 = POINTS): a block
    # with any hit is lit (thin branches survive), and its mean density adds
    # brightness on top (chaotic bands keep their shading).
    rgb = np.asarray(img, dtype=np.float32)
    ch = int(np.argmax(np.abs(np.subtract(POINTS, SURFACE))))
    alpha = np.clip((rgb[..., ch] - SURFACE[ch]) / (POINTS[ch] - SURFACE[ch]), 0, 1)
    fx, fy = max(1, img.width // w), max(1, img.height // h)
    H, W = (alpha.shape[0] // fy) * fy, (alpha.shape[1] // fx) * fx
    blocks = alpha[:H, :W].reshape(H // fy, fy, W // fx, fx)
    hit, mean = blocks.max(axis=(1, 3)), blocks.mean(axis=(1, 3))
    mean = np.sqrt(mean / (mean.max() or 1.0))
    a = np.where(hit > 0, 0.45 + 0.55 * mean, 0.0)
    a = np.asarray(Image.fromarray(a.astype(np.float32), "F")
                   .resize((w, h), Image.Resampling.BOX))[..., None]
    out = np.array(SURFACE) * (1 - a) + np.array(POINTS) * a
    # Few alpha levels: quantise them, so colour pairs stay well under 1024.
    out = np.round(out / 8) * 8
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def raster(png_path: str, columns: int, rows: int) -> dict:
    """
    A run image as terminal cells for the mod's Raster element: each cell is
    an upper half block, foreground = top pixel, background = bottom pixel.
    A period map keeps its aspect inside columns x (2*rows) pixels, so the
    returned columns/rows may be smaller than asked; a 1-D diagram (whose
    aspect means nothing) fills the whole box. Colours stay within the
    Raster's 1024 colour pairs.
    """
    import base64
    path = Path(png_path).expanduser().resolve()
    if STUDIO_ROOT.resolve() not in path.parents:
        raise ValueError(f"png_path must be inside {STUDIO_ROOT}")
    columns, rows = max(1, min(int(columns), 512)), max(1, min(int(rows), 256))
    img = Image.open(path).convert("RGB")
    is_map = _run_mode(path) == "period-map"
    if is_map:
        scale = min(columns / img.width, 2 * rows / img.height)
        w = max(1, min(columns, round(img.width * scale)))
        h = max(2, min(2 * rows, round(img.height * scale)))
        h += h % 2
    else:
        w, h = columns, 2 * rows
    px = np.asarray(_to_cells_size(img, w, h, is_map), dtype=np.uint32)
    rgb = (px[..., 0] << 16) | (px[..., 1] << 8) | px[..., 2]
    cells = np.empty((h // 2, w, 3), dtype="<u4")
    cells[..., 0] = 0x2580                     # ▀
    cells[..., 1] = rgb[0::2]
    cells[..., 2] = rgb[1::2]
    return {"columns": w, "rows": h // 2,
            "cells": base64.b64encode(cells.tobytes()).decode()}


def submit_run(rhs_path: str, session: str, params: dict | None = None,
               sweep: dict | None = None, run_options: dict | None = None) -> dict:
    rhs_path = str(Path(rhs_path).expanduser().resolve())
    desc, args = _resolve(rhs_path, params, sweep or {}, run_options)
    d = session_dir(session)
    run_id = _reserve_run(d)
    base = d / run_id
    meta = {
        "id": run_id, "state": "running", "created": time.time(),
        "system": desc["system"], "rhsPath": rhs_path,
        "mode": args["mode"], "params": args["params"], "sweep": args["sweep"],
        "options": args["options"],
        "axes": {"x": args["sweep"]["x"]["label"],
                 "y": (args["sweep"]["y"]["label"] if args["mode"] == "period-map"
                       else _observable(args["options"], desc))},
        "paths": {"json": str(base.with_suffix(".json")),
                  "npz": str(base.with_suffix(".npz"))},
    }
    _write_json(base.with_suffix(".json"), meta)
    job = Job(job_id=uuid.uuid4().hex, run_id=run_id, session_dir=d)
    with _lock:
        _jobs[job.job_id] = job
    threading.Thread(target=_worker, args=(job, desc, rhs_path, args, meta),
                     daemon=True).start()
    return {"jobId": job.job_id, "runId": run_id}


def _observable(opts: dict, desc: dict) -> str:
    c = opts["record_component"]
    name = desc["state"][c]["name"] if c < len(desc["state"]) else f"y[{c}]"
    return {"maxima": f"maxima of {name}", "tail": f"{name} (tail)",
            "isi": "inter-spike interval", "spike": f"{name} at spike"
            }[opts["record_mode"]]


def job_status(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise ValueError(f"unknown jobId {job_id!r} (server restarted? "
                         f"see list_runs for finished runs)")
    out = {"jobId": job_id, "runId": job.run_id, "state": job.state,
           "progress": job.progress, "message": job.message}
    if job.state == "error":
        out["error"] = job.error
    if job.state != "running":
        meta = json.loads((job.session_dir / f"{job.run_id}.json").read_text())
        out["result"] = {"paths": meta.get("paths", {}),
                         "summary": meta.get("summary")}
    return out


def cancel_job(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise ValueError(f"unknown jobId {job_id!r}")
    if job.state == "running":
        job.cancel.set()
        job.message = "Cancelling…"
    return {"jobId": job_id, "state": job.state}


def cleanup(older_than_hours: float = 24.0, keep_session: str = "") -> dict:
    """Delete session folders (and legacy job folders) untouched for that long."""
    cutoff = time.time() - older_than_hours * 3600
    removed = []
    roots = [STUDIO_ROOT, Path(tempfile.gettempdir()) / "bifurcation_jobs"]
    for root in roots:
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if not d.is_dir() or d.is_symlink() or d.name == keep_session:
                continue
            if root == STUDIO_ROOT and d.name == DRAFTS:
                # Drafts expire one by one; the folder stays.
                for f in d.glob("*.py"):
                    if f.stat().st_mtime < cutoff:
                        f.unlink(missing_ok=True)
                        removed.append(str(f))
                continue
            newest = max((f.stat().st_mtime for f in d.iterdir()),
                         default=d.stat().st_mtime)
            if newest < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed.append(str(d))
    return {"removed": len(removed), "paths": removed}
