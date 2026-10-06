"""
Compact, Claude-sized analysis of Bifurcation Studio runs (run-NNNN.npz).

Every command prints a few lines of text (or JSON with --json), never raw
arrays:

  python runinspect.py summary  run-0003.npz          what was run + period regions
  python runinspect.py slice    run-0003.npz c=5.7    one parameter value up close
  python runinspect.py compare  run-0003.npz run-0004.npz

"Period" is the smallest k for which the recorded sequence repeats,
|m[n+k] - m[n]| <= tol for all n (clustering.period_grid): k = period-k
orbit, "chaotic" = no k up to 32, "none" = too few values (quiescent or
diverged). Record mode tail (not in time order) falls back to DBSCAN
cluster counts. A period map's cells are read as the run stored them; check
the run's summary "rule" (maps made before the sequence rule counted
DBSCAN clusters).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import clustering as cl

MAX_PERIOD = 8
MAX_LINES = 40
# A column whose DBSCAN noise share exceeds this is chaotic, even if one
# accidental cluster formed (see clustering.cluster_grid, max_noise_frac).
MAX_NOISE_FRAC = 0.3


def label(n: int) -> str:
    if n == 0:
        return "none"
    if n < 0:
        return "chaotic"
    return f"period {n}"


def load(path: str) -> dict:
    """The run's arrays and metadata; 1-D runs get per-column periods."""
    npz = np.load(path)
    meta = json.loads(str(npz["meta"]))
    run = {"path": str(Path(path).resolve()), "meta": meta,
           "x": npz["params"], "mode": meta.get("mode", "bifurcation")}
    if run["mode"] == "period-map":
        run["y"] = npz["params2"]
        run["periods"] = npz["n_clusters"].reshape(len(run["y"]), len(run["x"]))
    else:
        run["values"], run["counts"] = npz["values"], npz["counts"]
        if meta.get("options", {}).get("record_mode") == "tail":
            clusters = cl.cluster_grid(npz["values"], npz["counts"], max_noise_frac=MAX_NOISE_FRAC)
            p = clusters["n_clusters"]
            run["periods"] = np.where(p > MAX_PERIOD, -1, p)   # many clusters: chaotic
        else:
            clusters = cl.period_grid(npz["values"], npz["counts"])
            run["periods"] = clusters["n_clusters"]
        run["rule"] = clusters.get("rule", "dbscan")
        run["eps"] = float(clusters["eps"])
    return run


def segments(xs: np.ndarray, periods: np.ndarray) -> list[dict]:
    """
    Run-length regions of equal label along a sweep. Runs shorter than
    ~0.25 % of the sweep are absorbed by their left neighbour (DBSCAN
    flicker at boundaries), so the list reads as the diagram does.
    """
    labels = [label(int(p)) for p in periods]
    min_len = max(2, len(labels) // 400)
    runs: list[list] = []                      # [label, start, end] indices
    for i, lab in enumerate(labels):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = i
        else:
            runs.append([lab, i, i])
    merged: list[list] = []
    for r in runs:
        if merged and (r[2] - r[1] + 1 < min_len or merged[-1][0] == r[0]):
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    return [{"label": lab, "from": float(xs[a]), "to": float(xs[b]), "steps": b - a + 1}
            for lab, a, b in merged]


def describe_run(run: dict) -> list[str]:
    m = run["meta"]
    sx = m["sweep"]["x"]
    o = m["options"]
    swept = {a["param"] for a in m["sweep"].values()}
    fixed = " ".join(f"{k}={v:g}" for k, v in m["params"].items() if k not in swept)
    axes = f"{sx['param']} {sx['min']:g}…{sx['max']:g} ({len(run['x'])} steps)"
    if run["mode"] == "period-map":
        sy = m["sweep"]["y"]
        axes += f" × {sy['param']} {sy['min']:g}…{sy['max']:g} ({len(run['y'])} steps)"
    return [
        f"{m['id']}  {m['system']['name']}  {run['mode']}  {axes}  fixed: {fixed or '–'}",
        f"observable: {m['axes']['y'] if run['mode'] != 'period-map' else o['record_mode']}"
        f" · {o['integrator']} dt={o['dt']:g} t_end={o['t_end']:g} transient={o['transient']:g}",
    ]


def summary(run: dict) -> dict:
    if run["mode"] == "period-map":
        flat = run["periods"].ravel()
        labels = np.array([label(int(p)) for p in flat])
        out = {}
        for lab in sorted(set(labels), key=lambda s: (s == "none", s == "chaotic", s)):
            j, i = np.nonzero((labels == lab).reshape(run["periods"].shape))
            out[lab] = {"fraction": round(float(len(i)) / flat.size, 4),
                        "x": [float(run["x"][i.min()]), float(run["x"][i.max()])],
                        "y": [float(run["y"][j.min()]), float(run["y"][j.max()])]}
        return {"run": run["meta"]["id"], "header": describe_run(run), "periods": out}
    segs = segments(run["x"], run["periods"])
    notes = []
    p = run["meta"]["sweep"]["x"]["param"]
    first2 = next((s for s in segs if s["label"] == "period 2"), None)
    chaos = next((s for s in segs if s["label"] == "chaotic"), None)
    if first2:
        notes.append(f"first period-2 region at {p}≈{first2['from']:.4g}")
    if chaos:
        notes.append(f"first chaotic region at {p}≈{chaos['from']:.4g}")
    return {"run": run["meta"]["id"], "header": describe_run(run),
            "eps": run["eps"], "regions": segs, "notes": notes}


def slice_at(run: dict, param: str, value: float) -> dict:
    m = run["meta"]
    sx, sy = m["sweep"]["x"], m["sweep"].get("y")
    if run["mode"] != "period-map":
        if param != sx["param"]:
            raise SystemExit(f"this run sweeps {sx['param']}, not {param}")
        i = int(np.abs(run["x"] - value).argmin())
        vals = np.sort(run["values"][i, :int(run["counts"][i])])
        centers = []
        if vals.size:
            gap = np.diff(vals) > max(run["eps"], 1e-12)
            for chunk in np.split(vals, np.nonzero(gap)[0] + 1):
                centers.append({"value": round(float(chunk.mean()), 6), "hits": int(chunk.size)})
        return {"run": m["id"], "at": {param: float(run["x"][i])},
                "period": label(int(run["periods"][i])), "recorded": int(vals.size),
                "clusters": centers[:20], "moreClusters": max(0, len(centers) - 20)}
    if param == sx["param"]:
        i = int(np.abs(run["x"] - value).argmin())
        along, line = run["y"], run["periods"][:, i]
        at, other = {param: float(run["x"][i])}, sy["param"]
    elif sy and param == sy["param"]:
        j = int(np.abs(run["y"] - value).argmin())
        along, line = run["x"], run["periods"][j, :]
        at, other = {param: float(run["y"][j])}, sx["param"]
    else:
        raise SystemExit(f"this run sweeps {sx['param']} and {sy['param'] if sy else '–'}, not {param}")
    return {"run": m["id"], "at": at, "along": other, "regions": segments(along, line)}


def compare(a: dict, b: dict) -> dict:
    out = {"runs": [a["meta"]["id"], b["meta"]["id"]],
           "headers": describe_run(a) + describe_run(b)}
    pa, pb = a["meta"]["params"], b["meta"]["params"]
    out["paramChanges"] = {k: [pa.get(k), pb.get(k)] for k in sorted(set(pa) | set(pb))
                           if pa.get(k) != pb.get(k)}
    if a["mode"] != b["mode"]:
        out["note"] = "different modes; compare the summaries instead"
        return out
    if a["mode"] == "period-map":
        if a["periods"].shape == b["periods"].shape:
            la = np.vectorize(lambda p: label(int(p)))(a["periods"])
            lb = np.vectorize(lambda p: label(int(p)))(b["periods"])
            out["cellsDiffering"] = round(float((la != lb).mean()), 4)
        out["periods"] = [summary(a)["periods"], summary(b)["periods"]]
        return out
    # 1-D: label both on a's grid (nearest column of b) and list where they differ.
    j = np.abs(b["x"][None, :] - a["x"][:, None]).argmin(axis=1)
    inside = (a["x"] >= b["x"].min()) & (a["x"] <= b["x"].max())
    la = np.array([label(int(p)) for p in a["periods"]])
    lb = np.array([label(int(p)) for p in b["periods"][j]])
    diff = (la != lb) & inside
    regions = []
    for seg in segments(a["x"], np.where(diff, 1, 0)):
        if seg["label"] == "period 1":           # 1 marks "differs"
            lo, hi = seg["from"], seg["to"]
            k = (a["x"] >= lo) & (a["x"] <= hi)
            mostly = lambda labs: max(set(labs), key=list(labs).count)
            regions.append({"from": lo, "to": hi, "differingShare": round(float(diff[k].mean()), 3),
                            "a": mostly(la[k]), "b": mostly(lb[k])})
    out["differs"] = regions
    out["sharedRange"] = [float(max(a["x"].min(), b["x"].min())), float(min(a["x"].max(), b["x"].max()))]
    return out


# ── text output ───────────────────────────────────────────────────────────────
def _fmt_regions(param: str, regions: list[dict]) -> list[str]:
    lines = [f"  {param} {r['from']:.5g}–{r['to']:.5g}  {r['label']}" for r in regions[:MAX_LINES]]
    if len(regions) > MAX_LINES:
        lines.append(f"  … {len(regions) - MAX_LINES} more regions (use --json)")
    return lines


def render(cmd: str, out: dict, run: dict | None = None) -> str:
    if cmd == "summary":
        lines = list(out["header"])
        if "regions" in out:
            p = run["meta"]["sweep"]["x"]["param"]
            rule = (f"period = sequence repeat, tol={out['eps']:.3g}" if run.get("rule") == "sequence"
                    else f"DBSCAN eps={out['eps']:.4g}")
            lines.append(f"regions ({rule}):")
            lines += _fmt_regions(p, out["regions"])
            lines += out["notes"]
        else:
            lines.append("periods (fraction of cells, x range, y range):")
            for lab, v in out["periods"].items():
                lines.append(f"  {lab:10s} {v['fraction']:7.2%}  x {v['x'][0]:.4g}–{v['x'][1]:.4g}"
                             f"  y {v['y'][0]:.4g}–{v['y'][1]:.4g}")
        return "\n".join(lines)
    if cmd == "slice":
        (k, v), = out["at"].items()
        if "clusters" in out:
            cs = ", ".join(f"{c['value']:.6g}×{c['hits']}" for c in out["clusters"])
            more = f" (+{out['moreClusters']} more)" if out["moreClusters"] else ""
            return f"{out['run']} at {k}={v:.6g}: {out['period']}, {out['recorded']} values\n  {cs}{more}"
        return "\n".join([f"{out['run']} at {k}={v:.6g}, along {out['along']}:"]
                         + _fmt_regions(out["along"], out["regions"]))
    if cmd == "compare":
        lines = list(out["headers"])
        if out["paramChanges"]:
            lines.append("changed: " + ", ".join(f"{k} {a}→{b}" for k, (a, b) in out["paramChanges"].items()))
        if "note" in out:
            lines.append(out["note"])
        elif "differs" in out:
            lines.append(f"labels differ on {len(out['differs'])} region(s) of the shared range "
                         f"{out['sharedRange'][0]:.5g}–{out['sharedRange'][1]:.5g}:")
            for r in out["differs"][:MAX_LINES]:
                change = (f"{r['a']} → {r['b']}" if r["a"] != r["b"]
                          else f"both mostly {r['a']}, labels differ on {r['differingShare']:.0%} of steps")
                lines.append(f"  {r['from']:.5g}–{r['to']:.5g}: {change}")
        else:
            if "cellsDiffering" in out:
                lines.append(f"cells with a different label: {out['cellsDiffering']:.2%}")
            for rid, per in zip(out["runs"], out["periods"]):
                lines.append(f"{rid}: " + ", ".join(f"{k} {v['fraction']:.1%}" for k, v in per.items()))
        return "\n".join(lines)
    raise ValueError(cmd)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="runinspect", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary"); s.add_argument("npz")
    s = sub.add_parser("slice"); s.add_argument("npz"); s.add_argument("at", help="param=value")
    s = sub.add_parser("compare"); s.add_argument("a"); s.add_argument("b")
    for p in sub.choices.values():
        p.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "summary":
        run = load(args.npz)
        out = summary(run)
    elif args.cmd == "slice":
        name, _, value = args.at.partition("=")
        run = load(args.npz)
        out = slice_at(run, name.strip(), float(value))
    else:
        run = None
        out = compare(load(args.a), load(args.b))
    print(json.dumps(out, indent=1) if args.json else render(args.cmd, out, run))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
