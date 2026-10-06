"""
Chart rendering and the period-map palette, shared by server.py (legacy
tools) and studio.py (Bifurcation Studio runs).
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex


# ── palette ───────────────────────────────────────────────────────────────────
SURFACE   = "#0f0f13"
INK       = "#e0e0e8"
INK_MUTED = "#a0a0b8"
INK_FAINT = "#606080"
EDGE      = "#2a2a3e"
POINTS    = "#7c6af7"

# Cluster count is ordinal magnitude, so it gets a single-hue sequential ramp
# (documented blue ramp, steps 600→100) running dark→light on the dark surface:
# period 1 sits nearest the surface, high periods glow. "No data" is a neutral
# below the ramp; the overflow bucket takes a contrasting hue because it is
# off the ordinal scale, not another step of it.
RAMP_ANCHORS = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
NO_DATA_COLOR  = "#2e2e3a"
OVERFLOW_COLOR = "#d95926"


def cluster_codes(n_clusters, max_clusters):
    """
    Map raw cluster counts to plot buckets:
      0                 -> "none"     (nothing recorded: quiescent or diverged)
      1..max_clusters   -> that period
      > max_clusters    -> top bucket (chaotic)
      -1 (all noise)    -> top bucket too: no repeating structure is the
                           aperiodic end of the scale, not the empty end.
    """
    codes = np.clip(n_clusters, 0, max_clusters + 1).astype(np.int32)
    codes[n_clusters < 0] = max_clusters + 1
    return codes


# ── chart renderers ───────────────────────────────────────────────────────────
def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_FAINT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE)


def _fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_chart(data: dict, labels: dict) -> bytes:
    """1-D bifurcation diagram: recorded values vs the swept parameter."""
    params, values, counts = data["params"], data["values"], data["counts"]
    xs, ys = [], []
    for i, c in enumerate(counts):
        n = int(c)
        if n > 0:
            xs.extend([params[i]] * n)
            ys.extend(values[i, :n].tolist())

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    _style_axes(ax)
    ax.scatter(xs, ys, s=0.3, c=POINTS, alpha=0.5, linewidths=0)
    ax.set_xlabel(labels["x"], color=INK_MUTED, fontsize=11)
    ax.set_ylabel(labels["y"], color=INK_MUTED, fontsize=11)
    ax.set_title(labels["title"], color=INK, fontsize=13, pad=12)
    fig.tight_layout()
    return _fig_to_png(fig)


def _cluster_cmap(n_levels: int):
    """[no-data] + n_levels steps of the sequential blue ramp + [overflow]."""
    ramp = LinearSegmentedColormap.from_list("bif_seq", RAMP_ANCHORS)
    steps = [to_hex(ramp(x)) for x in np.linspace(0.0, 1.0, max(n_levels, 2))][:n_levels]
    return ListedColormap([NO_DATA_COLOR] + steps + [OVERFLOW_COLOR])


def render_chart_2d(data: dict, clusters: dict, labels: dict,
                    max_clusters: int = 6) -> bytes:
    """
    2-parameter map. Each cell is one trajectory; its colour is the number of
    DBSCAN clusters found among the values recorded there — the period of the
    orbit. Cells that recorded nothing (quiescent or diverged) and cells above
    max_clusters or with no cluster structure at all get their own colours.
    """
    n_y, n_x = data["shape"]
    grid = clusters["n_clusters"].reshape(n_y, n_x).astype(np.int32)
    shown = cluster_codes(grid, max_clusters)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=SURFACE)
    _style_axes(ax)

    cmap = _cluster_cmap(max_clusters)
    im = ax.imshow(
        shown,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-0.5,
        vmax=max_clusters + 1.5,
        extent=[data["param_min"], data["param_max"],
                data["param2_min"], data["param2_max"]],
    )

    ax.set_xlabel(labels["x"], color=INK_MUTED, fontsize=11)
    ax.set_ylabel(labels["y"], color=INK_MUTED, fontsize=11)
    ax.set_title(labels["title"], color=INK, fontsize=13, pad=12)

    # Discrete colourbar doubles as the legend: identity is never colour-alone.
    cbar = fig.colorbar(im, ax=ax, ticks=range(0, max_clusters + 2), pad=0.02)
    tick_labels = ["none"] + [str(k) for k in range(1, max_clusters + 1)]
    tick_labels.append(f"≥{max_clusters + 1}\nor chaotic")
    cbar.ax.set_yticklabels(tick_labels)
    cbar.ax.tick_params(colors=INK_FAINT, labelsize=9)
    cbar.outline.set_edgecolor(EDGE)
    sequence = clusters.get("rule") == "sequence"
    cbar.set_label(
        f"orbit period of {labels['observable']}" if sequence
        else f"DBSCAN clusters of {labels['observable']}  (orbit period)",
        color=INK_MUTED, fontsize=10,
    )

    fig.text(
        0.01, 0.015,
        (f"period = smallest k with |m[n+k]-m[n]| <= {clusters['tol']:.3g}, k <= {clusters['max_period']}"
         if sequence else
         f"DBSCAN eps={clusters['eps']:.4g}, min_samples={clusters['min_samples']}")
        + f"   ·   {labels['subtitle']}",
        color=INK_FAINT, fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _fig_to_png(fig)



def cluster_colors(max_clusters: int) -> list:
    """[no-data] + max_clusters ramp steps + [overflow], as hex strings."""
    return list(_cluster_cmap(max_clusters).colors)


def write_chart(path, args: dict, desc: dict, data: dict,
                clusters: dict | None = None, max_clusters: int = 6) -> None:
    """Labelled chart for a studio run (axes named from the contract)."""
    sx = args["sweep"]["x"]
    name = desc["system"]["name"]
    o = args["options"]
    c = o["record_component"]
    state = desc["state"][c]["name"] if c < len(desc["state"]) else f"y[{c}]"
    observable = {"maxima": f"local maxima of {state}", "tail": f"{state} (tail)",
                  "isi": "inter-spike interval", "spike": f"{state} at each spike"
                  }[o["record_mode"]]
    fixed = ", ".join(f"{k}={v:g}" for k, v in args["params"].items()
                      if k not in {a["param"] for a in args["sweep"].values()})
    if clusters is not None:
        sy = args["sweep"]["y"]
        labels = {"x": sx["label"], "y": sy["label"], "observable": observable,
                  "title": f"{name}: period map",
                  "subtitle": "  ·  ".join(filter(None, [fixed, f"{o['integrator']}, dt={o['dt']:g}"]))}
        png = render_chart_2d(data, clusters, labels, max_clusters)
    else:
        labels = {"x": sx["label"], "y": observable,
                  "title": f"{name}: bifurcation diagram"
                           + (f"  ({fixed})" if fixed else "")}
        png = render_chart(data, labels)
    with open(path, "wb") as f:
        f.write(png)
