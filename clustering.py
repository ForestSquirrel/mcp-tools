"""
Exact 1-D DBSCAN, vectorised with numpy (no sklearn/scipy dependency).

Used to turn the set of recorded values (peaks / maxima / ISIs / spike states)
at one point of a 2-parameter grid into a single integer: how many distinct
clusters the attractor visits. That integer is the "period" of the orbit —
1 = fixed point / single peak, 2 = period-2, ... , large = chaotic — and it is
what the 2D diagram colour-codes.

Why DBSCAN rather than "count distinct values": recorded peaks are noisy
(finite dt, finite transient), so exact equality never happens; DBSCAN groups
peaks that lie within `eps` of each other and drops sparse outliers as noise
instead of inflating the period count.

In 1-D the algorithm has an exact O(n log n) formulation:
  * a point is CORE iff at least `min_samples` points (including itself) lie
    within eps -> found with two searchsorted calls on the sorted array;
  * two core points are density-connected iff they are within eps, and because
    the data is sorted this is true exactly when no gap > eps separates them,
    so clusters are the runs of core points between gaps > eps;
  * a non-core point is a BORDER point of the nearest core point within eps,
    otherwise it is NOISE.
Border/noise assignment never changes the cluster count, but the labels are
returned in full so callers can inspect the noise fraction.
"""

import numpy as np

__all__ = ["dbscan_1d", "count_clusters_1d", "auto_eps", "cluster_grid"]


def dbscan_1d(x, eps, min_samples=3):
    """
    Exact 1-D DBSCAN.

    x           : 1-D array of floats
    eps         : neighbourhood radius (same units as x)
    min_samples : minimum points in an eps-neighbourhood for a core point
                  (counting the point itself)

    Returns (labels, n_clusters): labels is int32, -1 for noise, otherwise a
    cluster index in [0, n_clusters).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return np.empty(0, dtype=np.int32), 0
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    min_samples = max(1, int(min_samples))

    order = np.argsort(x, kind="stable")
    xs = x[order]

    # Neighbourhood sizes: number of points within [xi-eps, xi+eps].
    lo = np.searchsorted(xs, xs - eps, side="left")
    hi = np.searchsorted(xs, xs + eps, side="right")
    core = (hi - lo) >= min_samples

    labels_sorted = np.full(n, -1, dtype=np.int32)
    core_idx = np.flatnonzero(core)
    if core_idx.size == 0:
        return np.zeros(n, dtype=np.int32) - 1, 0

    core_x = xs[core_idx]
    # A gap > eps between consecutive core points breaks density connectivity.
    breaks = np.diff(core_x) > eps
    core_labels = np.concatenate(([0], np.cumsum(breaks))).astype(np.int32)
    labels_sorted[core_idx] = core_labels
    n_clusters = int(core_labels[-1]) + 1

    # Border points: nearest core point, attached only if within eps.
    rest = np.flatnonzero(~core)
    if rest.size:
        rest_x = xs[rest]
        pos = np.searchsorted(core_x, rest_x)
        left = np.clip(pos - 1, 0, core_idx.size - 1)
        right = np.clip(pos, 0, core_idx.size - 1)
        d_left = np.abs(rest_x - core_x[left])
        d_right = np.abs(rest_x - core_x[right])
        take_left = d_left <= d_right
        nearest = np.where(take_left, left, right)
        dist = np.where(take_left, d_left, d_right)
        attach = dist <= eps
        labels_sorted[rest[attach]] = core_labels[nearest[attach]]

    labels = np.empty(n, dtype=np.int32)
    labels[order] = labels_sorted
    return labels, n_clusters


def count_clusters_1d(x, eps, min_samples=3):
    """Cluster count only — skips border assignment, so it is cheaper."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return 0
    min_samples = max(1, int(min_samples))
    xs = np.sort(x, kind="stable")
    lo = np.searchsorted(xs, xs - eps, side="left")
    hi = np.searchsorted(xs, xs + eps, side="right")
    core_x = xs[(hi - lo) >= min_samples]
    if core_x.size == 0:
        return 0
    return int(np.count_nonzero(np.diff(core_x) > eps)) + 1


def auto_eps(values, counts, eps_scale=0.01, lo_pct=1.0, hi_pct=99.0):
    """
    Pick a global eps as a fraction of the robust spread of every recorded
    value in the whole sweep.

    A single global eps (rather than a per-cell one) is deliberate: it keeps
    cluster counts comparable across the parameter plane, which is what makes
    the colour map meaningful. Percentile clipping keeps one diverging cell
    from blowing the scale out.
    """
    flat = _flatten_valid(values, counts)
    if flat.size == 0:
        return 1e-9
    lo, hi = np.percentile(flat, [lo_pct, hi_pct])
    spread = float(hi - lo)
    if not np.isfinite(spread) or spread <= 0.0:
        spread = float(np.abs(flat).max()) or 1.0
    return max(spread * float(eps_scale), 1e-12)


def _flatten_valid(values, counts):
    """Concatenate the used prefix of every row of `values`."""
    chunks = [values[i, : int(c)] for i, c in enumerate(counts) if int(c) > 0]
    if not chunks:
        return np.empty(0, dtype=np.float64)
    flat = np.concatenate(chunks)
    return flat[np.isfinite(flat)]


def cluster_grid(
    values,
    counts,
    eps=None,
    eps_scale=0.01,
    min_samples=3,
    min_points=4,
    progress_callback=None,
    max_noise_frac=None,
):
    """
    Run 1-D DBSCAN on every row of `values` (one row per parameter-grid point).

    values : (n_cells, record_n) float64 — recorded values, row-major
    counts : (n_cells,) int          — how many entries of each row are valid
    eps    : explicit radius, or None to derive one via auto_eps()
    min_points : rows with fewer than this many recorded values are reported
                 as 0 clusters ("no data" — silent / diverged / not enough
                 peaks to judge)
    max_noise_frac : when set, a row whose DBSCAN noise share exceeds it is
                 reported as -1 (aperiodic) even if some cluster formed: a
                 chaotic orbit can seed one accidental cluster and leave every
                 other point as noise, which plain counting reads as period 1.
                 None (default) keeps the plain count.

    Returns dict(n_clusters, n_points, n_noise, eps, min_samples), where
    n_noise is the DBSCAN noise count per row (only computed with
    max_noise_frac, else zeros) and n_clusters is:
        0   no usable data at this cell (nothing recorded / below min_points)
       -1   values were recorded but DBSCAN found no cluster at all — every
            point is noise, i.e. the orbit has no repeating structure at this
            eps. That is the aperiodic/chaotic end of the scale, NOT the empty
            end, and callers must not lump it in with 0.
       >0   the number of distinct clusters = the period of the orbit
    """
    counts = np.asarray(counts).astype(np.int64)
    n_cells = counts.shape[0]
    if eps is None:
        eps = auto_eps(values, counts, eps_scale=eps_scale)
    eps = float(eps)

    n_clusters = np.zeros(n_cells, dtype=np.int32)
    n_points = np.zeros(n_cells, dtype=np.int32)
    n_noise = np.zeros(n_cells, dtype=np.int32)

    report_every = max(1, n_cells // 20)
    for i in range(n_cells):
        c = int(counts[i])
        if c > 0:
            row = values[i, :c]
            row = row[np.isfinite(row)]
        else:
            row = np.empty(0, dtype=np.float64)
        n_points[i] = row.size
        if row.size >= min_points:
            if max_noise_frac is None:
                k = count_clusters_1d(row, eps, min_samples)
            else:
                labels, k = dbscan_1d(row, eps, min_samples)
                n_noise[i] = int((labels < 0).sum())
                if n_noise[i] > max_noise_frac * row.size:
                    k = 0
            # Points but no cluster == all noise == aperiodic, flagged as -1 so
            # it is never confused with an empty cell.
            n_clusters[i] = k if k > 0 else -1
        if progress_callback is not None and i % report_every == 0:
            progress_callback(i / n_cells)

    return {
        "n_clusters": n_clusters,
        "n_points": n_points,
        "n_noise": n_noise,
        "eps": eps,
        "min_samples": int(min_samples),
    }


def period_grid(
    values,
    counts,
    rel_tol=1e-3,
    max_period=32,
    min_points=4,
    progress_callback=None,
    chunk_rows=50_000,
):
    """
    Orbit period of every row from the order its values were recorded in.

    A period-k orbit repeats: |m[n+k] - m[n]| <= tol for every n. The period
    is the smallest such k (1..max_period); a row with no such k is
    aperiodic (-1). This tells chaos from a few-band attractor, which a
    cluster count cannot: a chaotic orbit whose maxima fall into 13 narrow
    bands has 13 clusters but never repeats.

    Needs values in time order (record modes maxima, isi, spike; not tail,
    whose ring buffer wraps). tol is rel_tol times the robust spread of all
    recorded values (auto_eps), one tolerance for the whole sweep. A row can
    only show period k if it holds at least 2k + 2 values; with fewer, k is
    not tested.

    Returns the dict cluster_grid returns, so callers and charts treat both
    alike: n_clusters (period, -1 aperiodic, 0 no data), n_points, n_noise
    (zeros), eps (= tol), min_samples (0), plus rule="sequence", tol and
    max_period.
    """
    values = np.asarray(values, dtype=np.float64)
    counts = np.asarray(counts).astype(np.int64)
    n_cells, rec = values.shape
    tol = auto_eps(values, counts, eps_scale=rel_tol)
    period = np.zeros(n_cells, dtype=np.int32)
    n_points = np.minimum(counts, rec).astype(np.int32)
    max_period = max(1, int(max_period))

    for start in range(0, n_cells, chunk_rows):
        stop = min(start + chunk_rows, n_cells)
        v = values[start:stop]
        c = n_points[start:stop].astype(np.int64)
        has = c >= min_points
        found = np.zeros(stop - start, dtype=np.int32)
        cols = np.arange(rec)
        for k in range(1, min(max_period, rec - 1) + 1):
            # Pairs (n, n+k) inside each row's recorded prefix.
            valid = cols[None, : rec - k] < (c - k)[:, None]
            gap = np.abs(v[:, k:] - v[:, :-k])
            worst = np.where(valid, gap, 0.0).max(axis=1)
            finite = np.isfinite(np.where(valid, gap, 0.0)).all(axis=1)
            testable = has & (c >= 2 * k + 2) & (found == 0)
            found[testable & finite & (worst <= tol)] = k
        # Recorded enough to judge, no k fitted: aperiodic.
        period[start:stop] = np.where(has, np.where(found > 0, found, -1), 0)
        if progress_callback is not None:
            progress_callback(stop / n_cells)

    return {
        "n_clusters": period,
        "n_points": n_points,
        "n_noise": np.zeros(n_cells, dtype=np.int32),
        "eps": float(tol),
        "min_samples": 0,
        "rule": "sequence",
        "tol": float(tol),
        "max_period": max_period,
    }
