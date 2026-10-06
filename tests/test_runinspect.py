"""
runinspect + clustering noise rule.  Run:  .venv/bin/python -m unittest discover tests
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import clustering as cl  # noqa: E402
import runinspect  # noqa: E402


def fake_1d(path: Path, run_id="run-0001", a=0.2, chaos_from=80):
    """100 sweep steps: period 1, then period 2, then chaos (spread values)."""
    rng = np.random.default_rng(1)
    n, rec = 100, 40
    params = np.linspace(0.0, 1.0, n)
    values = np.zeros((n, rec))
    for i in range(n):
        if i < 40:
            values[i] = 1.0
        elif i < chaos_from:
            values[i] = np.tile([1.0, 2.0], rec // 2)
        else:
            values[i] = rng.uniform(0.0, 4.0, rec)
    meta = {"id": run_id, "system": {"name": "toy"}, "mode": "bifurcation",
            "params": {"k": 0.5, "a": a}, "sweep": {"x": {"param": "k", "min": 0.0, "max": 1.0}},
            "options": {"integrator": "rk4", "dt": 0.01, "t_end": 10.0, "transient": 5.0,
                        "record_mode": "maxima"},
            "axes": {"x": "k", "y": "maxima of x"}}
    np.savez(path, meta=json.dumps(meta), params=params, values=values,
             counts=np.full(n, rec))
    return str(path)


class NoiseRule(unittest.TestCase):
    def test_accidental_cluster_in_noise_is_chaotic_only_when_asked(self):
        row = np.concatenate([[5.0, 5.0, 5.0], np.linspace(0, 100, 30)])[None, :]
        counts = np.array([row.shape[1]])
        plain = cl.cluster_grid(row, counts, eps=0.5)
        strict = cl.cluster_grid(row, counts, eps=0.5, max_noise_frac=0.3)
        self.assertEqual(int(plain["n_clusters"][0]), 1)        # map behaviour unchanged
        self.assertEqual(int(strict["n_clusters"][0]), -1)
        self.assertGreater(int(strict["n_noise"][0]), 20)


class SequenceRule(unittest.TestCase):
    def test_periods_from_time_order(self):
        rng = np.random.default_rng(2)
        rows = [
            np.tile([1.0], 40),                          # period 1
            np.tile([1.0, 3.0, 2.0], 14)[:40],           # period 3
            np.tile([1.0, 2.0], 20) + rng.normal(0, 1e-6, 40),   # period 2, numerical jitter
            1.0 + 0.5 * (np.arange(40) % 7 == 0),        # period 7
        ]
        v = np.array(rows)
        out = cl.period_grid(v, np.full(len(rows), 40))
        self.assertEqual(out["n_clusters"].tolist(), [1, 3, 2, 7])
        self.assertEqual(out["rule"], "sequence")

    def test_chaos_in_narrow_bands_is_aperiodic(self):
        # Values in 4 tight bands, visited in a non-repeating order: DBSCAN
        # counts 4 clusters ("period 4"); the sequence never repeats.
        rng = np.random.default_rng(3)
        bands = np.array([0.0, 1.0, 2.0, 3.0])
        row = bands[rng.integers(0, 4, 60)] + rng.uniform(-0.01, 0.01, 60)
        v, c = row[None, :], np.array([60])
        self.assertEqual(int(cl.cluster_grid(v, c, eps=0.05)["n_clusters"][0]), 4)
        self.assertEqual(int(cl.period_grid(v, c)["n_clusters"][0]), -1)

    def test_too_few_values_and_short_rows(self):
        v = np.zeros((2, 40))
        v[1, :6] = [1, 2, 1, 2, 1, 2]
        out = cl.period_grid(v, np.array([3, 6]))
        self.assertEqual(out["n_clusters"].tolist(), [0, 2])   # 3 < min_points; 6 >= 2*2+2


class Inspect(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_summary_regions(self):
        run = runinspect.load(fake_1d(self.dir / "a.npz"))
        out = runinspect.summary(run)
        labels = [r["label"] for r in out["regions"]]
        self.assertEqual(labels, ["period 1", "period 2", "chaotic"])
        self.assertAlmostEqual(out["regions"][1]["from"], 40 / 99, places=3)
        text = runinspect.render("summary", out, run)
        self.assertIn("first chaotic region at k≈0.8081", text)
        self.assertLess(len(text.splitlines()), 12)

    def test_slice_reports_cluster_centres(self):
        run = runinspect.load(fake_1d(self.dir / "a.npz"))
        out = runinspect.slice_at(run, "k", 0.5)
        self.assertEqual(out["period"], "period 2")
        self.assertEqual([c["value"] for c in out["clusters"]], [1.0, 2.0])
        with self.assertRaises(SystemExit):
            runinspect.slice_at(run, "zz", 1.0)

    def test_compare_lists_where_labels_change(self):
        a = runinspect.load(fake_1d(self.dir / "a.npz"))
        b = runinspect.load(fake_1d(self.dir / "b.npz", "run-0002", a=0.3, chaos_from=60))
        out = runinspect.compare(a, b)
        self.assertEqual(out["paramChanges"], {"a": [0.2, 0.3]})
        self.assertEqual(len(out["differs"]), 1)
        d = out["differs"][0]
        self.assertEqual((d["a"], d["b"]), ("period 2", "chaotic"))
        self.assertAlmostEqual(d["from"], 60 / 99, places=2)

    def test_cli_prints_text_and_json(self):
        import contextlib
        import io
        path = fake_1d(self.dir / "a.npz")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            runinspect.main(["summary", path, "--json"])
        self.assertEqual(json.loads(buf.getvalue())["run"], "run-0001")


if __name__ == "__main__":
    unittest.main()
