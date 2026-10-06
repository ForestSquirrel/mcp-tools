"""
Studio run API tests (GPU).  Run:  .venv/bin/python -m unittest discover tests
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

_root = tempfile.TemporaryDirectory()
os.environ["BIF_STUDIO_ROOT"] = _root.name

import studio  # noqa: E402


def wait(job_id, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        st = studio.job_status(job_id)
        if st["state"] != "running":
            return st
        time.sleep(0.1)
    raise TimeoutError(job_id)


class Resolve(unittest.TestCase):
    def test_defaults_and_sweep_range(self):
        _, a = studio._resolve(str(HERE / "systems" / "rossler.py"), None,
                               {"x": {"param": "c"}}, None)
        self.assertEqual(a["mode"], "bifurcation")
        self.assertEqual(a["sweep"]["x"]["min"], 2.0)
        self.assertEqual(a["options"]["param_steps"], 2000)
        self.assertNotIn("eps", a["options"])         # period-map only
        self.assertEqual(a["fixed"], [5.7, 0.2, 0.2])

    def test_period_map_mode_defaults(self):
        _, a = studio._resolve(str(HERE / "systems" / "izhikevich.py"), None,
                               {"x": {"param": "I"}, "y": {"param": "d"}},
                               {"mode": "period-map"})
        self.assertEqual(a["options"]["param_steps"], 300)
        self.assertEqual(a["options"]["record_n"], 64)
        self.assertEqual(a["options"]["integrator"], "euler")
        self.assertEqual(a["sweep"]["y"]["index"], 2)

    def test_rejections(self):
        rhs = str(HERE / "systems" / "rossler.py")
        for params, sweep, opts, needle in [
            (None, {}, None, "sweep.x.param"),
            ({"zz": 1}, {"x": {"param": "c"}}, None, "unknown parameters"),
            (None, {"x": {"param": "c"}}, {"warp": 9}, "unknown run options"),
            (None, {"x": {"param": "c"}}, {"record_mode": "isi"}, "record_mode must be"),
            (None, {"x": {"param": "c", "min": 3, "max": 1}}, None, "must be <"),
            (None, {"x": {"param": "c"}, "y": {"param": "a"}}, None, "extra"),
            (None, {"x": {"param": "c"}, "y": {"param": "c"}},
             {"mode": "period-map"}, "different parameters"),
        ]:
            with self.subTest(needle), self.assertRaisesRegex(ValueError, needle):
                studio._resolve(rhs, params, sweep, opts)

    def test_bad_session(self):
        with self.assertRaises(ValueError):
            studio.session_dir("../etc")


class Runs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from numba import cuda
        if not cuda.is_available():
            raise unittest.SkipTest("no CUDA device")

    def test_1d_run_writes_paths_only(self):
        r = studio.submit_run(str(HERE / "systems" / "rossler.py"), "t1d",
                              sweep={"x": {"param": "a", "min": 0.1, "max": 0.3}},
                              run_options={"param_steps": 200, "dt": 0.01,
                                           "t_end": 150.0, "transient": 100.0,
                                           "record_n": 50})
        st = wait(r["jobId"])
        self.assertEqual(st["state"], "done", st.get("error"))
        paths = st["result"]["paths"]
        for k in ("png", "preview", "chart", "npz", "json"):
            self.assertTrue(Path(paths[k]).is_file(), k)
        self.assertLess(Path(paths["preview"]).stat().st_size, 2 << 20)
        self.assertGreater(st["result"]["summary"]["points"], 200)
        self.assertNotIn("base64", json.dumps(st))
        import numpy as np
        meta = json.loads(str(np.load(paths["npz"])["meta"]))
        self.assertEqual(meta["sweep"]["x"]["param"], "a")
        runs = studio.list_runs("t1d")
        self.assertEqual([x["id"] for x in runs], [r["runId"]])
        self.assertEqual(r["runId"], "run-0001")

    def test_2d_run(self):
        r = studio.submit_run(str(HERE / "systems" / "izhikevich.py"), "t2d",
                              sweep={"x": {"param": "I", "min": 0, "max": 20},
                                     "y": {"param": "d", "min": 0, "max": 8}},
                              run_options={"mode": "period-map", "param_steps": 16,
                                           "param2_steps": 12, "t_end": 1000.0,
                                           "transient": 500.0, "record_n": 16})
        st = wait(r["jobId"])
        self.assertEqual(st["state"], "done", st.get("error"))
        self.assertEqual(st["result"]["summary"]["cells"], 16 * 12)
        from PIL import Image
        w, h = Image.open(st["result"]["paths"]["png"]).size
        self.assertEqual((w % 16, h % 12), (0, 0))   # nearest upscale, pixel-exact

    def test_cancel(self):
        r = studio.submit_run(str(HERE / "systems" / "lorenz.py"), "tcancel",
                              sweep={"x": {"param": "rho"}},
                              run_options={"param_steps": 20000, "t_end": 2000.0,
                                           "transient": 100.0, "record_n": 64})
        studio.cancel_job(r["jobId"])
        st = wait(r["jobId"])
        self.assertEqual(st["state"], "cancelled")
        self.assertNotIn("npz", st["result"]["paths"])

    def test_compile_error_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.py"
            bad.write_text(
                "from numba import cuda\nDIM = 1\nN_PARAMS = 1\n"
                "PARAMS = [dict(name='k', default=1.0, min=0.0, max=2.0, sweepable=True)]\n"
                "y0 = [1.0]\n@cuda.jit(device=True)\ndef rhs(t, y, p, dy):\n"
                "    dy[0] = undefined_name * y[0]\n")
            r = studio.submit_run(str(bad), "terr", sweep={"x": {"param": "k"}},
                                  run_options={"param_steps": 20, "t_end": 2.0,
                                               "transient": 1.0, "dt": 0.01})
            st = wait(r["jobId"])
        self.assertEqual(st["state"], "error")
        self.assertIn("undefined_name", st["error"])


class Preview(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name) / "run-0001"

    def tearDown(self):
        self._tmp.cleanup()

    def test_small_image_keeps_full_resolution(self):
        import numpy as np
        img = np.zeros((300, 400, 3), np.uint8)
        img[::7, ::5] = 255
        out = studio._write_images(self.base, img, nearest=True)
        self.assertEqual(out["previewSize"], [400, 300])

    def test_large_image_shrinks_to_fit_budget(self):
        import numpy as np
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (1200, 1600, 3), dtype=np.uint8)  # incompressible
        old = studio.PREVIEW_MAX_BYTES
        studio.PREVIEW_MAX_BYTES = 400_000
        try:
            out = studio._write_images(self.base, img, nearest=False)
        finally:
            studio.PREVIEW_MAX_BYTES = old
        self.assertLessEqual(out["previewBytes"], 400_000)
        self.assertLess(out["previewSize"][0], 1600)
        # Shrinks no further than needed: well above a fixed 960 box's area share.
        self.assertGreater(out["previewSize"][0] * out["previewSize"][1], 0.5 * 400_000 / 3)
        w, h = out["previewSize"]
        self.assertAlmostEqual(w / h, 1600 / 1200, places=2)

    def test_low_res_1d_columns_repeated_not_squeezed(self):
        import numpy as np
        params = np.linspace(0, 1, 200)
        values = np.tile(np.linspace(-1, 1, 5), (200, 1))
        counts = np.full(200, 5)
        img = studio._density_1d(params, values, counts)
        self.assertEqual(img.shape, (studio.FULL_1D_HEIGHT, 1600, 3))
        self.assertTrue((img[:, 0] == img[:, 7]).all())       # one sweep column = 8 px


class Raster(unittest.TestCase):
    def test_cells_fit_box_and_keep_aspect(self):
        import base64
        import numpy as np
        from PIL import Image
        d = studio.session_dir("traster")
        img = np.zeros((100, 200, 3), np.uint8)
        img[:50] = (255, 0, 0)                   # top half red, bottom blue
        img[50:] = (0, 0, 255)
        Image.fromarray(img).save(d / "x.png")
        (d / "x.json").write_text('{"mode": "period-map"}')   # keeps aspect
        out = studio.raster(str(d / "x.png"), 40, 40)
        self.assertEqual((out["columns"], out["rows"]), (40, 10))
        words = np.frombuffer(base64.b64decode(out["cells"]), "<u4").reshape(10, 40, 3)
        self.assertTrue((words[..., 0] == 0x2580).all())
        self.assertEqual(int(words[0, 0, 1]), 0xff0000)
        self.assertEqual(int(words[-1, 0, 2]), 0x0000ff)
        self.assertLessEqual(len({(int(a), int(b)) for a, b in words[..., 1:].reshape(-1, 2)}), 1024)

    def test_1d_fills_box_and_keeps_thin_branches(self):
        import base64
        import numpy as np
        from PIL import Image
        d = studio.session_dir("traster1d")
        img = np.zeros((1200, 4000, 3), np.uint8)
        img[600, :] = (124, 106, 247)                # one-pixel branch across
        Image.fromarray(img).save(d / "y.png")
        (d / "y.json").write_text('{"mode": "bifurcation"}')
        out = studio.raster(str(d / "y.png"), 80, 20)
        self.assertEqual((out["columns"], out["rows"]), (80, 20))
        words = np.frombuffer(base64.b64decode(out["cells"]), "<u4").reshape(20, 80, 3)
        lit = (words[..., 1:] > 0x202020).any(axis=2)
        self.assertTrue(lit.any(axis=0).all())     # branch visible in every column

    def test_refuses_paths_outside_studio(self):
        with self.assertRaises(ValueError):
            studio.raster("/etc/hostname", 10, 10)


class Cleanup(unittest.TestCase):
    def test_removes_only_old(self):
        old = studio.session_dir("old-sess")
        (old / "x.json").write_text("{}")
        os.utime(old / "x.json", (0, 0))
        studio.session_dir("new-sess")
        (studio.STUDIO_ROOT / "new-sess" / "y.json").write_text("{}")
        out = studio.cleanup(1.0)
        self.assertIn(str(old), out["paths"])
        self.assertTrue((studio.STUDIO_ROOT / "new-sess").is_dir())


if __name__ == "__main__":
    unittest.main()
