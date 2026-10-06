"""
Contract v2 / describe tests.  Run:  .venv/bin/python -m unittest discover tests
"""

import io
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import contract  # noqa: E402

HEADER = "from numba import cuda\n"
RHS = textwrap.dedent("""
    @cuda.jit(device=True)
    def rhs(t, y, p, dy):
        dy[0] = -p[0] * y[0]
""")


def write(tmp: Path, body: str, tail: str = "", name="sys.py") -> str:
    path = tmp / name
    path.write_text(HEADER + textwrap.dedent(body) + tail)
    return str(path)


class Builtins(unittest.TestCase):
    def test_examples_describe_ok(self):
        for name, kind in [("rossler", "ode"), ("izhikevich", "hybrid"),
                           ("lorenz", "ode")]:
            with self.subTest(name):
                d = contract.describe(str(HERE / "systems" / f"{name}.py"))
                self.assertTrue(d["ok"], d["errors"])
                self.assertEqual(d["system"]["kind"], kind)
                self.assertEqual(len(d["params"]), d["system"]["nParams"])
                self.assertEqual(len(d["state"]), d["system"]["dim"])
                self.assertTrue(d["system"]["rhsHash"].startswith("sha256:"))

    def test_hybrid_gets_reset_modes_and_run_defaults(self):
        d = contract.describe(str(HERE / "systems" / "izhikevich.py"))
        opts = {o["name"]: o for o in d["runOptions"]}
        self.assertIn("isi", opts["record_mode"]["options"])
        self.assertEqual(opts["record_mode"]["default"], "isi")
        self.assertEqual(opts["integrator"]["default"], "euler")
        self.assertEqual(opts["dt"]["default"], 0.25)

    def test_ode_hides_reset_modes(self):
        d = contract.describe(str(HERE / "systems" / "rossler.py"))
        modes = next(o for o in d["runOptions"] if o["name"] == "record_mode")
        self.assertNotIn("isi", modes["options"])
        self.assertNotIn("spike", modes["options"])

    def test_record_component_bounded_by_dim(self):
        d = contract.describe(str(HERE / "systems" / "izhikevich.py"))
        rc = next(o for o in d["runOptions"] if o["name"] == "record_component")
        self.assertEqual(rc["max"], 1)


class Errors(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def assertError(self, d, *needles):
        self.assertFalse(d["ok"])
        joined = "\n".join(d["errors"])
        for n in needles:
            self.assertIn(n, joined)

    def test_missing_file(self):
        self.assertError(contract.describe(str(self.tmp / "nope.py")),
                         "cannot read file")

    def test_syntax_error_has_line(self):
        self.assertError(contract.describe(write(self.tmp, "DIM = = 1\n")),
                         "line 2: SyntaxError")

    def test_runtime_error_has_line(self):
        path = write(self.tmp, "DIM = 1\nN_PARAMS = 1\nX = 1 / 0\n")
        self.assertError(contract.describe(path), "line 4: ZeroDivisionError")

    def test_param_missing_max_has_line(self):
        path = write(self.tmp, """
            DIM = 1
            N_PARAMS = 2
            PARAMS = [
                dict(name="k", default=1.0, min=0.0, max=2.0, sweepable=True),
                dict(name="b", default=1.0, min=0.0),
            ]
            y0 = [1.0]
        """, RHS)
        self.assertError(contract.describe(path),
                         "line 7: PARAMS[1] ('b'): missing 'max'")

    def test_param_count_and_range_checks(self):
        path = write(self.tmp, """
            DIM = 1
            N_PARAMS = 1
            PARAMS = [
                dict(name="k", default=5.0, min=0.0, max=2.0, sweepable=True),
                dict(name="k", default=1.0, min=0.0, max=2.0, colour="red"),
            ]
            y0 = [1.0]
        """, RHS)
        self.assertError(contract.describe(path),
                         "PARAMS has 2 entries, N_PARAMS is 1",
                         "default 5.0 is outside [0.0, 2.0]",
                         "duplicate name", "unknown keys ['colour']")

    def test_needs_a_sweepable_param(self):
        path = write(self.tmp, """
            DIM = 1
            N_PARAMS = 1
            PARAMS = [dict(name="k", default=1.0, min=0.0, max=2.0)]
            y0 = [1.0]
        """, RHS)
        self.assertError(contract.describe(path), "no parameter is sweepable")

    def test_core_fields(self):
        path = write(self.tmp, """
            DIM = 2
            N_PARAMS = 0
            STATE = ["x"]
            y0 = [1.0]

            def rhs(t, y, p, dy):
                pass
        """)
        self.assertError(contract.describe(path),
                         "N_PARAMS must be a positive int",
                         "rhs must be decorated with @cuda.jit(device=True)",
                         "y0 has 1 values, DIM is 2",
                         "STATE must be a list of DIM")

    def test_reset_signature(self):
        path = write(self.tmp, """
            DIM = 1
            N_PARAMS = 1
            PARAMS = [dict(name="k", default=1.0, min=0.0, max=2.0, sweepable=True)]
            y0 = [1.0]

            @cuda.jit(device=True)
            def reset(t, y):
                return 0
        """, RHS)
        self.assertError(contract.describe(path),
                         "reset must take 3 arguments (t, y, p), takes 2")

    def test_bad_run_defaults(self):
        path = write(self.tmp, """
            DIM = 1
            N_PARAMS = 1
            PARAMS = [dict(name="k", default=1.0, min=0.0, max=2.0, sweepable=True)]
            RUN_DEFAULTS = dict(record_mode="isi", speed=3, dt=-1.0)
            y0 = [1.0]
        """, RHS)
        self.assertError(contract.describe(path),
                         "RUN_DEFAULTS['record_mode']: 'isi' is not allowed",
                         "RUN_DEFAULTS['speed']: unknown run option",
                         "RUN_DEFAULTS['dt']: -1.0 is not allowed")

    def test_v1_file_describes_with_warning(self):
        path = write(self.tmp, "DIM = 1\nN_PARAMS = 2\ny0 = [1.0]\n" + RHS)
        d = contract.describe(path)
        self.assertTrue(d["ok"], d["errors"])
        self.assertEqual([q["name"] for q in d["params"]], ["p0", "p1"])
        self.assertIn("PARAMS is missing", d["warnings"][0])

    def test_cli_exit_codes(self):
        bad = write(self.tmp, "DIM = = 1\n")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(contract._main(["describe", str(HERE / "systems" / "lorenz.py")]), 0)
            self.assertEqual(contract._main(["describe", bad]), 1)
            self.assertEqual(contract._main([]), 2)


class GpuSmoke(unittest.TestCase):
    """The 3-parameter Rössler must still run with no fixed_params (PARAMS defaults)."""

    @classmethod
    def setUpClass(cls):
        from numba import cuda
        if not cuda.is_available():
            raise unittest.SkipTest("no CUDA device")
        import bifurcation_gpu
        cls.bif = bifurcation_gpu

    def test_rossler_1d_uses_param_defaults(self):
        d = self.bif.run_bifurcation(str(HERE / "systems" / "rossler.py"), 2.0, 8.0, 64,
                                     dt=0.01, t_end=150.0, transient=100.0,
                                     record_n=50)
        # a = b = 0 (the old zero fill) would give no oscillation at all.
        self.assertGreater(int(d["counts"].sum()), 64)

    def test_izhikevich_2d_runs(self):
        d = self.bif.run_bifurcation_2d(
            str(HERE / "systems" / "izhikevich.py"), 0.0, 20.0, 8, -65.0, -50.0, 8,
            param2_index=1, dt=0.25, t_end=1000.0, transient=500.0,
            record_mode="isi", integrator="euler", record_n=16)
        self.assertEqual(d["counts"].shape[0], 64)
        self.assertGreater(int((d["counts"] > 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()
