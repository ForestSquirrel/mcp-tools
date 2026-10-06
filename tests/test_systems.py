"""
Named systems: drafts, keep, list/resolve, expiry.  Run:  .venv/bin/python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import studio  # noqa: E402
import systems  # noqa: E402

GOOD = (HERE / "systems" / "lorenz.py").read_text()
BAD = GOOD.replace('dict(name="rho", default=28.0, min=0.0, max=250.0,',
                   'dict(name="rho", default=28.0, min=0.0,')


class Systems(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._saved = (studio.STUDIO_ROOT, systems.SYSTEMS_DIR)
        studio.STUDIO_ROOT = root / "studio"
        systems.SYSTEMS_DIR = root / "systems"

    def tearDown(self):
        studio.STUDIO_ROOT, systems.SYSTEMS_DIR = self._saved
        self._tmp.cleanup()

    def test_save_reports_contract_errors_with_lines(self):
        out = systems.save_system("lorenz_x", BAD)
        self.assertFalse(out["ok"])
        self.assertIn("missing 'max'", out["errors"][0])
        self.assertTrue(out["errors"][0].startswith("line "))
        self.assertIn("save_system again", out["next"])
        self.assertTrue(Path(out["path"]).is_file())          # kept so it can be fixed

    def test_save_then_keep(self):
        out = systems.save_system("lorenz_x", GOOD)
        self.assertTrue(out["ok"])
        self.assertIn("/bif lorenz_x", out["next"])
        self.assertIn("rho=28 [0, 250] sweepable", out["params"])
        kept = systems.keep_system("lorenz_x")
        self.assertEqual(Path(kept["path"]).parent, systems.SYSTEMS_DIR)
        self.assertFalse(Path(out["path"]).exists())           # moved, not copied

    def test_keep_refuses_broken_and_existing(self):
        systems.save_system("chua", BAD)
        with self.assertRaisesRegex(ValueError, "breaks the contract"):
            systems.keep_system("chua")
        systems.save_system("chua", GOOD)
        systems.keep_system("chua")
        out = systems.save_system("chua", GOOD)
        self.assertTrue(out["shadowsPermanent"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            systems.keep_system("chua")
        self.assertTrue(systems.keep_system("chua", overwrite=True)["replaced"])

    def test_names_are_checked(self):
        for bad in ["", "Chua", "../x", "a-b", "x" * 41, "1abc"]:
            with self.subTest(bad), self.assertRaises(ValueError):
                systems.save_system(bad, GOOD)
        with self.assertRaises(ValueError):
            systems.keep_system("never_saved")

    def test_list_and_resolve_prefer_draft(self):
        systems.save_system("lorenz_x", GOOD)
        systems.keep_system("lorenz_x")
        draft = systems.save_system("lorenz_x", GOOD)["path"]
        systems.save_system("broken", BAD)
        out = systems.list_systems()
        self.assertEqual(out["resolve"]["lorenz_x"], draft)
        broken = next(s for s in out["systems"] if s["name"] == "broken")
        self.assertEqual((broken["ok"], broken["where"]), (False, "draft"))

    def test_cleanup_expires_old_drafts_only(self):
        old = Path(systems.save_system("old_one", GOOD)["path"])
        new = Path(systems.save_system("new_one", GOOD)["path"])
        os.utime(old, (0, 0))
        out = studio.cleanup(1.0)
        self.assertIn(str(old), out["paths"])
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())
        self.assertTrue((studio.STUDIO_ROOT / studio.DRAFTS).is_dir())

    def test_drafts_folder_is_not_a_session(self):
        systems.save_system("x", GOOD)
        with self.assertRaises(ValueError):
            studio.session_dir(studio.DRAFTS)
        studio.session_dir("s-old")
        (studio.STUDIO_ROOT / "s-old" / "a.json").write_text("{}")
        os.utime(studio.STUDIO_ROOT / "s-old" / "a.json", (1, 1))
        studio.session_dir("s-new")
        (studio.STUDIO_ROOT / "s-new" / "a.json").write_text("{}")
        self.assertEqual(studio.latest_session(), "s-new")


if __name__ == "__main__":
    unittest.main()
