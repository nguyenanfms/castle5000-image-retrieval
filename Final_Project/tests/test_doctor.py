import contextlib
import importlib.metadata
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from castle.__main__ import main
from castle.config import Config


class DoctorExitTest(unittest.TestCase):
    """Diagnostics must fail the shell on a problem; CPU fallback remains valid."""

    def run_doctor(self, missing_data=False, missing_package=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config(root, root / "data", root / "artifacts", root / "cache", None)
            if not missing_data:
                from PIL import Image
                cfg.data_dir.mkdir()
                Image.new("RGB", (20, 20)).save(cfg.data_dir / "08_2.jpg")
            torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
            modules = {"torch": torch, "torchvision": types.ModuleType("torchvision"),
                       "open_clip": types.ModuleType("open_clip")}

            def version(name):
                if missing_package and name == "open_clip_torch":
                    raise importlib.metadata.PackageNotFoundError(name)
                return "test-version"

            with patch.dict(sys.modules, modules), patch("importlib.metadata.version", side_effect=version), \
                    patch("castle.__main__.load_config", return_value=cfg), \
                    patch.object(sys, "argv", ["castle", "doctor"]), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main()
            return raised.exception.code

    def test_missing_data_fails_cli(self):
        self.assertEqual(self.run_doctor(missing_data=True), 1)

    def test_missing_dependency_fails_cli(self):
        self.assertEqual(self.run_doctor(missing_package=True), 1)

    def test_cpu_fallback_is_not_a_failure(self):
        self.assertEqual(self.run_doctor(), 0)
