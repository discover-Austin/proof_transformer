from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class DesktopCommanderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["DESKTOP_COMMANDER_HOME"] = str(Path(self.tempdir.name) / "runtime_data")

    def tearDown(self) -> None:
        os.environ.pop("DESKTOP_COMMANDER_HOME", None)
        self.tempdir.cleanup()
