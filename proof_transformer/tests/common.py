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

from proof_transformer.db import get_connection
from proof_transformer.paths import get_brain_paths, reset_cached_paths


class BrainTestCase(unittest.TestCase):
    def setUp(self) -> None:
        sandbox_root = Path(os.environ.get("AUSTIN_AI_BRAIN_TEST_SANDBOX", "E:/AUSTIN_AI_BRAIN/temp/test_roots"))
        if sandbox_root.drive and not Path(f"{sandbox_root.drive}/").exists():
            sandbox_root = Path(tempfile.gettempdir()) / "AUSTIN_AI_BRAIN_TEST_ROOTS"
        sandbox_root.mkdir(parents=True, exist_ok=True)
        self.tempdir = tempfile.TemporaryDirectory(dir=str(sandbox_root))
        os.environ["AUSTIN_AI_BRAIN_ROOT"] = str(Path(self.tempdir.name) / "AUSTIN_AI_BRAIN")
        os.environ.pop("PROOF_TRANSFORMER_RANKER_LR", None)
        os.environ.pop("PROOF_TRANSFORMER_EMBED_BACKEND", None)
        os.environ.pop("PROOF_TRANSFORMER_EMBED_MODEL", None)
        reset_cached_paths()
        self.paths = get_brain_paths(refresh=True)
        self.conn = get_connection()

    def tearDown(self) -> None:
        try:
            self.conn.close()
        finally:
            reset_cached_paths()
            os.environ.pop("AUSTIN_AI_BRAIN_ROOT", None)
            os.environ.pop("PROOF_TRANSFORMER_RANKER_LR", None)
            os.environ.pop("PROOF_TRANSFORMER_EMBED_BACKEND", None)
            os.environ.pop("PROOF_TRANSFORMER_EMBED_MODEL", None)
            self.tempdir.cleanup()
