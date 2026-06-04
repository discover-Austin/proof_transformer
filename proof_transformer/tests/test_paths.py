from __future__ import annotations

import os
from pathlib import Path

from common import BrainTestCase
from proof_transformer.db import table_names
from proof_transformer.paths import BRAIN_SUBDIRS, DEFAULT_BRAIN_ROOT, build_brain_paths, reset_cached_paths


class PathTests(BrainTestCase):
    def test_default_root_uses_default_drive_or_workspace_fallback(self) -> None:
        original = os.environ.pop("AUSTIN_AI_BRAIN_ROOT", None)
        reset_cached_paths()
        try:
            paths = build_brain_paths()
            default_drive_exists = bool(DEFAULT_BRAIN_ROOT.drive) and Path(f"{DEFAULT_BRAIN_ROOT.drive}/").exists()
            if default_drive_exists:
                self.assertTrue(str(paths.root).replace('\\', '/').startswith('E:/AUSTIN_AI_BRAIN'))
            else:
                self.assertEqual(paths.root, Path(__file__).resolve().parents[2])
        finally:
            if original is not None:
                os.environ["AUSTIN_AI_BRAIN_ROOT"] = original
            reset_cached_paths()

    def test_env_override_creates_expected_directories_and_schema(self) -> None:
        for rel in BRAIN_SUBDIRS:
            self.assertTrue((self.paths.root / rel).exists(), rel)
        names = table_names(self.conn)
        required = {
            "memory_nodes",
            "memory_edges",
            "documents",
            "chunks",
            "chunk_embeddings",
            "retrieval_events",
            "learning_events",
            "training_candidates",
            "validator_results",
            "policy_snapshots",
            "learning_blocks",
            "ranker_weights",
            "adapter_registry",
            "task_queue",
            "task_runs",
            "workspace_roots",
            "workspace_files",
            "execution_runs",
            "repair_plans",
        }
        self.assertTrue(required.issubset(names))

    def test_invalid_env_drive_falls_back_to_workspace_root(self) -> None:
        original = os.environ.get("AUSTIN_AI_BRAIN_ROOT")
        os.environ["AUSTIN_AI_BRAIN_ROOT"] = "E:/AUSTIN_AI_BRAIN"
        reset_cached_paths()
        try:
            paths = build_brain_paths()
            if Path("E:/").exists():
                self.assertTrue(str(paths.root).replace('\\', '/').startswith('E:/AUSTIN_AI_BRAIN'))
            else:
                self.assertEqual(paths.root, Path(__file__).resolve().parents[2])
        finally:
            if original is None:
                os.environ.pop("AUSTIN_AI_BRAIN_ROOT", None)
            else:
                os.environ["AUSTIN_AI_BRAIN_ROOT"] = original
            reset_cached_paths()
