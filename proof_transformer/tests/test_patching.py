from __future__ import annotations

from common import BrainTestCase
from proof_transformer.agent.patching import PatchApplier


class PatchingTests(BrainTestCase):
    def test_patch_applier_updates_workspace_file(self) -> None:
        workspace = self.paths.temp / "patch_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        target = workspace / "module.py"
        target.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

        plan = {
            "edits": [
                {
                    "path": "module.py",
                    "old_text": "return a - b",
                    "new_text": "return a + b",
                    "reason": "Fix arithmetic bug",
                }
            ]
        }
        result = PatchApplier().apply_plan(workspace, plan)
        self.assertTrue(result["applied"])
        self.assertIn(str(target.resolve()), result["files_changed"])
        self.assertIn("return a + b", target.read_text(encoding="utf-8"))
