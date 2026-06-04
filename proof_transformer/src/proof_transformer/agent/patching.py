from __future__ import annotations

import os
from pathlib import Path


class PatchApplyError(RuntimeError):
    pass


class PatchApplier:
    def apply_plan(self, workspace_root: str | Path, plan: dict) -> dict:
        root = Path(workspace_root).resolve()
        edits = list(plan.get("edits") or [])
        if not edits:
            return {
                "workspace_root": str(root),
                "edit_count": 0,
                "files_changed": [],
                "applied": False,
                "reason": "No edits to apply.",
            }
        grouped: dict[Path, list[dict]] = {}
        for edit in edits:
            path = self._resolve_under_root(root, edit["path"])
            grouped.setdefault(path, []).append(edit)

        changed_files: list[str] = []
        total_edits = 0
        for path, file_edits in grouped.items():
            original = path.read_text(encoding="utf-8")
            replacements = self._locate_replacements(original, file_edits)
            updated = original
            for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
                updated = updated[:start] + replacement + updated[end:]
            if updated != original:
                path.write_text(updated, encoding="utf-8")
                changed_files.append(str(path))
            total_edits += len(file_edits)
        return {
            "workspace_root": str(root),
            "edit_count": total_edits,
            "files_changed": changed_files,
            "applied": True,
        }

    def _resolve_under_root(self, root: Path, relative_path: str) -> Path:
        candidate = (root / relative_path).resolve()
        try:
            common = os.path.commonpath([str(root), str(candidate)])
        except ValueError as exc:
            raise PatchApplyError(f"Invalid patch path: {relative_path}") from exc
        if common != str(root):
            raise PatchApplyError(f"Patch path escapes workspace root: {relative_path}")
        if not candidate.exists():
            raise PatchApplyError(f"Patch target does not exist: {relative_path}")
        return candidate

    def _locate_replacements(self, original: str, file_edits: list[dict]) -> list[tuple[int, int, str]]:
        ranges: list[tuple[int, int, str]] = []
        used_ranges: list[tuple[int, int]] = []
        for edit in file_edits:
            old_text = str(edit.get("old_text", ""))
            new_text = str(edit.get("new_text", ""))
            if not old_text:
                raise PatchApplyError("Patch edit old_text cannot be empty")
            first = original.find(old_text)
            if first < 0:
                raise PatchApplyError(f"old_text not found for patch path {edit['path']}")
            second = original.find(old_text, first + 1)
            if second >= 0:
                raise PatchApplyError(f"old_text is not unique for patch path {edit['path']}")
            start = first
            end = first + len(old_text)
            for used_start, used_end in used_ranges:
                if not (end <= used_start or start >= used_end):
                    raise PatchApplyError(f"Overlapping edits for patch path {edit['path']}")
            used_ranges.append((start, end))
            ranges.append((start, end, new_text))
        return ranges
