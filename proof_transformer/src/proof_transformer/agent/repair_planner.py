from __future__ import annotations

import json
import re
from pathlib import Path

from ..model.local_llm import LocalLLM


class RepairPlanner:
    TRACEBACK_RE = re.compile(r'File "([^"]+)", line (\d+)')

    def build_plan(
        self,
        *,
        workspace_root: str | Path,
        command: list[str],
        execution: dict,
        llm: LocalLLM,
    ) -> tuple[str, dict]:
        workspace = Path(workspace_root).resolve()
        combined_path = Path(execution["stdout_path"]).with_name("combined.txt")
        combined_text = combined_path.read_text(encoding="utf-8") if combined_path.exists() else ""
        suspected_files = self._suspected_files(workspace, combined_text)
        raw_plan = self._llm_plan(workspace, command, execution, combined_text, suspected_files, llm)
        if raw_plan is None:
            return self._heuristic_plan(workspace, command, execution, combined_text, suspected_files)
        normalized = self._normalize_plan(raw_plan, command, suspected_files)
        analysis = normalized.get("summary") or self._heuristic_summary(execution, suspected_files, combined_text)
        return analysis, normalized

    def _llm_plan(
        self,
        workspace: Path,
        command: list[str],
        execution: dict,
        combined_text: str,
        suspected_files: list[str],
        llm: LocalLLM,
    ) -> dict | None:
        if llm.backend_name == "fallback":
            return None
        file_contexts = []
        for relative_path in suspected_files[:3]:
            candidate = (workspace / relative_path).resolve()
            if candidate.exists():
                file_contexts.append(f"FILE: {relative_path}\n```\n{candidate.read_text(encoding='utf-8')[:8000]}\n```")
        prompt = (
            "You are generating a safe local repair plan for a coding workspace.\n"
            "Return JSON only with keys: summary, suspected_files, edits, post_commands.\n"
            "Each edit must have path, old_text, new_text, reason.\n"
            "Only emit an edit if old_text matches exactly once in the current file.\n"
            "If you cannot produce a safe exact patch, return edits as an empty list.\n\n"
            f"Workspace: {workspace}\n"
            f"Command: {' '.join(command)}\n"
            f"Execution status: {execution['status']} exit_code={execution['exit_code']}\n\n"
            f"Execution output:\n```\n{combined_text[:12000]}\n```\n\n"
            + ("\n\n".join(file_contexts) if file_contexts else "")
        )
        response = llm.generate(prompt)
        return self._extract_json_object(response)

    def _heuristic_plan(
        self,
        workspace: Path,
        command: list[str],
        execution: dict,
        combined_text: str,
        suspected_files: list[str],
    ) -> tuple[str, dict]:
        summary = self._heuristic_summary(execution, suspected_files, combined_text)
        plan = {
            "summary": summary,
            "suspected_files": suspected_files,
            "edits": [],
            "post_commands": [command],
        }
        return summary, plan

    def _heuristic_summary(self, execution: dict, suspected_files: list[str], combined_text: str) -> str:
        status = execution.get("status")
        exit_code = execution.get("exit_code")
        lines = [f"Execution status={status}, exit_code={exit_code}."]
        if suspected_files:
            lines.append("Suspected files: " + ", ".join(suspected_files))
        for pattern in ("Traceback", "AssertionError", "ModuleNotFoundError", "NameError", "SyntaxError"):
            if pattern in combined_text:
                lines.append(f"Detected failure pattern: {pattern}.")
                break
        lines.append("No exact safe patch was generated locally; manual review or a stronger local model backend may be needed.")
        return " ".join(lines)

    def _suspected_files(self, workspace: Path, combined_text: str) -> list[str]:
        suspects: list[str] = []
        for match in self.TRACEBACK_RE.finditer(combined_text):
            path = Path(match.group(1)).resolve()
            try:
                relative = path.relative_to(workspace)
            except ValueError:
                continue
            rel_text = relative.as_posix()
            if rel_text not in suspects:
                suspects.append(rel_text)
        return suspects

    def _normalize_plan(self, raw_plan: dict, command: list[str], suspected_files: list[str]) -> dict:
        edits = []
        for edit in raw_plan.get("edits", []):
            if not isinstance(edit, dict):
                continue
            path = str(edit.get("path", "")).strip()
            old_text = str(edit.get("old_text", ""))
            new_text = str(edit.get("new_text", ""))
            if not path or not old_text:
                continue
            edits.append(
                {
                    "path": path,
                    "old_text": old_text,
                    "new_text": new_text,
                    "reason": str(edit.get("reason", "")),
                }
            )
        post_commands = raw_plan.get("post_commands") or [command]
        normalized_commands: list[list[str]] = []
        for item in post_commands:
            if isinstance(item, list) and item:
                normalized_commands.append([str(part) for part in item])
        if not normalized_commands:
            normalized_commands = [command]
        files = [str(item) for item in (raw_plan.get("suspected_files") or suspected_files)]
        return {
            "summary": str(raw_plan.get("summary", "")).strip() or self._heuristic_summary({"status": None, "exit_code": None}, suspected_files, ""),
            "suspected_files": files,
            "edits": edits,
            "post_commands": normalized_commands,
        }

    def _extract_json_object(self, text: str) -> dict | None:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
