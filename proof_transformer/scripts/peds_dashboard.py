from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:  # pragma: no cover - depends on local Python installation
    tk = None
    ttk = None
    messagebox = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
STATUS_DIR = PROJECT_ROOT / "temp" / "operator_status"


def build_generation_command(generations: int, repo_root: Path | None = None) -> list[str]:
    root = repo_root or PROJECT_ROOT
    return [
        sys.executable,
        str(root / "examples" / "peds_trained_local_generations.py"),
        "--generations",
        str(generations),
    ]


def build_broad_validation_command(repo_root: Path | None = None) -> list[str]:
    _ = repo_root or PROJECT_ROOT
    return [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_peds*.py"]


def build_watcher_command(
    *,
    name: str,
    command: list[str],
    repo_root: Path | None = None,
    status_dir: Path | None = None,
) -> list[str]:
    root = repo_root or PROJECT_ROOT
    output_dir = status_dir or STATUS_DIR
    return [
        sys.executable,
        str(root / "scripts" / "watch_command.py"),
        "--name",
        name,
        "--cwd",
        str(root),
        "--status-dir",
        str(output_dir),
        "--",
        *command,
    ]


def build_latest_report_command(repo_root: Path | None = None) -> list[str]:
    root = repo_root or PROJECT_ROOT
    return [sys.executable, str(root / "scripts" / "peds_latest_report.py"), "--json"]


def load_latest_report(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or PROJECT_ROOT
    result = subprocess.run(
        build_latest_report_command(root),
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {"found": False, "passed": False, "blocker": result.stderr.strip() or "report command failed"}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"found": False, "passed": False, "blocker": "latest report command returned invalid JSON"}


def load_latest_watcher_status(status_dir: Path | None = None) -> dict[str, Any] | None:
    output_dir = status_dir or STATUS_DIR
    if not output_dir.exists():
        return None
    candidates = list(output_dir.glob("*.status.json"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: (path.stat().st_mtime, path.name))
    try:
        status = json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "failed", "path": str(latest), "blocker": "invalid status JSON"}
    status["path"] = str(latest)
    return status


class PEDSDashboard:
    def __init__(self, root: "tk.Tk", repo_root: Path = PROJECT_ROOT) -> None:
        self.root = root
        self.repo_root = repo_root
        self.root.title("PEDS Operator Cockpit")
        self.root.geometry("980x680")
        self.status_var = tk.StringVar(value="Ready")
        self.report_text = tk.StringVar(value="")
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x")
        ttk.Label(header, text="Repo Root").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text=str(self.repo_root)).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(header, textvariable=self.status_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        buttons = ttk.Frame(container)
        buttons.pack(fill="x", pady=10)
        ttk.Button(buttons, text="Run 1 Generation", command=lambda: self.run_generation(1)).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Run 3 Generations", command=lambda: self.run_generation(3)).pack(side="left", padx=6)
        ttk.Button(buttons, text="Run Broad PEDS Validation", command=self.run_broad_validation).pack(side="left", padx=6)
        ttk.Button(buttons, text="Refresh", command=self.refresh).pack(side="left", padx=6)
        ttk.Button(buttons, text="Open Latest Run Folder", command=self.open_latest_run_folder).pack(side="left", padx=6)

        self.text = tk.Text(container, wrap="word", height=30)
        self.text.pack(fill="both", expand=True)

    def run_generation(self, generations: int) -> None:
        command = build_generation_command(generations, self.repo_root)
        watcher = build_watcher_command(
            name=f"peds_generation_{generations}",
            command=command,
            repo_root=self.repo_root,
        )
        self._launch_watcher(watcher, f"Started {generations} generation run")

    def run_broad_validation(self) -> None:
        command = build_broad_validation_command(self.repo_root)
        watcher = build_watcher_command(name="peds_validation", command=command, repo_root=self.repo_root)
        self._launch_watcher(watcher, "Started broad PEDS validation")

    def refresh(self) -> None:
        report = load_latest_report(self.repo_root)
        status = load_latest_watcher_status()
        text = _format_dashboard_text(report, status)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self.status_var.set("Refreshed")

    def open_latest_run_folder(self) -> None:
        report = load_latest_report(self.repo_root)
        run_root = report.get("run_root")
        if not run_root or not Path(run_root).exists():
            if messagebox is not None:
                messagebox.showinfo("PEDS", "No latest run folder found.")
            return
        _open_path(Path(run_root))

    def _launch_watcher(self, command: list[str], status_message: str) -> None:
        def worker() -> None:
            subprocess.Popen(command, cwd=str(self.repo_root))

        threading.Thread(target=worker, daemon=True).start()
        self.status_var.set(status_message)


def _format_dashboard_text(report: dict[str, Any], status: dict[str, Any] | None) -> str:
    lines = [
        "Latest PEDS Report",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "Latest Watcher Status",
        json.dumps(status or {"found": False}, ensure_ascii=False, indent=2, sort_keys=True),
    ]
    return "\n".join(lines)


def _open_path(path: Path) -> None:
    if os.name == "nt" and hasattr(os, "startfile"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        webbrowser.open(path.resolve().as_uri())


def create_app(repo_root: Path = PROJECT_ROOT) -> "tk.Tk":
    if tk is None or ttk is None:
        raise RuntimeError("tkinter is not available in this Python environment")
    root = tk.Tk()
    PEDSDashboard(root, repo_root=repo_root)
    return root


def main() -> int:
    root = create_app()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
