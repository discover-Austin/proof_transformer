from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS_DIR = PROJECT_ROOT / "temp" / "operator_status"
LAST_LINE_COUNT = 20


def main(argv: list[str] | None = None) -> int:
    args, command = _parse_args(argv if argv is not None else sys.argv[1:])
    return run_watched_command(
        name=args.name,
        command=command,
        cwd=args.cwd,
        status_dir=args.status_dir,
    )


def run_watched_command(
    *,
    name: str,
    command: list[str],
    cwd: Path | str | None = None,
    status_dir: Path | str | None = None,
) -> int:
    if not command:
        raise ValueError("command must not be empty")
    safe_name = _safe_name(name)
    run_cwd = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    output_dir = Path(status_dir).resolve() if status_dir is not None else DEFAULT_STATUS_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = output_dir / f"{safe_name}.stdout.log"
    stderr_path = output_dir / f"{safe_name}.stderr.log"
    status_path = output_dir / f"{safe_name}.status.json"
    started = datetime.now(timezone.utc)
    last_stdout: deque[str] = deque(maxlen=LAST_LINE_COUNT)
    last_stderr: deque[str] = deque(maxlen=LAST_LINE_COUNT)
    lines_lock = threading.Lock()

    def write_status(status: str, returncode: int | None = None, ended: datetime | None = None) -> None:
        now = ended or datetime.now(timezone.utc)
        with lines_lock:
            payload = {
                "command": command,
                "cwd": str(run_cwd),
                "started_at": _format_time(started),
                "ended_at": _format_time(ended) if ended else None,
                "elapsed_seconds": round((now - started).total_seconds(), 3),
                "status": status,
                "returncode": returncode,
                "last_stdout_lines": list(last_stdout),
                "last_stderr_lines": list(last_stderr),
            }
        tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(status_path)

    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr_file:
        write_status("running")
        process = subprocess.Popen(
            command,
            cwd=str(run_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        threads = [
            threading.Thread(
                target=_stream_lines,
                args=(process.stdout, stdout_file, sys.stdout, last_stdout, lines_lock),
                daemon=True,
            ),
            threading.Thread(
                target=_stream_lines,
                args=(process.stderr, stderr_file, sys.stderr, last_stderr, lines_lock),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        while process.poll() is None:
            write_status("running")
            time.sleep(0.5)
        for thread in threads:
            thread.join()
        returncode = int(process.returncode)
        ended = datetime.now(timezone.utc)
        write_status("passed" if returncode == 0 else "failed", returncode=returncode, ended=ended)
        return returncode


def _stream_lines(
    source: TextIO | None,
    log_file: TextIO,
    console: TextIO,
    last_lines: deque[str],
    lines_lock: threading.Lock,
) -> None:
    if source is None:
        return
    try:
        for line in source:
            log_file.write(line)
            log_file.flush()
            console.write(line)
            console.flush()
            with lines_lock:
                last_lines.append(line.rstrip("\r\n"))
    finally:
        source.close()


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" not in argv:
        raise SystemExit("Usage: watch_command.py --name <name> [--cwd <path>] [--status-dir <path>] -- <command...>")
    split_at = argv.index("--")
    option_args = argv[:split_at]
    command = argv[split_at + 1 :]
    parser = argparse.ArgumentParser(description="Run a command while writing logs and a live status JSON file.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--status-dir", type=Path, default=None)
    args = parser.parse_args(option_args)
    if not command:
        parser.error("command after -- is required")
    return args, command


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "command"


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
