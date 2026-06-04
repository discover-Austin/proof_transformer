# scout_repo.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".turbo",
    ".cache",
    ".idea",
    ".vscode",
}

SECRET_NAME_PARTS = {
    ".env",
    "secret",
    "secrets",
    "token",
    "tokens",
    "credential",
    "credentials",
    "private",
    "key.pem",
    "id_rsa",
    "id_ed25519",
    ".p12",
    ".pfx",
}

TEXT_EXTS = {
    ".py", ".sql", ".md", ".txt", ".toml", ".yaml", ".yml", ".json",
    ".ini", ".cfg", ".sh", ".ps1", ".bat", ".js", ".ts", ".tsx", ".jsx",
    ".html", ".css", ".scss", ".rs", ".go", ".java", ".c", ".cpp", ".h",
}

IMPORTANT_NAMES = {
    "README.md",
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
    "peds_schema.sql",
}

DEFAULT_FOCUS_PATHS = [
    "src/proof_transformer/peds",
    "tests",
    "peds_schema.sql",
    "src/proof_transformer/db.py",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_secret_path(path: Path) -> bool:
    lower = str(path).replace("\\", "/").lower()
    name = path.name.lower()

    if name == ".env" or name.startswith(".env."):
        return True

    return any(part in lower for part in SECRET_NAME_PARTS)


def is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & EXCLUDE_DIRS) or is_secret_path(path)


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTS or path.name in IMPORTANT_NAMES


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return (result.stdout or result.stderr).strip()
    except Exception as exc:
        return f"<git unavailable: {exc}>"


def safe_read_text(path: Path, max_chars: int) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"error": str(exc), "text": ""}

    truncated = len(text) > max_chars
    return {
        "text": text[:max_chars],
        "truncated": truncated,
        "chars_total": len(text),
    }


def collect_tree(root: Path, max_files: int) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)

        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS and not is_secret_path(current / d)
        ]

        for filename in sorted(filenames):
            path = current / filename
            rel = path.relative_to(root)

            if is_excluded(rel) or is_secret_path(rel):
                continue

            try:
                stat = path.stat()
            except OSError:
                continue

            files.append({
                "path": rel.as_posix(),
                "size_bytes": stat.st_size,
                "suffix": path.suffix.lower(),
                "is_text": is_text_file(path),
            })

            if len(files) >= max_files:
                return files

    return files


def collect_focus_files(root: Path, focus_paths: list[str], max_chars: int, max_file_bytes: int) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()

    candidates: list[Path] = []

    for focus in focus_paths:
        p = root / focus
        if not p.exists():
            continue

        if p.is_file():
            candidates.append(p)
        else:
            for child in p.rglob("*"):
                if child.is_file():
                    candidates.append(child)

    for path in sorted(candidates):
        rel = path.relative_to(root)

        if rel.as_posix() in seen:
            continue
        seen.add(rel.as_posix())

        if is_excluded(rel) or is_secret_path(rel):
            continue

        try:
            stat = path.stat()
        except OSError:
            continue

        item: dict[str, Any] = {
            "path": rel.as_posix(),
            "size_bytes": stat.st_size,
            "sha256": None,
            "content": None,
            "skipped_reason": None,
        }

        if stat.st_size > max_file_bytes:
            item["skipped_reason"] = f"file larger than {max_file_bytes} bytes"
            collected.append(item)
            continue

        if not is_text_file(path):
            item["skipped_reason"] = "non-text or unsupported extension"
            collected.append(item)
            continue

        item["sha256"] = file_sha256(path)
        item["content"] = safe_read_text(path, max_chars)
        collected.append(item)

    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sanitized repo scout bundle for review.")
    parser.add_argument("--root", default=".", help="Repo root. Default: current directory.")
    parser.add_argument("--out", default="repo_scout.json", help="Output JSON file.")
    parser.add_argument("--max-files", type=int, default=3000)
    parser.add_argument("--max-chars-per-file", type=int, default=30000)
    parser.add_argument("--max-file-bytes", type=int, default=250_000)
    parser.add_argument(
        "--focus",
        action="append",
        default=[],
        help="Additional focus path. Can be used multiple times.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()

    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    focus_paths = DEFAULT_FOCUS_PATHS + args.focus

    bundle = {
        "generated_at": utc_now(),
        "root_name": root.name,
        "absolute_root_redacted": str(root.drive) + "/.../" + root.name if root.drive else ".../" + root.name,
        "safety": {
            "excluded_dirs": sorted(EXCLUDE_DIRS),
            "secret_name_parts": sorted(SECRET_NAME_PARTS),
            "note": "This scout intentionally excludes likely secrets, .git internals, caches, venvs, node_modules, and large/binary files.",
        },
        "git": {
            "branch": run_git(root, ["branch", "--show-current"]),
            "status_short": run_git(root, ["status", "--short"]),
            "last_commit": run_git(root, ["log", "-1", "--oneline"]),
            "tracked_files_count": len(run_git(root, ["ls-files"]).splitlines())
            if "git unavailable" not in run_git(root, ["rev-parse", "--is-inside-work-tree"])
            else None,
        },
        "tree": collect_tree(root, args.max_files),
        "focus_paths": focus_paths,
        "focus_files": collect_focus_files(
            root=root,
            focus_paths=focus_paths,
            max_chars=args.max_chars_per_file,
            max_file_bytes=args.max_file_bytes,
        ),
    }

    out = root / args.out
    out.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote sanitized scout bundle: {out}")
    print("Upload repo_scout.json here, or paste selected sections from it.")


if __name__ == "__main__":
    main()