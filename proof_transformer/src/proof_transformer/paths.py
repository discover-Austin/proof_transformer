from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BRAIN_ROOT = Path("E:/AUSTIN_AI_BRAIN")
BRAIN_SUBDIRS = (
    "models",
    "memory",
    "memory/documents",
    "memory/vectors",
    "memory/graph",
    "ledger",
    "datasets",
    "evals",
    "checkpoints",
    "adapters",
    "cache",
    "temp",
    "logs",
    "sessions",
    "proof_transformer",
)


@dataclass(frozen=True)
class BrainPaths:
    root: Path
    models: Path
    memory: Path
    documents: Path
    vectors: Path
    graph: Path
    ledger: Path
    datasets: Path
    evals: Path
    checkpoints: Path
    adapters: Path
    cache: Path
    temp: Path
    logs: Path
    sessions: Path
    repo: Path
    db: Path

    def ensure(self) -> "BrainPaths":
        for path in (
            self.root,
            self.models,
            self.memory,
            self.documents,
            self.vectors,
            self.graph,
            self.ledger,
            self.datasets,
            self.evals,
            self.checkpoints,
            self.adapters,
            self.cache,
            self.temp,
            self.logs,
            self.sessions,
            self.repo,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "models": str(self.models),
            "memory": str(self.memory),
            "documents": str(self.documents),
            "vectors": str(self.vectors),
            "graph": str(self.graph),
            "ledger": str(self.ledger),
            "datasets": str(self.datasets),
            "evals": str(self.evals),
            "checkpoints": str(self.checkpoints),
            "adapters": str(self.adapters),
            "cache": str(self.cache),
            "temp": str(self.temp),
            "logs": str(self.logs),
            "sessions": str(self.sessions),
            "repo": str(self.repo),
            "db": str(self.db),
        }


_cached_paths: BrainPaths | None = None


def _workspace_root_fallback() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if candidate.name.lower() == "proof_transformer" and (candidate / "pyproject.toml").exists():
            return candidate.parent
        if (candidate / "proof_transformer" / "pyproject.toml").exists():
            return candidate
    return cwd / "AUSTIN_AI_BRAIN"


def _root_from_env() -> Path:
    raw = os.getenv("AUSTIN_AI_BRAIN_ROOT")
    if raw:
        path = Path(raw).expanduser()
        path = path if path.is_absolute() else path.absolute()
        if path.drive and not Path(f"{path.drive}/").exists():
            return _workspace_root_fallback()
        return path
    default_drive = DEFAULT_BRAIN_ROOT.drive
    if default_drive and not Path(f"{default_drive}/").exists():
        return _workspace_root_fallback()
    path = DEFAULT_BRAIN_ROOT.expanduser()
    return path if path.is_absolute() else path.absolute()


def build_brain_paths(root: Path | None = None) -> BrainPaths:
    base = root or _root_from_env()
    return BrainPaths(
        root=base,
        models=base / "models",
        memory=base / "memory",
        documents=base / "memory" / "documents",
        vectors=base / "memory" / "vectors",
        graph=base / "memory" / "graph",
        ledger=base / "ledger",
        datasets=base / "datasets",
        evals=base / "evals",
        checkpoints=base / "checkpoints",
        adapters=base / "adapters",
        cache=base / "cache",
        temp=base / "temp",
        logs=base / "logs",
        sessions=base / "sessions",
        repo=base / "proof_transformer",
        db=base / "memory" / "brain.sqlite3",
    )


def get_brain_paths(refresh: bool = False) -> BrainPaths:
    global _cached_paths
    if refresh or _cached_paths is None:
        _cached_paths = build_brain_paths().ensure()
    return _cached_paths


def reset_cached_paths() -> None:
    global _cached_paths
    _cached_paths = None


def brain_path(*parts: str) -> Path:
    return get_brain_paths().root.joinpath(*parts)


def is_under_brain_root(path: str | Path) -> bool:
    candidate = Path(path).absolute()
    root = get_brain_paths().root.absolute()
    try:
        os.path.commonpath([str(candidate), str(root)])
    except ValueError:
        return False
    return str(candidate).lower().startswith(str(root).lower())
