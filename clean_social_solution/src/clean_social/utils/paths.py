from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Resolve project root by locating pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate project root containing pyproject.toml")
