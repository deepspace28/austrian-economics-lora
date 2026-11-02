"""Durable checkpoints for the long-running dataset-generation pipeline."""

import json
import os
import tempfile
from pathlib import Path


def load_progress(path: Path, total: int | None = None) -> int:
    """Return the saved chunk index, or zero when the checkpoint is unusable."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle).get("last_chunk", 0)
    except (OSError, ValueError, AttributeError, TypeError):
        return 0

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    if total is not None:
        return min(value, total)
    return value


def save_progress(index: int, path: Path) -> None:
    """Atomically persist a completed chunk index next to the target file."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("progress index must be a non-negative integer")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"last_chunk": index}, handle, indent=4)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
