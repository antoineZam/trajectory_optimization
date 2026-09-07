"""Run logging helpers.

The codebase reports progress through `print()` rather than the `logging`
module, so Hydra's file handler never captured anything -- every `main.log`
in `outputs/` is 0 bytes. Rather than rewrite several hundred print calls,
`capture_console` tees stdout/stderr to a file inside the Hydra run directory,
which also captures Stable-Baselines3's own `verbose=1` tables.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


class _Tee:
    """Write-through stream duplicating everything to a second sink."""

    def __init__(self, primary: TextIO, mirror: TextIO):
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._mirror.write(data)
        # Flush the mirror eagerly: a run killed mid-training must still leave
        # a readable log behind.
        self._mirror.flush()
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        return self._primary.isatty()

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


@contextmanager
def capture_console(log_path: str | Path):
    """Tee stdout and stderr into `log_path` for the duration of the block.

    Args:
        log_path: File to append console output to. Parent dirs are created.

    Yields:
        The resolved Path of the log file.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    with path.open("a", encoding="utf-8") as handle:
        sys.stdout = _Tee(original_stdout, handle)
        sys.stderr = _Tee(original_stderr, handle)
        try:
            yield path
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
