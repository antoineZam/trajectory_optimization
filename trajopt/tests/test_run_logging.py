"""Console capture must never be able to kill the job it is logging.

A single emoji in a "new best lap time" message raised UnicodeEncodeError on
the cp1252 Windows console and took down a whole training run. It surfaced
only after the agent first completed a lap, because that branch had never
executed before -- a landmine on a success path.
"""
from __future__ import annotations

import io

from utils.run_logging import _Tee, capture_console


class _Cp1252Stream(io.TextIOBase):
    """Stdout that rejects anything cp1252 cannot represent, like a Windows console."""

    def __init__(self):
        self.written: list[str] = []

    def write(self, data: str) -> int:
        data.encode("cp1252")  # raises UnicodeEncodeError on emoji
        self.written.append(data)
        return len(data)


def test_tee_survives_unencodable_output():
    """An emoji must degrade, not raise."""
    console = _Cp1252Stream()
    mirror = io.StringIO()
    tee = _Tee(console, mirror)

    tee.write("plain text\n")
    tee.write("\U0001f3c6 NEW BEST LAP TIME! 26.9s\n")  # the exact crash

    # The console got a readable, escaped form...
    assert any("NEW BEST LAP TIME" in chunk for chunk in console.written)
    # ...and the log file keeps the original character.
    assert "\U0001f3c6" in mirror.getvalue()


def test_capture_console_writes_and_restores(tmp_path):
    """The log file is written, and stdout is put back afterwards."""
    import sys

    log = tmp_path / "console.log"
    before = sys.stdout

    with capture_console(log):
        print("captured line")
        assert sys.stdout is not before

    assert sys.stdout is before
    assert "captured line" in log.read_text(encoding="utf-8")


def test_capture_console_creates_parent_directories(tmp_path):
    """Hydra run directories are nested, so parents must be created."""
    log = tmp_path / "a" / "b" / "console.log"
    with capture_console(log):
        print("nested")
    assert log.exists()
