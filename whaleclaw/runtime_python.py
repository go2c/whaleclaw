"""Runtime helpers for enforcing project-embedded Python."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REEXEC_ENV = "WHALECLAW_EMBEDDED_PY_REEXEC"


def _project_python() -> Path:
    """Return the expected embedded Python path under project root."""
    return Path(__file__).resolve().parents[1] / "python" / "bin" / "python3.12"


def ensure_embedded_python(*, module: str) -> None:
    """Re-exec current process with embedded Python when available."""
    if os.environ.get("WHALECLAW_DISABLE_EMBEDDED_PYTHON") == "1":
        return
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    if os.environ.get(_REEXEC_ENV) == "1":
        return

    target = _project_python()
    if not target.is_file():
        return

    try:
        current = Path(sys.executable).resolve()
        target_resolved = target.resolve()
    except OSError:
        return

    if current == target_resolved:
        return

    os.environ[_REEXEC_ENV] = "1"
    os.execv(
        str(target_resolved),
        [str(target_resolved), "-m", module, *sys.argv[1:]],
    )
