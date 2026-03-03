"""Allow running WhaleClaw as ``python -m whaleclaw``."""

import importlib

from whaleclaw.runtime_python import ensure_embedded_python

ensure_embedded_python(module="whaleclaw")

cli_main = importlib.import_module("whaleclaw.cli.main")
cli_main.app()
