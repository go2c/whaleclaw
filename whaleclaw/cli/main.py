"""WhaleClaw CLI — Typer application entry point."""

from __future__ import annotations

import importlib

import typer

from whaleclaw.runtime_python import ensure_embedded_python

ensure_embedded_python(module="whaleclaw")

app = typer.Typer(
    name="whaleclaw",
    help="WhaleClaw — Personal AI Assistant",
    no_args_is_help=True,
)

gateway_module = importlib.import_module("whaleclaw.cli.gateway_cmd")
app.add_typer(gateway_module.gateway_app, name="gateway", help="Gateway 服务管理")


@app.callback()
def _root_callback() -> None:
    """Ensure runtime uses the project-embedded Python."""
    ensure_embedded_python(module="whaleclaw")


def main() -> None:
    """CLI callable for script entry points."""
    app()


if __name__ == "__main__":
    main()
