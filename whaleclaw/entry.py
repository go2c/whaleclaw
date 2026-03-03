"""Main entry point — starts the Gateway via uvicorn."""

from __future__ import annotations

from whaleclaw.runtime_python import ensure_embedded_python


def main() -> None:
    """Bootstrap and run the WhaleClaw Gateway."""
    ensure_embedded_python(module="whaleclaw.entry")

    import uvicorn

    from whaleclaw.config.loader import load_config
    from whaleclaw.config.paths import ensure_dirs
    from whaleclaw.gateway.app import create_app
    from whaleclaw.utils.log import setup_logging

    ensure_dirs()
    config = load_config()
    setup_logging(verbose=config.gateway.verbose)
    app = create_app(config)

    uvicorn.run(
        app,
        host=config.gateway.bind,
        port=config.gateway.port,
        log_level="debug" if config.gateway.verbose else "info",
    )


if __name__ == "__main__":
    main()
