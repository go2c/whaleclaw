"""Feishu long-connection (WebSocket) client bridge.

Uses the official ``lark-oapi`` SDK to maintain a persistent WebSocket
connection with the Feishu server.  Events are dispatched to the
FeishuBot in the main asyncio event loop via ``run_coroutine_threadsafe``.

The SDK's ``lark.ws.Client.start()`` blocks its own event loop, so we
run it in a dedicated daemon thread.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any

import lark_oapi as lark
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

from whaleclaw.utils.log import get_logger

log = get_logger(__name__)


class FeishuWSBridge:
    """Bridge between ``lark.ws.Client`` and the async FeishuBot."""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        on_message: Callable[[dict[str, Any]], Awaitable[None]],
        main_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._on_message = on_message
        self._main_loop = main_loop
        self._thread: threading.Thread | None = None
        self._ws_client: lark.ws.Client | None = None

    def _handle_message_receive(self, data: P2ImMessageReceiveV1) -> None:
        """Called by the SDK in its own thread when a message arrives."""
        try:
            raw = lark.JSON.marshal(data)
            import json
            body: dict[str, Any] = json.loads(raw)
            asyncio.run_coroutine_threadsafe(
                self._on_message(body), self._main_loop,
            )
        except Exception:
            log.exception("feishu.ws.handle_error")

    def _run(self) -> None:
        """Entry point for the daemon thread.

        The SDK caches a module-level ``loop`` obtained via
        ``asyncio.get_event_loop()`` at import time, which will be the
        main uvicorn loop.  We must replace it with a fresh loop that
        belongs to *this* thread so ``run_until_complete`` works.
        """
        import lark_oapi.ws.client as _ws_mod

        thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_loop)
        _ws_mod.loop = thread_loop

        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._handle_message_receive)
            .build()
        )
        self._ws_client = lark.ws.Client(
            self._app_id,
            self._app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.INFO,
        )
        log.info("feishu.ws.connecting")
        try:
            self._ws_client.start()
        except Exception:
            log.exception("feishu.ws.connection_failed")
        finally:
            thread_loop.close()

    def start(self) -> None:
        """Start the long-connection in a background daemon thread."""
        self._thread = threading.Thread(
            target=self._run, name="feishu-ws", daemon=True,
        )
        self._thread.start()
        log.info("feishu.ws.started")

    def stop(self) -> None:
        """Best-effort cleanup (SDK thread is daemon, dies with process)."""
        log.info("feishu.ws.stopped")
