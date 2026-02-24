"""Feishu bot — core message handling logic."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from whaleclaw.channels.feishu.allowlist import FeishuAllowList
from whaleclaw.channels.feishu.card import FeishuCard
from whaleclaw.channels.feishu.client import FeishuClient
from whaleclaw.channels.feishu.config import FeishuConfig
from whaleclaw.channels.feishu.dedup import MessageDedup
from whaleclaw.channels.feishu.mention import is_bot_mentioned, strip_bot_mention
from whaleclaw.utils.log import get_logger

_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_FILE_RE = re.compile(r"(?<!!)\[([^\]]+)\]\((/[^)]+)\)")
_BARE_PATH_RE = re.compile(r"(?:^|[\s`])(/[\w./-]+\.(?:pptx|ppt|pdf|docx|doc|xlsx|xls))(?:[\s`]|$)", re.MULTILINE)
_FILE_EXTS = {".pptx", ".ppt", ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".zip", ".tar", ".gz"}

if TYPE_CHECKING:
    from whaleclaw.config.schema import WhaleclawConfig
    from whaleclaw.sessions.manager import SessionManager
    from whaleclaw.tools.registry import ToolRegistry

log = get_logger(__name__)


class FeishuBot:
    """Process incoming Feishu messages and route to the Agent."""

    def __init__(
        self,
        client: FeishuClient,
        config: FeishuConfig,
        allowlist: FeishuAllowList | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._dedup = MessageDedup()
        self._allowlist = allowlist or FeishuAllowList()
        self._pairing_codes: dict[str, str] = {}
        self._bot_open_id = ""
        self._whaleclaw_config: WhaleclawConfig | None = None
        self._session_manager: SessionManager | None = None
        self._tool_registry: ToolRegistry | None = None

    def set_bot_open_id(self, bot_open_id: str) -> None:
        self._bot_open_id = bot_open_id

    def bind_agent(
        self,
        config: WhaleclawConfig,
        session_manager: SessionManager,
        registry: ToolRegistry,
    ) -> None:
        """Inject Agent dependencies so handle_message can run the full loop."""
        self._whaleclaw_config = config
        self._session_manager = session_manager
        self._tool_registry = registry

    async def handle_event(
        self, event_type: str, body: dict[str, Any]
    ) -> None:
        """Dispatch an event to the appropriate handler."""
        if event_type == "im.message.receive_v1":
            event = body.get("event", {})
            await self.handle_message(event)

    async def handle_message(self, event: dict[str, Any]) -> None:
        """Process a received message event."""
        message = event.get("message", {})
        msg_id = message.get("message_id", "")
        chat_type = message.get("chat_type", "")
        sender = event.get("sender", {}).get("sender_id", {})
        open_id = sender.get("open_id", "")

        if self._dedup.is_duplicate(msg_id):
            return
        self._dedup.mark(msg_id)

        text = self.extract_text(message)
        if not text:
            return

        if chat_type == "group":
            group_cfg = self._config.groups.get(
                message.get("chat_id", "")
            )
            need_mention = (
                group_cfg.require_mention if group_cfg else True
            )
            if need_mention and not is_bot_mentioned(message, self._bot_open_id):
                return
            text = strip_bot_mention(text, "")

        not_allowed = (
            chat_type == "p2p"
            and self._config.dm_policy != "open"
            and not self._allowlist.is_allowed(open_id)
        )
        if not_allowed:
            if self._config.dm_policy == "closed":
                return
            await self._send_pairing_prompt(open_id, msg_id)
            return

        log.info(
            "feishu.message",
            chat_type=chat_type,
            open_id=open_id,
            text_len=len(text),
        )

        card = FeishuCard.streaming_card()
        resp = await self._client.reply_message(msg_id, "interactive", card)
        reply_msg_id = resp.get("data", {}).get("message_id", "")

        if not reply_msg_id:
            content = json.dumps({"text": "处理中..."})
            await self._client.reply_message(msg_id, "text", content)
            return

        await self._run_agent_and_reply(text, open_id, reply_msg_id)

    async def _run_agent_and_reply(
        self, text: str, peer_id: str, card_msg_id: str
    ) -> None:
        """Run the Agent loop and stream results back via Feishu card + WebChat."""
        if not self._whaleclaw_config or not self._session_manager or not self._tool_registry:
            card = FeishuCard.text_card(text)
            await self._client.update_message(card_msg_id, card)
            return

        from whaleclaw.agent.loop import run_agent
        from whaleclaw.gateway.protocol import make_message
        from whaleclaw.gateway.ws import broadcast_all
        from whaleclaw.providers.router import ModelRouter

        session = await self._session_manager.get_or_create("feishu", peer_id)
        await self._session_manager.add_message(session, "user", text)

        await broadcast_all(make_message(session.id, f"📨 **飞书** `{peer_id[:8]}…`:\n{text}"))

        router = ModelRouter(self._whaleclaw_config.models)

        try:
            reply = await run_agent(
                message=text,
                session_id=session.id,
                config=self._whaleclaw_config,
                session=session,
                router=router,
                registry=self._tool_registry,
                session_manager=self._session_manager,
                session_store=self._session_manager._store,  # noqa: SLF001
            )
            log.info("feishu.agent_reply", reply_len=len(reply), preview=reply[:200])
        except Exception as exc:
            log.error("feishu.agent_error", error=str(exc))
            error_card = FeishuCard.error_card(f"处理失败: {exc}")
            await self._client.update_message(card_msg_id, error_card)
            await broadcast_all(make_message(session.id, f"❌ **飞书处理失败**: {exc}"))
            return

        if not reply.strip():
            fallback = FeishuCard.text_card("任务执行中但未返回结果，请稍后重试或查看 WebChat。")
            await self._client.update_message(card_msg_id, fallback)
            return

        try:
            await self._session_manager.add_message(session, "assistant", reply)
            final_card, file_paths = await self._build_reply_card(reply)
            await self._client.update_message(card_msg_id, final_card)
            for fp in file_paths:
                await self._send_file_to_peer(peer_id, fp)
            await broadcast_all(make_message(session.id, f"🤖 **飞书回复**:\n{reply}"))
        except Exception as exc:
            log.exception("feishu.reply_failed")
            fallback = FeishuCard.text_card(reply or f"回复发送失败: {exc}")
            await self._client.update_message(card_msg_id, fallback)

    async def _build_reply_card(self, reply: str) -> tuple[str, list[Path]]:
        """Build a Feishu card, uploading images inline. Returns (card_json, file_paths)."""
        images: list[tuple[str, str]] = []
        for match in _IMG_RE.finditer(reply):
            path = match.group(2)
            local = Path(path)
            if local.is_file() and local.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                try:
                    image_key = await self._client.upload_image(local.read_bytes())
                    if image_key:
                        images.append((match.group(0), image_key))
                except Exception:
                    log.warning("feishu.image_upload_failed", path=path)

        file_paths: list[Path] = []
        file_replacements: list[tuple[str, str]] = []
        seen_paths: set[str] = set()

        for match in _FILE_RE.finditer(reply):
            name, path = match.group(1), match.group(2)
            local = Path(path)
            if local.is_file() and local.suffix.lower() in _FILE_EXTS and path not in seen_paths:
                seen_paths.add(path)
                file_paths.append(local)
                file_replacements.append((match.group(0), f"📎 {name}"))

        for match in _BARE_PATH_RE.finditer(reply):
            path = match.group(1)
            local = Path(path)
            if local.is_file() and path not in seen_paths:
                seen_paths.add(path)
                file_paths.append(local)
                file_replacements.append((path, f"📎 {local.name}"))

        log.info("feishu.reply_files", count=len(file_paths), paths=[str(p) for p in file_paths])

        clean_text = reply
        for md_str, _ in images:
            clean_text = clean_text.replace(md_str, "")
        for md_str, label in file_replacements:
            clean_text = clean_text.replace(md_str, label)
        clean_text = clean_text.strip()

        elements: list[dict[str, Any]] = []
        if clean_text:
            elements.append({"tag": "div", "text": {"tag": "lark_md", "content": clean_text}})
        for _, image_key in images:
            elements.append({"tag": "img", "img_key": image_key, "alt": {"tag": "plain_text", "content": " "}})

        if not elements:
            elements.append({"tag": "div", "text": {"tag": "lark_md", "content": reply}})

        card: dict[str, Any] = {"config": {"wide_screen_mode": True}, "elements": elements}
        return json.dumps(card, ensure_ascii=False), file_paths

    async def _send_file_to_peer(self, peer_id: str, file_path: Path) -> None:
        """Upload a local file to Feishu and send as a file message."""
        try:
            log.info("feishu.file_uploading", name=file_path.name, size=file_path.stat().st_size)
            data = file_path.read_bytes()
            file_key = await self._client.upload_file(data, file_path.name, "stream")
            log.info("feishu.file_uploaded", file_key=file_key)
            if file_key:
                content = json.dumps({"file_key": file_key})
                await self._client.send_message(peer_id, "file", content)
                log.info("feishu.file_sent", name=file_path.name)
            else:
                log.warning("feishu.file_upload_no_key", name=file_path.name)
        except Exception:
            log.exception("feishu.file_send_failed", path=str(file_path))

    @staticmethod
    def extract_text(message: dict[str, Any]) -> str:
        """Extract plain text from a Feishu message."""
        msg_type = message.get("message_type", "text")
        content_str = message.get("content", "{}")
        try:
            content = json.loads(content_str)
        except (json.JSONDecodeError, TypeError):
            return ""

        if msg_type == "text":
            return content.get("text", "")
        if msg_type == "post":
            parts: list[str] = []
            blocks = content.get("content") or [[]]
            for line in blocks:
                for elem in line:
                    if elem.get("tag") == "text":
                        parts.append(elem.get("text", ""))
            return " ".join(parts)
        return ""

    async def _send_pairing_prompt(
        self, open_id: str, msg_id: str
    ) -> None:
        import random
        import string

        code = "".join(random.choices(string.digits, k=6))  # noqa: S311
        self._pairing_codes[code] = open_id
        card = FeishuCard.text_card(
            f"请将此配对码发送给管理员进行验证:\n\n**{code}**",
            title="配对验证",
        )
        await self._client.reply_message(msg_id, "interactive", card)

    def approve_pairing(self, code: str) -> str | None:
        """Approve a pairing code and add the user to the allowlist."""
        open_id = self._pairing_codes.pop(code, None)
        if open_id:
            self._allowlist.add(open_id)
        return open_id
