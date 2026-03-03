"""Skill routing by keyword matching."""

from __future__ import annotations

import re

from whaleclaw.skills.parser import Skill


class SkillRouter:
    """Route user messages to skills by keyword matching."""

    def route(
        self,
        user_message: str,
        available_skills: list[Skill],
        max_skills: int = 2,
    ) -> list[Skill]:
        """Select top skills by /use command or keyword score."""
        msg = user_message.strip()
        lower = msg.lower()
        if msg.startswith("/use "):
            skill_id = msg[5:].strip().lower()
            for s in available_skills:
                if s.id.lower() == skill_id:
                    return [s]

        # Explicit skill mention in natural language:
        # e.g. "用 ppt-generator 这个技能" / "use skill ppt-generator".
        if any(marker in lower for marker in ("技能", "skill")):
            explicit: list[Skill] = []
            for s in available_skills:
                if self._mentions_skill(msg, s):
                    explicit.append(s)
            if explicit:
                explicit.sort(key=lambda x: x.id)
                return explicit[:max_skills]

        scored = [(self._score(msg, s), s) for s in available_skills]
        scored = [(score, s) for score, s in scored if score > 0]
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return [s for _, s in scored[:max_skills]]

    def _score(self, message: str, skill: Skill) -> float:
        """Return hit_count / total_triggers, 0 if no triggers."""
        if not skill.triggers:
            return 0.0
        lower = message.lower()
        hits = sum(1 for t in skill.triggers if t.lower() in lower)
        return hits / len(skill.triggers)

    @staticmethod
    def _norm_text(text: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.lower())

    def _mentions_skill(self, message: str, skill: Skill) -> bool:
        lower = message.lower()
        msg_norm = self._norm_text(message)
        for raw in (skill.id, skill.name):
            token = raw.strip().lower()
            if not token:
                continue
            if token in lower:
                return True
            norm = self._norm_text(token)
            if len(norm) >= 5 and norm in msg_norm:
                return True
        return False
