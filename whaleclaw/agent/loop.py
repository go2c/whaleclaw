"""Backward-compatible alias for the single-agent runtime module."""

from __future__ import annotations

import sys

from whaleclaw.agent import single_agent as _single_agent

sys.modules[__name__] = _single_agent
