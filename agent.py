"""AIOS-Local Agent Engine v3 — backward-compatible re-export."""

from core.agent import Agent
from core.config import (
    ALLOWED_WEB_DOMAINS,
    BASE_DIR,
    DESTRUCTIVE_RE,
    SAFETY_MODE,
)
from core.knowledge_sync import KnowledgeSync
from core.learning_engine import LearningEngine
from core.memory import MemoryManager
from core.parser_utils import find_balanced_json_end
from core.rag_engine import RAGEngine
from core.tools import _run_goose_cli, _tool_goose_run, execute_tool
from core.utils import _blocked_by_mode, _domain_allowed, _safe_resolve_path

# Parser aliases used by Agent
_find_balanced_json_end = find_balanced_json_end
_parse_tool_call = Agent._parse_tool_call
_parse_delegate = Agent._parse_delegate
_parse_rag = Agent._parse_rag

__all__ = [
    "Agent",
    "ALLOWED_WEB_DOMAINS",
    "BASE_DIR",
    "DESTRUCTIVE_RE",
    "KnowledgeSync",
    "LearningEngine",
    "MemoryManager",
    "RAGEngine",
    "SAFETY_MODE",
    "_blocked_by_mode",
    "_domain_allowed",
    "_find_balanced_json_end",
    "_parse_delegate",
    "_parse_rag",
    "_parse_tool_call",
    "_run_goose_cli",
    "_safe_resolve_path",
    "_tool_goose_run",
    "execute_tool",
]
