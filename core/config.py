"""AIOS-Local configuration and constants."""

import os
import re
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.resolve()
MEMORY_DIR = BASE_DIR / "memory"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
EPISODES_DIR = MEMORY_DIR / "episodes"
LEARNING_DIR = MEMORY_DIR / "learning"
CONFIG_DIR = BASE_DIR / "config"

for d in [MEMORY_DIR, KNOWLEDGE_DIR, EPISODES_DIR, LEARNING_DIR, CONFIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MEMORY_FILE = MEMORY_DIR / "MEMORY.md"
SOUL_FILE = CONFIG_DIR / "SOUL.md"
LEARNING_FILE = LEARNING_DIR / "patterns.json"
OUTCOMES_FILE = LEARNING_DIR / "outcomes.json"
TELEMETRY_FILE = MEMORY_DIR / "telemetry.jsonl"

# ── External services ───────────────────────────────────────────────────────
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")

# ── Goose CLI ──────────────────────────────────────────────────────────────
GOOSE_EXE = Path(os.getenv("GOOSE_EXE", Path.home() / ".local" / "bin" / "goose.exe"))
GOOSE_TIMEOUT_SEC = int(os.getenv("AIOS_GOOSE_TIMEOUT", "300"))
GOOSE_MAX_TURNS = int(os.getenv("AIOS_GOOSE_MAX_TURNS", "20"))

# ── Runtime Controls ───────────────────────────────────────────────────────
SAFETY_MODE = os.getenv("AIOS_SAFETY_MODE", "trusted_local")
ALLOWED_WEB_DOMAINS = set(filter(None, os.getenv("AIOS_ALLOWED_DOMAINS", "").split(",")))
MAX_DELEGATE_DEPTH = int(os.getenv("AIOS_MAX_DELEGATE_DEPTH", "3"))
MAX_TOOL_TIMEOUT_SEC = int(os.getenv("AIOS_MAX_TOOL_TIMEOUT", "60"))
MAX_SYNC_ERRORS_TRACKED = int(os.getenv("AIOS_MAX_SYNC_ERRORS_TRACKED", "20"))
SWARM_PARALLEL = os.getenv("AIOS_SWARM_PARALLEL", "false").lower() == "true"
LIVE_ADAPT_WINDOW_SEC = int(os.getenv("AIOS_LIVE_ADAPT_WINDOW_SEC", "900"))
MAX_WRITE_FILE_BYTES = int(os.getenv("AIOS_MAX_WRITE_FILE_BYTES", str(256 * 1024)))
HIGH_RISK_TOOL_SET = {"bash", "write_file", "edit_file", "goose_run"}
REQUIRE_HIGH_RISK_APPROVAL = os.getenv("AIOS_REQUIRE_HIGH_RISK_APPROVAL", "").lower() in {"1", "true", "yes", "on"}
BASH_ALLOWED_PREFIXES = tuple(
    p.strip().lower()
    for p in os.getenv("AIOS_BASH_ALLOWED_PREFIXES", "git ,python ,py ,pip ,pytest ,dir ,ls ,type ").split(",")
    if p.strip()
)

# ── Patterns ───────────────────────────────────────────────────────────────
PATTERN_README = "README.md"
PATTERN_MARKDOWN_ALL = "**/*.md*"
PATTERN_TEXT_ALL = "**/*.txt"

# ── Shared constants ───────────────────────────────────────────────────────
_MSG_RAG_UNAVAILABLE = "RAG not available"
_MSG_KSYNC_UNAVAILABLE = "Knowledge sync not available"
_ISO_UTC_SUFFIX = "+00:00"
_COPILOT_INSTRUCTIONS = "copilot-instructions.md"
_YAML_EXTS = {".yaml", ".yml"}
_SENSITIVE_FILENAMES = {
    ".env",
    ".env.local",
    "secrets.json",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}
_HIGH_RISK_REASON = (
    "High-risk tool execution requires explicit approval "
    "(set AIOS_REQUIRE_HIGH_RISK_APPROVAL=false to disable gate)."
)

# ── Agent Definitions ──────────────────────────────────────────────────────
AGENT_ROLES = {
    "coordinator": {
        "name": "Coordinator",
        "emoji": "🧠",
        "description": "Analyzes tasks, delegates to specialists, merges results.",
        "system_suffix": """
You are the Coordinator agent. Your job is to:
1. Analyze the user's request
2. Break it into subtasks
3. Delegate each subtask to the best specialist agent
4. Merge all results into a coherent response
When you need to delegate, output: DELEGATE: {"agent": "agent_name", "task": "description"}
You can delegate to: coder, researcher, writer, analyst, or handle things yourself.
""",
    },
    "coder": {
        "name": "Coder",
        "emoji": "💻",
        "description": "Writes, edits, and debugs code.",
        "system_suffix": """
You are the Coder agent. You write clean, well-documented code.
- Explain approach briefly before code
- Use best practices and defensive checks
- Handle edge cases gracefully
""",
    },
    "researcher": {
        "name": "Researcher",
        "emoji": "🔍",
        "description": "Searches the web and gathers information.",
        "system_suffix": """
You are the Researcher agent.
- Use web_search and crawl_url when needed
- Cross-reference sources
- Distinguish facts from assumptions
""",
    },
    "writer": {
        "name": "Writer",
        "emoji": "✍️",
        "description": "Creates documentation and written content.",
        "system_suffix": """
You are the Writer agent.
- Keep writing clear and structured
- Match user tone and target audience
""",
    },
    "analyst": {
        "name": "Analyst",
        "emoji": "📊",
        "description": "Analyzes data and provides insights.",
        "system_suffix": """
You are the Analyst agent.
- Use evidence and quantify where possible
- Surface conclusions and confidence
""",
    },
    "goose": {
        "name": "Goose",
        "emoji": "🪿",
        "description": "External AI agent (Goose CLI) with its own extensions and tool ecosystem.",
        "system_suffix": """
You are delegating to Goose, an external AI agent running locally via Ollama.
Goose has its own developer tools, file editing, shell access, and extensions.
Use DELEGATE: {"agent": "goose", "task": "description"} for complex coding tasks
that benefit from Goose's autonomous execution loop and built-in extensions.
""",
        "external": True,
    },
}

DEFAULT_SYSTEM = """You are AIOS-Local, a powerful on-device AI assistant.

Capabilities:
- Autonomous execution loop
- Tool use (shell, files, web, browser)
- RAG retrieval and ingestion
- Multi-agent delegation
- Learning and adaptation

Tool usage format:
- TOOL_CALL: {"tool": "name", "args": {"key": "value"}}
- DELEGATE: {"agent": "role", "task": "description"}
- RAG_QUERY: {"query": "search terms", "top_k": 5}
"""

# ── Tool System Constants ──────────────────────────────────────────────────
DESTRUCTIVE_RE = re.compile(
    r"\b(rm\s+-rf|del\s+/[sf]|format\s+|remove-item\s+-recurse|"
    r"stop-computer|restart-computer|net\s+user\s+/delete|"
    r"diskpart|bcdedit|reg\s+delete)\b",
    re.IGNORECASE,
)

TOOLS = {
    "bash": {"description": "Execute a PowerShell command", "args": {"command": "string"}},
    "read_file": {"description": "Read a file", "args": {"path": "string"}},
    "write_file": {
        "description": "Write/create a file",
        "args": {"path": "string", "content": "string"},
    },
    "edit_file": {
        "description": "Replace text in a file",
        "args": {"path": "string", "old_text": "string", "new_text": "string"},
    },
    "list_files": {"description": "List directory contents", "args": {"path": "string"}},
    "web_search": {"description": "Search the web (keyless)", "args": {"query": "string"}},
    "crawl_url": {"description": "Fetch and extract webpage", "args": {"url": "string"}},
    "memory_read": {"description": "Read long-term memory", "args": {}},
    "memory_write": {"description": "Write to long-term memory", "args": {"content": "string"}},
    "rag_query": {
        "description": "Search the knowledge base",
        "args": {"query": "string", "top_k": "integer"},
    },
    "rag_ingest": {
        "description": "Add content to the knowledge base",
        "args": {"content": "string", "source": "string"},
    },
    "knowledge_sync_now": {
        "description": "Run immediate knowledge sync",
        "args": {"mode": "string", "repo": "string"},
    },
    "knowledge_clear_repo": {
        "description": "Remove indexed chunks for one repository",
        "args": {"repo": "string"},
    },
    "knowledge_prune": {
        "description": "Prune stale knowledge chunks",
        "args": {"max_age_hours": "integer"},
    },
    "knowledge_status": {"description": "Get knowledge sync status", "args": {}},
    "knowledge_adapt_live": {
        "description": "Real-time lookup of repo pattern signals and optional adaptive sync execution",
        "args": {"top_n": "integer", "run_sync": "boolean"},
    },
    "plan_task": {"description": "Create a step-by-step plan", "args": {"task": "string"}},
    "delegate": {
        "description": "Delegate a subtask to a specialist agent",
        "args": {"agent": "string", "task": "string"},
    },
    "swarm_execute": {
        "description": "Autonomously split and execute a task across specialist agents",
        "args": {"task": "string"},
    },
    "a2a_send": {
        "description": "Send packet to another agent",
        "args": {"to_agent": "string", "task": "string"},
    },
    "a2a_status": {"description": "Get A2A queue status", "args": {}},
    "browser_open": {"description": "Open page and extract body text", "args": {"url": "string"}},
    "browser_extract": {"description": "Extract body text using browser", "args": {"url": "string"}},
    "goose_run": {
        "description": "Dispatch a task to the Goose AI agent (local, via Ollama). Goose has its own developer extensions and autonomous execution loop.",
        "args": {"task": "string", "with_builtins": "string"},
    },
}

TOOL_REQUIRED_ARGS = {
    "bash": ["command"],
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "edit_file": ["path", "old_text", "new_text"],
    "list_files": [],
    "web_search": ["query"],
    "crawl_url": ["url"],
    "memory_read": [],
    "memory_write": ["content"],
    "rag_query": ["query"],
    "rag_ingest": ["content"],
    "knowledge_sync_now": [],
    "knowledge_clear_repo": ["repo"],
    "knowledge_prune": [],
    "knowledge_status": [],
    "knowledge_adapt_live": [],
    "plan_task": ["task"],
    "delegate": ["agent", "task"],
    "swarm_execute": ["task"],
    "a2a_send": ["to_agent", "task"],
    "a2a_status": [],
    "browser_open": ["url"],
    "browser_extract": ["url"],
    "goose_run": ["task"],
}

TOOLS_SCHEMA = []
for name, spec in TOOLS.items():
    props = {}
    required = TOOL_REQUIRED_ARGS.get(name, [])
    for k, v in spec.get("args", {}).items():
        props[k] = {"type": v}
    TOOLS_SCHEMA.append(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }
    )
