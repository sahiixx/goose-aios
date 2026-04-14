"""
AIOS-Local Agent Engine v3 - Autonomous, RAG, Swarm, Learning
"""

import asyncio
import fnmatch
import hashlib
import importlib
import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from core.parser_utils import extract_prefixed_json
from core.safety import blocked_by_mode, command_policy_block_reason, safe_resolve_path, validate_write_target

try:
    import httpx  # pyright: ignore[reportMissingImports]
except ImportError:
    httpx = None

try:
    import numpy as np  # pyright: ignore[reportMissingImports]
except ImportError:
    np = None

try:
    from bs4 import BeautifulSoup  # pyright: ignore[reportMissingImports]
except ImportError:
    BeautifulSoup = None

try:
    from duckduckgo_search import DDGS  # pyright: ignore[reportMissingImports]
except ImportError:
    DDGS = None

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler  # pyright: ignore[reportMissingImports]
    from apscheduler.triggers.cron import CronTrigger  # pyright: ignore[reportMissingImports]
except ImportError:
    AsyncIOScheduler = None
    CronTrigger = None

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
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


def _telemetry(event: str, **fields):
    payload = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    try:
        with TELEMETRY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return

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


# ═══════════════════════════════════════════════════════════════════════════
#  TOOL SYSTEM
# ═══════════════════════════════════════════════════════════════════════════
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

TOOLS_SCHEMA = []
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


def _domain_allowed(url: str) -> bool:
    if not ALLOWED_WEB_DOMAINS:
        return True
    if httpx is None:
        return False
    try:
        host = httpx.URL(url).host or ""
    except Exception:
        return False
    return any(host == d or host.endswith(f".{d}") for d in ALLOWED_WEB_DOMAINS)


async def crawl_url(url: str, request_timeout_sec: int = 20) -> dict:
    if not _domain_allowed(url):
        return {"url": url, "error": "domain_not_allowed"}
    if httpx is None:
        return {"url": url, "error": "httpx_not_available"}
    try:
        async with httpx.AsyncClient(timeout=request_timeout_sec, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "AIOS-Local/3.0"})
            r.raise_for_status()
        html = r.text
        if BeautifulSoup is None:
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return {"url": url, "title": "", "text": text[:12000]}

        soup = BeautifulSoup(html, "lxml")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return {"url": url, "title": title, "text": text[:12000]}
    except Exception as e:
        return {"url": url, "error": str(e)}


async def web_search(query: str) -> str:
    if not query.strip():
        return "Search error: empty query"
    if DDGS is None:
        return "Search error: duckduckgo-search is not installed"

    try:
        results = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=10):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", ""),
                    }
                )

        if not results:
            return "No results"

        crawled = await asyncio.gather(
            *(crawl_url(item["url"]) for item in results[:5] if item.get("url")),
            return_exceptions=True,
        )

        lines = []
        for i, item in enumerate(results[:8], 1):
            lines.append(f"[{i}] {item['title']}\n{item['snippet']}\n{item['url']}")

        excerpts = []
        for page in crawled:
            if isinstance(page, dict) and page.get("text"):
                title = page.get("title", "")
                excerpts.append(f"{title}\n{page['url']}\n{page['text'][:350]}")

        if excerpts:
            lines.append("\nCrawled Excerpts:\n" + "\n\n".join(excerpts[:3]))
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


async def browser_extract(url: str) -> str:
    if not _domain_allowed(url):
        return "Blocked: domain not allowed"
    try:
        async_playwright = importlib.import_module("playwright.async_api").async_playwright
    except Exception:
        return "Browser error: playwright is not installed"

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            text = await page.inner_text("body")
            await browser.close()
            return text[:12000]
    except Exception as e:
        return f"Browser error: {e}"


class A2ABus:
    def __init__(self):
        self.queues: dict[str, deque] = defaultdict(deque)

    def send(self, to_agent: str, payload: dict):
        self.queues[to_agent].append(payload)

    def receive(self, agent_name: str) -> Optional[dict]:
        if self.queues[agent_name]:
            return self.queues[agent_name].popleft()
        return None

    def status(self) -> dict:
        return {k: len(v) for k, v in self.queues.items()}


class KnowledgeSync:
    REPO_SCAN_LIMIT = 5000
    _GOOSE_PROFILE_PATH = CONFIG_DIR / "goose_profile.json"
    _STATE_PATH = KNOWLEDGE_DIR / ".knowledge_sync_state.json"

    def __init__(self, rag_engine: "RAGEngine"):
        self.rag = rag_engine
        self.goose_profile = self._load_goose_profile()
        self.last_sync = None
        self.sources = [
            "https://news.ycombinator.com/",
            "https://arxiv.org/list/cs.AI/recent",
            "https://openai.com/news/",
        ]
        self.external_repo_configs = self._build_external_repo_configs()
        self.external_docs = self._discover_external_docs()
        self.repo_profiles = {}
        self.repo_watermarks = {}
        self.repo_fingerprints = {}
        self.doc_discovery_cache = {}
        self.sync_running = False
        self.last_error = None
        self.last_duration_sec = 0.0
        self.sync_successes = 0
        self.sync_failures = 0
        self.error_history = []
        self.adaptation_history = []
        self._sync_lock = asyncio.Lock()
        self.scheduler = AsyncIOScheduler() if AsyncIOScheduler else None
        self._load_state()

    @classmethod
    def _load_goose_profile(cls) -> dict:
        try:
            if cls._GOOSE_PROFILE_PATH.exists():
                return json.loads(cls._GOOSE_PROFILE_PATH.read_text("utf-8"))
        except Exception:
            pass
        return {}

    def _load_state(self):
        try:
            if self._STATE_PATH.exists():
                data = json.loads(self._STATE_PATH.read_text("utf-8"))
                self.repo_watermarks = data.get("repo_watermarks", {}) or {}
                self.repo_fingerprints = data.get("repo_fingerprints", {}) or {}
                self.doc_discovery_cache = data.get("doc_discovery_cache", {}) or {}
        except Exception:
            self.repo_watermarks = {}
            self.repo_fingerprints = {}
            self.doc_discovery_cache = {}

    def _save_state(self):
        payload = {
            "repo_watermarks": self.repo_watermarks,
            "repo_fingerprints": self.repo_fingerprints,
            "doc_discovery_cache": self.doc_discovery_cache,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        except Exception:
            return

    def _build_external_repo_configs(self) -> list[dict]:
        external_root = BASE_DIR / ".external"
        if not external_root.exists():
            return []

        overrides = {
            "goose": {"patterns": [PATTERN_README, "documentation/docs/**/*.md*"], "limit": 40},
            "open-saas": {
                "patterns": [PATTERN_README, "CONTRIBUTING.md", PATTERN_MARKDOWN_ALL, PATTERN_TEXT_ALL],
                "limit": 30,
            },
            "agency-agents": {"patterns": [PATTERN_README, "docs/**/*.md*", PATTERN_MARKDOWN_ALL], "limit": 30},
        }

        configs = []
        excluded = set(self.goose_profile.get("excluded_repos", []))
        profile_limits = self.goose_profile.get("repo_limits", {})
        for repo_path in sorted(external_root.iterdir(), key=lambda p: p.name.lower()):
            if not repo_path.is_dir() or repo_path.name.startswith("."):
                continue
            repo_name = repo_path.name
            if repo_name in excluded:
                continue
            base = {
                "patterns": [
                    PATTERN_README,
                    "CONTRIBUTING.md",
                    "AGENTS.md",
                    _COPILOT_INSTRUCTIONS,
                    PATTERN_MARKDOWN_ALL,
                    PATTERN_TEXT_ALL,
                    "**/*.instructions.md",
                    "**/*.prompt.md",
                    "**/*.agent.md",
                    "**/*.json",
                    "**/*.yaml",
                    "**/*.yml",
                    "**/*.toml",
                    "**/*.ini",
                    "**/*.py",
                    "**/*.ts",
                    "**/*.tsx",
                    "**/*.js",
                    "**/*.jsx",
                    "**/*.sh",
                ],
                "exclude_globs": [
                    "**/.git/**",
                    "**/.venv/**",
                    "**/venv/**",
                    "**/node_modules/**",
                    "**/dist/**",
                    "**/build/**",
                    "**/__pycache__/**",
                    "**/.next/**",
                ],
                "limit": 140,
            }
            base.update(overrides.get(repo_name, {}))
            if repo_name in profile_limits:
                base["limit"] = profile_limits[repo_name]
            configs.append(
                {
                    "name": repo_name,
                    "path": repo_path,
                    "patterns": base["patterns"],
                    "exclude_globs": base["exclude_globs"],
                    "limit": base["limit"],
                }
            )

        # Sort by goose_profile repo_priority (lower index = higher priority)
        priority_list = self.goose_profile.get("repo_priority", [])
        if priority_list:
            priority_map = {name: idx for idx, name in enumerate(priority_list)}
            configs.sort(key=lambda c: priority_map.get(c["name"], len(priority_list)))

        return configs

    @staticmethod
    def _is_ignored_path(path: Path, exclude_globs: list[str]) -> bool:
        normalized = path.as_posix()
        return any(fnmatch.fnmatch(normalized, g) for g in exclude_globs)

    @staticmethod
    def _doc_tier(path: Path) -> str:
        name = path.name.lower()
        ext = path.suffix.lower()
        if name in {"agents.md", _COPILOT_INSTRUCTIONS, "readme.md", "contributing.md"}:
            return "high"
        if name.endswith(".instructions.md") or name.endswith(".prompt.md") or name.endswith(".agent.md"):
            return "high"
        if ext in {".json", ".toml", ".ini"} | _YAML_EXTS:
            return "config"
        if ext in {".py", ".ts", ".tsx", ".js", ".jsx", ".sh"}:
            return "code"
        return "other"

    @staticmethod
    def _doc_priority(path: Path) -> tuple:
        tier_weight = {"high": 0, "config": 1, "code": 2, "other": 3}
        tier = KnowledgeSync._doc_tier(path)
        return (tier_weight[tier], len(path.as_posix()), path.as_posix().lower())

    @staticmethod
    def _is_binary_bytes(raw: bytes) -> bool:
        if not raw:
            return False
        if b"\x00" in raw:
            return True
        sample = raw[:4096]
        non_text = sum(1 for b in sample if b < 9 or (13 < b < 32))
        return (non_text / max(len(sample), 1)) > 0.2

    def _read_text_if_safe(self, path: Path, max_bytes: int = 256000) -> Optional[str]:
        try:
            if path.stat().st_size > max_bytes:
                return None
            raw = path.read_bytes()
            if self._is_binary_bytes(raw):
                return None
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return None

    def _build_repo_profile(self, repo_cfg: dict) -> dict:
        repo_path = repo_cfg["path"]
        if not repo_path.exists():
            return self._empty_repo_profile(repo_cfg["name"])

        extension_counts = defaultdict(int)
        markers = set()
        workflows = 0
        files_scanned = 0

        for path in repo_path.rglob("*"):
            if not path.is_file():
                continue
            files_scanned += 1
            if files_scanned > self.REPO_SCAN_LIMIT:
                break

            workflows += self._update_repo_profile_from_file(path, extension_counts, markers)

        return {
            "name": repo_cfg["name"],
            "present": True,
            "files_scanned": files_scanned,
            "extension_counts": dict(sorted(extension_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]),
            "markers": sorted(markers),
            "workflows": workflows,
        }

    @staticmethod
    def _empty_repo_profile(repo_name: str) -> dict:
        return {
            "name": repo_name,
            "present": False,
            "files_scanned": 0,
            "extension_counts": {},
            "markers": [],
            "workflows": 0,
        }

    @staticmethod
    def _file_markers() -> dict:
        return {
            "agents.md": "agent_manifest",
            _COPILOT_INSTRUCTIONS: "copilot_instructions",
            "azure.yaml": "azure_project",
            "dockerfile": "containerization",
            "docker-compose.yml": "compose",
            "docker-compose.yaml": "compose",
        }

    def _update_repo_profile_from_file(self, path: Path, extension_counts: defaultdict, markers: set) -> int:
        name_lower = path.name.lower()
        ext = path.suffix.lower() or "(no_ext)"
        extension_counts[ext] += 1

        mapped = self._file_markers().get(name_lower)
        if mapped:
            markers.add(mapped)

        workflows = 0
        path_posix = path.as_posix().lower()
        if "/.github/workflows/" in path_posix and ext in _YAML_EXTS:
            workflows = 1
            markers.add("github_actions")

        if "prompt" in name_lower or name_lower.endswith(".instructions.md"):
            markers.add("prompt_assets")
        if ext in {".tf", ".bicep"}:
            markers.add("iac")
        if ext in {".py", ".ts", ".tsx", ".js", ".jsx"}:
            markers.add("source_code")
        return workflows

    def _profile_to_text(self, profile: dict) -> str:
        return json.dumps(profile, ensure_ascii=False, indent=2)

    def _prepare_ingest_text(self, path: Path, text: str) -> str:
        ext = path.suffix.lower()
        if ext in {".py", ".ts", ".tsx", ".js", ".jsx", ".sh"}:
            return f"Code pattern from {path.name}:\n\n{text[:8000]}"
        if ext in {".json", ".toml", ".ini"} | _YAML_EXTS:
            return f"Configuration pattern from {path.name}:\n\n{text[:10000]}"
        return text[:12000]

    @staticmethod
    def _metadata_for_doc(repo_name: str, rel_path: str, path: Path, profile: dict) -> dict:
        return {
            "repo": repo_name,
            "rel_path": rel_path,
            "file_type": path.suffix.lower() or "(no_ext)",
            "tier": KnowledgeSync._doc_tier(path),
            "file_size": path.stat().st_size if path.exists() else 0,
            "markers": profile.get("markers", []),
            "ingest_kind": "repo_doc",
        }

    def _select_docs_by_tier(self, docs: list[Path], limit: int) -> list[Path]:
        tier_limits = {
            "high": max(int(limit * 0.45), 1),
            "config": max(int(limit * 0.2), 1),
            "code": max(int(limit * 0.25), 1),
            "other": max(int(limit * 0.1), 1),
        }
        by_tier: dict[str, list[Path]] = {"high": [], "config": [], "code": [], "other": []}
        for d in docs:
            by_tier[self._doc_tier(d)].append(d)

        selected: list[Path] = []
        for tier in ("high", "config", "code", "other"):
            selected.extend(by_tier[tier][: tier_limits[tier]])

        if len(selected) < limit:
            extras = [d for d in docs if d not in selected]
            selected.extend(extras[: limit - len(selected)])
        return selected[:limit]

    def _discover_repo_docs(self, repo_cfg: dict) -> list[Path]:
        repo_path = repo_cfg["path"]
        if not repo_path.exists():
            return []
        repo_name = repo_cfg["name"]
        try:
            repo_mtime = repo_path.stat().st_mtime
        except Exception:
            repo_mtime = 0.0
        cache_key = f"{repo_name}:{repo_mtime:.3f}:{repo_cfg.get('limit', 20)}"
        cached = self.doc_discovery_cache.get(cache_key)
        if cached:
            docs = [Path(p) for p in cached if Path(p).exists()]
            if docs:
                return docs
        seen: set[str] = set()
        docs: list[Path] = []
        exclude_globs = repo_cfg.get("exclude_globs", [])
        for pattern in repo_cfg.get("patterns", [PATTERN_MARKDOWN_ALL]):
            for path in repo_path.glob(pattern):
                if not path.is_file() or self._is_ignored_path(path, exclude_globs):
                    continue
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    docs.append(path)
        docs.sort(key=self._doc_priority)
        selected = self._select_docs_by_tier(docs, repo_cfg.get("limit", 20))
        self.doc_discovery_cache = {k: v for k, v in self.doc_discovery_cache.items() if k.startswith(f"{repo_name}:")}
        self.doc_discovery_cache[cache_key] = [str(p) for p in selected]
        return selected

    def _repo_fingerprint(self, repo_cfg: dict) -> str:
        repo_path = repo_cfg["path"]
        if not repo_path.exists():
            return "missing"
        try:
            stat = repo_path.stat()
        except Exception:
            return "unknown"
        docs = self.external_docs.get(repo_cfg["name"], [])
        docs_sig = "|".join(f"{p.name}:{int(p.stat().st_mtime)}" for p in docs if p.exists())
        return hashlib.md5(
            f"{repo_cfg['name']}|{int(stat.st_mtime)}|{len(docs)}|{docs_sig[:4000]}".encode("utf-8")
        ).hexdigest()

    def _discover_external_docs(self) -> dict[str, list[Path]]:
        return {cfg["name"]: self._discover_repo_docs(cfg) for cfg in self.external_repo_configs}

    def _recent_doc_changes(self, repo_docs: list[Path], window_sec: int = LIVE_ADAPT_WINDOW_SEC) -> list[str]:
        cutoff = time.time() - max(window_sec, 1)
        recent = []
        for p in repo_docs:
            try:
                if p.stat().st_mtime >= cutoff:
                    recent.append(p.name)
            except Exception:
                continue
        return recent[:20]

    def lookup_live_adaptation(self, top_n: int = 3) -> dict:
        self.external_repo_configs = self._build_external_repo_configs()
        self.external_docs = self._discover_external_docs()

        candidates = []
        for cfg in self.external_repo_configs:
            repo_name = cfg["name"]
            docs = self.external_docs.get(repo_name, [])
            profile = self._build_repo_profile(cfg)
            self.repo_profiles[repo_name] = profile

            indexed_docs = len(docs)
            watermarked_docs = sum(1 for k in self.repo_watermarks if k.startswith(f"{repo_name}:"))
            pending_docs = max(indexed_docs - watermarked_docs, 0)
            recent_files = self._recent_doc_changes(docs)

            marker_score = len(profile.get("markers", [])) * 1.2
            workflow_score = profile.get("workflows", 0) * 2.0
            freshness_score = len(recent_files) * 3.0
            pending_score = pending_docs * 1.5
            score = round(marker_score + workflow_score + freshness_score + pending_score, 2)

            reasons = []
            if pending_docs:
                reasons.append(f"{pending_docs} pending docs")
            if recent_files:
                reasons.append(f"{len(recent_files)} recent changes")
            if profile.get("markers"):
                reasons.append(f"markers={','.join(profile['markers'][:4])}")
            if profile.get("workflows", 0):
                reasons.append(f"workflows={profile['workflows']}")

            candidates.append(
                {
                    "repo": repo_name,
                    "score": score,
                    "indexed_docs": indexed_docs,
                    "watermarked_docs": watermarked_docs,
                    "pending_docs": pending_docs,
                    "recent_files": recent_files,
                    "markers": profile.get("markers", []),
                    "workflows": profile.get("workflows", 0),
                    "reasons": reasons,
                }
            )

        candidates.sort(key=lambda x: (-x["score"], x["repo"]))
        selected = [c["repo"] for c in candidates[: max(top_n, 1)] if c["score"] > 0]
        if not selected:
            selected = [c["repo"] for c in candidates[: max(top_n, 1)]]

        snapshot = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "top_n": max(top_n, 1),
            "selected": selected,
            "candidates": candidates,
        }
        self.adaptation_history.append(snapshot)
        self.adaptation_history = self.adaptation_history[-50:]
        return snapshot

    async def adapt_live(self, top_n: int = 3) -> dict:
        lookup = self.lookup_live_adaptation(top_n=top_n)
        runs = []
        for repo_name in lookup.get("selected", []):
            runs.append(await self.sync_once(mode="incremental", repo=repo_name))
        return {
            "status": "ok",
            "lookup": lookup,
            "sync_runs": runs,
            "last_sync": self.last_sync,
        }

    async def _sync_web_sources(self, mode: str):
        if not self.goose_profile.get("enable_web_sync", True):
            return
        if mode == "full":
            self.rag.delete_source_prefix("sync:")
        for url in self.sources:
            page = await crawl_url(url)
            if page.get("text"):
                self.rag.ingest(
                    page["text"],
                    source=f"sync:{url}",
                    metadata={"ingest_kind": "web_sync", "url": url, "provenance": "external_web", "trust": "untrusted"},
                    save=False,
                )

    def _sync_repo_docs(self, cfg: dict, mode: str) -> int:
        repo_name = cfg["name"]
        repo_root = cfg["path"]
        if mode == "full":
            self.rag.delete_source_prefix(f"{repo_name}:")

        repo_fp = self._repo_fingerprint(cfg)
        previous_fp = self.repo_fingerprints.get(repo_name)
        if mode != "full" and previous_fp and previous_fp == repo_fp:
            return 0

        profile = self._build_repo_profile(cfg)
        self.repo_profiles[repo_name] = profile
        self.rag.ingest(
            self._profile_to_text(profile),
            source=f"{repo_name}:_repo_profile",
            metadata={"ingest_kind": "repo_profile", "repo": repo_name, "provenance": "external_repo", "trust": "medium"},
            save=False,
        )

        ingested = 0
        for doc in self.external_docs.get(repo_name, []):
            try:
                rel = doc.relative_to(repo_root)
                rel_posix = rel.as_posix()
                source_key = f"{repo_name}:{rel_posix}"
                mtime = doc.stat().st_mtime
                if mode != "full" and self.repo_watermarks.get(source_key, 0) >= mtime:
                    continue
                text = self._read_text_if_safe(doc)
                if text and text.strip():
                    prepared = self._prepare_ingest_text(doc, text)
                    self.rag.ingest(
                        prepared,
                        source=source_key,
                        metadata={
                            **self._metadata_for_doc(repo_name, rel_posix, doc, profile),
                            "provenance": "external_repo",
                            "trust": "medium",
                        },
                        save=False,
                    )
                    self.repo_watermarks[source_key] = mtime
                    ingested += 1
            except Exception:
                continue
        self.repo_fingerprints[repo_name] = repo_fp
        return ingested

    async def sync_once(self, mode: str = "incremental", repo: Optional[str] = None) -> dict:
        if self.sync_running:
            return {"status": "busy", "last_sync": self.last_sync}

        async with self._sync_lock:
            self.sync_running = True
            start = time.time()
            self.last_error = None
            ingested_docs = 0
            touched_repos: list[str] = []

            try:
                self.external_repo_configs = self._build_external_repo_configs()
                self.external_docs = self._discover_external_docs()
                self.repo_profiles = {}

                await self._sync_web_sources(mode)

                targets = self.external_repo_configs
                if repo:
                    targets = [cfg for cfg in targets if cfg["name"] == repo]

                for cfg in targets:
                    touched_repos.append(cfg["name"])
                    ingested_docs += self._sync_repo_docs(cfg, mode)
                self.rag.save()
                self._save_state()

                self.last_sync = datetime.now(timezone.utc).isoformat()
                self.sync_successes += 1
                self.last_duration_sec = round(time.time() - start, 3)
                _telemetry(
                    "knowledge_sync",
                    status="ok",
                    mode=mode,
                    repo=repo,
                    touched_repos=touched_repos,
                    ingested_docs=ingested_docs,
                    duration_sec=self.last_duration_sec,
                )
                return {
                    "status": "ok",
                    "mode": mode,
                    "repo": repo,
                    "ingested_docs": ingested_docs,
                    "touched_repos": touched_repos,
                    "duration_sec": self.last_duration_sec,
                    "last_sync": self.last_sync,
                }
            except Exception as e:
                self.last_error = str(e)
                self.sync_failures += 1
                self.last_duration_sec = round(time.time() - start, 3)
                self.error_history.append(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "error": self.last_error,
                    }
                )
                self.error_history = self.error_history[-MAX_SYNC_ERRORS_TRACKED:]
                _telemetry(
                    "knowledge_sync",
                    status="error",
                    mode=mode,
                    repo=repo,
                    error=self.last_error,
                    duration_sec=self.last_duration_sec,
                )
                return {
                    "status": "error",
                    "mode": mode,
                    "repo": repo,
                    "error": self.last_error,
                    "duration_sec": self.last_duration_sec,
                    "last_sync": self.last_sync,
                }
            finally:
                self.sync_running = False

    def clear_repo_index(self, repo_name: str) -> dict:
        removed = self.rag.delete_source_prefix(f"{repo_name}:")
        for k in [k for k in self.repo_watermarks if k.startswith(f"{repo_name}:")]:
            del self.repo_watermarks[k]
        self.repo_fingerprints.pop(repo_name, None)
        self._save_state()
        return {"repo": repo_name, "removed_chunks": removed}

    def start(self):
        if not self.scheduler or self.scheduler.running:
            return
        self.scheduler.add_job(
            lambda: asyncio.create_task(self.sync_once()),
            "interval",
            minutes=5,
            id="knowledge_sync_5m",
            replace_existing=True,
        )
        if CronTrigger:
            self.scheduler.add_job(
                lambda: asyncio.create_task(self.sync_once()),
                CronTrigger(hour="*/1"),
                id="knowledge_sync_hourly",
                replace_existing=True,
            )
        self.scheduler.start()

    def status(self) -> dict:
        jobs = []
        if self.scheduler:
            jobs = [j.id for j in self.scheduler.get_jobs()]
        repo_status = []
        for cfg in self.external_repo_configs:
            repo_status.append(
                {
                    "name": cfg["name"],
                    "present": cfg["path"].exists(),
                    "indexed_docs": len(self.external_docs.get(cfg["name"], [])),
                    "markers": self.repo_profiles.get(cfg["name"], {}).get("markers", []),
                    "workflows": self.repo_profiles.get(cfg["name"], {}).get("workflows", 0),
                    "fingerprint_known": cfg["name"] in self.repo_fingerprints,
                }
            )
        return {
            "last_sync": self.last_sync,
            "sync_running": self.sync_running,
            "last_error": self.last_error,
            "last_duration_sec": self.last_duration_sec,
            "sync_successes": self.sync_successes,
            "sync_failures": self.sync_failures,
            "source_count": len(self.sources),
            "external_repos": repo_status,
            "jobs": jobs,
        }


async def _tool_bash(args: dict, _agent: "Agent" = None) -> str:
    cmd = args.get("command", "")
    block_reason = command_policy_block_reason(cmd, DESTRUCTIVE_RE, BASH_ALLOWED_PREFIXES)
    if block_reason:
        return f"⚠️ BLOCKED: {block_reason}: `{cmd}`"
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=MAX_TOOL_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        return f"⏱️ Timed out ({MAX_TOOL_TIMEOUT_SEC}s) — process killed"
    result = (out.decode("utf-8", errors="replace").strip() + "\n" + err.decode("utf-8", errors="replace").strip()).strip()
    return result[:8000] or "(no output)"


def _tool_read_file(args: dict, _agent: "Agent" = None) -> str:
    p = safe_resolve_path(args.get("path", ""), BASE_DIR)
    if not p.exists():
        return f"❌ Not found: {p}"
    return p.read_text("utf-8", errors="replace")[:8000]


def _tool_write_file(args: dict, _agent: "Agent" = None) -> str:
    p = safe_resolve_path(args.get("path", ""), BASE_DIR)
    content = args.get("content", "")
    if len(content.encode("utf-8")) > MAX_WRITE_FILE_BYTES:
        return f"❌ Content too large ({MAX_WRITE_FILE_BYTES} bytes max)"
    path_warning = validate_write_target(p, _SENSITIVE_FILENAMES)
    if path_warning:
        return f"⚠️ BLOCKED: {path_warning}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, "utf-8")
    return f"✅ Written {len(content)} chars -> {p}"


def _tool_edit_file(args: dict, _agent: "Agent" = None) -> str:
    p = safe_resolve_path(args.get("path", ""), BASE_DIR)
    if not p.exists():
        return f"❌ Not found: {p}"
    path_warning = validate_write_target(p, _SENSITIVE_FILENAMES)
    if path_warning:
        return f"⚠️ BLOCKED: {path_warning}"
    old = args.get("old_text", "")
    new = args.get("new_text", "")
    content = p.read_text("utf-8", errors="replace")
    if old not in content:
        return "❌ Text not found in file"
    p.write_text(content.replace(old, new, 1), "utf-8")
    return f"✅ Edited {p}"


def _tool_list_files(args: dict, _agent: "Agent" = None) -> str:
    p = safe_resolve_path(args.get("path", "."), BASE_DIR)
    if not p.exists():
        return f"❌ Not found: {p}"
    if p.is_file():
        return f"📄 {p.name} ({p.stat().st_size:,} bytes)"
    items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))[:80]
    lines = []
    for item in items:
        pre = "📁" if item.is_dir() else "📄"
        size = f" ({item.stat().st_size:,})" if item.is_file() else ""
        lines.append(f"  {pre} {item.name}{size}")
    return "\n".join(lines) or "(empty)"


async def _tool_web_search(args: dict, agent: "Agent" = None) -> str:
    query = args.get("query", "")
    result = await web_search(query)
    if agent and agent.rag and result and not result.startswith("Search error"):
        agent.rag.ingest(
            result,
            source=f"web_search:{query[:50]}",
            metadata={"ingest_kind": "web_search", "provenance": "external_web", "trust": "untrusted"},
        )
    return result


async def _tool_crawl_url(args: dict, _agent: "Agent" = None) -> str:
    page = await crawl_url(args.get("url", ""))
    if page.get("text"):
        return f"{page.get('title', '')}\n{page['url']}\n\n{page['text'][:4000]}"
    return f"Crawl failed: {page.get('error', 'unknown')}"


def _tool_memory_read(args: dict, _agent: "Agent" = None) -> str:
    return MEMORY_FILE.read_text("utf-8") if MEMORY_FILE.exists() else "(empty)"


_MAX_MEMORY_FILE_BYTES = 512 * 1024  # 512 KB cap


def _tool_memory_write(args: dict, _agent: "Agent" = None) -> str:
    content = args.get("content", "")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    existing = MEMORY_FILE.read_text("utf-8") if MEMORY_FILE.exists() else ""
    if len(existing.encode("utf-8")) > _MAX_MEMORY_FILE_BYTES:
        return "⚠️ Memory file size limit reached. Please prune old entries first."
    new_blob = f"{existing}\n## {ts}\n{content}\n"
    if len(new_blob.encode("utf-8")) > _MAX_MEMORY_FILE_BYTES:
        return "⚠️ Memory append rejected: resulting file would exceed size limit."
    MEMORY_FILE.write_text(new_blob, "utf-8")
    return "✅ Memory updated"


def _tool_rag_query(args: dict, agent: "Agent" = None) -> str:
    if not (agent and agent.rag):
        return _MSG_RAG_UNAVAILABLE
    top_k = min(int(args.get("top_k", 5)), 20)
    results = agent.rag.search_with_citations(args.get("query", ""), top_k=top_k)
    if not results:
        return "No relevant documents found."
    lines = []
    for i, item in enumerate(results, 1):
        lines.append(
            f"[{i}] Score: {item['score']:.2f} | Source: {item['source']}\n"
            f"Citation: {item['citation']}\n"
            f"{item['text'][:500]}"
        )
    return "\n\n---\n\n".join(lines)


def _tool_rag_ingest(args: dict, agent: "Agent" = None) -> str:
    if not (agent and agent.rag):
        return _MSG_RAG_UNAVAILABLE
    agent.rag.ingest(args.get("content", ""), source=args.get("source", "user_input"))
    return "✅ Added to knowledge base"


async def _tool_knowledge_sync_now(args: dict, agent: "Agent" = None) -> str:
    if not (agent and agent.knowledge_sync):
        return _MSG_KSYNC_UNAVAILABLE
    mode = args.get("mode", "incremental")
    repo = args.get("repo")
    result = await agent.knowledge_sync.sync_once(mode=mode, repo=repo)
    return "✅ Knowledge sync completed\n" + json.dumps(result, indent=2)


def _tool_knowledge_status(args: dict, agent: "Agent" = None) -> str:
    if not (agent and agent.knowledge_sync):
        return _MSG_KSYNC_UNAVAILABLE
    return json.dumps(agent.knowledge_sync.status(), indent=2)


async def _tool_knowledge_adapt_live(args: dict, agent: "Agent" = None) -> str:
    if not (agent and agent.knowledge_sync):
        return _MSG_KSYNC_UNAVAILABLE
    top_n = max(int(args.get("top_n", 3)), 1)
    run_sync = args.get("run_sync", True)
    if isinstance(run_sync, str):
        run_sync = run_sync.strip().lower() not in {"0", "false", "no", "off"}

    if run_sync:
        result = await agent.knowledge_sync.adapt_live(top_n=top_n)
    else:
        result = agent.knowledge_sync.lookup_live_adaptation(top_n=top_n)
    return json.dumps(result, indent=2)


def _tool_knowledge_clear_repo(args: dict, agent: "Agent" = None) -> str:
    if not (agent and agent.knowledge_sync):
        return _MSG_KSYNC_UNAVAILABLE
    repo = args.get("repo", "")
    if not repo:
        return "Missing required arg: repo"
    return json.dumps(agent.knowledge_sync.clear_repo_index(repo), indent=2)


def _tool_knowledge_prune(args: dict, agent: "Agent" = None) -> str:
    if not (agent and agent.rag):
        return _MSG_RAG_UNAVAILABLE
    max_age_hours = int(args.get("max_age_hours", 24 * 30))
    removed = agent.rag.prune_stale(max_age_hours=max_age_hours)
    return json.dumps({"removed_chunks": removed, "rag": agent.rag.stats()}, indent=2)


def _tool_plan_task(args: dict, _agent: "Agent" = None) -> str:
    return f"📋 Planning: {args.get('task', '')}\n→ Decomposing into steps..."


def _tool_delegate(args: dict, _agent: "Agent" = None) -> str:
    return f"🔀 Delegating to {args.get('agent', 'coder')}: {args.get('task', '')}"


async def _tool_swarm_execute(args: dict, agent: "Agent" = None) -> str:
    if not agent:
        return "Swarm executor not available"
    task = args.get("task", "").strip()
    if not task:
        return "Missing required arg: task"
    return await agent.run_swarm(task)


def _tool_a2a_send(args: dict, agent: "Agent" = None) -> str:
    if not (agent and agent.a2a):
        return "A2A not available"
    agent.a2a.send(
        args.get("to_agent", "coordinator"),
        {"task": args.get("task", ""), "ts": datetime.now(timezone.utc).isoformat()},
    )
    return "✅ A2A packet sent"


def _tool_a2a_status(args: dict, agent: "Agent" = None) -> str:
    if not (agent and agent.a2a):
        return "A2A not available"
    return json.dumps(agent.a2a.status(), indent=2)


async def _tool_browser_extract(args: dict, _agent: "Agent" = None) -> str:
    return await browser_extract(args.get("url", ""))


# ── Goose CLI integration ─────────────────────────────────────────────────
async def _run_goose_cli(
    task: str,
    with_builtins: str = "developer",
    timeout_sec: int | None = None,
    max_turns: int | None = None,
) -> dict:
    """Run a task through the Goose CLI and return parsed results."""
    if not GOOSE_EXE.exists():
        return {"ok": False, "error": f"Goose not found at {GOOSE_EXE}"}

    timeout_sec = timeout_sec or GOOSE_TIMEOUT_SEC
    max_turns = max_turns or GOOSE_MAX_TURNS

    cmd = [
        str(GOOSE_EXE), "run",
        "--no-session",
        "--output-format", "json",
        "--quiet",
        "--provider", "ollama",
        "--model", "qwen2.5-coder:7b",
        "--max-turns", str(max_turns),
        "-t", task,
    ]
    if with_builtins:
        cmd.extend(["--with-builtin", with_builtins])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(BASE_DIR),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)

        raw_out = stdout.decode("utf-8", errors="replace").strip()
        raw_err = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            return {"ok": False, "error": raw_err or f"Goose exited with code {proc.returncode}"}

        # Parse JSON output
        try:
            data = json.loads(raw_out)
            # Extract the last assistant message
            messages = data.get("messages", [])
            response_parts = []
            for msg in messages:
                if msg.get("role") == "assistant":
                    for part in msg.get("content", []):
                        if isinstance(part, dict) and part.get("type") == "text":
                            response_parts.append(part["text"])
                        elif isinstance(part, str):
                            response_parts.append(part)
            response_text = "\n".join(response_parts) if response_parts else raw_out
            return {
                "ok": True,
                "response": response_text[:8000],
                "metadata": data.get("metadata", {}),
            }
        except json.JSONDecodeError:
            # Fallback to raw text
            return {"ok": True, "response": raw_out[:8000]}

    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return {"ok": False, "error": f"Goose timed out after {timeout_sec}s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _tool_goose_run(args: dict, _agent: "Agent" = None) -> str:
    task = args.get("task", "").strip()
    if not task:
        return "❌ Missing required arg: task"
    builtins = args.get("with_builtins", "developer")
    result = await _run_goose_cli(task, with_builtins=builtins)
    if not result["ok"]:
        return f"❌ Goose error: {result['error']}"
    meta = result.get("metadata", {})
    prefix = "🪿 Goose completed"
    if meta.get("total_tokens"):
        prefix += f" ({meta['total_tokens']} tokens)"
    return f"{prefix}:\n{result['response']}"


TOOL_HANDLERS = {
    "bash": _tool_bash,
    "read_file": _tool_read_file,
    "write_file": _tool_write_file,
    "edit_file": _tool_edit_file,
    "list_files": _tool_list_files,
    "web_search": _tool_web_search,
    "crawl_url": _tool_crawl_url,
    "memory_read": _tool_memory_read,
    "memory_write": _tool_memory_write,
    "rag_query": _tool_rag_query,
    "rag_ingest": _tool_rag_ingest,
    "knowledge_sync_now": _tool_knowledge_sync_now,
    "knowledge_clear_repo": _tool_knowledge_clear_repo,
    "knowledge_prune": _tool_knowledge_prune,
    "knowledge_status": _tool_knowledge_status,
    "knowledge_adapt_live": _tool_knowledge_adapt_live,
    "plan_task": _tool_plan_task,
    "delegate": _tool_delegate,
    "swarm_execute": _tool_swarm_execute,
    "a2a_send": _tool_a2a_send,
    "a2a_status": _tool_a2a_status,
    "browser_open": _tool_browser_extract,
    "browser_extract": _tool_browser_extract,
    "goose_run": _tool_goose_run,
}


async def execute_tool(name: str, args: dict, agent: "Agent" = None) -> str:
    try:
        blocked = blocked_by_mode(
            name,
            SAFETY_MODE,
            HIGH_RISK_TOOL_SET,
            REQUIRE_HIGH_RISK_APPROVAL,
            _HIGH_RISK_REASON,
        )
        if blocked:
            return f"⚠️ {blocked}"
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            return f"❌ Unknown tool: {name}"
        result = handler(args, agent)
        if asyncio.iscoroutine(result):
            return await result
        return result
    except asyncio.TimeoutError:
        return f"⏱️ Timed out ({MAX_TOOL_TIMEOUT_SEC}s)"
    except PermissionError as e:
        return f"⚠️ BLOCKED: {e}"
    except Exception as e:
        return f"❌ Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════
#  RAG ENGINE — Retrieval-Augmented Generation
# ═══════════════════════════════════════════════════════════════════════════
class RAGEngine:
    """Local RAG with TF-IDF vectorization and cosine similarity."""

    _tfidf_cls = None
    _cosine_fn = None

    @classmethod
    def _load_sklearn(cls):
        if cls._tfidf_cls is None:
            cls._tfidf_cls = importlib.import_module("sklearn.feature_extraction.text").TfidfVectorizer
            cls._cosine_fn = importlib.import_module("sklearn.metrics.pairwise").cosine_similarity

    def __init__(self):
        self.chunks: list[dict] = []
        self.chunk_ids: set[str] = set()
        self._index_path = KNOWLEDGE_DIR / ".rag_index.json"
        self._vectorizer = None
        self._doc_matrix = None
        self._index_dirty = True
        self._load()
        self._index_dirty = True

    def ingest(
        self,
        content: str,
        source: str = "unknown",
        chunk_size: int = 800,
        overlap: int = 100,
        metadata: Optional[dict] = None,
        save: bool = True,
    ):
        chunks = self._chunk_text(content, chunk_size, overlap)
        changed = False
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
            if chunk_id in self.chunk_ids:
                continue
            self.chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "source": source,
                    "metadata": metadata or {},
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self.chunk_ids.add(chunk_id)
            changed = True
        if changed:
            self._index_dirty = True
        if save:
            self._save()

    def ingest_file(self, path: str):
        p = Path(path)
        if not p.exists():
            return False
        text = p.read_text("utf-8", errors="replace")
        self.ingest(text, source=p.name)
        return True

    def _ensure_index(self) -> bool:
        if not self.chunks or np is None:
            self._vectorizer = None
            self._doc_matrix = None
            return False
        try:
            self._load_sklearn()
        except Exception:
            return False
        if not self._index_dirty and self._vectorizer is not None and self._doc_matrix is not None:
            return True
        texts = [c["text"] for c in self.chunks]
        self._vectorizer = self._tfidf_cls(max_features=5000, stop_words="english", ngram_range=(1, 2))
        try:
            self._doc_matrix = self._vectorizer.fit_transform(texts)
        except ValueError:
            self._vectorizer = None
            self._doc_matrix = None
            return False
        self._index_dirty = False
        return True

    def _ranked_search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Return list of (chunk_index, adjusted_score) ranked by relevance."""
        if not query.strip() or not self._ensure_index():
            return []
        try:
            query_matrix = self._vectorizer.transform([query])
        except ValueError:
            return []
        scores = RAGEngine._cosine_fn(query_matrix, self._doc_matrix)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        now = datetime.now(timezone.utc)
        ranked = []
        for idx in top_indices:
            if scores[idx] <= 0.05:
                continue
            c = self.chunks[idx]
            freshness_bonus = 0.0
            ts = c.get("ingested_at")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", _ISO_UTC_SUFFIX))
                    hours = max((now - dt).total_seconds() / 3600.0, 0.0)
                    freshness_bonus = max(0.0, 0.15 - min(hours / 240.0, 0.15))
                except Exception:
                    pass
            ranked.append((int(idx), float(scores[idx]) + freshness_bonus))
        return ranked

    def search(self, query: str, top_k: int = 5) -> list:
        results = []
        for idx, score in self._ranked_search(query, top_k):
            c = self.chunks[idx]
            results.append((c["text"], score, c["source"]))
        return results

    def search_with_citations(self, query: str, top_k: int = 5) -> list[dict]:
        results = []
        for idx, score in self._ranked_search(query, top_k):
            c = self.chunks[idx]
            results.append(
                {
                    "text": c["text"],
                    "score": score,
                    "source": c["source"],
                    "citation": self._citation_for_chunk(c),
                    "metadata": c.get("metadata", {}),
                }
            )
        return results

    @staticmethod
    def _citation_for_chunk(chunk: dict) -> str:
        md = chunk.get("metadata", {}) or {}
        repo = md.get("repo")
        rel = md.get("rel_path")
        if repo and rel:
            return f"{repo}/{rel}"
        if repo:
            return repo
        return chunk.get("source", "unknown")

    def list_sources(self) -> list:
        return list({c["source"] for c in self.chunks})

    def delete_source(self, source: str):
        self.chunks = [c for c in self.chunks if c["source"] != source]
        self.chunk_ids = {c["id"] for c in self.chunks}
        self._index_dirty = True
        self._save()

    def delete_source_prefix(self, source_prefix: str) -> int:
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if not c["source"].startswith(source_prefix)]
        self.chunk_ids = {c["id"] for c in self.chunks}
        self._index_dirty = True
        self._save()
        return before - len(self.chunks)

    def prune_stale(self, max_age_hours: int = 24 * 30) -> int:
        now = datetime.now(timezone.utc)
        kept = []
        removed = 0
        for c in self.chunks:
            ts = c.get("ingested_at")
            if not ts:
                kept.append(c)
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", _ISO_UTC_SUFFIX))
                age_hours = (now - dt).total_seconds() / 3600.0
                if age_hours > max_age_hours:
                    removed += 1
                else:
                    kept.append(c)
            except Exception:
                kept.append(c)
        self.chunks = kept
        self.chunk_ids = {c["id"] for c in self.chunks}
        self._index_dirty = True
        self._save()
        return removed

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "sources": len(self.list_sources()),
            "latest_ingested_at": max((c.get("ingested_at") for c in self.chunks), default=None),
        }

    def _chunk_text(self, text: str, size: int, overlap: int) -> list:
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > size and current:
                chunks.append(current.strip())
                words = current.split()[-overlap // 4 :] if overlap else []
                current = " ".join(words) + "\n\n" + para
            else:
                current += "\n\n" + para if current else para
        if current.strip():
            chunks.append(current.strip())
        return [c for c in chunks if len(c) > 20]

    def _save(self):
        self._index_path.write_text(json.dumps({"chunks": self.chunks}, ensure_ascii=False, indent=2), "utf-8")

    def save(self):
        self._save()

    def _load(self):
        if self._index_path.exists():
            try:
                self.chunks = json.loads(self._index_path.read_text("utf-8")).get("chunks", [])
                self.chunk_ids = {c.get("id") for c in self.chunks if c.get("id")}
            except Exception:
                self.chunks = []
                self.chunk_ids = set()
        self._index_dirty = True


# ═══════════════════════════════════════════════════════════════════════════
#  MEMORY MANAGER — Three-Layer Memory
# ═══════════════════════════════════════════════════════════════════════════
class MemoryManager:
    def __init__(self):
        self.working = []
        self.episodes_path = EPISODES_DIR

    def add_working(self, role: str, content: str):
        self.working.append({"role": role, "content": content, "ts": datetime.now(timezone.utc).isoformat()})
        if len(self.working) > 100:
            self.working = self.working[-100:]

    def get_working_context(self, max_chars: int = 8000) -> str:
        recent = self.working[-20:]
        lines = [f"{m['role']}: {m['content'][:500]}" for m in recent]
        text = "\n".join(lines)
        return text[-max_chars:] if len(text) > max_chars else text

    def save_episode(self, task: str, outcome: str, tools_used: list, agent_role: str = "main"):
        episode = {
            "id": hashlib.md5(f"{task}{time.time()}".encode()).hexdigest()[:12],
            "task": task,
            "outcome": outcome,
            "tools_used": tools_used,
            "agent": agent_role,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_count": len(self.working),
        }
        path = self.episodes_path / f"{episode['id']}.json"
        path.write_text(json.dumps(episode, ensure_ascii=False, indent=2), "utf-8")
        self._prune_episodes(max_keep=200)
        return episode

    def _prune_episodes(self, max_keep: int = 200):
        files = sorted(self.episodes_path.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if len(files) <= max_keep:
            return
        for f in files[: len(files) - max_keep]:
            try:
                f.unlink()
            except Exception:
                pass

    def get_relevant_episodes(self, query: str, limit: int = 3) -> list:
        episodes = []
        for f in self.episodes_path.glob("*.json"):
            try:
                episodes.append(json.loads(f.read_text("utf-8")))
            except Exception:
                pass
        if not episodes:
            return []
        query_words = set(query.lower().split())
        scored = []
        for ep in episodes:
            task_words = set(ep.get("task", "").lower().split())
            overlap = len(query_words & task_words) / max(len(query_words), 1)
            scored.append((ep, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, s in scored[:limit] if s > 0.1]

    def read_long_term(self) -> str:
        return MEMORY_FILE.read_text("utf-8") if MEMORY_FILE.exists() else ""


# ═══════════════════════════════════════════════════════════════════════════
#  LEARNING ENGINE — Adaptation Loop
# ═══════════════════════════════════════════════════════════════════════════
class LearningEngine:
    """Tracks outcomes, learns patterns, adapts agent behavior."""

    _save_lock = threading.Lock()

    def __init__(self):
        self.patterns = self._load_json(
            LEARNING_FILE,
            {
                "tool_success": defaultdict(int),
                "tool_failure": defaultdict(int),
                "agent_success": defaultdict(int),
                "preferences": {},
            },
        )
        self.outcomes = self._load_json(OUTCOMES_FILE, [])
        if not isinstance(self.patterns.get("tool_success"), defaultdict):
            self.patterns["tool_success"] = defaultdict(int, self.patterns.get("tool_success", {}))
            self.patterns["tool_failure"] = defaultdict(int, self.patterns.get("tool_failure", {}))
            self.patterns["agent_success"] = defaultdict(int, self.patterns.get("agent_success", {}))

    def record_tool(self, tool: str, success: bool):
        key = "tool_success" if success else "tool_failure"
        self.patterns[key][tool] = self.patterns[key].get(tool, 0) + 1
        self._save()

    def record_agent(self, agent: str, success: bool):
        key = "agent_success" if success else "agent_failure"
        self.patterns.setdefault(key, {})
        self.patterns[key][agent] = self.patterns[key].get(agent, 0) + 1
        self._save()

    def record_outcome(self, task: str, outcome: str, steps: int, duration: float):
        self.outcomes.append(
            {
                "task": task[:200],
                "outcome": outcome,
                "steps": steps,
                "duration": round(duration, 1),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(self.outcomes) > 500:
            self.outcomes = self.outcomes[-500:]
        self._save()

    def get_tool_ranking(self) -> list:
        rankings = []
        tools = set(list(self.patterns["tool_success"].keys()) + list(self.patterns["tool_failure"].keys()))
        for t in tools:
            s = self.patterns["tool_success"].get(t, 0)
            f = self.patterns["tool_failure"].get(t, 0)
            total = s + f
            rate = s / total if total > 0 else 0
            rankings.append((t, s, f, rate))
        rankings.sort(key=lambda x: x[3], reverse=True)
        return rankings

    def get_adaptation_hints(self) -> str:
        hints = []
        ranking = self.get_tool_ranking()
        if ranking:
            top_tools = [t for t, s, f, r in ranking[:3] if r > 0.7]
            if top_tools:
                hints.append(f"Tools with high success rate: {', '.join(top_tools)}")
            weak_tools = [t for t, s, f, r in ranking if r < 0.5 and s + f >= 3]
            if weak_tools:
                hints.append(f"Tools to use carefully: {', '.join(weak_tools)}")
        return "\n".join(hints) if hints else ""

    def _load_json(self, path, default):
        if Path(path).exists():
            try:
                return json.loads(Path(path).read_text("utf-8"))
            except Exception:
                return default
        return default

    def _save(self):
        with self._save_lock:
            p = self.patterns
            LEARNING_FILE.write_text(
                json.dumps({k: dict(v) if isinstance(v, defaultdict) else v for k, v in p.items()}, indent=2),
                "utf-8",
            )
            OUTCOMES_FILE.write_text(json.dumps(self.outcomes, ensure_ascii=False, indent=2), "utf-8")


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT — Main Agent Class
# ═══════════════════════════════════════════════════════════════════════════


class Agent:
    """AIOS-Local Agent with autonomous execution, RAG, swarm, and learning."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        role: str = "coordinator",
        delegate_depth: int = 0,
        start_services: bool = True,
        a2a: Optional[A2ABus] = None,
        rag: Optional["RAGEngine"] = None,
    ):
        self.model = model
        self.role = role
        self.delegate_depth = delegate_depth
        self.rag = rag or RAGEngine()
        self.memory = MemoryManager()
        self.learning = LearningEngine()
        self.messages: list[dict] = []
        self.tools_used: list[str] = []
        self.start_time = None
        self.a2a = a2a or A2ABus()
        self.knowledge_sync = KnowledgeSync(self.rag)
        if start_services:
            self.knowledge_sync.start()

    def build_system_prompt(self) -> str:
        soul = SOUL_FILE.read_text("utf-8") if SOUL_FILE.exists() else ""
        role_info = AGENT_ROLES.get(self.role, {})
        role_suffix = role_info.get("system_suffix", "")
        long_term = self.memory.read_long_term()
        adaptation = self.learning.get_adaptation_hints()
        relevant_episodes = self.memory.get_relevant_episodes(
            self.messages[-1].get("content", "") if self.messages else "", 2
        )

        episode_text = ""
        if relevant_episodes:
            episode_text = "\n<relevant_past_experiences>"
            for ep in relevant_episodes:
                episode_text += f"\n- Task: {ep['task'][:100]} -> {ep['outcome']}"
                if ep.get("tools_used"):
                    episode_text += f" (used: {', '.join(ep['tools_used'])})"
            episode_text += "\n</relevant_past_experiences>"

        return f"""<system_context>
Current time: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Model: {self.model} | Role: {self.role} | Agent: {role_info.get('name', 'AIOS')}
Platform: Windows 11, PowerShell
Safety mode: {SAFETY_MODE}
</system_context>

<personality>
{soul}
</personality>

{role_suffix}

<persistent_memory>
{long_term}
</persistent_memory>

{episode_text}

{f"<learning_hints>\n{adaptation}\n</learning_hints>" if adaptation else ""}

<instructions>
You are autonomous. Plan your approach, use tools as needed, and deliver results.
- Think step-by-step before acting
- Use tools for external actions
- If a tool fails, analyze and retry with an alternative
- Use DELEGATE for specialist subtasks
- Use swarm_execute when a request needs multi-agent execution and automation
- Use knowledge_adapt_live for real-time repository pattern adaptation decisions
- Use RAG_QUERY when retrieval is needed
- Use goose_run or DELEGATE to "goose" for complex coding/dev tasks — Goose is a separate AI agent with its own tools and extensions running locally via Ollama
Available specialist agents: {', '.join(AGENT_ROLES.keys())}
</instructions>"""

    def _build_api_messages(self, user_message: str, rounds: int) -> list[dict]:
        api_messages = [{"role": "system", "content": self.build_system_prompt()}]

        if rounds == 1:
            rag_results = self.rag.search(user_message, top_k=3)
            if rag_results:
                rag_context = "\n<retrieved_knowledge>\n"
                for text, score, source in rag_results:
                    rag_context += f"[{source}] (score={score:.2f}): {text[:400]}\n---\n"
                rag_context += "</retrieved_knowledge>\n"
                api_messages.append({"role": "system", "content": rag_context})

        incoming = self.a2a.receive(self.role)
        if incoming:
            api_messages.append(
                {
                    "role": "system",
                    "content": f"Pending A2A packet: {json.dumps(incoming, ensure_ascii=False)[:1000]}",
                }
            )

        api_messages.extend(self.messages)
        return api_messages

    async def _stream_model_response(self, api_messages: list[dict], stream):
        chunk_text = ""
        native_tool_calls = []
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": self.model,
                    "messages": api_messages,
                    "stream": True,
                    "tools": TOOLS_SCHEMA,
                    "options": {"num_predict": 4096, "temperature": 0.6},
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        msg = chunk.get("message", {})
                        content = msg.get("content", "")
                        if msg.get("tool_calls"):
                            native_tool_calls.extend(msg["tool_calls"])
                        if content:
                            chunk_text += content
                            if stream:
                                await stream({"type": "token", "content": content})
                        if chunk.get("done"):
                            break
                    except Exception:
                        continue
        return chunk_text, native_tool_calls

    @staticmethod
    def _normalize_native_args(raw_args: Any) -> dict:
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args) if raw_args.strip() else {}
            except Exception:
                _telemetry("native_tool_args_parse_error", raw_type="str", preview=raw_args[:200])
                return {}
        if isinstance(raw_args, dict):
            return raw_args
        _telemetry("native_tool_args_parse_error", raw_type=type(raw_args).__name__)
        return {}

    async def _process_native_tool_calls(self, chunk_text: str, native_tool_calls: list, stream):
        self.messages.append({"role": "assistant", "content": chunk_text})
        for tc in native_tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            args = self._normalize_native_args(fn.get("arguments", {}))
            result = await self._execute_and_stream(tool_name, args, stream)
            self.messages.append({"role": "tool", "content": result, "name": tool_name})

    async def _process_text_actions(self, chunk_text: str, stream):
        tool_name, tool_args, remaining = self._parse_tool_call(chunk_text)
        if tool_name:
            self.messages.append({"role": "assistant", "content": chunk_text})
            result = await self._execute_and_stream(tool_name, tool_args, stream)
            self.messages.append({"role": "tool", "name": tool_name, "content": result})
            return True, remaining

        agent_name, task_desc, remaining = self._parse_delegate(chunk_text)
        if agent_name:
            self.messages.append({"role": "assistant", "content": chunk_text})
            delegate_result = await self._delegate(agent_name, task_desc, stream)
            self.messages.append({"role": "system", "content": f"[Agent {agent_name} result]:\n{delegate_result}"})
            return True, remaining

        rag_q, rag_a, remaining = self._parse_rag(chunk_text)
        if rag_q:
            self.messages.append({"role": "assistant", "content": chunk_text})
            top_k = rag_a.get("top_k", 5)
            results = self.rag.search(rag_q, top_k=top_k)
            if results:
                rag_text = "Retrieved knowledge:\n" + "\n---\n".join(
                    f"[{s}]: {t[:400]}" for t, sc, s in results
                )
            else:
                rag_text = "No relevant documents found."
            if stream:
                await stream({"type": "rag_result", "query": rag_q, "results": rag_text[:500]})
            self.messages.append({"role": "system", "content": f"[RAG results]:\n{rag_text}"})
            return True, remaining

        return False, chunk_text

    async def _stream_error(self, message: str, stream):
        if stream:
            await stream({"type": "token", "content": message})

    async def chat(self, user_message: str, stream=None, max_rounds: int = 20) -> str:
        self.start_time = time.time()
        self.tools_used = []
        self.memory.add_working("user", user_message)
        self.messages = [{"role": "user", "content": user_message}]

        full_response = ""
        rounds = 0

        while rounds < max_rounds:
            rounds += 1
            api_messages = self._build_api_messages(user_message, rounds)

            try:
                chunk_text, native_tool_calls = await self._stream_model_response(api_messages, stream)

                if native_tool_calls:
                    await self._process_native_tool_calls(chunk_text, native_tool_calls, stream)
                    full_response = chunk_text
                    continue

                handled, remaining = await self._process_text_actions(chunk_text, stream)
                if handled:
                    full_response = remaining
                    continue

                full_response = chunk_text
                break

            except httpx.ConnectError:
                error = "❌ Cannot connect to Ollama. Make sure it's running."
                await self._stream_error(error, stream)
                full_response = error
                break
            except Exception as e:
                error = f"❌ Error: {e}"
                await self._stream_error(error, stream)
                full_response = error
                break

        duration = time.time() - self.start_time if self.start_time else 0
        self.memory.add_working("assistant", full_response)
        outcome = "success" if not full_response.startswith("❌") else "failure"
        self.learning.record_outcome(user_message[:200], outcome, len(self.tools_used), duration)
        if self.tools_used:
            self.memory.save_episode(user_message[:200], outcome, self.tools_used, self.role)

        return full_response

    async def _execute_and_stream(self, tool_name: str, args: dict, stream) -> str:
        self.tools_used.append(tool_name)
        if stream:
            await stream({"type": "tool_call", "tool": tool_name, "args": args})
        result = await execute_tool(tool_name, args, self)
        self.learning.record_tool(tool_name, not result.startswith("❌") and not result.startswith("⚠️"))
        if stream:
            await stream({"type": "tool_result", "tool": tool_name, "result": result[:3000]})
        return result

    async def _delegate(self, agent_name: str, task: str, stream=None) -> str:
        if self.delegate_depth >= MAX_DELEGATE_DEPTH:
            return f"❌ Delegate depth limit reached ({MAX_DELEGATE_DEPTH})"
        if agent_name not in AGENT_ROLES:
            return f"❌ Unknown agent: {agent_name}. Available: {', '.join(AGENT_ROLES.keys())}"
        if stream:
            await stream({"type": "delegate_start", "agent": agent_name, "task": task})

        # External agent: dispatch to Goose CLI
        role_def = AGENT_ROLES[agent_name]
        if role_def.get("external"):
            self.tools_used.append("goose_run")
            result_data = await _run_goose_cli(task, with_builtins="developer")
            if result_data["ok"]:
                result = f"🪿 Goose result:\n{result_data['response']}"
            else:
                result = f"❌ Goose error: {result_data['error']}"
            if stream:
                await stream({"type": "delegate_end", "agent": agent_name})
            self.learning.record_agent(agent_name, result_data["ok"])
            return result

        specialist = Agent(
            model=self.model,
            role=agent_name,
            delegate_depth=self.delegate_depth + 1,
            start_services=False,
            a2a=self.a2a,
            rag=self.rag,
        )
        result = await specialist.chat(task, stream=stream, max_rounds=10)
        if stream:
            await stream({"type": "delegate_end", "agent": agent_name})
        self.learning.record_agent(agent_name, not result.startswith("❌"))
        return result

    @staticmethod
    def _split_swarm_subtasks(task: str) -> list[tuple[str, str]]:
        text = task.strip()
        if not text:
            return []

        # If user already provided explicit steps, map them by intent.
        explicit = [p.strip(" -\t") for p in re.split(r"[\n;]+", text) if p.strip()]
        if len(explicit) >= 2:
            subtasks = []
            for p in explicit[:6]:
                low = p.lower()
                if any(k in low for k in ["research", "find", "compare", "search"]):
                    role = "researcher"
                elif any(k in low for k in ["write", "doc", "explain", "summary"]):
                    role = "writer"
                elif any(k in low for k in ["analy", "metric", "evaluate", "risk"]):
                    role = "analyst"
                elif any(k in low for k in ["code", "build", "implement", "fix", "refactor"]):
                    role = "coder"
                else:
                    role = "analyst"
                subtasks.append((role, p))
            return subtasks

        return [
            ("researcher", f"Gather relevant context, references, and constraints for: {text}"),
            ("analyst", f"Decompose execution strategy, dependencies, and risk checks for: {text}"),
            ("coder", f"Execute implementation and automation steps for: {text}"),
            ("writer", f"Produce an operator-ready summary and final output for: {text}"),
        ]

    @staticmethod
    def _format_swarm_results(task: str, results) -> str:
        lines = [f"Swarm execution completed for: {task}"]
        for idx, (role, subtask, result) in enumerate(results, 1):
            lines.append(
                f"\n[{idx}] {role.upper()}\n"
                f"Subtask: {subtask}\n"
                f"Result:\n{(result or '').strip()[:2500]}"
            )
        return "\n".join(lines)

    async def run_swarm(self, task: str, stream=None) -> str:
        subtasks = self._split_swarm_subtasks(task)
        if not subtasks:
            return "❌ Swarm task is empty"

        if stream:
            await stream({"type": "swarm_start", "task": task, "agents": [r for r, _ in subtasks]})

        async def _run_one(role: str, subtask: str):
            last = ""
            for attempt in range(2):
                result = await self._delegate(role, subtask, stream=stream)
                last = result or ""
                if "500 Internal Server Error" in last and attempt == 0:
                    await asyncio.sleep(0.8)
                    continue
                return role, subtask, result
            return role, subtask, last

        if SWARM_PARALLEL:
            results = await asyncio.gather(*(_run_one(role, subtask) for role, subtask in subtasks))
        else:
            results = [await _run_one(role, subtask) for role, subtask in subtasks]

        if stream:
            await stream({"type": "swarm_end", "task": task})
        return self._format_swarm_results(task, results)

    def _parse_tool_call(self, text: str):
        obj, remaining = extract_prefixed_json(text, "TOOL_CALL:")
        if not obj:
            return None, None, text
        return obj.get("tool"), obj.get("args", {}), remaining

    def _parse_delegate(self, text: str):
        obj, remaining = extract_prefixed_json(text, "DELEGATE:")
        if not obj:
            return None, None, text
        return obj.get("agent"), obj.get("task", ""), remaining

    def _parse_rag(self, text: str):
        obj, remaining = extract_prefixed_json(text, "RAG_QUERY:")
        if not obj:
            return None, None, text
        return obj.get("query", ""), obj, remaining
