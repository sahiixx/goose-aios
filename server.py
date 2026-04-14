"""
AIOS-Local — A local AI assistant powered by Ollama
Runs entirely on-device with tool execution, memory, and web search.
"""

import asyncio as _asyncio
import json
import importlib
import logging
import os
import re as _re
import time as _time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

httpx = importlib.import_module("httpx")
_fastapi = importlib.import_module("fastapi")
FastAPI = _fastapi.FastAPI
WebSocket = _fastapi.WebSocket
WebSocketDisconnect = _fastapi.WebSocketDisconnect
CORSMiddleware = importlib.import_module("fastapi.middleware.cors").CORSMiddleware
FileResponse = importlib.import_module("fastapi.responses").FileResponse
StaticFiles = importlib.import_module("fastapi.staticfiles").StaticFiles

_agent_runtime = importlib.import_module("agent")
RuntimeAgent = _agent_runtime.Agent

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aios-local")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
CONVERSATIONS_DIR = BASE_DIR / "conversations"
CONVERSATIONS_DIR.mkdir(exist_ok=True)
CONV_INDEX_FILE = CONVERSATIONS_DIR / "_index.json"

# ── Config ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "qwen2.5-coder:7b"
AVAILABLE_MODELS = ["qwen2.5-coder:7b", "llama3.1:8b", "deepseek-coder:6.7b"]
OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GOOSE_REPO_DIR = BASE_DIR / ".external" / "goose"
MAX_AGENT_CACHE = 4
MAX_WS_MESSAGE_KB = 256
MAX_CONVERSATION_MESSAGES = 200
DEFAULT_CONV_TITLE = "New Chat"

# Auth: set AIOS_API_KEY env var to enable; empty/unset = no auth
API_KEY = os.environ.get("AIOS_API_KEY", "")

# Rate limiting: per-IP, sliding window
RATE_LIMIT_WINDOW = 60       # seconds
RATE_LIMIT_MAX_REQUESTS = 30 # max requests per window
_rate_ledger: dict[str, list[float]] = {}


def get_external_repos() -> dict[str, Path]:
    external_root = BASE_DIR / ".external"
    if not external_root.exists():
        return {}
    repos = {}
    for repo_dir in sorted(external_root.iterdir(), key=lambda p: p.name.lower()):
        if repo_dir.is_dir() and not repo_dir.name.startswith("."):
            repos[repo_dir.name] = repo_dir
    return repos

# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    yield
    # Shutdown: stop knowledge sync schedulers in all cached agents
    for agent in AGENT_CACHE.values():
        try:
            ks = agent.knowledge_sync
            if hasattr(ks, '_scheduler') and ks._scheduler and ks._scheduler.running:
                ks._scheduler.shutdown(wait=False)
        except Exception:
            pass
    AGENT_CACHE.clear()

def _check_rate_limit(ip: str) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = _time.time()
    timestamps = _rate_ledger.get(ip, [])
    cutoff = now - RATE_LIMIT_WINDOW
    timestamps = [t for t in timestamps if t > cutoff]
    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        _rate_ledger[ip] = timestamps
        return False
    timestamps.append(now)
    _rate_ledger[ip] = timestamps
    return True

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="AIOS-Local", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

Request = _fastapi.Request
Response = importlib.import_module("fastapi.responses").JSONResponse


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    path = request.url.path
    # Skip auth/rate-limit for static files and health
    if path.startswith("/static") or path == "/api/health" or path == "/":
        return await call_next(request)

    # API key check (only when AIOS_API_KEY is set)
    if API_KEY:
        provided = request.headers.get("x-api-key", "") or request.query_params.get("api_key", "")
        if provided != API_KEY:
            logger.warning("Auth rejected for %s %s from %s", request.method, path, request.client.host if request.client else "?")
            return Response(content={"error": "Invalid or missing API key"}, status_code=401)

    # Rate limit
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        logger.warning("Rate limited %s on %s %s", client_ip, request.method, path)
        return Response(content={"error": "Rate limit exceeded"}, status_code=429)

    return await call_next(request)


AGENT_CACHE: OrderedDict[str, RuntimeAgent] = OrderedDict()


def get_runtime_agent(model: str) -> RuntimeAgent:
    if model in AGENT_CACHE:
        AGENT_CACHE.move_to_end(model)
        return AGENT_CACHE[model]
    if len(AGENT_CACHE) >= MAX_AGENT_CACHE:
        _evict_model, evicted = AGENT_CACHE.popitem(last=False)
        try:
            ks = evicted.knowledge_sync
            if hasattr(ks, '_scheduler') and ks._scheduler and ks._scheduler.running:
                ks._scheduler.shutdown(wait=False)
        except Exception:
            pass
    AGENT_CACHE[model] = RuntimeAgent(model=model, role="coordinator")
    logger.info("Created agent for model %s (cache size: %d)", model, len(AGENT_CACHE))
    return AGENT_CACHE[model]


def _build_context_prompt(messages: list[dict], user_msg: str, max_items: int = 8) -> str:
    recent = messages[-max_items:]
    if not recent:
        return user_msg
    lines = []
    for m in recent:
        role = m.get("role", "user")
        content = (m.get("content", "") or "").strip()
        if content:
            lines.append(f"{role}: {content[:500]}")
    context = "\n".join(lines)
    return f"Conversation context:\n{context}\n\nUser request:\n{user_msg}"


async def chat_with_runtime_agent(user_msg: str, messages: list, model: str, websocket=None, cancel_event=None) -> str:
    agent = get_runtime_agent(model)
    prompt = _build_context_prompt(messages, user_msg)

    async def stream_bridge(event: dict):
        if cancel_event and cancel_event.is_set():
            raise _asyncio.CancelledError("User cancelled")
        if not websocket:
            return
        event_type = event.get("type")
        if event_type == "token":
            await websocket.send_json({"type": "token", "content": event.get("content", "")})
        elif event_type == "tool_call":
            await websocket.send_json(
                {"type": "tool_call", "tool": event.get("tool"), "args": event.get("args", {})}
            )
        elif event_type == "tool_result":
            await websocket.send_json(
                {"type": "tool_result", "tool": event.get("tool"), "result": event.get("result", "")}
            )
        elif event_type in {"rag_result", "delegate_start", "delegate_end"}:
            await websocket.send_json(event)

    return await agent.chat(prompt, stream=stream_bridge)


# ── Conversation Storage ──────────────────────────────────────────────────
_CONV_ID_RE = _re.compile(r"^[a-zA-Z0-9_-]+$")


def _sanitize_conv_id(conv_id: str) -> str:
    if not _CONV_ID_RE.match(conv_id):
        raise ValueError(f"Invalid conversation id: {conv_id!r}")
    return conv_id


def save_conversation(conv_id: str, messages: list):
    """Save conversation to disk and update index."""
    safe_id = _sanitize_conv_id(conv_id)
    path = CONVERSATIONS_DIR / f"{safe_id}.json"
    path.write_text(json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8")
    _update_conv_index(safe_id, messages)

def load_conversation(conv_id: str) -> list:
    """Load conversation from disk"""
    path = CONVERSATIONS_DIR / f"{_sanitize_conv_id(conv_id)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []

def list_conversations() -> list:
    """List all saved conversations using index for speed."""
    index = _load_conv_index()
    if index:
        return sorted(index.values(), key=lambda c: c.get("updated", 0), reverse=True)[:50]
    # Fallback: rebuild index from disk
    convs = []
    for f in sorted(CONVERSATIONS_DIR.glob("*.json"), reverse=True):
        if f.name.startswith("_"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            title = DEFAULT_CONV_TITLE
            for msg in data:
                if msg.get("role") == "user":
                    title = msg.get("content", DEFAULT_CONV_TITLE)[:60]
                    break
            convs.append({
                "id": f.stem,
                "title": title,
                "updated": f.stat().st_mtime,
            })
        except Exception:
            pass
    _save_conv_index({c["id"]: c for c in convs})
    return convs[:50]


def _load_conv_index() -> dict:
    if CONV_INDEX_FILE.exists():
        try:
            return json.loads(CONV_INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_conv_index(index: dict):
    CONV_INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_conv_index(conv_id: str, messages: list):
    index = _load_conv_index()
    title = DEFAULT_CONV_TITLE
    for msg in messages:
        if msg.get("role") == "user":
            title = msg.get("content", DEFAULT_CONV_TITLE)[:60]
            break
    index[conv_id] = {"id": conv_id, "title": title, "updated": _time.time()}
    _save_conv_index(index)


# ── REST Endpoints ────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/api/conversations")
async def get_conversations():
    return {"conversations": list_conversations()}

@app.get("/api/conversation/{conv_id}")
async def get_conversation(conv_id: str):
    messages = load_conversation(conv_id)
    return {"messages": messages}

@app.delete("/api/conversation/{conv_id}")
async def delete_conversation(conv_id: str):
    safe_id = _sanitize_conv_id(conv_id)
    path = CONVERSATIONS_DIR / f"{safe_id}.json"
    if path.exists():
        path.unlink()
    index = _load_conv_index()
    index.pop(safe_id, None)
    _save_conv_index(index)
    return {"ok": True}

@app.get("/api/models")
async def get_models():
    """Get available models from Ollama"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"models": models}
    except Exception:
        return {"models": AVAILABLE_MODELS}


@app.get("/api/integrations/goose")
async def goose_integration_status():
    readme = GOOSE_REPO_DIR / "README.md"
    docs_dir = GOOSE_REPO_DIR / "documentation" / "docs"
    doc_count = 0
    if docs_dir.exists():
        doc_count = sum(1 for p in docs_dir.rglob("*.md*") if p.is_file())
    return {
        "cloned": GOOSE_REPO_DIR.exists(),
        "path": str(GOOSE_REPO_DIR),
        "readme_present": readme.exists(),
        "docs_count": doc_count,
        "integrated_with_runtime_sync": True,
    }


@app.get("/api/integrations")
async def integrations_status():
    integrations = []
    for name, repo_dir in get_external_repos().items():
        readme = repo_dir / "README.md"
        docs_count = 0
        if repo_dir.exists():
            docs_count = sum(1 for p in repo_dir.rglob("*.md*") if p.is_file())
        integrations.append(
            {
                "name": name,
                "cloned": repo_dir.exists(),
                "path": str(repo_dir),
                "readme_present": readme.exists(),
                "docs_count": docs_count,
                "integrated_with_runtime_sync": True,
            }
        )
    return {"integrations": integrations}


@app.get("/api/health")
async def health_check():
    """Runtime health endpoint."""
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": ollama_ok,
        "agents_cached": len(AGENT_CACHE),
        "conversations": len(list(CONVERSATIONS_DIR.glob("*.json"))),
    }


# ── WebSocket Chat ────────────────────────────────────────────────────────
@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    cancel_event = _asyncio.Event()

    try:
        # Wait for initial config
        init = await websocket.receive_json()
        model = init.get("model", DEFAULT_MODEL)
        conv_id = init.get("conversation_id", str(uuid.uuid4()))
        messages = load_conversation(conv_id)
        client_ip = websocket.client.host if websocket.client else "unknown"

        # Auth check for WebSocket
        if API_KEY:
            ws_key = init.get("api_key", "")
            if ws_key != API_KEY:
                logger.warning("WS auth rejected from %s", client_ip)
                await websocket.send_json({"type": "error", "message": "Invalid API key"})
                await websocket.close(code=4001)
                return

        # Rate limit for WebSocket
        if not _check_rate_limit(client_ip):
            logger.warning("WS rate limited %s", client_ip)
            await websocket.send_json({"type": "error", "message": "Rate limit exceeded"})
            await websocket.close(code=4029)
            return

        logger.info("WS connected: ip=%s model=%s conv=%s", client_ip, model, conv_id[:8])

        # Welcome message
        await websocket.send_json({
            "type": "system",
            "message": f"AIOS-Local ready. Model: {model} | Conversation: {conv_id[:8]}",
        })

        while True:
            raw = await websocket.receive_text()
            if len(raw) > MAX_WS_MESSAGE_KB * 1024:
                await websocket.send_json({"type": "error", "message": "Message too large"})
                continue
            data = json.loads(raw)

            if data.get("type") == "cancel":
                cancel_event.set()
                continue

            if data.get("type") == "message":
                user_msg = data.get("content", "").strip()
                if not user_msg:
                    continue

                cancel_event.clear()

                # Add user message
                messages.append({"role": "user", "content": user_msg})
                # Cap conversation length
                if len(messages) > MAX_CONVERSATION_MESSAGES:
                    messages = messages[-MAX_CONVERSATION_MESSAGES:]
                save_conversation(conv_id, messages)

                # Stream AI response
                await websocket.send_json({"type": "start"})

                try:
                    response = await chat_with_runtime_agent(
                        user_msg, messages, model, websocket, cancel_event
                    )
                except _asyncio.CancelledError:
                    response = "(generation stopped by user)"

                await websocket.send_json({"type": "end"})

                # Save assistant response
                messages.append({"role": "assistant", "content": response})
                save_conversation(conv_id, messages)

            elif data.get("type") == "switch_model":
                model = data.get("model", DEFAULT_MODEL)
                await websocket.send_json({
                    "type": "system",
                    "message": f"Switched to {model}",
                })

    except WebSocketDisconnect:
        logger.info("WS disconnected: conv=%s", conv_id[:8] if 'conv_id' in dir() else "?")
    except json.JSONDecodeError:
        logger.warning("WS received invalid JSON")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn = importlib.import_module("uvicorn")
    logger.info("Starting AIOS-Local on http://localhost:8765 (model=%s, auth=%s)", DEFAULT_MODEL, "on" if API_KEY else "off")
    print("=" * 50)
    print("  AIOS-Local \u2014 Starting server...")
    print("  URL: http://localhost:8765")
    print(f"  Default model: {DEFAULT_MODEL}")
    print(f"  Auth: {'API key required' if API_KEY else 'disabled'}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
