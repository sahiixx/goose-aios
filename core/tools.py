"""Tool handlers and execution dispatcher."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .config import (
    GOOSE_EXE,
    GOOSE_MAX_TURNS,
    GOOSE_TIMEOUT_SEC,
    MAX_TOOL_TIMEOUT_SEC,
    MAX_WRITE_FILE_BYTES,
    MEMORY_FILE,
    _MSG_KSYNC_UNAVAILABLE,
    _MSG_RAG_UNAVAILABLE,
)
from .telemetry import _telemetry
from .utils import (
    _blocked_by_mode,
    _command_policy_block_reason,
    _safe_resolve_path,
    _validate_write_target,
    browser_extract,
    crawl_url,
    web_search,
)

if TYPE_CHECKING:
    from .agent import Agent

_MAX_MEMORY_FILE_BYTES = 512 * 1024  # 512 KB cap


async def _tool_bash(args: dict, _agent: Agent | None = None) -> str:
    cmd = args.get("command", "")
    block_reason = _command_policy_block_reason(cmd)
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


def _tool_read_file(args: dict, _agent: Agent | None = None) -> str:
    p = _safe_resolve_path(args.get("path", ""))
    if not p.exists():
        return f"❌ Not found: {p}"
    return p.read_text("utf-8", errors="replace")[:8000]


def _tool_write_file(args: dict, _agent: Agent | None = None) -> str:
    p = _safe_resolve_path(args.get("path", ""))
    content = args.get("content", "")
    if len(content.encode("utf-8")) > MAX_WRITE_FILE_BYTES:
        return f"❌ Content too large ({MAX_WRITE_FILE_BYTES} bytes max)"
    path_warning = _validate_write_target(p)
    if path_warning:
        return f"⚠️ BLOCKED: {path_warning}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, "utf-8")
    return f"✅ Written {len(content)} chars -> {p}"


def _tool_edit_file(args: dict, _agent: Agent | None = None) -> str:
    p = _safe_resolve_path(args.get("path", ""))
    if not p.exists():
        return f"❌ Not found: {p}"
    path_warning = _validate_write_target(p)
    if path_warning:
        return f"⚠️ BLOCKED: {path_warning}"
    old = args.get("old_text", "")
    new = args.get("new_text", "")
    content = p.read_text("utf-8", errors="replace")
    if old not in content:
        return "❌ Text not found in file"
    p.write_text(content.replace(old, new, 1), "utf-8")
    return f"✅ Edited {p}"


def _tool_list_files(args: dict, _agent: Agent | None = None) -> str:
    p = _safe_resolve_path(args.get("path", "."))
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


async def _tool_web_search(args: dict, agent: Agent | None = None) -> str:
    query = args.get("query", "")
    result = await web_search(query)
    if agent and agent.rag and result and not result.startswith("Search error"):
        agent.rag.ingest(
            result,
            source=f"web_search:{query[:50]}",
            metadata={"ingest_kind": "web_search", "provenance": "external_web", "trust": "untrusted"},
        )
    return result


async def _tool_crawl_url(args: dict, _agent: Agent | None = None) -> str:
    page = await crawl_url(args.get("url", ""))
    if page.get("text"):
        return f"{page.get('title', '')}\n{page['url']}\n\n{page['text'][:4000]}"
    return f"Crawl failed: {page.get('error', 'unknown')}"


def _tool_memory_read(args: dict, _agent: Agent | None = None) -> str:
    return MEMORY_FILE.read_text("utf-8") if MEMORY_FILE.exists() else "(empty)"


def _tool_memory_write(args: dict, _agent: Agent | None = None) -> str:
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


def _tool_rag_query(args: dict, agent: Agent | None = None) -> str:
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


def _tool_rag_ingest(args: dict, agent: Agent | None = None) -> str:
    if not (agent and agent.rag):
        return _MSG_RAG_UNAVAILABLE
    agent.rag.ingest(args.get("content", ""), source=args.get("source", "user_input"))
    return "✅ Added to knowledge base"


async def _tool_knowledge_sync_now(args: dict, agent: Agent | None = None) -> str:
    if not (agent and agent.knowledge_sync):
        return _MSG_KSYNC_UNAVAILABLE
    mode = args.get("mode", "incremental")
    repo = args.get("repo")
    result = await agent.knowledge_sync.sync_once(mode=mode, repo=repo)
    return "✅ Knowledge sync completed\n" + json.dumps(result, indent=2)


def _tool_knowledge_status(args: dict, agent: Agent | None = None) -> str:
    if not (agent and agent.knowledge_sync):
        return _MSG_KSYNC_UNAVAILABLE
    return json.dumps(agent.knowledge_sync.status(), indent=2)


async def _tool_knowledge_adapt_live(args: dict, agent: Agent | None = None) -> str:
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


def _tool_knowledge_clear_repo(args: dict, agent: Agent | None = None) -> str:
    if not (agent and agent.knowledge_sync):
        return _MSG_KSYNC_UNAVAILABLE
    repo = args.get("repo", "")
    if not repo:
        return "Missing required arg: repo"
    return json.dumps(agent.knowledge_sync.clear_repo_index(repo), indent=2)


def _tool_knowledge_prune(args: dict, agent: Agent | None = None) -> str:
    if not (agent and agent.rag):
        return _MSG_RAG_UNAVAILABLE
    max_age_hours = int(args.get("max_age_hours", 24 * 30))
    removed = agent.rag.prune_stale(max_age_hours=max_age_hours)
    return json.dumps({"removed_chunks": removed, "rag": agent.rag.stats()}, indent=2)


def _tool_plan_task(args: dict, _agent: Agent | None = None) -> str:
    return f"📋 Planning: {args.get('task', '')}\n→ Decomposing into steps..."


def _tool_delegate(args: dict, _agent: Agent | None = None) -> str:
    return f"🔀 Delegating to {args.get('agent', 'coder')}: {args.get('task', '')}"


async def _tool_swarm_execute(args: dict, agent: Agent | None = None) -> str:
    if not agent:
        return "Swarm executor not available"
    task = args.get("task", "").strip()
    if not task:
        return "Missing required arg: task"
    return await agent.run_swarm(task)


def _tool_a2a_send(args: dict, agent: Agent | None = None) -> str:
    if not (agent and agent.a2a):
        return "A2A not available"
    agent.a2a.send(
        args.get("to_agent", "coordinator"),
        {"task": args.get("task", ""), "ts": datetime.now(timezone.utc).isoformat()},
    )
    return "✅ A2A packet sent"


def _tool_a2a_status(args: dict, agent: Agent | None = None) -> str:
    if not (agent and agent.a2a):
        return "A2A not available"
    return json.dumps(agent.a2a.status(), indent=2)


async def _tool_browser_extract(args: dict, _agent: Agent | None = None) -> str:
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
        "--model", "kimi-k2.6:cloud",
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
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)

        raw_out = stdout.decode("utf-8", errors="replace").strip()
        raw_err = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            return {"ok": False, "error": raw_err or f"Goose exited with code {proc.returncode}"}

        try:
            data = json.loads(raw_out)
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
            return {"ok": True, "response": raw_out[:8000]}

    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return {"ok": False, "error": f"Goose timed out after {timeout_sec}s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _tool_goose_run(args: dict, _agent: Agent | None = None) -> str:
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


async def execute_tool(name: str, args: dict, agent: Agent | None = None) -> str:
    try:
        blocked = _blocked_by_mode(name)
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
