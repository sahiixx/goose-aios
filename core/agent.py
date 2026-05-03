"""Main Agent class with autonomous execution, RAG, swarm, and learning."""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

from .a2a import A2ABus
from .config import (
    AGENT_ROLES,
    BASE_DIR,
    DEFAULT_SYSTEM,
    MAX_DELEGATE_DEPTH,
    MEMORY_FILE,
    OLLAMA_BASE,
    SAFETY_MODE,
    SOUL_FILE,
    SWARM_PARALLEL,
    TOOLS_SCHEMA,
)
from .learning_engine import LearningEngine
from .memory import MemoryManager as _CoreMemoryManager
from .parser_utils import extract_prefixed_json
from .rag_engine import RAGEngine
from .telemetry import _telemetry
from .tools import _run_goose_cli, execute_tool


try:
    import httpx  # pyright: ignore[reportMissingImports]
    _ConnectError = httpx.ConnectError
except ImportError:
    httpx = None
    class _ConnectError(Exception):
        pass


class MemoryManager(_CoreMemoryManager):
    def __init__(self):
        from .config import EPISODES_DIR, MEMORY_FILE
        super().__init__(episodes_path=EPISODES_DIR, memory_file=MEMORY_FILE)


class Agent:
    """AIOS-Local Agent with autonomous execution, RAG, swarm, and learning."""

    def __init__(
        self,
        model: str = "kimi-k2.6:cloud",
        role: str = "coordinator",
        delegate_depth: int = 0,
        start_services: bool = True,
        a2a: Optional[A2ABus] = None,
        rag: Optional[RAGEngine] = None,
    ):
        from .knowledge_sync import KnowledgeSync
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

            except _ConnectError:
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
