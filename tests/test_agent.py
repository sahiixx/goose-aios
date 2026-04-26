"""Tests for core agent.py functions."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import pytest


# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent


def _run(coro):
    """Run a coroutine in a fresh event loop (avoids DeprecationWarning)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── _safe_resolve_path ────────────────────────────────────────────────────

class TestSafeResolvePath:
    def test_relative_path_within_workspace(self):
        result = agent._safe_resolve_path("README.md")
        assert result.is_relative_to(agent.BASE_DIR)

    def test_absolute_path_within_workspace(self):
        p = agent.BASE_DIR / "agent.py"
        result = agent._safe_resolve_path(str(p))
        assert result == p.resolve()

    def test_path_outside_workspace_raises(self):
        with pytest.raises(PermissionError, match="outside workspace"):
            agent._safe_resolve_path("C:/Windows/System32/cmd.exe")

    def test_path_traversal_blocked(self):
        with pytest.raises(PermissionError, match="outside workspace"):
            agent._safe_resolve_path("../../etc/passwd")


# ── DESTRUCTIVE_RE ────────────────────────────────────────────────────────

class TestDestructiveRegex:
    @pytest.mark.parametrize("cmd", [
        "rm -rf /",
        "del /s /q C:\\",
        "format C:",
        "Remove-Item -Recurse C:\\Users",
        "Stop-Computer",
        "Restart-Computer",
        "diskpart",
        "bcdedit /set",
        "reg delete HKLM",
    ])
    def test_blocks_destructive(self, cmd):
        assert agent.DESTRUCTIVE_RE.search(cmd) is not None

    @pytest.mark.parametrize("cmd", [
        "ls -la",
        "dir",
        "Get-Content file.txt",
        "echo hello",
        "python script.py",
    ])
    def test_allows_safe(self, cmd):
        assert agent.DESTRUCTIVE_RE.search(cmd) is None


# ── _blocked_by_mode ─────────────────────────────────────────────────────

class TestBlockedByMode:
    def test_trusted_local_allows_bash(self):
        original = agent.SAFETY_MODE
        agent.SAFETY_MODE = "trusted_local"
        try:
            assert agent._blocked_by_mode("bash") is None
        finally:
            agent.SAFETY_MODE = original

    def test_read_only_blocks_bash(self):
        original = agent.SAFETY_MODE
        agent.SAFETY_MODE = "read_only"
        try:
            result = agent._blocked_by_mode("bash")
            assert result is not None
            assert "read_only" in result
        finally:
            agent.SAFETY_MODE = original

    def test_read_only_allows_read_file(self):
        original = agent.SAFETY_MODE
        agent.SAFETY_MODE = "read_only"
        try:
            assert agent._blocked_by_mode("read_file") is None
        finally:
            agent.SAFETY_MODE = original


# ── RAGEngine ─────────────────────────────────────────────────────────────

class TestRAGEngine:
    def setup_method(self):
        self.rag = agent.RAGEngine()
        # Clear any persistent state for test isolation
        self.rag.chunks = []
        self.rag.chunk_ids = set()
        self.rag._vectorizer = None
        self.rag._doc_matrix = None
        self.rag._index_dirty = True

    def test_ingest_and_search(self):
        self.rag.ingest("Python is a programming language", "test-doc")
        results = self.rag.search("Python programming", top_k=1)
        assert len(results) >= 1
        assert "Python" in results[0][0]

    def test_search_empty_returns_empty(self):
        fresh = agent.RAGEngine()
        results = fresh.search("anything")
        assert results == []

    def test_stats(self):
        self.rag.ingest("This is a longer chunk of text that should definitely create at least one chunk in the system", "test-source")
        s = self.rag.stats()
        assert s["chunks"] >= 1
        assert s["sources"] >= 1

    def test_delete_source(self):
        self.rag.ingest("Hello world", "to-delete")
        self.rag.delete_source("to-delete")
        assert "to-delete" not in self.rag.list_sources()

    def test_list_sources(self):
        self.rag.ingest("This is a sufficiently long content block for source A testing", "src-a")
        self.rag.ingest("This is a sufficiently long content block for source B testing", "src-b")
        sources = self.rag.list_sources()
        assert "src-a" in sources
        assert "src-b" in sources


# ── _parse_tool_call / _parse_delegate / _parse_rag ──────────────────────

class TestAgentParsing:
    def setup_method(self):
        self.agent = agent.Agent(start_services=False)

    def test_parse_tool_call(self):
        text = 'Let me check. TOOL_CALL: {"tool": "bash", "args": {"command": "ls"}}'
        tool, args, remaining = self.agent._parse_tool_call(text)
        assert tool == "bash"
        assert args == {"command": "ls"}
        assert "Let me check." in remaining

    def test_parse_tool_call_none(self):
        tool, args, remaining = self.agent._parse_tool_call("No tool here")
        assert tool is None

    def test_parse_delegate(self):
        text = 'DELEGATE: {"agent": "coder", "task": "write tests"}'
        name, task, _ = self.agent._parse_delegate(text)
        assert name == "coder"
        assert task == "write tests"

    def test_parse_rag(self):
        text = 'RAG_QUERY: {"query": "how to test", "top_k": 3}'
        query, args, _ = self.agent._parse_rag(text)
        assert query == "how to test"
        assert args.get("top_k") == 3


# ── _split_swarm_subtasks ────────────────────────────────────────────────

class TestSwarmSubtasks:
    def test_empty_returns_empty(self):
        result = agent.Agent._split_swarm_subtasks("")
        assert result == []

    def test_single_task_gets_default_roles(self):
        result = agent.Agent._split_swarm_subtasks("Build a website")
        assert len(result) == 4
        roles = [r for r, _ in result]
        assert "researcher" in roles
        assert "coder" in roles

    def test_explicit_steps_get_role_mapped(self):
        task = "research best frameworks\ncode the API\nwrite documentation"
        result = agent.Agent._split_swarm_subtasks(task)
        assert len(result) == 3
        assert result[0][0] == "researcher"
        assert result[1][0] == "coder"
        assert result[2][0] == "writer"


# ── MemoryManager ────────────────────────────────────────────────────────

class TestMemoryManager:
    def test_working_memory_cap(self):
        mm = agent.MemoryManager()
        for i in range(120):
            mm.add_working("user", f"msg {i}")
        assert len(mm.working) == 100

    def test_episode_pruning(self):
        with tempfile.TemporaryDirectory() as tmp:
            mm = agent.MemoryManager()
            mm.episodes_path = Path(tmp)
            for i in range(10):
                (Path(tmp) / f"ep_{i}.json").write_text(
                    json.dumps({"id": f"ep_{i}", "task": "t", "outcome": "ok"}), "utf-8"
                )
            mm._prune_episodes(max_keep=5)
            remaining = list(Path(tmp).glob("*.json"))
            assert len(remaining) == 5


# ── _domain_allowed ──────────────────────────────────────────────────────

class TestDomainAllowed:
    def test_empty_domains_allows_all(self):
        original = agent.ALLOWED_WEB_DOMAINS
        agent.ALLOWED_WEB_DOMAINS = set()
        try:
            assert agent._domain_allowed("https://example.com") is True
        finally:
            agent.ALLOWED_WEB_DOMAINS = original

    def test_restricted_domain_blocks(self):
        original = agent.ALLOWED_WEB_DOMAINS
        agent.ALLOWED_WEB_DOMAINS = {"example.com"}
        try:
            assert agent._domain_allowed("https://evil.com") is False
            assert agent._domain_allowed("https://example.com/page") is True
        finally:
            agent.ALLOWED_WEB_DOMAINS = original


# ── _find_balanced_json_end ──────────────────────────────────────────────

class TestBalancedJson:
    def test_simple_object(self):
        text = '{"key": "value"} extra'
        end = agent._find_balanced_json_end(text, 0)
        assert end == 15
        assert json.loads(text[: end + 1]) == {"key": "value"}

    def test_nested_object(self):
        text = '{"a": {"b": 1}} rest'
        end = agent._find_balanced_json_end(text, 0)
        assert json.loads(text[: end + 1]) == {"a": {"b": 1}}

    def test_string_with_braces(self):
        text = '{"a": "hello {world}"} rest'
        end = agent._find_balanced_json_end(text, 0)
        assert json.loads(text[: end + 1]) == {"a": "hello {world}"}

    def test_no_closing_brace(self):
        text = '{"key": "value"'
        end = agent._find_balanced_json_end(text, 0)
        assert end == -1


# ── execute_tool ─────────────────────────────────────────────────────────

class TestExecuteTool:
    def test_unknown_tool(self):
        result = _run(
            agent.execute_tool("nonexistent_tool", {})
        )
        assert "Unknown tool" in result

    def test_blocked_in_read_only(self):
        original = agent.SAFETY_MODE
        agent.SAFETY_MODE = "read_only"
        try:
            result = _run(
                agent.execute_tool("bash", {"command": "ls"})
            )
            assert "Blocked" in result or "BLOCKED" in result
        finally:
            agent.SAFETY_MODE = original


# ── Goose CLI integration ────────────────────────────────────────────────

class TestRunGooseCli:
    """Tests for _run_goose_cli and _tool_goose_run."""

    def test_goose_not_found(self, monkeypatch):
        monkeypatch.setattr(agent, "GOOSE_EXE", Path("/nonexistent/goose.exe"))
        result = _run(
            agent._run_goose_cli("say hello")
        )
        assert result["ok"] is False
        assert "not found" in result["error"]

    def test_goose_success_json(self, monkeypatch):
        """Simulate a successful Goose run with JSON output."""
        goose_output = json.dumps({
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "Hello from Goose!"}]}
            ],
            "metadata": {"total_tokens": 42},
        })

        async def fake_exec(*cmd, stdout, stderr, cwd):
            class FakeProc:
                returncode = 0
                async def communicate(self):
                    return goose_output.encode(), b""
            return FakeProc()

        monkeypatch.setattr(agent, "GOOSE_EXE", Path(__file__))  # exists
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        result = _run(
            agent._run_goose_cli("say hello")
        )
        assert result["ok"] is True
        assert "Hello from Goose!" in result["response"]
        assert result["metadata"]["total_tokens"] == 42

    def test_goose_nonzero_exit(self, monkeypatch):
        async def fake_exec(*cmd, stdout, stderr, cwd):
            class FakeProc:
                returncode = 1
                async def communicate(self):
                    return b"", b"model not found"
            return FakeProc()

        monkeypatch.setattr(agent, "GOOSE_EXE", Path(__file__))
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        result = _run(
            agent._run_goose_cli("fail task")
        )
        assert result["ok"] is False
        assert "model not found" in result["error"]

    def test_goose_timeout(self, monkeypatch):
        async def fake_exec(*cmd, stdout, stderr, cwd):
            class FakeProc:
                returncode = 0
                async def communicate(self):
                    await asyncio.sleep(999)
                    return b"", b""
                def kill(self):
                    pass
            return FakeProc()

        monkeypatch.setattr(agent, "GOOSE_EXE", Path(__file__))
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        result = _run(
            agent._run_goose_cli("slow task", timeout_sec=0.1)
        )
        assert result["ok"] is False
        assert "timed out" in result["error"]

    def test_goose_raw_text_fallback(self, monkeypatch):
        """When Goose returns non-JSON, fall back to raw text."""
        async def fake_exec(*cmd, stdout, stderr, cwd):
            class FakeProc:
                returncode = 0
                async def communicate(self):
                    return b"Just plain text output", b""
            return FakeProc()

        monkeypatch.setattr(agent, "GOOSE_EXE", Path(__file__))
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        result = _run(
            agent._run_goose_cli("plain task")
        )
        assert result["ok"] is True
        assert "Just plain text output" in result["response"]


class TestToolGooseRun:
    def test_missing_task(self):
        result = _run(
            agent._tool_goose_run({}, None)
        )
        assert "Missing" in result

    def test_success_formatting(self, monkeypatch):
        async def fake_run(task, with_builtins="developer", timeout_sec=None, max_turns=None):
            return {"ok": True, "response": "Done!", "metadata": {"total_tokens": 10}}

        monkeypatch.setattr(agent, "_run_goose_cli", fake_run)
        result = _run(
            agent._tool_goose_run({"task": "test"}, None)
        )
        assert "Goose completed" in result
        assert "10 tokens" in result
        assert "Done!" in result

    def test_error_formatting(self, monkeypatch):
        async def fake_run(task, with_builtins="developer", timeout_sec=None, max_turns=None):
            return {"ok": False, "error": "connection refused"}

        monkeypatch.setattr(agent, "_run_goose_cli", fake_run)
        result = _run(
            agent._tool_goose_run({"task": "test"}, None)
        )
        assert "Goose error" in result
        assert "connection refused" in result


class TestDelegateGoose:
    """Test that _delegate routes 'goose' to _run_goose_cli."""

    def test_delegate_to_goose(self, monkeypatch):
        async def fake_run(task, with_builtins="developer"):
            return {"ok": True, "response": "Goose did it"}

        monkeypatch.setattr(agent, "_run_goose_cli", fake_run)
        a = agent.Agent(start_services=False)
        result = _run(
            a._delegate("goose", "build a thing")
        )
        assert "Goose result" in result
        assert "Goose did it" in result
        assert "goose_run" in a.tools_used

    def test_delegate_to_goose_error(self, monkeypatch):
        async def fake_run(task, with_builtins="developer"):
            return {"ok": False, "error": "Goose crashed"}

        monkeypatch.setattr(agent, "_run_goose_cli", fake_run)
        a = agent.Agent(start_services=False)
        result = _run(
            a._delegate("goose", "do something")
        )
        assert "Goose error" in result
        assert "Goose crashed" in result
