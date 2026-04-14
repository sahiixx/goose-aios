"""Tests for server.py functions."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


# ── _sanitize_conv_id ────────────────────────────────────────────────────

class TestSanitizeConvId:
    def test_valid_uuid_style(self):
        assert server._sanitize_conv_id("abc-123_DEF") == "abc-123_DEF"

    def test_rejects_slashes(self):
        with pytest.raises(ValueError):
            server._sanitize_conv_id("../etc/passwd")

    def test_rejects_dots(self):
        with pytest.raises(ValueError):
            server._sanitize_conv_id("hello.world")

    def test_rejects_spaces(self):
        with pytest.raises(ValueError):
            server._sanitize_conv_id("hello world")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            server._sanitize_conv_id("")


# ── Conversation index ───────────────────────────────────────────────────

class TestConversationIndex:
    def test_update_and_load_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = server.CONV_INDEX_FILE
            server.CONV_INDEX_FILE = Path(tmp) / "_index.json"
            try:
                server._update_conv_index("test-1", [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ])
                index = server._load_conv_index()
                assert "test-1" in index
                assert index["test-1"]["title"] == "Hello"
            finally:
                server.CONV_INDEX_FILE = original

    def test_delete_removes_from_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_idx = server.CONV_INDEX_FILE
            original_dir = server.CONVERSATIONS_DIR
            server.CONV_INDEX_FILE = Path(tmp) / "_index.json"
            server.CONVERSATIONS_DIR = Path(tmp)
            try:
                # Create a conversation file
                (Path(tmp) / "del-test.json").write_text(
                    json.dumps([{"role": "user", "content": "bye"}]), "utf-8"
                )
                server._update_conv_index("del-test", [{"role": "user", "content": "bye"}])
                # Delete it
                index = server._load_conv_index()
                index.pop("del-test", None)
                server._save_conv_index(index)
                assert "del-test" not in server._load_conv_index()
            finally:
                server.CONV_INDEX_FILE = original_idx
                server.CONVERSATIONS_DIR = original_dir


# ── _build_context_prompt ────────────────────────────────────────────────

class TestBuildContextPrompt:
    def test_empty_messages(self):
        result = server._build_context_prompt([], "hello")
        assert result == "hello"

    def test_with_history(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        result = server._build_context_prompt(msgs, "second")
        assert "first" in result
        assert "reply" in result
        assert "second" in result

    def test_truncates_long_content(self):
        msgs = [{"role": "user", "content": "x" * 1000}]
        result = server._build_context_prompt(msgs, "q")
        # Content per message is capped at 500 chars
        assert len(result) < 1000


# ── get_external_repos ───────────────────────────────────────────────────

class TestGetExternalRepos:
    def test_returns_dict(self):
        repos = server.get_external_repos()
        assert isinstance(repos, dict)

    def test_skips_hidden_dirs(self):
        repos = server.get_external_repos()
        for name in repos:
            assert not name.startswith(".")
