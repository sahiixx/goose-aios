"""Integration tests for server.py HTTP and WebSocket endpoints."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server  # noqa: E402

starlette_testclient = __import__("starlette.testclient", fromlist=["TestClient"])
TestClient = starlette_testclient.TestClient


class _FakeAgent:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "test-model")
        self.knowledge_sync = type("KS", (), {"_scheduler": None})()

    async def chat(self, prompt, stream=None):
        if stream:
            await stream({"type": "token", "content": "Hello "})
            await stream({"type": "token", "content": "World"})
        return "Hello World"


@pytest.fixture()
def client():
    with TestClient(server.app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clean_conversations(tmp_path, monkeypatch):
    """Redirect conversations to a temp dir so tests don't pollute real data."""
    monkeypatch.setattr(server, "CONVERSATIONS_DIR", tmp_path)
    monkeypatch.setattr(server, "CONV_INDEX_FILE", tmp_path / "_index.json")


@pytest.fixture(autouse=True)
def _fake_agent(monkeypatch):
    """Replace RuntimeAgent with a fake that doesn't need Ollama."""
    monkeypatch.setattr(server, "RuntimeAgent", _FakeAgent)
    # Clear agent cache so the fake is used for new agents
    server.AGENT_CACHE.clear()
    yield


# ── HTTP Endpoint Tests ──────────────────────────────────────────────────

class TestRootEndpoint:
    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


class TestConversationsAPI:
    def test_list_empty(self, client):
        resp = client.get("/api/conversations")
        assert resp.status_code == 200
        assert resp.json()["conversations"] == []

    def test_create_and_list(self, client):
        # Save a conversation directly
        server.save_conversation("test-conv-1", [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ])
        resp = client.get("/api/conversations")
        data = resp.json()
        assert len(data["conversations"]) == 1
        assert data["conversations"][0]["id"] == "test-conv-1"
        assert data["conversations"][0]["title"] == "Hello"

    def test_get_conversation(self, client):
        server.save_conversation("test-conv-2", [
            {"role": "user", "content": "Test message"},
        ])
        resp = client.get("/api/conversation/test-conv-2")
        assert resp.status_code == 200
        msgs = resp.json()["messages"]
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Test message"

    def test_get_nonexistent_conversation(self, client):
        resp = client.get("/api/conversation/nonexistent-id")
        assert resp.status_code == 200
        assert resp.json()["messages"] == []

    def test_delete_conversation(self, client):
        server.save_conversation("to-delete", [
            {"role": "user", "content": "Bye"},
        ])
        resp = client.delete("/api/conversation/to-delete")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        # Verify gone
        resp2 = client.get("/api/conversation/to-delete")
        assert resp2.json()["messages"] == []

    def test_invalid_conv_id_rejected(self, client):
        resp = client.get("/api/conversation/../../etc/passwd")
        assert resp.status_code in (400, 404, 422, 500)


class TestModelsAPI:
    def test_models_fallback(self, client):
        """When Ollama is unreachable, returns fallback models."""
        resp = client.get("/api/models")
        assert resp.status_code == 200
        assert "models" in resp.json()


class TestHealthAPI:
    def test_health_returns_status(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "ollama" in data
        assert "agents_cached" in data


class TestIntegrationsAPI:
    def test_integrations_returns_list(self, client):
        resp = client.get("/api/integrations")
        assert resp.status_code == 200
        assert "integrations" in resp.json()


# ── WebSocket Tests ───────────────────────────────────────────────────────

class TestWebSocketChat:
    def test_ws_connect_and_receive_system(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"model": "test-model", "conversation_id": "ws-test-1"})
            msg = ws.receive_json()
            assert msg["type"] == "system"
            assert "AIOS-Local ready" in msg["message"]

    def test_ws_send_message_gets_response(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"model": "test-model", "conversation_id": "ws-test-2"})
            ws.receive_json()  # system message

            ws.send_text(json.dumps({"type": "message", "content": "Hi"}))
            # Expect: start, tokens, end
            start = ws.receive_json()
            assert start["type"] == "start"

            tokens = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "end":
                    break
                if msg["type"] == "token":
                    tokens.append(msg["content"])

            assert len(tokens) >= 1
            assert "".join(tokens) == "Hello World"

    def test_ws_switch_model(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"model": "test-model", "conversation_id": "ws-test-3"})
            ws.receive_json()  # system message

            ws.send_text(json.dumps({"type": "switch_model", "model": "other-model"}))
            msg = ws.receive_json()
            assert msg["type"] == "system"
            assert "other-model" in msg["message"]

    def test_ws_empty_message_ignored(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"model": "test-model", "conversation_id": "ws-test-4"})
            ws.receive_json()  # system

            ws.send_text(json.dumps({"type": "message", "content": "   "}))
            # Send a real message to verify connection still works
            ws.send_text(json.dumps({"type": "message", "content": "Real msg"}))
            start = ws.receive_json()
            assert start["type"] == "start"

    def test_ws_oversized_message_rejected(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"model": "test-model", "conversation_id": "ws-test-5"})
            ws.receive_json()  # system

            # Send a message larger than MAX_WS_MESSAGE_KB
            big = "x" * (server.MAX_WS_MESSAGE_KB * 1024 + 1)
            ws.send_text(big)
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "too large" in msg["message"].lower()
