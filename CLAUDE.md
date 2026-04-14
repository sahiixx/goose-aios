# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Setup:**
```bash
pip install -r requirements.txt        # Install dependencies
```

**Run tests:**
```bash
pytest tests/ -v                      # Run all tests
pytest tests/test_agent.py -v          # Run agent tests only
pytest tests/test_server.py -v         # Run server tests only
```

**Start server:**
```bash
python server.py                       # Start FastAPI server
python server.py --port 8080          # Custom port
```

**Start with Ollama:**
```bash
ollama serve                           # Start Ollama (in separate terminal)
ollama pull qwen2.5-coder:7b          # Pull default model
python start.bat                       # Or use batch file
```

## Project Architecture

**Entry Points:**
- `server.py` — FastAPI web server with WebSocket support
- `agent.py` (~84KB) — Core autonomous agent engine
- `start.bat` / `goose.bat` — Windows batch launchers

**Core Components:**

- `agent.py` — Main agent runtime with:
  - Autonomous task execution
  - RAG (Retrieval-Augmented Generation)
  - Swarm coordination
  - Learning from episodes
  - Tool use (web search, browser, scheduling)
  - Memory management (episodic + semantic)

- `server.py` — FastAPI server providing:
  - REST API for chat
  - WebSocket for real-time streaming
  - Conversation persistence
  - Static file serving
  - Rate limiting
  - Agent caching (max 4 agents)

**Supporting Directories:**

- `config/` — Configuration files
- `core/` — Core modules (empty, placeholder)
- `memory/` — Episodic memory storage
  - `episodes/` — Stored conversation episodes
- `knowledge/` — Knowledge base for RAG
- `tools/` — Tool implementations
- `conversations/` — Persisted conversation history
- `static/` — Frontend static files
- `tests/` — Test suite
  - `test_agent.py` — Agent unit tests
  - `test_server.py` — Server unit tests
  - `test_server_integration.py` — Integration tests

**External Dependencies:**
- `.external/` — External repositories (goose, etc.)

## Code Patterns

**Graceful Imports:** The codebase uses try/except for optional dependencies:
```python
try:
    import numpy as np
except ImportError:
    np = None
```

**Lazy Import Pattern:** `server.py` uses `importlib` to avoid circular imports with `agent.py`:
```python
_agent_runtime = importlib.import_module("agent")
RuntimeAgent = _agent_runtime.Agent
```

**Agent Architecture:**
- `Agent` class in `agent.py` is self-contained with memory, tools, and learning
- Agents are cached in `AGENT_CACHE` (max 4) in server.py
- Each agent has episodic memory (`MEMORY_DIR/episodes/`) and semantic knowledge

**Ollama Integration:**
- Default model: `qwen2.5-coder:7b`
- Available models: `qwen2.5-coder:7b`, `llama3.1:8b`, `deepseek-coder:6.7b`
- Ollama endpoint configured via `OLLAMA_HOST` env var (default: `http://localhost:11434`)

## Important Notes

- Requires Ollama running locally (or accessible via `OLLAMA_HOST`)
- Python 3.10+ recommended
- Uses FastAPI + WebSockets for real-time communication
- Static files served from `static/` directory
- Conversations persisted to `conversations/` as JSON
- Rate limiting: 30 requests per 60 seconds per IP
- API key auth optional via `AIOS_API_KEY` env var
- Tool system includes: web search (DDG), browser (Playwright), scheduling (APScheduler), file operations
- Memory system uses episodic (conversation history) + semantic (knowledge base) storage
