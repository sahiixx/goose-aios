# AIOS-Local

A local-first AI assistant powered by [Ollama](https://ollama.com). Runs entirely on your device — no cloud, no API keys, no data leaving your machine.

## Features

- **Autonomous agent** — multi-step task execution with tool use
- **RAG (Retrieval-Augmented Generation)** — semantic search over a local knowledge base
- **Swarm coordination** — parallel sub-agents with role assignment
- **Episodic memory** — persistent conversation history and learning from past outcomes
- **Tool suite** — web search (DuckDuckGo), browser (Playwright), file I/O, shell, scheduling (APScheduler)
- **Real-time streaming** — WebSocket-based chat with token-by-token responses
- **Conversation persistence** — conversations saved to disk as JSON
- **Optional API key auth** and per-IP rate limiting
- **Docker-ready** — single `docker build` gets you a running server

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (or accessible via `OLLAMA_HOST`)

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Pull the default model
ollama pull qwen2.5-coder:7b

# 3. Start Ollama (in a separate terminal if not already running)
ollama serve

# 4. Start the server
python server.py
```

Open **http://localhost:8765** in your browser.

### Windows batch launchers

```bat
start.bat   # starts Ollama + server
goose.bat   # starts with Goose integration
```

### Custom port

```bash
# Set via uvicorn directly
python -c "import uvicorn; uvicorn.run('server:app', host='0.0.0.0', port=8080)"
```

## Docker

```bash
# Build
docker build -t aios-local .

# Run (assumes Ollama is on the host)
docker run -p 8765:8765 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  aios-local
```

The server listens on port **8765** by default. A built-in healthcheck hits `/api/health` every 30 seconds.

## Configuration

All settings are controlled through environment variables.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `AIOS_API_KEY` | _(empty — auth disabled)_ | Set to require an API key on every request |
| `AIOS_SAFETY_MODE` | `trusted_local` | `trusted_local` · `read_only` · `paranoid` |
| `AIOS_REQUIRE_HIGH_RISK_APPROVAL` | `false` | Gate high-risk tools (`bash`, `write_file`, `goose_run`) |
| `AIOS_BASH_ALLOWED_PREFIXES` | `git ,python ,py ,pip ,pytest ,dir ,ls ,type ` | Comma-separated command prefixes allowed in `bash` tool |
| `AIOS_ALLOWED_DOMAINS` | _(empty — all allowed)_ | Comma-separated allowlist for web requests |
| `AIOS_MAX_DELEGATE_DEPTH` | `3` | Maximum swarm delegation depth |
| `AIOS_MAX_TOOL_TIMEOUT` | `60` | Per-tool timeout in seconds |
| `AIOS_MAX_WRITE_FILE_BYTES` | `262144` (256 KB) | Max size for `write_file` tool |
| `AIOS_SWARM_PARALLEL` | `false` | Run swarm sub-agents in parallel |
| `AIOS_GOOSE_TIMEOUT` | `300` | Goose CLI task timeout in seconds |
| `AIOS_GOOSE_MAX_TURNS` | `20` | Max Goose CLI conversation turns |
| `GOOSE_EXE` | `~/.local/bin/goose.exe` | Path to the Goose executable |

## Available Models

| Model | Pull command |
|---|---|
| `qwen2.5-coder:7b` _(default)_ | `ollama pull qwen2.5-coder:7b` |
| `llama3.1:8b` | `ollama pull llama3.1:8b` |
| `deepseek-coder:6.7b` | `ollama pull deepseek-coder:6.7b` |

Switch models at runtime via the UI or by sending `{"type": "switch_model", "model": "<name>"}` over the WebSocket.

## Project Structure

```
aios-local/
├── agent.py              # Core agent engine (autonomous execution, RAG, swarm, learning)
├── server.py             # FastAPI server (REST + WebSocket)
├── core/                 # Shared utility modules
│   ├── a2a.py            #   Agent-to-agent message bus
│   ├── memory.py         #   MemoryManager (episodic + working memory)
│   ├── parser_utils.py   #   JSON parsing helpers
│   └── safety.py         #   Path safety, command policy, mode gates
├── tools/
│   └── healthcheck.py    # Healthcheck tool
├── config/
│   ├── SOUL.md           # Agent personality definition
│   └── goose_profile.json
├── static/               # Frontend (index.html + vendored JS/CSS)
├── knowledge/            # RAG knowledge base (auto-indexed)
├── memory/
│   ├── episodes/         # Conversation episode files
│   └── learning/         # Learned patterns and outcomes
├── conversations/        # Persisted conversation JSON files
├── tests/
│   ├── test_agent.py
│   ├── test_server.py
│   └── test_server_integration.py
├── Dockerfile
└── requirements.txt
```

## API

### REST

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve the chat UI |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/models` | List available models |
| `GET` | `/api/conversations` | List all conversations |
| `POST` | `/api/conversations` | Create a new conversation |
| `GET` | `/api/conversations/{id}` | Get a conversation |
| `DELETE` | `/api/conversations/{id}` | Delete a conversation |
| `GET` | `/api/integrations` | List configured integrations |

### WebSocket

Connect to `ws://localhost:8765/ws/{conversation_id}`.

**Send a message:**
```json
{"type": "message", "content": "Your message here", "model": "qwen2.5-coder:7b"}
```

**Switch model:**
```json
{"type": "switch_model", "model": "llama3.1:8b"}
```

**Receive events:** `system` · `token` · `end` · `error`

## Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest

# Run all tests
pytest tests/ -v

# Compile check
python -m py_compile agent.py server.py

# Run specific test file
pytest tests/test_agent.py -v
pytest tests/test_server.py -v
pytest tests/test_server_integration.py -v
```

CI runs on every push/PR to `main` via GitHub Actions (see `.github/workflows/ci.yml`).

## Customizing the Agent Personality

Edit `config/SOUL.md` to change how the agent presents itself, its communication style, and behavioral boundaries.

## License

See repository for license information.
