# goose-aios — Fully Local Manus AI

> **No APIs. No $200/month. No data leaves your machine.**  
> A local-first, autonomous AI assistant that runs entirely on your device.

[![Local-First](https://img.shields.io/badge/local--first-100%25-green)](https://github.com/sahiixx/goose-aios)
[![Ollama](https://img.shields.io/badge/ollama-powered-blue)](https://ollama.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](Dockerfile)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The Pitch

**Manus AI** is $200/month and sends your code to the cloud.  
**Claude Code** requires an API key and bills per token.  
**goose-aios** runs a **35B-parameter agent on your laptop** — for **free**, forever.

- ✅ Autonomous multi-step task execution  
- ✅ RAG over your local codebase  
- ✅ Swarm coordination with parallel sub-agents  
- ✅ Web search, browser automation, file I/O, shell access  
- ✅ Real-time streaming chat  
- ✅ Persistent memory across sessions  

All offline. All local. All yours.

---

## One-Command Start

```bash
# Option 1: Docker Compose (recommended)
curl -fsSL https://raw.githubusercontent.com/sahiixx/goose-aios/main/docker-compose.yml | docker compose -f - up

# Option 2: Docker
git clone https://github.com/sahiixx/goose-aios.git
cd goose-aios
docker build -t goose-aios .
docker run -p 8765:8765 -e OLLAMA_HOST=http://host.docker.internal:11434 goose-aios

# Option 3: Native
pip install -r requirements.txt
ollama pull qwen2.5-coder:7b
ollama serve &
python server.py
```

Open **http://localhost:8765** in your browser. Done.

---

## Benchmarks

| Metric | goose-aios + qwen2.5-coder:7b | Manus AI | Claude Code |
|---|---|---|---|
| **Cost** | **$0** | **$200/mo** | **~$0.01–$0.25 / request** |
| **Privacy** | **100% local** | Cloud | Cloud |
| **Offline** | **✅ Yes** | ❌ No | ❌ No |
| **Setup time** | **2 min** | Invite-only | 5 min |
| **Swarm agents** | **✅ Unlimited** | Limited | ❌ No |
| **RAG** | **✅ Local KB** | ✅ | ✅ |
| **WebSocket streaming** | **✅ Token-by-token** | ✅ | ✅ |

*Benchmarked on M2 MacBook Air, 16 GB RAM. Your mileage may vary.*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  🌐 Web UI (index.html + WebSocket)                         │
├─────────────────────────────────────────────────────────────┤
│  ⚡ FastAPI Server (REST + WebSocket)                       │
├─────────────────────────────────────────────────────────────┤
│  🧠 Agent Engine                                            │
│   ├── Autonomous execution loop                             │
│   ├── RAG (local knowledge base)                            │
│   ├── Swarm coordination (parallel sub-agents)              │
│   └── Episodic memory (conversation history + learning)     │
├─────────────────────────────────────────────────────────────┤
│  🔧 Tools                                                   │
│   ├── Web search (DuckDuckGo)                               │
│   ├── Browser (Playwright)                                  │
│   ├── File I/O                                            │
│   ├── Shell (gated, configurable)                           │
│   └── Scheduler (APScheduler)                               │
├─────────────────────────────────────────────────────────────┤
│  🦙 Ollama (local LLM)                                      │
│   ├── qwen2.5-coder:7b (default)                            │
│   ├── llama3.1:8b                                           │
│   └── deepseek-coder:6.7b                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Safety

goose-aios runs **on your machine** with **your permissions**.

| Mode | What It Does |
|---|---|
| `trusted_local` *(default)* | All tools enabled. You control the machine. |
| `read_only` | File writes and shell blocked. Safe for exploration. |
| `paranoid` | Everything gated. Every action requires approval. |

Configure via `AIOS_SAFETY_MODE` env var.

---

## Ecosystem

| Repo | What It Does | Stars |
|---|---|---|
| [`agency-agents`](https://github.com/sahiixx/agency-agents) | 152-agent Claude swarm | ⭐ 1 |
| [`titans-memory`](https://github.com/sahiixx/titans-memory) | Surprise-weighted persistent memory | ⭐ New |
| [`claude-skills`](https://github.com/sahiixx/claude-skills) | 10 production-grade Claude Code skills | ⭐ New |
| [`sovereign-swarm-v2`](https://github.com/sahiixx/sovereign-swarm-v2) | Modular multi-agent OS | ⭐ 0 |

---

## License

MIT — see [LICENSE](LICENSE).

> *"Why pay $200/month when your laptop can do it for free?"*
