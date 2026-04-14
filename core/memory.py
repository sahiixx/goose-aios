import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


class MemoryManager:
    def __init__(self, episodes_path: Path, memory_file: Path):
        self.working = []
        self.episodes_path = episodes_path
        self.memory_file = memory_file

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
        return self.memory_file.read_text("utf-8") if self.memory_file.exists() else ""
