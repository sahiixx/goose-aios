"""Learning engine — tracks outcomes and adapts behavior."""

import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import LEARNING_FILE, OUTCOMES_FILE


class LearningEngine:
    """Tracks outcomes, learns patterns, adapts agent behavior."""

    _save_lock = threading.Lock()

    def __init__(self):
        self.patterns = self._load_json(
            LEARNING_FILE,
            {
                "tool_success": defaultdict(int),
                "tool_failure": defaultdict(int),
                "agent_success": defaultdict(int),
                "preferences": {},
            },
        )
        self.outcomes = self._load_json(OUTCOMES_FILE, [])
        if not isinstance(self.patterns.get("tool_success"), defaultdict):
            self.patterns["tool_success"] = defaultdict(int, self.patterns.get("tool_success", {}))
            self.patterns["tool_failure"] = defaultdict(int, self.patterns.get("tool_failure", {}))
            self.patterns["agent_success"] = defaultdict(int, self.patterns.get("agent_success", {}))

    def record_tool(self, tool: str, success: bool):
        key = "tool_success" if success else "tool_failure"
        self.patterns[key][tool] = self.patterns[key].get(tool, 0) + 1
        self._save()

    def record_agent(self, agent: str, success: bool):
        key = "agent_success" if success else "agent_failure"
        self.patterns.setdefault(key, {})
        self.patterns[key][agent] = self.patterns[key].get(agent, 0) + 1
        self._save()

    def record_outcome(self, task: str, outcome: str, steps: int, duration: float):
        self.outcomes.append(
            {
                "task": task[:200],
                "outcome": outcome,
                "steps": steps,
                "duration": round(duration, 1),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(self.outcomes) > 500:
            self.outcomes = self.outcomes[-500:]
        self._save()

    def get_tool_ranking(self) -> list:
        rankings = []
        tools = set(list(self.patterns["tool_success"].keys()) + list(self.patterns["tool_failure"].keys()))
        for t in tools:
            s = self.patterns["tool_success"].get(t, 0)
            f = self.patterns["tool_failure"].get(t, 0)
            total = s + f
            rate = s / total if total > 0 else 0
            rankings.append((t, s, f, rate))
        rankings.sort(key=lambda x: x[3], reverse=True)
        return rankings

    def get_adaptation_hints(self) -> str:
        hints = []
        ranking = self.get_tool_ranking()
        if ranking:
            top_tools = [t for t, s, f, r in ranking[:3] if r > 0.7]
            if top_tools:
                hints.append(f"Tools with high success rate: {', '.join(top_tools)}")
            weak_tools = [t for t, s, f, r in ranking if r < 0.5 and s + f >= 3]
            if weak_tools:
                hints.append(f"Tools to use carefully: {', '.join(weak_tools)}")
        return "\n".join(hints) if hints else ""

    def _load_json(self, path, default):
        if Path(path).exists():
            try:
                return json.loads(Path(path).read_text("utf-8"))
            except Exception:
                return default
        return default

    def _save(self):
        with self._save_lock:
            p = self.patterns
            LEARNING_FILE.write_text(
                json.dumps({k: dict(v) if isinstance(v, defaultdict) else v for k, v in p.items()}, indent=2),
                "utf-8",
            )
            OUTCOMES_FILE.write_text(json.dumps(self.outcomes, ensure_ascii=False, indent=2), "utf-8")
