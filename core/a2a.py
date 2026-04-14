from collections import defaultdict, deque
from typing import Optional


class A2ABus:
    def __init__(self):
        self.queues: dict[str, deque] = defaultdict(deque)

    def send(self, to_agent: str, payload: dict):
        self.queues[to_agent].append(payload)

    def receive(self, agent_name: str) -> Optional[dict]:
        if self.queues[agent_name]:
            return self.queues[agent_name].popleft()
        return None

    def status(self) -> dict:
        return {k: len(v) for k, v in self.queues.items()}
