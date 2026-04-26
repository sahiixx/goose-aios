"""Goose-AIOS Bridge — Replaces in-process A2A with HTTP calls to sahiixx-bus.

Goose's original A2ABus (core/a2a.py) uses in-process deques. This module
replaces it with HTTP calls to the sahiixx-bus A2A router, enabling Goose
to dispatch tasks to agency-agents, FRIDAY, Fixfizx, and other ecosystem
services.

It also feeds Goose's RAG results into sovereign-swarm's SwarmMemory for
shared context across the ecosystem.

Usage in goose-aios/agent.py:
    from bridge import GooseBridge
    bridge = GooseBridge()
    result = await bridge.dispatch("Search for Dubai real estate trends")
    await bridge.sync_memory("dubai_market_trends", result, tags="dubai real-estate")
"""
import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

# Add paths
sys.path.insert(0, "/mnt/c/Users/Sahil Khan/Downloads")
sys.path.insert(0, "/home/sahiix/sahiixx-bus")

from sahiixx_bus.a2a_router import A2ARouter
from sahiixx_bus.bridge import AgencyBridge, FridayBridge, GooseBridge as GooseClient, FixfizxBridge, MoltBridge
from sovereign_swarm import SwarmBus, SwarmMemory, SafetyCouncil
from pathlib import Path

logger = logging.getLogger("goose.bridge")


class GooseBridge:
    """Bridge between Goose-AIOS and the SAHIIXX ecosystem.

    Provides:
    - A2A task dispatch to agency-agents, FRIDAY, Fixfizx
    - Shared memory sync with sovereign-swarm
    - Safety scanning via sovereign-swarm's SafetyCouncil
    - Local fallback (Goose handles it internally if ecosystem is unavailable)
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.router = A2ARouter()
        self.safety = SafetyCouncil()
        self._bus: Optional[SwarmBus] = None
        self._memory: Optional[SwarmMemory] = None
        self._initialized = False

        # Register ecosystem bridges
        self.router.register("agency", AgencyBridge(), priority=10)
        self.router.register("friday", FridayBridge(), priority=5)
        self.router.register("goose", GooseClient(), priority=3)
        self.router.register("fixfizx", FixfizxBridge(), priority=7)
        self.router.register("molt", MoltBridge(), priority=1)

    async def init(self):
        """Initialize bus and memory subsystems."""
        if self._initialized:
            return
        self._bus = SwarmBus(self.data_dir / "goose_bus.db")
        self._memory = SwarmMemory(self.data_dir / "goose_memory.db")
        await self._bus.init()
        await self._memory.init()
        self._initialized = True

    async def dispatch(self, task: str, skills: Optional[List[str]] = None,
                       preferred_service: Optional[str] = None) -> Dict:
        """Dispatch a task to the best available ecosystem agent.

        If no agents are available, returns a local fallback indicator.
        """
        if not self._initialized:
            await self.init()

        # Safety scan first
        scan = self.safety.scan(task)
        if scan["blocked"]:
            return {"error": "blocked_by_safety", "rule": scan["rule"]}

        results = await self.router.route(task, skills=skills, preferred_service=preferred_service)
        if not results:
            return {"error": "no_agents_available", "local_fallback": True}

        # Publish to swarm bus for ecosystem visibility
        await self._bus.publish("goose.dispatch", {"task": task, "results": results})

        return results[0]

    async def sync_memory(self, key: str, value: Any, tags: str = "") -> None:
        """Sync a result to shared SwarmMemory for ecosystem access."""
        if not self._initialized:
            await self.init()
        await self._memory.store(key, value, tags=f"goose {tags}")

    async def search_memory(self, query: str, limit: int = 10) -> List[Dict]:
        """Search shared memory for relevant context."""
        if not self._initialized:
            await self.init()
        return await self._memory.search(query, limit)

    async def publish_event(self, topic: str, payload: Dict) -> None:
        """Publish an event to the swarm bus."""
        if not self._initialized:
            await self.init()
        await self._bus.publish(topic, payload)

    async def subscribe(self, topic: str, callback) -> None:
        """Subscribe to swarm bus events."""
        if not self._initialized:
            await self.init()
        await self._bus.subscribe(topic, callback)

    async def health(self) -> Dict:
        """Check health of ecosystem services."""
        return await self.router.health_check()

    async def shutdown(self):
        """Gracefully shutdown."""
        if self._bus:
            await self._bus.close()
        if self._memory:
            await self._memory.close()