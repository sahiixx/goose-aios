"""Knowledge synchronization for external repositories and web sources."""

import asyncio
import fnmatch
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler  # pyright: ignore[reportMissingImports]
    from apscheduler.triggers.cron import CronTrigger  # pyright: ignore[reportMissingImports]
except ImportError:
    AsyncIOScheduler = None
    CronTrigger = None

from .config import (
    BASE_DIR,
    CONFIG_DIR,
    KNOWLEDGE_DIR,
    LIVE_ADAPT_WINDOW_SEC,
    MAX_SYNC_ERRORS_TRACKED,
    PATTERN_MARKDOWN_ALL,
    PATTERN_README,
    PATTERN_TEXT_ALL,
    _COPILOT_INSTRUCTIONS,
    _YAML_EXTS,
)
from .telemetry import _telemetry
from .utils import crawl_url


class KnowledgeSync:
    """Synchronizes knowledge from external repos and web sources into RAG."""

    REPO_SCAN_LIMIT = 5000
    _GOOSE_PROFILE_PATH = CONFIG_DIR / "goose_profile.json"
    _STATE_PATH = KNOWLEDGE_DIR / ".knowledge_sync_state.json"

    def __init__(self, rag_engine):
        self.rag = rag_engine
        self.goose_profile = self._load_goose_profile()
        self.last_sync = None
        self.sources = [
            "https://news.ycombinator.com/",
            "https://arxiv.org/list/cs.AI/recent",
            "https://openai.com/news/",
        ]
        self.doc_discovery_cache = {}
        self.external_repo_configs = self._build_external_repo_configs()
        self.external_docs = self._discover_external_docs()
        self.repo_profiles = {}
        self.repo_watermarks = {}
        self.repo_fingerprints = {}
        self.sync_running = False
        self.last_error = None
        self.last_duration_sec = 0.0
        self.sync_successes = 0
        self.sync_failures = 0
        self.error_history = []
        self.adaptation_history = []
        self._sync_lock = asyncio.Lock()
        self.scheduler = AsyncIOScheduler() if AsyncIOScheduler else None
        self._load_state()

    @classmethod
    def _load_goose_profile(cls) -> dict:
        try:
            if cls._GOOSE_PROFILE_PATH.exists():
                return json.loads(cls._GOOSE_PROFILE_PATH.read_text("utf-8"))
        except Exception:
            pass
        return {}

    def _load_state(self):
        try:
            if self._STATE_PATH.exists():
                data = json.loads(self._STATE_PATH.read_text("utf-8"))
                self.repo_watermarks = data.get("repo_watermarks", {}) or {}
                self.repo_fingerprints = data.get("repo_fingerprints", {}) or {}
                self.doc_discovery_cache = data.get("doc_discovery_cache", {}) or {}
        except Exception:
            self.repo_watermarks = {}
            self.repo_fingerprints = {}
            self.doc_discovery_cache = {}

    def _save_state(self):
        payload = {
            "repo_watermarks": self.repo_watermarks,
            "repo_fingerprints": self.repo_fingerprints,
            "doc_discovery_cache": self.doc_discovery_cache,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        except Exception:
            return

    def _build_external_repo_configs(self) -> list[dict]:
        external_root = BASE_DIR / ".external"
        if not external_root.exists():
            return []

        overrides = {
            "goose": {"patterns": [PATTERN_README, "documentation/docs/**/*.md*"], "limit": 40},
            "open-saas": {
                "patterns": [PATTERN_README, "CONTRIBUTING.md", PATTERN_MARKDOWN_ALL, PATTERN_TEXT_ALL],
                "limit": 30,
            },
            "agency-agents": {"patterns": [PATTERN_README, "docs/**/*.md*", PATTERN_MARKDOWN_ALL], "limit": 30},
        }

        configs = []
        excluded = set(self.goose_profile.get("excluded_repos", []))
        profile_limits = self.goose_profile.get("repo_limits", {})
        for repo_path in sorted(external_root.iterdir(), key=lambda p: p.name.lower()):
            if not repo_path.is_dir() or repo_path.name.startswith("."):
                continue
            repo_name = repo_path.name
            if repo_name in excluded:
                continue
            base = {
                "patterns": [
                    PATTERN_README,
                    "CONTRIBUTING.md",
                    "AGENTS.md",
                    _COPILOT_INSTRUCTIONS,
                    PATTERN_MARKDOWN_ALL,
                    PATTERN_TEXT_ALL,
                    "**/*.instructions.md",
                    "**/*.prompt.md",
                    "**/*.agent.md",
                    "**/*.json",
                    "**/*.yaml",
                    "**/*.yml",
                    "**/*.toml",
                    "**/*.ini",
                    "**/*.py",
                    "**/*.ts",
                    "**/*.tsx",
                    "**/*.js",
                    "**/*.jsx",
                    "**/*.sh",
                ],
                "exclude_globs": [
                    "**/.git/**",
                    "**/.venv/**",
                    "**/venv/**",
                    "**/node_modules/**",
                    "**/dist/**",
                    "**/build/**",
                    "**/__pycache__/**",
                    "**/.next/**",
                ],
                "limit": 140,
            }
            base.update(overrides.get(repo_name, {}))
            if repo_name in profile_limits:
                base["limit"] = profile_limits[repo_name]
            configs.append(
                {
                    "name": repo_name,
                    "path": repo_path,
                    "patterns": base["patterns"],
                    "exclude_globs": base["exclude_globs"],
                    "limit": base["limit"],
                }
            )

        priority_list = self.goose_profile.get("repo_priority", [])
        if priority_list:
            priority_map = {name: idx for idx, name in enumerate(priority_list)}
            configs.sort(key=lambda c: priority_map.get(c["name"], len(priority_list)))

        return configs

    @staticmethod
    def _is_ignored_path(path: Path, exclude_globs: list[str]) -> bool:
        normalized = path.as_posix()
        return any(fnmatch.fnmatch(normalized, g) for g in exclude_globs)

    @staticmethod
    def _doc_tier(path: Path) -> str:
        name = path.name.lower()
        ext = path.suffix.lower()
        if name in {"agents.md", _COPILOT_INSTRUCTIONS, "readme.md", "contributing.md"}:
            return "high"
        if name.endswith(".instructions.md") or name.endswith(".prompt.md") or name.endswith(".agent.md"):
            return "high"
        if ext in {".json", ".toml", ".ini"} | _YAML_EXTS:
            return "config"
        if ext in {".py", ".ts", ".tsx", ".js", ".jsx", ".sh"}:
            return "code"
        return "other"

    @staticmethod
    def _doc_priority(path: Path) -> tuple:
        tier_weight = {"high": 0, "config": 1, "code": 2, "other": 3}
        tier = KnowledgeSync._doc_tier(path)
        return (tier_weight[tier], len(path.as_posix()), path.as_posix().lower())

    @staticmethod
    def _is_binary_bytes(raw: bytes) -> bool:
        if not raw:
            return False
        if b"\x00" in raw:
            return True
        sample = raw[:4096]
        non_text = sum(1 for b in sample if b < 9 or (13 < b < 32))
        return (non_text / max(len(sample), 1)) > 0.2

    def _read_text_if_safe(self, path: Path, max_bytes: int = 256000) -> Optional[str]:
        try:
            if path.stat().st_size > max_bytes:
                return None
            raw = path.read_bytes()
            if self._is_binary_bytes(raw):
                return None
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return None

    def _build_repo_profile(self, repo_cfg: dict) -> dict:
        repo_path = repo_cfg["path"]
        if not repo_path.exists():
            return self._empty_repo_profile(repo_cfg["name"])

        extension_counts = defaultdict(int)
        markers = set()
        workflows = 0
        files_scanned = 0

        for path in repo_path.rglob("*"):
            if not path.is_file():
                continue
            files_scanned += 1
            if files_scanned > self.REPO_SCAN_LIMIT:
                break

            workflows += self._update_repo_profile_from_file(path, extension_counts, markers)

        return {
            "name": repo_cfg["name"],
            "present": True,
            "files_scanned": files_scanned,
            "extension_counts": dict(sorted(extension_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]),
            "markers": sorted(markers),
            "workflows": workflows,
        }

    @staticmethod
    def _empty_repo_profile(repo_name: str) -> dict:
        return {
            "name": repo_name,
            "present": False,
            "files_scanned": 0,
            "extension_counts": {},
            "markers": [],
            "workflows": 0,
        }

    @staticmethod
    def _file_markers() -> dict:
        return {
            "agents.md": "agent_manifest",
            _COPILOT_INSTRUCTIONS: "copilot_instructions",
            "azure.yaml": "azure_project",
            "dockerfile": "containerization",
            "docker-compose.yml": "compose",
            "docker-compose.yaml": "compose",
        }

    def _update_repo_profile_from_file(self, path: Path, extension_counts: defaultdict, markers: set) -> int:
        name_lower = path.name.lower()
        ext = path.suffix.lower() or "(no_ext)"
        extension_counts[ext] += 1

        mapped = self._file_markers().get(name_lower)
        if mapped:
            markers.add(mapped)

        workflows = 0
        path_posix = path.as_posix().lower()
        if "/.github/workflows/" in path_posix and ext in _YAML_EXTS:
            workflows = 1
            markers.add("github_actions")

        if "prompt" in name_lower or name_lower.endswith(".instructions.md"):
            markers.add("prompt_assets")
        if ext in {".tf", ".bicep"}:
            markers.add("iac")
        if ext in {".py", ".ts", ".tsx", ".js", ".jsx"}:
            markers.add("source_code")
        return workflows

    def _profile_to_text(self, profile: dict) -> str:
        return json.dumps(profile, ensure_ascii=False, indent=2)

    def _prepare_ingest_text(self, path: Path, text: str) -> str:
        ext = path.suffix.lower()
        if ext in {".py", ".ts", ".tsx", ".js", ".jsx", ".sh"}:
            return f"Code pattern from {path.name}:\n\n{text[:8000]}"
        if ext in {".json", ".toml", ".ini"} | _YAML_EXTS:
            return f"Configuration pattern from {path.name}:\n\n{text[:10000]}"
        return text[:12000]

    @staticmethod
    def _metadata_for_doc(repo_name: str, rel_path: str, path: Path, profile: dict) -> dict:
        return {
            "repo": repo_name,
            "rel_path": rel_path,
            "file_type": path.suffix.lower() or "(no_ext)",
            "tier": KnowledgeSync._doc_tier(path),
            "file_size": path.stat().st_size if path.exists() else 0,
            "markers": profile.get("markers", []),
            "ingest_kind": "repo_doc",
        }

    def _select_docs_by_tier(self, docs: list[Path], limit: int) -> list[Path]:
        tier_limits = {
            "high": max(int(limit * 0.45), 1),
            "config": max(int(limit * 0.2), 1),
            "code": max(int(limit * 0.25), 1),
            "other": max(int(limit * 0.1), 1),
        }
        by_tier: dict[str, list[Path]] = {"high": [], "config": [], "code": [], "other": []}
        for d in docs:
            by_tier[self._doc_tier(d)].append(d)

        selected: list[Path] = []
        for tier in ("high", "config", "code", "other"):
            selected.extend(by_tier[tier][: tier_limits[tier]])

        if len(selected) < limit:
            extras = [d for d in docs if d not in selected]
            selected.extend(extras[: limit - len(selected)])
        return selected[:limit]

    def _discover_repo_docs(self, repo_cfg: dict) -> list[Path]:
        repo_path = repo_cfg["path"]
        if not repo_path.exists():
            return []
        repo_name = repo_cfg["name"]
        try:
            repo_mtime = repo_path.stat().st_mtime
        except Exception:
            repo_mtime = 0.0
        cache_key = f"{repo_name}:{repo_mtime:.3f}:{repo_cfg.get('limit', 20)}"
        cached = getattr(self, "doc_discovery_cache", {}).get(cache_key)
        if cached:
            docs = [Path(p) for p in cached if Path(p).exists()]
            if docs:
                return docs
        seen: set[str] = set()
        docs: list[Path] = []
        exclude_globs = repo_cfg.get("exclude_globs", [])
        for pattern in repo_cfg.get("patterns", [PATTERN_MARKDOWN_ALL]):
            for path in repo_path.glob(pattern):
                if not path.is_file() or self._is_ignored_path(path, exclude_globs):
                    continue
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    docs.append(path)
        docs.sort(key=self._doc_priority)
        selected = self._select_docs_by_tier(docs, repo_cfg.get("limit", 20))
        self.doc_discovery_cache = {k: v for k, v in self.doc_discovery_cache.items() if k.startswith(f"{repo_name}:")}
        self.doc_discovery_cache[cache_key] = [str(p) for p in selected]
        return selected

    def _repo_fingerprint(self, repo_cfg: dict) -> str:
        repo_path = repo_cfg["path"]
        if not repo_path.exists():
            return "missing"
        try:
            stat = repo_path.stat()
        except Exception:
            return "unknown"
        docs = self.external_docs.get(repo_cfg["name"], [])
        docs_sig = "|".join(f"{p.name}:{int(p.stat().st_mtime)}" for p in docs if p.exists())
        return hashlib.md5(
            f"{repo_cfg['name']}|{int(stat.st_mtime)}|{len(docs)}|{docs_sig[:4000]}".encode("utf-8")
        ).hexdigest()

    def _discover_external_docs(self) -> dict[str, list[Path]]:
        return {cfg["name"]: self._discover_repo_docs(cfg) for cfg in self.external_repo_configs}

    def _recent_doc_changes(self, repo_docs: list[Path], window_sec: int = LIVE_ADAPT_WINDOW_SEC) -> list[str]:
        cutoff = time.time() - max(window_sec, 1)
        recent = []
        for p in repo_docs:
            try:
                if p.stat().st_mtime >= cutoff:
                    recent.append(p.name)
            except Exception:
                continue
        return recent[:20]

    def lookup_live_adaptation(self, top_n: int = 3) -> dict:
        self.external_repo_configs = self._build_external_repo_configs()
        self.external_docs = self._discover_external_docs()

        candidates = []
        for cfg in self.external_repo_configs:
            repo_name = cfg["name"]
            docs = self.external_docs.get(repo_name, [])
            profile = self._build_repo_profile(cfg)
            self.repo_profiles[repo_name] = profile

            indexed_docs = len(docs)
            watermarked_docs = sum(1 for k in self.repo_watermarks if k.startswith(f"{repo_name}:"))
            pending_docs = max(indexed_docs - watermarked_docs, 0)
            recent_files = self._recent_doc_changes(docs)

            marker_score = len(profile.get("markers", [])) * 1.2
            workflow_score = profile.get("workflows", 0) * 2.0
            freshness_score = len(recent_files) * 3.0
            pending_score = pending_docs * 1.5
            score = round(marker_score + workflow_score + freshness_score + pending_score, 2)

            reasons = []
            if pending_docs:
                reasons.append(f"{pending_docs} pending docs")
            if recent_files:
                reasons.append(f"{len(recent_files)} recent changes")
            if profile.get("markers"):
                reasons.append(f"markers={','.join(profile['markers'][:4])}")
            if profile.get("workflows", 0):
                reasons.append(f"workflows={profile['workflows']}")

            candidates.append(
                {
                    "repo": repo_name,
                    "score": score,
                    "indexed_docs": indexed_docs,
                    "watermarked_docs": watermarked_docs,
                    "pending_docs": pending_docs,
                    "recent_files": recent_files,
                    "markers": profile.get("markers", []),
                    "workflows": profile.get("workflows", 0),
                    "reasons": reasons,
                }
            )

        candidates.sort(key=lambda x: (-x["score"], x["repo"]))
        selected = [c["repo"] for c in candidates[: max(top_n, 1)] if c["score"] > 0]
        if not selected:
            selected = [c["repo"] for c in candidates[: max(top_n, 1)]]

        snapshot = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "top_n": max(top_n, 1),
            "selected": selected,
            "candidates": candidates,
        }
        self.adaptation_history.append(snapshot)
        self.adaptation_history = self.adaptation_history[-50:]
        return snapshot

    async def adapt_live(self, top_n: int = 3) -> dict:
        lookup = self.lookup_live_adaptation(top_n=top_n)
        runs = []
        for repo_name in lookup.get("selected", []):
            runs.append(await self.sync_once(mode="incremental", repo=repo_name))
        return {
            "status": "ok",
            "lookup": lookup,
            "sync_runs": runs,
            "last_sync": self.last_sync,
        }

    async def _sync_web_sources(self, mode: str):
        if not self.goose_profile.get("enable_web_sync", True):
            return
        if mode == "full":
            self.rag.delete_source_prefix("sync:")
        for url in self.sources:
            page = await crawl_url(url)
            if page.get("text"):
                self.rag.ingest(
                    page["text"],
                    source=f"sync:{url}",
                    metadata={"ingest_kind": "web_sync", "url": url, "provenance": "external_web", "trust": "untrusted"},
                    save=False,
                )

    def _sync_repo_docs(self, cfg: dict, mode: str) -> int:
        repo_name = cfg["name"]
        repo_root = cfg["path"]
        if mode == "full":
            self.rag.delete_source_prefix(f"{repo_name}:")

        repo_fp = self._repo_fingerprint(cfg)
        previous_fp = self.repo_fingerprints.get(repo_name)
        if mode != "full" and previous_fp and previous_fp == repo_fp:
            return 0

        profile = self._build_repo_profile(cfg)
        self.repo_profiles[repo_name] = profile
        self.rag.ingest(
            self._profile_to_text(profile),
            source=f"{repo_name}:_repo_profile",
            metadata={"ingest_kind": "repo_profile", "repo": repo_name, "provenance": "external_repo", "trust": "medium"},
            save=False,
        )

        ingested = 0
        for doc in self.external_docs.get(repo_name, []):
            try:
                rel = doc.relative_to(repo_root)
                rel_posix = rel.as_posix()
                source_key = f"{repo_name}:{rel_posix}"
                mtime = doc.stat().st_mtime
                if mode != "full" and self.repo_watermarks.get(source_key, 0) >= mtime:
                    continue
                text = self._read_text_if_safe(doc)
                if text and text.strip():
                    prepared = self._prepare_ingest_text(doc, text)
                    self.rag.ingest(
                        prepared,
                        source=source_key,
                        metadata={
                            **self._metadata_for_doc(repo_name, rel_posix, doc, profile),
                            "provenance": "external_repo",
                            "trust": "medium",
                        },
                        save=False,
                    )
                    self.repo_watermarks[source_key] = mtime
                    ingested += 1
            except Exception:
                continue
        self.repo_fingerprints[repo_name] = repo_fp
        return ingested

    async def sync_once(self, mode: str = "incremental", repo: Optional[str] = None) -> dict:
        if self.sync_running:
            return {"status": "busy", "last_sync": self.last_sync}

        async with self._sync_lock:
            self.sync_running = True
            start = time.time()
            self.last_error = None
            ingested_docs = 0
            touched_repos: list[str] = []

            try:
                self.external_repo_configs = self._build_external_repo_configs()
                self.external_docs = self._discover_external_docs()
                self.repo_profiles = {}

                await self._sync_web_sources(mode)

                targets = self.external_repo_configs
                if repo:
                    targets = [cfg for cfg in targets if cfg["name"] == repo]

                for cfg in targets:
                    touched_repos.append(cfg["name"])
                    ingested_docs += self._sync_repo_docs(cfg, mode)
                self.rag.save()
                self._save_state()

                self.last_sync = datetime.now(timezone.utc).isoformat()
                self.sync_successes += 1
                self.last_duration_sec = round(time.time() - start, 3)
                _telemetry(
                    "knowledge_sync",
                    status="ok",
                    mode=mode,
                    repo=repo,
                    touched_repos=touched_repos,
                    ingested_docs=ingested_docs,
                    duration_sec=self.last_duration_sec,
                )
                return {
                    "status": "ok",
                    "mode": mode,
                    "repo": repo,
                    "ingested_docs": ingested_docs,
                    "touched_repos": touched_repos,
                    "duration_sec": self.last_duration_sec,
                    "last_sync": self.last_sync,
                }
            except Exception as e:
                self.last_error = str(e)
                self.sync_failures += 1
                self.last_duration_sec = round(time.time() - start, 3)
                self.error_history.append(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "error": self.last_error,
                    }
                )
                self.error_history = self.error_history[-MAX_SYNC_ERRORS_TRACKED:]
                _telemetry(
                    "knowledge_sync",
                    status="error",
                    mode=mode,
                    repo=repo,
                    error=self.last_error,
                    duration_sec=self.last_duration_sec,
                )
                return {
                    "status": "error",
                    "mode": mode,
                    "repo": repo,
                    "error": self.last_error,
                    "duration_sec": self.last_duration_sec,
                    "last_sync": self.last_sync,
                }
            finally:
                self.sync_running = False

    def clear_repo_index(self, repo_name: str) -> dict:
        removed = self.rag.delete_source_prefix(f"{repo_name}:")
        for k in [k for k in self.repo_watermarks if k.startswith(f"{repo_name}:")]:
            del self.repo_watermarks[k]
        self.repo_fingerprints.pop(repo_name, None)
        self._save_state()
        return {"repo": repo_name, "removed_chunks": removed}

    def start(self):
        if not self.scheduler or self.scheduler.running:
            return
        self.scheduler.add_job(
            lambda: asyncio.create_task(self.sync_once()),
            "interval",
            minutes=5,
            id="knowledge_sync_5m",
            replace_existing=True,
        )
        if CronTrigger:
            self.scheduler.add_job(
                lambda: asyncio.create_task(self.sync_once()),
                CronTrigger(hour="*/1"),
                id="knowledge_sync_hourly",
                replace_existing=True,
            )
        self.scheduler.start()

    def status(self) -> dict:
        jobs = []
        if self.scheduler:
            jobs = [j.id for j in self.scheduler.get_jobs()]
        repo_status = []
        for cfg in self.external_repo_configs:
            repo_status.append(
                {
                    "name": cfg["name"],
                    "present": cfg["path"].exists(),
                    "indexed_docs": len(self.external_docs.get(cfg["name"], [])),
                    "markers": self.repo_profiles.get(cfg["name"], {}).get("markers", []),
                    "workflows": self.repo_profiles.get(cfg["name"], {}).get("workflows", 0),
                    "fingerprint_known": cfg["name"] in self.repo_fingerprints,
                }
            )
        return {
            "last_sync": self.last_sync,
            "sync_running": self.sync_running,
            "last_error": self.last_error,
            "last_duration_sec": self.last_duration_sec,
            "sync_successes": self.sync_successes,
            "sync_failures": self.sync_failures,
            "source_count": len(self.sources),
            "external_repos": repo_status,
            "jobs": jobs,
        }
