"""Local RAG with TF-IDF vectorization and cosine similarity."""

import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import numpy as np  # pyright: ignore[reportMissingImports]
except ImportError:
    np = None

from .config import KNOWLEDGE_DIR, _ISO_UTC_SUFFIX


class RAGEngine:
    """Local RAG with TF-IDF vectorization and cosine similarity."""

    _tfidf_cls = None
    _cosine_fn = None

    @classmethod
    def _load_sklearn(cls):
        if cls._tfidf_cls is None:
            cls._tfidf_cls = importlib.import_module("sklearn.feature_extraction.text").TfidfVectorizer
            cls._cosine_fn = importlib.import_module("sklearn.metrics.pairwise").cosine_similarity

    def __init__(self):
        self.chunks: list[dict] = []
        self.chunk_ids: set[str] = set()
        self._index_path = KNOWLEDGE_DIR / ".rag_index.json"
        self._vectorizer = None
        self._doc_matrix = None
        self._index_dirty = True
        self._load()
        self._index_dirty = True

    def ingest(
        self,
        content: str,
        source: str = "unknown",
        chunk_size: int = 800,
        overlap: int = 100,
        metadata: Optional[dict] = None,
        save: bool = True,
    ):
        chunks = self._chunk_text(content, chunk_size, overlap)
        changed = False
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
            if chunk_id in self.chunk_ids:
                continue
            self.chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "source": source,
                    "metadata": metadata or {},
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self.chunk_ids.add(chunk_id)
            changed = True
        if changed:
            self._index_dirty = True
        if save:
            self._save()

    def ingest_file(self, path: str):
        p = Path(path)
        if not p.exists():
            return False
        text = p.read_text("utf-8", errors="replace")
        self.ingest(text, source=p.name)
        return True

    def _ensure_index(self) -> bool:
        if not self.chunks or np is None:
            self._vectorizer = None
            self._doc_matrix = None
            return False
        try:
            self._load_sklearn()
        except Exception:
            return False
        if not self._index_dirty and self._vectorizer is not None and self._doc_matrix is not None:
            return True
        texts = [c["text"] for c in self.chunks]
        self._vectorizer = self._tfidf_cls(max_features=5000, stop_words="english", ngram_range=(1, 2))
        try:
            self._doc_matrix = self._vectorizer.fit_transform(texts)
        except ValueError:
            self._vectorizer = None
            self._doc_matrix = None
            return False
        self._index_dirty = False
        return True

    def _ranked_search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Return list of (chunk_index, adjusted_score) ranked by relevance."""
        if not query.strip() or not self._ensure_index():
            return []
        try:
            query_matrix = self._vectorizer.transform([query])
        except ValueError:
            return []
        scores = RAGEngine._cosine_fn(query_matrix, self._doc_matrix)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        now = datetime.now(timezone.utc)
        ranked = []
        for idx in top_indices:
            if scores[idx] <= 0.05:
                continue
            c = self.chunks[idx]
            freshness_bonus = 0.0
            ts = c.get("ingested_at")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", _ISO_UTC_SUFFIX))
                    hours = max((now - dt).total_seconds() / 3600.0, 0.0)
                    freshness_bonus = max(0.0, 0.15 - min(hours / 240.0, 0.15))
                except Exception:
                    pass
            ranked.append((int(idx), float(scores[idx]) + freshness_bonus))
        return ranked

    def search(self, query: str, top_k: int = 5) -> list:
        results = []
        for idx, score in self._ranked_search(query, top_k):
            c = self.chunks[idx]
            results.append((c["text"], score, c["source"]))
        return results

    def search_with_citations(self, query: str, top_k: int = 5) -> list[dict]:
        results = []
        for idx, score in self._ranked_search(query, top_k):
            c = self.chunks[idx]
            results.append(
                {
                    "text": c["text"],
                    "score": score,
                    "source": c["source"],
                    "citation": self._citation_for_chunk(c),
                    "metadata": c.get("metadata", {}),
                }
            )
        return results

    @staticmethod
    def _citation_for_chunk(chunk: dict) -> str:
        md = chunk.get("metadata", {}) or {}
        repo = md.get("repo")
        rel = md.get("rel_path")
        if repo and rel:
            return f"{repo}/{rel}"
        if repo:
            return repo
        return chunk.get("source", "unknown")

    def list_sources(self) -> list:
        return list({c["source"] for c in self.chunks})

    def delete_source(self, source: str):
        self.chunks = [c for c in self.chunks if c["source"] != source]
        self.chunk_ids = {c["id"] for c in self.chunks}
        self._index_dirty = True
        self._save()

    def delete_source_prefix(self, source_prefix: str) -> int:
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if not c["source"].startswith(source_prefix)]
        self.chunk_ids = {c["id"] for c in self.chunks}
        self._index_dirty = True
        self._save()
        return before - len(self.chunks)

    def prune_stale(self, max_age_hours: int = 24 * 30) -> int:
        now = datetime.now(timezone.utc)
        kept = []
        removed = 0
        for c in self.chunks:
            ts = c.get("ingested_at")
            if not ts:
                kept.append(c)
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", _ISO_UTC_SUFFIX))
                age_hours = (now - dt).total_seconds() / 3600.0
                if age_hours > max_age_hours:
                    removed += 1
                else:
                    kept.append(c)
            except Exception:
                kept.append(c)
        self.chunks = kept
        self.chunk_ids = {c["id"] for c in self.chunks}
        self._index_dirty = True
        self._save()
        return removed

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "sources": len(self.list_sources()),
            "latest_ingested_at": max((c.get("ingested_at") for c in self.chunks), default=None),
        }

    def _chunk_text(self, text: str, size: int, overlap: int) -> list:
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > size and current:
                chunks.append(current.strip())
                words = current.split()[-overlap // 4 :] if overlap else []
                current = " ".join(words) + "\n\n" + para
            else:
                current += "\n\n" + para if current else para
        if current.strip():
            chunks.append(current.strip())
        return [c for c in chunks if len(c) > 20]

    def _save(self):
        self._index_path.write_text(json.dumps({"chunks": self.chunks}, ensure_ascii=False, indent=2), "utf-8")

    def save(self):
        self._save()

    def _load(self):
        if self._index_path.exists():
            try:
                self.chunks = json.loads(self._index_path.read_text("utf-8")).get("chunks", [])
                self.chunk_ids = {c.get("id") for c in self.chunks if c.get("id")}
            except Exception:
                self.chunks = []
                self.chunk_ids = set()
        self._index_dirty = True
