from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

# Load config from JSON
_config_path = Path(__file__).parent / "generate_config.json"
with open(_config_path, "r") as f:
    _config = json.load(f)

CHROMA_COLLECTION = _config["common"]["chroma_collection"]
VECTOR_SIM_THRESHOLD = _config["common"]["vector_sim_threshold"]
CHROMA_PATH = str(Path(__file__).parent.parent / "data" / "chroma_db")


def pair_document(user_text: str, assistant_text: str) -> str:
    return f"USER:{user_text}\nASSISTANT:{assistant_text}"


class QAStore:
    """ChromaDB store: metadata + document text search + vector similarity in one place."""

    def __init__(self, persist_path: str = CHROMA_PATH, collection_name: str = CHROMA_COLLECTION) -> None:
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _next_id(self) -> str:
        existing = self.collection.get(include=[])
        ids = existing.get("ids") or []
        numeric = [int(i) for i in ids if str(i).isdigit()]
        return str(max(numeric, default=0) + 1)

    def insert_sample(
        self,
        user_text: str | None,
        assistant_text: str | None,
        status: str,
        raw_response: str,
        reason: str | None = None,
        nearest_selected_id: str | None = None,
        embedding: np.ndarray | None = None,
    ) -> str:
        sample_id = self._next_id()
        document = ""
        if user_text is not None and assistant_text is not None:
            document = pair_document(user_text, assistant_text)

        metadata: dict[str, Any] = {
            "user": user_text or "",
            "assistant": assistant_text or "",
            "status": status,
            "raw_response": raw_response,
            "reason": reason or "",
            "nearest_selected_id": nearest_selected_id or "",
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }

        kwargs: dict[str, Any] = {
            "ids": [sample_id],
            "documents": [document],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [embedding.reshape(-1).tolist()]

        self.collection.add(**kwargs)
        return sample_id

    def _count_where(self, where: dict[str, Any]) -> int:
        rows = self.collection.get(where=where, include=[])
        return len(rows.get("ids") or [])

    def count_selected(self) -> int:
        return self._count_where({"status": "selected"})

    def is_duplicate(self, embedding: np.ndarray) -> tuple[bool, float, str | None]:
        if self.count_selected() == 0:
            return False, 0.0, None

        result = self.collection.query(
            query_embeddings=[embedding.reshape(-1).tolist()],
            n_results=1,
            where={"status": "selected"},
            include=["metadatas", "distances"],
        )

        ids = result.get("ids") or [[]]
        distances = result.get("distances") or [[]]
        if not ids[0]:
            return False, 0.0, None

        distance = float(distances[0][0])
        similarity = 1.0 - distance
        nearest_id = ids[0][0]
        return similarity >= VECTOR_SIM_THRESHOLD, similarity, nearest_id

    def search_vector(
        self,
        embedding: np.ndarray,
        n_results: int = 5,
        status: str | None = "selected",
    ) -> dict[str, Any]:
        where = {"status": status} if status else None
        return self.collection.query(
            query_embeddings=[embedding.reshape(-1).tolist()],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def search_text(
        self,
        keyword: str,
        n_results: int = 10,
        status: str | None = None,
    ) -> dict[str, Any]:
        where: dict[str, Any] | None = {"status": status} if status else None
        return self.collection.get(
            where_document={"$contains": keyword},
            where=where,
            limit=n_results,
            include=["documents", "metadatas"],
        )

    def search_semantic(
        self,
        query_text: str,
        embedder,
        n_results: int = 5,
        status: str | None = "selected",
    ) -> dict[str, Any]:
        emb = embedder.encode([query_text])[0]
        vec = (emb / (np.linalg.norm(emb) + 1e-12)).astype(np.float32)
        return self.search_vector(vec, n_results=n_results, status=status)

    def list_by_status(self, status: str, limit: int | None = None) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "where": {"status": status},
            "include": ["metadatas", "documents"],
        }
        if limit is not None:
            kwargs["limit"] = limit

        rows = self.collection.get(**kwargs)
        items: list[dict[str, Any]] = []
        for sample_id, meta in zip(rows.get("ids") or [], rows.get("metadatas") or []):
            meta = meta or {}
            items.append(
                {
                    "id": sample_id,
                    "user": meta.get("user", ""),
                    "assistant": meta.get("assistant", ""),
                    "status": meta.get("status", ""),
                    "reason": meta.get("reason", ""),
                    "nearest_selected_id": meta.get("nearest_selected_id", ""),
                    "created_at": meta.get("created_at", ""),
                    "raw_response": meta.get("raw_response", ""),
                }
            )

        items.sort(key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else 0)
        return items

    def get_stats(self) -> dict[str, Any]:
        statuses = ("selected", "duplicate_context", "invalid_json", "bad_rule")
        status_counts = {s: self._count_where({"status": s}) for s in statuses}
        total = sum(status_counts.values())

        recent_rows = self.list_by_status("selected")
        recent = [
            {"user": r["user"], "assistant": r["assistant"], "time": r["created_at"]}
            for r in reversed(recent_rows[-10:])
        ]

        dup_rows = self.list_by_status("duplicate_context")
        recent_duplicates = []
        selected_map = {r["id"]: r for r in self.list_by_status("selected")}
        for r in reversed(dup_rows[-10:]):
            nearest_id = r.get("nearest_selected_id") or ""
            nearest = selected_map.get(nearest_id, {})
            recent_duplicates.append(
                {
                    "user": r["user"],
                    "assistant": r["assistant"],
                    "reason": r["reason"],
                    "nearest_selected_id": nearest_id,
                    "nearest_user": nearest.get("user", ""),
                    "nearest_assistant": nearest.get("assistant", ""),
                    "time": r["created_at"],
                }
            )

        return {
            "status_counts": status_counts,
            "total": total,
            "recent": recent,
            "recent_duplicates": recent_duplicates,
        }

    def close(self) -> None:
        pass
