"""ChromaDB QA browser + CRUD (Flask). Run: python data/view_data.py"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from generate.chroma_db import QAStore, pair_document

# Load config from JSON
_config_path = Path(_ROOT) / "generate" / "generate_config.json"
with open(_config_path, "r") as f:
    _config = json.load(f)

CHROMA_COLLECTION = _config["common"]["chroma_collection"]
EMBED_MODEL = _config["common"]["embed_model"]
CHROMA_PATH = str(Path(_ROOT) / "data" / "chroma_db")

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_STATUSES = ("selected", "duplicate_context", "invalid_json", "bad_rule")
TRAIN_STATUS = "selected"

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _pair_embedding(user_text: str, assistant_text: str) -> np.ndarray:
    text = pair_document(user_text, assistant_text)
    emb = _get_embedder().encode([text])[0]
    vec = np.asarray(emb, dtype=np.float32).reshape(1, -1)
    return (vec / (np.linalg.norm(vec) + 1e-12)).astype(np.float32)


def _rows_to_items(ids: list[str], metadatas: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for sample_id, meta in zip(ids, metadatas):
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


class DataIndex:
    """Load, filter, and mutate QA samples in ChromaDB."""

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = CHROMA_COLLECTION,
    ) -> None:
        self.store = QAStore(persist_path=chroma_path, collection_name=collection_name)

    def query(
        self,
        status: str | None = None,
        keyword: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        sample_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if sample_id:
            rows = self.store.collection.get(ids=[sample_id], include=["metadatas"])
            ids = rows.get("ids") or []
            if not ids:
                return []
            return _rows_to_items(ids, rows.get("metadatas") or [])

        where: dict[str, Any] | None = {"status": status} if status else None
        kwargs: dict[str, Any] = {"include": ["metadatas", "documents"]}
        if where:
            kwargs["where"] = where
        if keyword:
            kwargs["where_document"] = {"$contains": keyword}

        rows = self.store.collection.get(**kwargs)
        items = _rows_to_items(rows.get("ids") or [], rows.get("metadatas") or [])

        if offset:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]
        return items

    def count(self, status: str | None = None, keyword: str | None = None) -> int:
        return len(self.query(status=status, keyword=keyword, limit=None, offset=0))

    def summary(self) -> dict[str, Any]:
        stats = self.store.get_stats()
        counts = stats["status_counts"]
        return {
            "total": stats["total"],
            "selected": counts.get("selected", 0),
            "duplicate_context": counts.get("duplicate_context", 0),
            "invalid_json": counts.get("invalid_json", 0),
            "bad_rule": counts.get("bad_rule", 0),
        }

    def load_training_rows(self, status: str = TRAIN_STATUS) -> list[dict[str, str]]:
        rows = self.query(status=status, limit=None)
        return [
            {"user": r["user"], "assistant": r["assistant"]}
            for r in rows
            if r.get("user") and r.get("assistant")
        ]

    def create(
        self,
        user: str,
        assistant: str,
        status: str = TRAIN_STATUS,
        reason: str = "",
    ) -> dict[str, Any]:
        if status not in ALL_STATUSES:
            raise ValueError(f"Invalid status: {status}")

        raw_response = json.dumps({"user": user, "assistant": assistant}, ensure_ascii=False)
        embedding = None
        if status == TRAIN_STATUS and user and assistant:
            embedding = _pair_embedding(user, assistant)

        sample_id = self.store.insert_sample(
            user_text=user,
            assistant_text=assistant,
            status=status,
            raw_response=raw_response,
            reason=reason or None,
            embedding=embedding,
        )
        rows = self.query(sample_id=sample_id)
        if not rows:
            raise RuntimeError("Failed to load sample after create")
        return rows[0]

    def update(self, sample_id: str, **fields: Any) -> dict[str, Any] | None:
        rows = self.store.collection.get(ids=[sample_id], include=["metadatas"])
        ids = rows.get("ids") or []
        if not ids:
            return None

        meta = dict(rows["metadatas"][0] or {})
        user = fields["user"] if "user" in fields else meta.get("user", "")
        assistant = fields["assistant"] if "assistant" in fields else meta.get("assistant", "")
        status = fields["status"] if "status" in fields else meta.get("status", "")
        reason = fields["reason"] if "reason" in fields else meta.get("reason", "")

        if status not in ALL_STATUSES:
            raise ValueError(f"Invalid status: {status}")

        document = pair_document(user, assistant) if user and assistant else ""
        raw_response = fields["raw_response"] if "raw_response" in fields else meta.get("raw_response", "")
        raw_json = json.dumps({"user": user, "assistant": assistant}, ensure_ascii=False)
        meta.update(
            {
                "user": user,
                "assistant": assistant,
                "status": status,
                "reason": reason or "",
                "raw_response": raw_response,
            }
        )

        update_kwargs: dict[str, Any] = {
            "ids": [sample_id],
            "documents": [document],
            "metadatas": [meta],
        }
        if status == TRAIN_STATUS and user and assistant:
            vec = _pair_embedding(user, assistant)
            update_kwargs["embeddings"] = [vec.reshape(-1).tolist()]

        self.store.collection.update(**update_kwargs)
        updated = self.query(sample_id=sample_id)
        return updated[0] if updated else None

    def delete(self, sample_id: str) -> bool:
        rows = self.store.collection.get(ids=[sample_id], include=[])
        if not rows.get("ids"):
            return False
        self.store.collection.delete(ids=[sample_id])
        return True

    def delete_by_status(self, status: str) -> int:
        if status not in ALL_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        rows = self.store.collection.get(where={"status": status}, include=[])
        ids = rows.get("ids") or []
        if not ids:
            return 0
        self.store.collection.delete(ids=ids)
        return len(ids)

    def close(self) -> None:
        self.store.close()


def load_training_dataset(status: str = TRAIN_STATUS):
    from datasets import Dataset

    index = DataIndex()
    try:
        rows = index.load_training_rows(status=status)
    finally:
        index.close()

    if not rows:
        raise RuntimeError(
            "No training samples found in ChromaDB. Run generate first (status=selected)."
        )
    return Dataset.from_list(rows)


app = Flask(__name__, template_folder=_DATA_DIR)


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/api/summary")
def api_summary():
    index = DataIndex()
    try:
        return jsonify(index.summary())
    finally:
        index.close()


@app.route("/api/samples", methods=["GET"])
def api_samples_list():
    status = request.args.get("status") or None
    keyword = request.args.get("keyword") or None
    sample_id = request.args.get("id") or None
    limit = request.args.get("limit", type=int, default=50)
    offset = request.args.get("offset", type=int, default=0)

    index = DataIndex()
    try:
        if sample_id:
            items = index.query(sample_id=sample_id)
            total = len(items)
        else:
            total = index.count(status=status, keyword=keyword)
            items = index.query(
                status=status,
                keyword=keyword,
                limit=limit,
                offset=offset,
            )
        return jsonify(
            {
                "items": items,
                "total_matched": total,
                "offset": offset,
                "limit": limit,
            }
        )
    finally:
        index.close()


@app.route("/api/samples", methods=["POST"])
def api_samples_create():
    body = request.get_json(silent=True) or {}
    user = str(body.get("user", "")).strip()
    assistant = str(body.get("assistant", "")).strip()
    status = str(body.get("status", TRAIN_STATUS)).strip()
    reason = str(body.get("reason", "")).strip()

    index = DataIndex()
    try:
        item = index.create(user=user, assistant=assistant, status=status, reason=reason)
        return jsonify(item), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        index.close()


@app.route("/api/samples/<sample_id>", methods=["PUT"])
def api_samples_update(sample_id: str):
    body = request.get_json(silent=True) or {}
    fields: dict[str, Any] = {}
    if "user" in body:
        fields["user"] = str(body["user"]).strip()
    if "assistant" in body:
        fields["assistant"] = str(body["assistant"]).strip()
    if "status" in body:
        fields["status"] = str(body["status"]).strip()
    if "reason" in body:
        fields["reason"] = str(body["reason"]).strip()

    if not fields:
        return jsonify({"error": "No fields to update"}), 400

    index = DataIndex()
    try:
        item = index.update(sample_id, **fields)
        if item is None:
            return jsonify({"error": "Sample not found"}), 404
        return jsonify(item)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        index.close()


@app.route("/api/samples/<sample_id>", methods=["DELETE"])
def api_samples_delete(sample_id: str):
    index = DataIndex()
    try:
        if not index.delete(sample_id):
            return jsonify({"error": "Sample not found"}), 404
        return jsonify({"ok": True, "id": sample_id})
    finally:
        index.close()


@app.route("/api/samples/by-status/<status>", methods=["DELETE"])
def api_samples_delete_by_status(status: str):
    index = DataIndex()
    try:
        deleted = index.delete_by_status(status)
        return jsonify({"ok": True, "status": status, "deleted": deleted})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        index.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="QA data viewer (ChromaDB)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()
    print(f"[VIEW_DATA] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
