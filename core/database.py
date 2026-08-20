"""
Face Database
=============
Persistent vector store for criminal face embeddings using ChromaDB.

ChromaDB provides:
- Fast approximate nearest-neighbor search
- Persistent on-disk storage (survives restarts)
- Metadata filtering (search by name, case ID, etc.)
- Built-in cosine similarity distance

Each enrolled criminal gets a record with:
- 512-dim ArcFace embedding (for matching)
- Metadata: name, criminal_id, case_number, notes, enrollment date
- Optional: base64 face thumbnail for quick display
"""

import os
import json
import base64
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
from rich.console import Console

console = Console()

# Lazy-load chromadb
_chromadb = None


def _get_chromadb():
    global _chromadb
    if _chromadb is None:
        import chromadb
        _chromadb = chromadb
    return _chromadb


class FaceDatabase:
    """
    ChromaDB-backed vector store for criminal face embeddings.

    Usage:
        db = FaceDatabase("./chroma_db")
        db.enroll("CRIM001", embedding, {"name": "John Doe", "case": "2024-001"})
        results = db.search(query_embedding, top_k=5)
    """

    COLLECTION_NAME = "criminal_faces"

    def __init__(self, persist_dir: str = "./data/chromadb"):
        """
        Initialize the face database.

        Args:
            persist_dir: Directory for ChromaDB persistent storage.
        """
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _ensure_initialized(self):
        """Lazy initialization of ChromaDB client and collection."""
        chromadb = _get_chromadb()
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir)

        if self._collection is not None:
            try:
                self._collection.count()
                return
            except Exception:
                self._collection = None

        console.print(
            f"[cyan][*] Initializing ChromaDB[/cyan] at {self.persist_dir}"
        )

        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        count = self._collection.count()
        console.print(
            f"[green][+] Database ready[/green] — "
            f"{count} criminal face(s) enrolled"
        )

    def _get_collection(self):
        """Safely returns the collection, reconnecting if invalidated."""
        self._ensure_initialized()
        try:
            self._collection.count()
            return self._collection
        except Exception:
            self._collection = None
            self._ensure_initialized()
            return self._collection

    def enroll(
        self,
        criminal_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        face_image: Optional[np.ndarray] = None,
        thumbnail: Optional[str] = None,
    ) -> bool:
        """
        Enroll a new criminal face into the database.

        Args:
            criminal_id: Unique identifier for the criminal (e.g., "CRIM001").
            embedding: 512-dim normalized ArcFace embedding.
            metadata: Optional metadata dict (name, case_number, notes, etc.).
                      Values must be strings, ints, floats, or bools (ChromaDB limitation).
            face_image: Optional BGR face crop to store as base64 thumbnail.
            thumbnail: Optional pre-encoded base64 thumbnail string.

        Returns:
            True if enrolled successfully.
        """
        self._ensure_initialized()

        # Build metadata
        meta = {
            "criminal_id": criminal_id,
            "enrolled_at": datetime.datetime.now().isoformat(),
        }
        if metadata:
            # ChromaDB only supports str/int/float/bool metadata values
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)

        # Store pre-encoded thumbnail or generate from face_image
        if thumbnail:
            meta["thumbnail_b64"] = thumbnail
        elif face_image is not None:
            try:
                # Resize to HD studio thumbnail (256x256) with 95% JPEG quality for crisp UI display
                h, w = face_image.shape[:2]
                max_dim = max(h, w)
                if max_dim > 256:
                    scale = 256.0 / max_dim
                    thumb = cv2.resize(face_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                else:
                    thumb = face_image
                _, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 95])
                meta["thumbnail_b64"] = base64.b64encode(buffer).decode("utf-8")
            except Exception:
                pass  # Skip thumbnail on error

        # Use upsert to handle re-enrollment
        self._get_collection().upsert(
            ids=[criminal_id],
            embeddings=[embedding.tolist()],
            metadatas=[meta],
        )

        return True

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the criminal database for faces similar to the query.

        Args:
            query_embedding: 512-dim normalized ArcFace embedding to search for.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity threshold (0-1).
                       Results below this are filtered out.
                       Note: ChromaDB cosine distance = 1 - cosine_similarity.
            metadata_filter: Optional ChromaDB where filter
                             (e.g., {"name": "John"}).

        Returns:
            List of match dicts, each containing:
                - criminal_id: str
                - similarity: float (0-1, higher = more similar)
                - distance: float (ChromaDB cosine distance)
                - metadata: dict (name, case_number, etc.)
        """
        query_kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["metadatas", "distances"],
        }
        if metadata_filter:
            query_kwargs["where"] = metadata_filter

        results = self._get_collection().query(**query_kwargs)

        matches = []
        if results and results["ids"] and results["ids"][0]:
            for i, cid in enumerate(results["ids"][0]):
                # ChromaDB cosine distance = 1 - cosine_similarity
                distance = results["distances"][0][i]
                similarity = 1.0 - distance

                # Apply threshold filter
                if threshold is not None and similarity < threshold:
                    continue

                meta = results["metadatas"][0][i] if results["metadatas"] else {}

                matches.append({
                    "criminal_id": cid,
                    "similarity": round(similarity, 4),
                    "distance": round(distance, 4),
                    "metadata": meta,
                })

        return matches

    def delete(self, criminal_id: str) -> bool:
        """
        Remove a criminal from the database.

        Args:
            criminal_id: ID of the criminal to remove.

        Returns:
            True if deleted.
        """
        self._get_collection().delete(ids=[criminal_id])
        return True

    def get(self, criminal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific criminal's record.

        Args:
            criminal_id: ID of the criminal.

        Returns:
            Dict with criminal data, or None if not found.
        """
        result = self._get_collection().get(
            ids=[criminal_id],
            include=["metadatas", "embeddings"],
        )
        if result and result["ids"]:
            return {
                "criminal_id": result["ids"][0],
                "metadata": result["metadatas"][0] if result["metadatas"] else {},
                "has_embedding": result["embeddings"] is not None and len(result["embeddings"]) > 0,
            }
        return None

    def count(self) -> int:
        """Get total number of enrolled criminals."""
        return self._get_collection().count()

    def list_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all enrolled criminals.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of criminal records with metadata.
        """
        return self.get_all(include_thumbnails=False, limit=limit)

    def get_all(self, include_thumbnails: bool = True, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all enrolled criminal records.

        Args:
            include_thumbnails: Whether to retain base64 thumbnails in records.
            limit: Maximum number of records to return.

        Returns:
            List of dicts with 'id', 'name', 'criminal_id', 'metadata', 'thumbnail'.
        """
        kwargs = {"include": ["metadatas"]}
        if limit:
            kwargs["limit"] = limit

        result = self._get_collection().get(**kwargs)
        records = []
        if result and result["ids"]:
            for i, cid in enumerate(result["ids"]):
                meta = dict(result["metadatas"][i]) if result["metadatas"] else {}
                name = meta.get("name", cid)
                thumb = meta.get("thumbnail_b64") or meta.get("thumbnail", "")

                rec = {
                    "id": cid,
                    "criminal_id": cid,
                    "name": name,
                    "metadata": meta,
                    "thumbnail": thumb if include_thumbnails else None,
                }
                records.append(rec)
        return records

    def clear(self) -> int:
        """
        Delete ALL records from the database.

        Returns:
            Number of records deleted.
        """
        count = self._get_collection().count()
        # Recreate the collection to clear it
        self._ensure_initialized()
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return count
