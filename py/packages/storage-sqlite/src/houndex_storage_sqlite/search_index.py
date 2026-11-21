"""Search over documents in one SQLite file (or in memory), using only the standard library.

``SqliteSearchIndex`` keeps documents with named text fields and an optional vector:

- ``lexical``: FTS5 full-text search. Every query word matches as a prefix (``"word"*``, joined
  with ``OR``), results are ranked by bm25 over all fields, and ties break by document id, or by
  the newest ``order`` first.
- ``similar``: vector search over the stored vectors, best first, ties by id.

Two helpers work on documents you already hold, without a store:

- ``fuzzy_matches``: typo-tolerant matching of a query against names, by similarity ratio.
- ``rank_by_similarity``: rank (id, vector) pairs against a query vector.

Combine any of these with ``houndex_core.reciprocal_rank_fusion``.
"""

from __future__ import annotations

import math
import operator
import os
import re
import sqlite3
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal

_TOKENS = re.compile(r"\w+", re.UNICODE)
_FIELD = re.compile(r"[a-z][a-z0-9_]*")
# The index's own columns, plus names FTS5 reserves or uses for hidden columns.
_RESERVED = {"document_id", "document_order", "documents", "rank", "rowid", "oid", "_rowid_"}

Measure = Literal["dot", "cosine"]


@dataclass(frozen=True)
class SearchDocument:
    """A document to index: its id, its text fields by name, an ordering key for "newest first"
    ties, and an optional vector."""

    id: str
    fields: dict[str, str] = field(default_factory=dict)
    order: int = 0
    vector: Sequence[float] | None = None


def normalize(text: str) -> str:
    """Lower-case words joined by single spaces; punctuation is dropped."""
    return " ".join(_TOKENS.findall(text.lower()))


def match_expression(query: str) -> str | None:
    """The FTS5 query for ``query``: each word as a quoted prefix, joined with ``OR``; ``None``
    when the query has no words."""
    tokens = _TOKENS.findall(query.lower())
    return " OR ".join(f'"{token}"*' for token in tokens) if tokens else None


def fuzzy_matches(
    query: str, names: Iterable[tuple[str, str]], *, threshold: float, minimum_length: int = 1
) -> list[str]:
    """Ids whose name is at least ``threshold`` similar to the query (difflib ratio over the
    normalized texts), most similar first, ties by id. A normalized query shorter than
    ``minimum_length`` matches nothing."""
    normalized = normalize(query)
    if len(normalized) < minimum_length:
        return []
    scored = [
        (SequenceMatcher(None, normalized, normalize(name)).ratio(), identifier)
        for identifier, name in names
    ]
    return [
        identifier
        for score, identifier in sorted(scored, key=lambda item: (-item[0], item[1]))
        if score >= threshold
    ]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    # math.sumprod (Python 3.12+) is implemented in C; older versions use the plain loop.
    sumprod = getattr(math, "sumprod", None)
    return sumprod(left, right) if sumprod else sum(map(operator.mul, left, right))


def _similarity(query: Sequence[float], vector: Sequence[float], measure: Measure) -> float:
    if measure == "dot":
        return _dot(query, vector)
    norms = math.hypot(*query) * math.hypot(*vector)
    return _dot(query, vector) / norms if norms else 0.0


def rank_by_similarity(
    query_vector: Sequence[float],
    candidates: Iterable[tuple[str, Sequence[float]]],
    *,
    measure: Measure = "cosine",
) -> list[tuple[str, float]]:
    """``(id, score)`` for every candidate, best first, ties by id. ``dot`` suits vectors that
    are already unit length; ``cosine`` normalizes them."""
    if measure not in ("dot", "cosine"):
        raise ValueError("measure must be dot or cosine")
    scored = [
        (identifier, _similarity(query_vector, vector, measure))
        for identifier, vector in candidates
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


class SqliteSearchIndex:
    """Documents with named text fields, searchable by FTS5 and by vector.

    ``fields`` names the text fields, in order (lower-case identifiers). Re-adding a document id
    replaces it. One index is used by one thread at a time; pass ``check_same_thread=False``
    to hand it between threads that take turns.
    """

    def __init__(
        self,
        path: str = ":memory:",
        fields: Sequence[str] = ("text",),
        *,
        check_same_thread: bool = True,
    ) -> None:
        names = list(fields)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not _FIELD.fullmatch(name) or name in _RESERVED for name in names)
        ):
            raise ValueError("fields must be distinct lower-case identifiers")
        if path.startswith("file:"):
            raise ValueError("pass a filesystem path or ':memory:', not a SQLite URI")
        self.fields = tuple(names)
        if path != ":memory:":
            # A new index file is readable and writable by its owner only.
            os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
        self._db = sqlite3.connect(path, check_same_thread=check_same_thread)
        columns = ", ".join(f'"{name}"' for name in names)
        self._db.executescript(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5("
            f"document_order UNINDEXED, {columns});"
            "CREATE TABLE IF NOT EXISTS document_ids ("
            " id TEXT PRIMARY KEY, row INTEGER NOT NULL UNIQUE);"
            "CREATE TABLE IF NOT EXISTS document_vectors ("
            " id TEXT PRIMARY KEY, vector BLOB NOT NULL);"
        )
        # Whether any vector may be held, so documents without vectors skip that table.
        self._vectors = (
            self._db.execute("SELECT 1 FROM document_vectors LIMIT 1").fetchone() is not None
        )

    def add(self, documents: Iterable[SearchDocument]) -> None:
        """Index documents in one transaction, replacing any already held under the same id.
        A field the index doesn't have is an error, never silently dropped."""
        placeholders = ", ".join("?" for _ in range(len(self.fields) + 1))
        columns = ", ".join(f'"{name}"' for name in self.fields)
        known = set(self.fields)
        with self._db:
            for document in documents:
                unknown = set(document.fields) - known
                if unknown:
                    raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")
                self._remove(document.id)
                cursor = self._db.execute(
                    f"INSERT INTO documents (document_order, {columns}) VALUES ({placeholders})",
                    (document.order, *(document.fields.get(name, "") for name in self.fields)),
                )
                self._db.execute(
                    "INSERT INTO document_ids (id, row) VALUES (?, ?)",
                    (document.id, cursor.lastrowid),
                )
                if document.vector is not None:
                    self._vectors = True
                    self._db.execute(
                        "INSERT INTO document_vectors (id, vector) VALUES (?, ?)",
                        (document.id, array("f", document.vector).tobytes()),
                    )

    def remove(self, ids: Iterable[str]) -> None:
        """Forget the documents with these ids (a list of ids, never a single id string)."""
        if isinstance(ids, str):
            raise TypeError("pass a list of ids, not a single id string")
        with self._db:
            for identifier in ids:
                self._remove(identifier)

    def _remove(self, identifier: str) -> None:
        row = self._db.execute(
            "SELECT row FROM document_ids WHERE id = ?", (identifier,)
        ).fetchone()
        if row is not None:
            self._db.execute("DELETE FROM documents WHERE rowid = ?", (row[0],))
            self._db.execute("DELETE FROM document_ids WHERE id = ?", (identifier,))
        if self._vectors:
            self._db.execute("DELETE FROM document_vectors WHERE id = ?", (identifier,))

    def clear(self) -> None:
        with self._db:
            self._db.execute("DELETE FROM documents")
            self._db.execute("DELETE FROM document_ids")
            self._db.execute("DELETE FROM document_vectors")

    def lexical(
        self,
        query: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        ties: Literal["id", "newest"] = "id",
    ) -> list[str]:
        """Ids of documents sharing any query word prefix, best bm25 first."""
        expression = match_expression(query)
        if expression is None or (limit is not None and limit <= 0):
            return []
        tie = "ids.id" if ties == "id" else "documents.document_order DESC"
        rows = self._db.execute(
            "SELECT ids.id FROM documents JOIN document_ids AS ids ON ids.row = documents.rowid "
            f"WHERE documents MATCH ? ORDER BY bm25(documents), {tie} LIMIT ? OFFSET ?",
            (expression, -1 if limit is None else limit, max(offset, 0)),
        ).fetchall()
        return [row[0] for row in rows]

    def similar(
        self,
        query_vector: Sequence[float],
        *,
        limit: int | None = None,
        minimum: float | None = None,
        measure: Measure = "cosine",
    ) -> list[tuple[str, float]]:
        """``(id, score)`` of documents with vectors, best first, at or above ``minimum``."""
        stored = [
            (identifier, array("f", blob))
            for identifier, blob in self._db.execute("SELECT id, vector FROM document_vectors")
        ]
        ranked = rank_by_similarity(query_vector, stored, measure=measure)
        kept = [item for item in ranked if minimum is None or item[1] >= minimum]
        return kept if limit is None else kept[: max(limit, 0)]

    def close(self) -> None:
        self._db.close()
