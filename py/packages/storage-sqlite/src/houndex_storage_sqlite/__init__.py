"""houndex-storage-sqlite — durable single-file SQLite StorageAdapter and search index."""

from __future__ import annotations

from .search_index import (
    SearchDocument,
    SqliteSearchIndex,
    fuzzy_matches,
    match_expression,
    normalize,
    rank_by_similarity,
)
from .sqlite_storage_adapter import SqliteStorageAdapter

__all__ = [
    "SearchDocument",
    "SqliteSearchIndex",
    "SqliteStorageAdapter",
    "fuzzy_matches",
    "match_expression",
    "normalize",
    "rank_by_similarity",
]
