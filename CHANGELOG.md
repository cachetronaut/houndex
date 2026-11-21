# Changelog

## Unreleased

- Consolidated TypeScript and Python implementations into one primitive repository.

## Python 0.3.0

- Added `houndex_storage_sqlite.SqliteSearchIndex`: FTS5 lexical search (word prefixes, bm25 ranking, ties by id or newest first) and vector search (cosine or dot product) over documents in a SQLite file or in memory, using only the standard library.
- Added `houndex_storage_sqlite.fuzzy_matches` for typo-tolerant name matching and `rank_by_similarity` for ranking vectors you already hold.
- Added `houndex_core.reciprocal_rank_fusion`, which merges ranked lists and records each list's rank for every hit.

## Python 0.2.0

- Added `houndex_storage_sqlite.SqliteStorageAdapter`, a durable single-file SQLite `StorageAdapter` using only the standard library, with the same behavior as the local adapter.
- Added `houndex_pipeline.verify_trace`, which resolves an `OutputEnvelope`'s cited claims against a store and returns a `CitationVerdict` plus the claim ids that do not resolve.
