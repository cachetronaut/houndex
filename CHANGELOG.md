# Changelog

## Unreleased

- Consolidated TypeScript and Python implementations into one primitive repository.

## Python 0.2.0

- Added `houndex_storage_sqlite.SqliteStorageAdapter`, a durable single-file SQLite `StorageAdapter` using only the standard library, with the same behavior as the local adapter.
- Added `houndex_pipeline.verify_trace`, which resolves an `OutputEnvelope`'s cited claims against a store and returns a `CitationVerdict` plus the claim ids that do not resolve.
