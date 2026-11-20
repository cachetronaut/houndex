# houndex-storage-sqlite (Python)

Durable, zero-service implementation of the `houndex-core` `StorageAdapter`
protocol on a single SQLite file. It uses only the Python standard library, so it
needs no database server and no network, and the evidence store stays in one file
on the machine that runs it.

## What it provides

- A `SqliteStorageAdapter` class that implements every `StorageAdapter` method:
  tenants, runs, claims, sources, edges, curation suggestions, knowledge-base
  entries, and verification overrides.
- The same behavior as `houndex-storage-local`: idempotent content-addressed
  upserts, insertion-ordered results, and cosine vector search over claim
  embeddings, but persisted across runs.
- Tenant partitioning by construction: every row carries its tenant and every
  query filters on it. A write whose record belongs to a different tenant than the
  write itself raises `ValueError`.
- A new database file is created readable and writable by its owner only. It uses
  a rollback journal rather than a write-ahead log, so every committed write is in
  that one file.
- To copy a store that may be in use, use SQLite's online backup API, for example
  `sqlite3.connect(path).backup(sqlite3.connect(copy_path))`. Copy the raw file only
  when no connection has it open.
- Requires SQLite's JSON functions, built into SQLite since 3.38 and present in the
  standard `sqlite3` module of current Python releases.
- Calls run on a worker thread, so disk I/O never blocks the event loop. Run status
  changes are single atomic statements, safe alongside other connections.

## Usage

```python
from houndex_storage_sqlite import SqliteStorageAdapter
from houndex_core.storage import EnsureTenantInput

adapter = SqliteStorageAdapter("evidence.sqlite3")  # or ":memory:"
await adapter.ensure_tenant(EnsureTenantInput(tenant=tenant))
...
adapter.close()
```
