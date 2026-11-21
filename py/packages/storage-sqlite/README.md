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

## Search

`SqliteSearchIndex` keeps documents (an id, named text fields, an ordering key and an optional
vector) in a SQLite file or in memory and searches them:

- `lexical(query, limit=, offset=, ties=)`: FTS5 full-text search. Each query word matches as a
  prefix, results rank by bm25 over all fields, and ties break by id or by newest `order` first.
- `similar(query_vector, limit=, minimum=, measure=)`: vector search, by cosine or dot product.

`fuzzy_matches` (typo-tolerant name matching) and `rank_by_similarity` work on documents you
already hold. Merge any ranked lists with `houndex_core.reciprocal_rank_fusion`, which keeps each
list's rank for every hit.

```python
from houndex_core import reciprocal_rank_fusion
from houndex_storage_sqlite import SearchDocument, SqliteSearchIndex

index = SqliteSearchIndex(fields=("title", "body"))
index.add([SearchDocument("a", {"title": "Charmander", "body": "fire lizard"}, vector=[1, 0])])
fused = reciprocal_rank_fusion({
    "lexical": index.lexical("fire"),
    "vector": [identifier for identifier, _ in index.similar([1, 0])],
})
```

## Usage

```python
from houndex_storage_sqlite import SqliteStorageAdapter
from houndex_core.storage import EnsureTenantInput

adapter = SqliteStorageAdapter("evidence.sqlite3")  # or ":memory:"
await adapter.ensure_tenant(EnsureTenantInput(tenant=tenant))
...
adapter.close()
```
