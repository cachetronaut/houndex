# houndex

Python implementation of Houndex.

For product-level context, shared contracts, and cross-language repository information, see the public repository: https://github.com/cachetronaut/houndex.

## Install

```sh
pip install houndex
```

## Import

```python
import houndex
```

## Evidence storage and citation checks

- `houndex_storage_sqlite.SqliteStorageAdapter` keeps the evidence store in one SQLite file,
  using only the standard library (`":memory:"` for an in-process store).
- `houndex_pipeline.verify_trace` resolves every claim an `OutputEnvelope` cites against a
  store and returns a `TraceVerification`: its `verdict` is a `CitationVerdict` (green when all
  resolve, yellow when some do, red when none do or nothing is cited), `resolved` holds the
  claims that resolved and `unresolved` the cited ids that did not.

```python
from houndex_pipeline import verify_trace
from houndex_storage_sqlite import SqliteStorageAdapter

store = SqliteStorageAdapter("evidence.sqlite3")
result = await verify_trace(store, tenant, envelope)
if not result.grounded:
    print(result.unresolved)
```

## Development

Run from `py/`:

```sh
uv sync --dev
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .
uv run --with ty ty check
uv run --with pytest --with pytest-asyncio python -m pytest
```

## License

MIT
