# Houndex

A storage-decoupled, last-mile RAG framework: typed output envelopes, tenant-aware evidence stores, citation verification, and pluggable storage adapters.

This repository contains the TypeScript and Python implementations for the Houndex primitive. The shared repository keeps the public contract, fixtures, and release history aligned across both languages.

## Packages

- npm: `houndex`
- PyPI: `houndex`

## Install

```sh
npm install houndex
pip install houndex
```

## Layout

- `ts/` - TypeScript implementation and npm package.
- `py/` - Python implementation and PyPI package.
- `fixtures/` - Shared conformance and parity fixtures when the primitive needs them.

## Development

Run TypeScript checks from `ts/`:

```sh
pnpm verify
```

Run Python checks from `py/`:

```sh
uv sync --dev
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .
uv run --with ty ty check
uv run --with pytest --with pytest-asyncio python -m pytest
```

## License

MIT
