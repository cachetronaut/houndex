"""Durable, zero-service ``StorageAdapter`` on a single SQLite file, using only the
standard library. Every method is scoped to ``input.tenant.tenant_id``; each row
carries its tenant and every query filters on it, so a read for one tenant can
never return another tenant's records. Behavior matches ``LocalStorageAdapter``:
idempotent content-addressed upserts, insertion-ordered results, and cosine
vector search over stored claim embeddings.
"""

from __future__ import annotations

import asyncio
import math
import os
import sqlite3
import sys
import threading
import time
import weakref
from array import array
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypeVar

from houndex_core.schemas import Claim, Run, Source, canonicalize_url, edge_idempotency_key
from houndex_core.storage import (
    ClaimSearchInput,
    CompleteRunInput,
    CreateRunInput,
    CurationSuggestionInput,
    DecideSuggestionInput,
    EnsureTenantInput,
    FailRunInput,
    GetClaimInput,
    KbEntryRecord,
    ListKbEntriesInput,
    UpsertClaimInput,
    UpsertEdgeInput,
    UpsertKbEntryInput,
    UpsertResult,
    UpsertSourceInput,
    VerificationOverrideInput,
)

_T = TypeVar("_T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (tenant_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS runs (
    tenant_id TEXT NOT NULL, run_id TEXT NOT NULL, record TEXT NOT NULL,
    PRIMARY KEY (tenant_id, run_id));
CREATE TABLE IF NOT EXISTS sources (
    tenant_id TEXT NOT NULL, url TEXT NOT NULL, record TEXT NOT NULL,
    PRIMARY KEY (tenant_id, url));
CREATE TABLE IF NOT EXISTS claims (
    tenant_id TEXT NOT NULL, claim_id TEXT NOT NULL, subject TEXT NOT NULL,
    category TEXT NOT NULL, record TEXT NOT NULL, embedding BLOB,
    PRIMARY KEY (tenant_id, claim_id));
CREATE INDEX IF NOT EXISTS claims_by_subject ON claims (tenant_id, subject);
CREATE TABLE IF NOT EXISTS edges (
    tenant_id TEXT NOT NULL, edge_key TEXT NOT NULL, PRIMARY KEY (tenant_id, edge_key));
CREATE TABLE IF NOT EXISTS suggestions (
    tenant_id TEXT NOT NULL, suggestion_id TEXT NOT NULL, claim TEXT NOT NULL,
    status TEXT NOT NULL, rationale TEXT, PRIMARY KEY (tenant_id, suggestion_id));
CREATE TABLE IF NOT EXISTS kb_entries (
    tenant_id TEXT NOT NULL, entry_id TEXT NOT NULL, record TEXT NOT NULL,
    subject TEXT NOT NULL, category TEXT NOT NULL, PRIMARY KEY (tenant_id, entry_id));
CREATE TABLE IF NOT EXISTS overrides (
    tenant_id TEXT NOT NULL, claim_id TEXT NOT NULL, verdict TEXT NOT NULL,
    reason TEXT NOT NULL);
"""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _pack(vector: Sequence[float] | None) -> bytes | None:
    """Doubles in little-endian order, so a store reads the same on any machine."""
    if vector is None:
        return None
    values = array("d", vector)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _unpack(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    values = array("d")
    values.frombytes(blob)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def _owned(tenant_id: str, record_tenant_id: str, kind: str) -> None:
    """A write stores only records that belong to the tenant it is made for."""
    if record_tenant_id != tenant_id:
        raise ValueError(f"{kind} belongs to a different tenant than the write")


def _cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    length = min(len(vector_a), len(vector_b))
    dot = sum(vector_a[index] * vector_b[index] for index in range(length))
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SqliteStorageAdapter:
    """A tenant-partitioned evidence store in one SQLite file.

    ``path`` is a file path, or ``":memory:"`` for a private in-process database.
    A new file is created readable and writable by its owner only, and every
    committed write is in that one file (no write-ahead log); copy a store in use
    with SQLite's online backup API. Calls run on a worker thread so the
    event loop is never blocked by disk I/O; one connection is shared and
    serialized by a lock. A write whose record belongs to another tenant than the
    write's own raises ``ValueError``. Call ``close`` when done.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        location = os.fspath(path)
        if location != ":memory:" and not Path(location).exists():
            Path(location).parent.mkdir(parents=True, exist_ok=True)
            os.close(os.open(location, os.O_CREAT | os.O_WRONLY, 0o600))
        self._connection = sqlite3.connect(location, check_same_thread=False)
        self._lock = threading.Lock()
        # Runs handed out by ``create_run``, so a status change updates the caller's own
        # handle as the reference adapter does.
        self._runs: weakref.WeakValueDictionary[tuple[str, str], Run] = (
            weakref.WeakValueDictionary()
        )
        with self._lock:
            # A rollback journal, never a write-ahead log: a commit lands in the one file.
            self._connection.execute("PRAGMA journal_mode=DELETE")
            with self._connection:
                self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    async def _run(
        self,
        operation: Callable[[sqlite3.Connection], _T],
        after: Callable[[_T], None] | None = None,
    ) -> _T:
        """Run ``operation`` in one transaction on the worker; ``after`` runs right after the
        commit, in the same locked section, so it happens even if the caller is cancelled."""

        def locked() -> _T:
            with self._lock:
                with self._connection:
                    result = operation(self._connection)
                if after is not None:
                    after(result)
                return result

        return await asyncio.to_thread(locked)

    @staticmethod
    def _ensure(connection: sqlite3.Connection, tenant_id: str) -> None:
        connection.execute("INSERT OR IGNORE INTO tenants (tenant_id) VALUES (?)", (tenant_id,))

    # ── tenant + run lifecycle ─────────────────────────────────────────

    async def ensure_tenant(self, input: EnsureTenantInput) -> None:
        tenant_id = input.tenant.tenant_id
        await self._run(lambda db: self._ensure(db, tenant_id))

    async def create_run(self, input: CreateRunInput) -> Run:
        run = Run(
            tenant_id=input.tenant.tenant_id,
            run_id=input.run_id,
            subject=input.subject,
            signal=input.signal,
            status="running",
            created_at=_now_ms(),
        )
        key, record = (run.tenant_id, run.run_id), run.model_dump_json()

        def write(db: sqlite3.Connection) -> None:
            self._ensure(db, key[0])
            db.execute(
                "INSERT INTO runs (tenant_id, run_id, record) VALUES (?, ?, ?) "
                "ON CONFLICT (tenant_id, run_id) DO UPDATE SET record = excluded.record",
                (*key, record),
            )

        def register(_: None) -> None:
            self._runs[key] = run

        await self._run(write, register)
        return run

    async def complete_run(self, input: CompleteRunInput) -> None:
        await self._set_run_status(input.tenant.tenant_id, input.run_id, "complete")

    async def fail_run(self, input: FailRunInput) -> None:
        await self._set_run_status(input.tenant.tenant_id, input.run_id, "failed")

    async def _set_run_status(
        self, tenant_id: str, run_id: str, status: Literal["complete", "failed"]
    ) -> None:
        # One statement, so a run replaced meanwhile keeps everything but its new status.
        def write(db: sqlite3.Connection) -> int:
            return db.execute(
                "UPDATE runs SET record = json_set(record, '$.status', ?) "
                "WHERE tenant_id = ? AND run_id = ?",
                (status, tenant_id, run_id),
            ).rowcount

        def follow(changed: int) -> None:
            handle = self._runs.get((tenant_id, run_id))
            if changed and handle is not None:
                handle.status = status

        await self._run(write, follow)

    # ── evidence store: sources, claims, edges ─────────────────────────
    # Every write takes its tenant and record as they are when called, checks that the
    # record belongs to that tenant, and stores exactly that snapshot.

    async def upsert_source(self, input: UpsertSourceInput) -> Source:
        tenant_id = input.tenant.tenant_id
        source = input.source.model_copy(update={"url": canonicalize_url(input.source.url)})
        _owned(tenant_id, source.tenant_id, "source")
        record = source.model_dump_json()

        def write(db: sqlite3.Connection) -> None:
            self._ensure(db, tenant_id)
            db.execute(
                "INSERT INTO sources (tenant_id, url, record) VALUES (?, ?, ?) "
                "ON CONFLICT (tenant_id, url) DO UPDATE SET record = excluded.record",
                (tenant_id, source.url, record),
            )

        await self._run(write)
        return Source.model_validate_json(record)

    async def upsert_claim(self, input: UpsertClaimInput) -> UpsertResult:
        tenant_id = input.tenant.tenant_id
        claim = input.claim.model_copy(deep=True)
        _owned(tenant_id, claim.tenant_id, "claim")
        row = (
            tenant_id,
            claim.claim_id,
            claim.subject,
            claim.category,
            claim.model_dump_json(),
            _pack(input.embedding),
        )

        def write(db: sqlite3.Connection) -> bool:
            self._ensure(db, tenant_id)
            existing = db.execute(
                "SELECT 1 FROM claims WHERE tenant_id = ? AND claim_id = ?", (tenant_id, row[1])
            ).fetchone()
            # Updating in place keeps the claim's original insertion order.
            db.execute(
                "INSERT INTO claims (tenant_id, claim_id, subject, category, record, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (tenant_id, claim_id) DO UPDATE SET "
                "subject = excluded.subject, category = excluded.category, "
                "record = excluded.record, embedding = excluded.embedding",
                row,
            )
            return existing is None

        created = await self._run(write)
        return UpsertResult(id=row[1], created=created)

    async def upsert_edge(self, input: UpsertEdgeInput) -> UpsertResult:
        tenant_id = input.tenant.tenant_id
        edge = input.edge.model_copy(deep=True)
        _owned(tenant_id, edge.tenant_id, "edge")
        key = edge_idempotency_key(src_id=edge.src_id, dst_id=edge.dst_id, kind=edge.kind)

        def write(db: sqlite3.Connection) -> bool:
            self._ensure(db, tenant_id)
            cursor = db.execute(
                "INSERT OR IGNORE INTO edges (tenant_id, edge_key) VALUES (?, ?)", (tenant_id, key)
            )
            return cursor.rowcount == 1

        created = await self._run(write)
        return UpsertResult(id=key, created=created)

    @staticmethod
    def _limit(limit: int | None) -> int | None:
        return None if limit is None else max(limit, 0)

    async def search_claims(self, input: ClaimSearchInput) -> list[Claim]:
        tenant_id = input.tenant.tenant_id
        target = list(input.query_vector) if input.query_vector is not None else None
        limit = self._limit(input.limit)
        query = [
            f"SELECT record, {'embedding' if target is not None else 'NULL'} "
            "FROM claims WHERE tenant_id = ?"
        ]
        parameters: list[str] = [tenant_id]
        if input.subject is not None:
            query.append("AND subject = ?")
            parameters.append(input.subject)
        if input.category is not None:
            query.append("AND category = ?")
            parameters.append(input.category)
        query.append("ORDER BY rowid")

        def read(db: sqlite3.Connection) -> list[tuple[Claim, list[float] | None]]:
            # Only records that belong to this tenant, whatever else the file holds. Without a
            # query vector the order is insertion order, so the scan stops at the limit.
            found: list[tuple[Claim, list[float] | None]] = []
            for record, blob in db.execute(" ".join(query), parameters):
                if target is None and limit is not None and len(found) >= limit:
                    break
                claim = Claim.model_validate_json(record)
                if claim.tenant_id == tenant_id:
                    found.append((claim, _unpack(blob)))
            return found

        matches = await self._run(read)

        if target is not None:

            def score(entry: tuple[Claim, list[float] | None]) -> float:
                return _cosine_similarity(target, entry[1]) if entry[1] else -1.0

            matches.sort(key=score, reverse=True)

        claims = [claim for claim, _ in matches]
        return claims[:limit] if limit is not None else claims

    async def get_claim(self, input: GetClaimInput) -> Claim | None:
        tenant_id, claim_id = input.tenant.tenant_id, input.claim_id
        row = await self._run(
            lambda db: db.execute(
                "SELECT record FROM claims WHERE tenant_id = ? AND claim_id = ?",
                (tenant_id, claim_id),
            ).fetchone()
        )
        if row is None:
            return None
        claim = Claim.model_validate_json(row[0])
        return claim if claim.tenant_id == tenant_id and claim.claim_id == claim_id else None

    # ── human curation ─────────────────────────────────────────────────

    async def create_curation_suggestion(self, input: CurationSuggestionInput) -> None:
        tenant_id, suggestion_id, rationale = (
            input.tenant.tenant_id,
            input.suggestion_id,
            input.rationale,
        )
        claim = input.claim.model_copy(deep=True)
        _owned(tenant_id, claim.tenant_id, "claim")
        record = claim.model_dump_json()

        def write(db: sqlite3.Connection) -> None:
            self._ensure(db, tenant_id)
            db.execute(
                "INSERT INTO suggestions (tenant_id, suggestion_id, claim, status, rationale) "
                "VALUES (?, ?, ?, 'pending', ?) ON CONFLICT (tenant_id, suggestion_id) DO UPDATE "
                "SET claim = excluded.claim, status = 'pending', rationale = excluded.rationale",
                (tenant_id, suggestion_id, record, rationale),
            )

        await self._run(write)

    async def decide_suggestion(self, input: DecideSuggestionInput) -> None:
        tenant_id, suggestion_id, status = (
            input.tenant.tenant_id,
            input.suggestion_id,
            input.status,
        )
        edited = None
        if input.edited_claim is not None:
            claim = input.edited_claim.model_copy(deep=True)
            _owned(tenant_id, claim.tenant_id, "claim")
            edited = claim.model_dump_json()

        def write(db: sqlite3.Connection) -> None:
            if edited is None:
                db.execute(
                    "UPDATE suggestions SET status = ? WHERE tenant_id = ? AND suggestion_id = ?",
                    (status, tenant_id, suggestion_id),
                )
            else:
                db.execute(
                    "UPDATE suggestions SET status = ?, claim = ? "
                    "WHERE tenant_id = ? AND suggestion_id = ?",
                    (status, edited, tenant_id, suggestion_id),
                )

        await self._run(write)

    # ── knowledge base ─────────────────────────────────────────────────

    async def upsert_kb_entry(self, input: UpsertKbEntryInput) -> None:
        tenant_id = input.tenant.tenant_id
        claim = input.claim.model_copy(deep=True)
        _owned(tenant_id, claim.tenant_id, "claim")
        entry = KbEntryRecord(
            tenant_id=tenant_id,
            entry_id=input.entry_id,
            claim=claim,
            status="rejected" if input.action == "rejected" else "approved",
            updated_at=_now_ms(),
        )
        row = (tenant_id, entry.entry_id, entry.model_dump_json(), claim.subject, claim.category)

        def write(db: sqlite3.Connection) -> None:
            self._ensure(db, tenant_id)
            db.execute(
                "INSERT INTO kb_entries (tenant_id, entry_id, record, subject, category) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (tenant_id, entry_id) DO UPDATE SET "
                "record = excluded.record, subject = excluded.subject, "
                "category = excluded.category",
                row,
            )

        await self._run(write)

    async def list_kb_entries(self, input: ListKbEntriesInput) -> list[KbEntryRecord]:
        tenant_id = input.tenant.tenant_id
        query = ["SELECT record FROM kb_entries WHERE tenant_id = ?"]
        parameters: list[str] = [tenant_id]
        if input.subject is not None:
            query.append("AND subject = ?")
            parameters.append(input.subject)
        if input.category is not None:
            query.append("AND category = ?")
            parameters.append(input.category)
        query.append("ORDER BY rowid")
        rows = await self._run(lambda db: db.execute(" ".join(query), parameters).fetchall())
        entries = (KbEntryRecord.model_validate_json(row[0]) for row in rows)
        return [
            entry
            for entry in entries
            if entry.tenant_id == tenant_id and entry.claim.tenant_id == tenant_id
        ]

    # ── verification overrides ─────────────────────────────────────────

    async def record_verification_override(self, input: VerificationOverrideInput) -> None:
        row = (input.tenant.tenant_id, input.claim_id, input.verdict, input.reason)

        def write(db: sqlite3.Connection) -> None:
            self._ensure(db, row[0])
            db.execute(
                "INSERT INTO overrides (tenant_id, claim_id, verdict, reason) VALUES (?, ?, ?, ?)",
                row,
            )

        await self._run(write)


__all__ = ["SqliteStorageAdapter"]
