from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import stat
from pathlib import Path
from typing import Any

import pytest
from houndex_core import compute_claim_id, tenant_primary, tenant_secondary
from houndex_core.schemas import Claim, Edge, Run, Source, canonicalize_url
from houndex_core.storage import (
    ClaimSearchInput,
    CompleteRunInput,
    CreateRunInput,
    CurationSuggestionInput,
    DecideSuggestionInput,
    FailRunInput,
    GetClaimInput,
    ListKbEntriesInput,
    UpsertClaimInput,
    UpsertEdgeInput,
    UpsertKbEntryInput,
    UpsertSourceInput,
    VerificationOverrideInput,
)
from houndex_storage_local import LocalStorageAdapter
from houndex_storage_sqlite import SqliteStorageAdapter


def _make_claim(**overrides: Any) -> Claim:
    subject = overrides.pop("subject", "Acme")
    claim_text = overrides.pop("claim_text", "Has an audit log")
    source_url = overrides.pop("source_url", "https://example.com/security")
    tenant_id = overrides.pop("tenant_id", "primary")
    data: dict[str, Any] = {
        "tenant_id": tenant_id,
        "claim_id": compute_claim_id(
            tenant_id=tenant_id, subject=subject, claim_text=claim_text, source_url=source_url
        ),
        "subject": subject,
        "category": "security",
        "polarity": "positive",
        "scope": "global",
        "claim_text": claim_text,
        "evidence_text": "evidence",
        "confidence": "stated",
        "source_url": source_url,
        "source_tier": "tier_2",
        "extracted_at": 1_700_000_000_000,
    }
    data.update(overrides)
    return Claim.model_validate(data)


@pytest.fixture(params=["memory", "file"])
def adapter(request: pytest.FixtureRequest, tmp_path: Path):
    store = SqliteStorageAdapter(":memory:" if request.param == "memory" else tmp_path / "e.db")
    yield store
    store.close()


def test_upsert_is_idempotent(adapter: SqliteStorageAdapter) -> None:
    async def run() -> None:
        tenant = tenant_primary()
        claim = _make_claim()
        assert (await adapter.upsert_claim(UpsertClaimInput(tenant=tenant, claim=claim))).created
        assert not (
            await adapter.upsert_claim(UpsertClaimInput(tenant=tenant, claim=claim))
        ).created

    asyncio.run(run())


def test_get_and_search_filter(adapter: SqliteStorageAdapter) -> None:
    async def run() -> None:
        tenant = tenant_primary()
        await adapter.upsert_claim(UpsertClaimInput(tenant=tenant, claim=_make_claim()))
        await adapter.upsert_claim(
            UpsertClaimInput(
                tenant=tenant, claim=_make_claim(subject="Globex", claim_text="different")
            )
        )
        acme = await adapter.search_claims(ClaimSearchInput(tenant=tenant, subject="Acme"))
        assert [claim.subject for claim in acme] == ["Acme"]
        security = await adapter.search_claims(
            ClaimSearchInput(tenant=tenant, category="security", limit=1)
        )
        assert len(security) == 1

        first = _make_claim()
        got = await adapter.get_claim(GetClaimInput(tenant=tenant, claim_id=first.claim_id))
        assert got == first
        missing = await adapter.get_claim(GetClaimInput(tenant=tenant, claim_id="0123456789abcdef"))
        assert missing is None

    asyncio.run(run())


def test_vector_search_orders_by_similarity(adapter: SqliteStorageAdapter) -> None:
    async def run() -> None:
        tenant = tenant_primary()
        near = _make_claim(claim_text="near", source_url="https://example.com/near")
        far = _make_claim(claim_text="far", source_url="https://example.com/far")
        plain = _make_claim(claim_text="plain", source_url="https://example.com/plain")
        await adapter.upsert_claim(UpsertClaimInput(tenant=tenant, claim=plain))
        await adapter.upsert_claim(UpsertClaimInput(tenant=tenant, claim=far, embedding=[0.0, 1.0]))
        await adapter.upsert_claim(
            UpsertClaimInput(tenant=tenant, claim=near, embedding=[1.0, 0.0])
        )
        results = await adapter.search_claims(
            ClaimSearchInput(tenant=tenant, query_vector=[1.0, 0.0])
        )
        assert [claim.claim_id for claim in results] == [
            near.claim_id,
            far.claim_id,
            plain.claim_id,
        ]

    asyncio.run(run())


def test_matches_local_adapter_ordering(adapter: SqliteStorageAdapter) -> None:
    async def run() -> None:
        tenant = tenant_primary()
        local = LocalStorageAdapter()
        claims = [
            _make_claim(claim_text=f"claim {index}", source_url=f"https://example.com/{index}")
            for index in range(5)
        ]
        vectors = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.6, 0.8], [1.0, 0.0]]
        for store in (adapter, local):
            for claim, vector in zip(claims, vectors, strict=True):
                await store.upsert_claim(
                    UpsertClaimInput(tenant=tenant, claim=claim, embedding=vector)
                )
            # A repeat write keeps the claim's original position.
            await store.upsert_claim(
                UpsertClaimInput(tenant=tenant, claim=claims[0], embedding=vectors[0])
            )
        for search in (
            ClaimSearchInput(tenant=tenant),
            ClaimSearchInput(tenant=tenant, query_vector=[0.6, 0.8], limit=3),
        ):
            assert await adapter.search_claims(search) == await local.search_claims(search)

    asyncio.run(run())


def test_tenant_isolation(adapter: SqliteStorageAdapter) -> None:
    async def run() -> None:
        primary = tenant_primary()
        secondary = tenant_secondary()
        claim = _make_claim(tenant_id="primary")
        await adapter.upsert_claim(UpsertClaimInput(tenant=primary, claim=claim))

        assert (
            await adapter.get_claim(GetClaimInput(tenant=secondary, claim_id=claim.claim_id))
        ) is None
        assert len(await adapter.search_claims(ClaimSearchInput(tenant=secondary))) == 0
        assert len(await adapter.search_claims(ClaimSearchInput(tenant=primary))) == 1

    asyncio.run(run())


def test_runs_edges_curation_kb_overrides(adapter: SqliteStorageAdapter) -> None:
    async def run() -> None:
        tenant = tenant_primary()

        created_run = await adapter.create_run(
            CreateRunInput(tenant=tenant, run_id="r1", subject="Acme")
        )
        assert created_run.status == "running"
        await adapter.complete_run(CompleteRunInput(tenant=tenant, run_id="r1"))
        await adapter.fail_run(FailRunInput(tenant=tenant, run_id="missing"))

        edge = Edge(
            tenant_id="primary",
            src_id="claim:0000000000000001",
            dst_id="claim:0000000000000002",
            kind="reinforces",
        )
        assert (await adapter.upsert_edge(UpsertEdgeInput(tenant=tenant, edge=edge))).created
        assert not (await adapter.upsert_edge(UpsertEdgeInput(tenant=tenant, edge=edge))).created

        source = await adapter.upsert_source(
            UpsertSourceInput(
                tenant=tenant,
                source=Source(tenant_id="primary", url="https://Example.com/a/?x=1", fetched_at=1),
            )
        )
        assert source.url == canonicalize_url("https://Example.com/a/?x=1")

        claim = _make_claim()
        await adapter.create_curation_suggestion(
            CurationSuggestionInput(tenant=tenant, suggestion_id="s1", claim=claim)
        )
        await adapter.decide_suggestion(
            DecideSuggestionInput(tenant=tenant, suggestion_id="s1", status="approved")
        )
        await adapter.upsert_kb_entry(
            UpsertKbEntryInput(tenant=tenant, entry_id="e1", claim=claim, action="approved")
        )
        await adapter.upsert_kb_entry(
            UpsertKbEntryInput(tenant=tenant, entry_id="e2", claim=claim, action="rejected")
        )
        entries = await adapter.list_kb_entries(ListKbEntriesInput(tenant=tenant, subject="Acme"))
        assert [(entry.entry_id, entry.status) for entry in entries] == [
            ("e1", "approved"),
            ("e2", "rejected"),
        ]

        await adapter.record_verification_override(
            VerificationOverrideInput(
                tenant=tenant, claim_id="abcdef0123456789", verdict="green", reason="verified"
            )
        )

    asyncio.run(run())


def test_survives_reopen_and_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "evidence.sqlite3"
    tenant = tenant_primary()
    claim = _make_claim()

    async def write() -> None:
        store = SqliteStorageAdapter(path)
        await store.upsert_claim(UpsertClaimInput(tenant=tenant, claim=claim, embedding=[0.5]))
        await store.create_run(CreateRunInput(tenant=tenant, run_id="r1", subject="Acme"))
        await store.complete_run(CompleteRunInput(tenant=tenant, run_id="r1"))
        store.close()

    async def read() -> None:
        store = SqliteStorageAdapter(path)
        assert await store.get_claim(GetClaimInput(tenant=tenant, claim_id=claim.claim_id)) == claim
        found = await store.search_claims(ClaimSearchInput(tenant=tenant, query_vector=[1.0]))
        assert found == [claim]
        store.close()

    asyncio.run(write())
    asyncio.run(read())
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with sqlite3.connect(path) as connection:
        record = connection.execute("SELECT record FROM runs").fetchone()[0]
    assert '"status":"complete"' in record


def test_rejects_records_of_another_tenant(adapter: SqliteStorageAdapter) -> None:
    async def run() -> None:
        theirs = _make_claim(tenant_id="secondary")
        with pytest.raises(ValueError):
            await adapter.upsert_claim(UpsertClaimInput(tenant=tenant_primary(), claim=theirs))
        with pytest.raises(ValueError):
            await adapter.upsert_kb_entry(
                UpsertKbEntryInput(
                    tenant=tenant_primary(), entry_id="e1", claim=theirs, action="approved"
                )
            )
        edge = Edge(tenant_id="secondary", src_id="claim:a", dst_id="claim:b", kind="reinforces")
        with pytest.raises(ValueError):
            await adapter.upsert_edge(UpsertEdgeInput(tenant=tenant_primary(), edge=edge))
        assert await adapter.search_claims(ClaimSearchInput(tenant=tenant_primary())) == []

    asyncio.run(run())


def test_run_handle_follows_status(adapter: SqliteStorageAdapter) -> None:
    async def run() -> None:
        tenant = tenant_primary()
        handle = await adapter.create_run(CreateRunInput(tenant=tenant, run_id="r1", subject="A"))
        await adapter.complete_run(CompleteRunInput(tenant=tenant, run_id="r1"))
        assert handle.status == "complete"
        failed = await adapter.create_run(CreateRunInput(tenant=tenant, run_id="r2", subject="A"))
        await adapter.fail_run(FailRunInput(tenant=tenant, run_id="r2"))
        assert failed.status == "failed"

    asyncio.run(run())


def test_status_change_keeps_a_concurrent_replacement(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite3"
    tenant = tenant_primary()

    async def run() -> None:
        first, second = SqliteStorageAdapter(path), SqliteStorageAdapter(path)
        await first.create_run(CreateRunInput(tenant=tenant, run_id="r1", subject="old"))
        await second.create_run(CreateRunInput(tenant=tenant, run_id="r1", subject="new"))
        await first.complete_run(CompleteRunInput(tenant=tenant, run_id="r1"))
        first.close()
        second.close()

    asyncio.run(run())
    with sqlite3.connect(path) as connection:
        record = connection.execute("SELECT record FROM runs").fetchone()[0]
    assert '"subject":"new"' in record and '"status":"complete"' in record


def test_limit_without_vector_and_single_file(tmp_path: Path) -> None:
    path = tmp_path / "one.sqlite3"
    tenant = tenant_primary()

    async def run() -> None:
        store = SqliteStorageAdapter(path)
        for index in range(3):
            claim = _make_claim(claim_text=f"c{index}", source_url=f"https://example.com/{index}")
            await store.upsert_claim(UpsertClaimInput(tenant=tenant, claim=claim, embedding=[1.0]))
        limited = await store.search_claims(ClaimSearchInput(tenant=tenant, limit=2))
        assert [claim.claim_text for claim in limited] == ["c0", "c1"]
        assert await store.search_claims(ClaimSearchInput(tenant=tenant, limit=0)) == []
        # Every commit is in the one file: no write-ahead log beside it.
        assert not Path(f"{path}-wal").exists()
        store.close()

    asyncio.run(run())


def test_foreign_rows_already_on_disk_are_never_returned(tmp_path: Path) -> None:
    path = tmp_path / "planted.sqlite3"
    theirs = _make_claim(tenant_id="secondary", claim_text="theirs")
    SqliteStorageAdapter(path).close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO claims (tenant_id, claim_id, subject, category, record) "
            "VALUES ('primary', ?, 'Acme', 'security', ?)",
            (theirs.claim_id, theirs.model_dump_json()),
        )
        connection.execute(
            "INSERT INTO kb_entries (tenant_id, entry_id, record, subject, category) "
            "VALUES ('primary', 'e1', ?, 'Acme', 'security')",
            (
                '{"tenant_id":"secondary","entry_id":"e1","claim":'
                + theirs.model_dump_json()
                + ',"status":"approved","updated_at":1}',
            ),
        )

    async def run() -> None:
        store = SqliteStorageAdapter(path)
        tenant = tenant_primary()
        mine = _make_claim()
        await store.upsert_claim(UpsertClaimInput(tenant=tenant, claim=mine))
        assert await store.get_claim(GetClaimInput(tenant=tenant, claim_id=theirs.claim_id)) is None
        assert await store.search_claims(ClaimSearchInput(tenant=tenant, limit=1)) == [mine]
        assert await store.list_kb_entries(ListKbEntriesInput(tenant=tenant)) == []
        store.close()

    asyncio.run(run())


def test_write_stores_the_record_as_it_was_when_called(adapter: SqliteStorageAdapter) -> None:
    async def run() -> None:
        tenant = tenant_primary()
        claim = _make_claim()
        task = asyncio.create_task(
            adapter.upsert_claim(UpsertClaimInput(tenant=tenant, claim=claim))
        )
        await asyncio.sleep(0)
        claim.tenant_id = "secondary"
        claim.claim_text = "changed later"
        await task
        stored = await adapter.get_claim(GetClaimInput(tenant=tenant, claim_id=claim.claim_id))
        assert stored is not None
        assert stored.tenant_id == "primary"
        assert stored.claim_text == "Has an audit log"

    asyncio.run(run())


def test_cancelled_status_change_keeps_handle_and_row_in_step(tmp_path: Path) -> None:
    path = tmp_path / "cancel.sqlite3"
    tenant = tenant_primary()

    async def run() -> list[tuple[str, str]]:
        store = SqliteStorageAdapter(path)
        seen = []
        for index in range(20):
            run_id = f"r{index}"
            handle = await store.create_run(
                CreateRunInput(tenant=tenant, run_id=run_id, subject="A")
            )
            task = asyncio.create_task(
                store.complete_run(CompleteRunInput(tenant=tenant, run_id=run_id))
            )
            for _ in range(index % 4):
                await asyncio.sleep(0)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            # Any change already handed to the worker finishes before the next call runs.
            await store.search_claims(ClaimSearchInput(tenant=tenant))
            with sqlite3.connect(path) as connection:
                record = connection.execute(
                    "SELECT record FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()[0]
            seen.append((handle.status, Run.model_validate_json(record).status))
        store.close()
        return seen

    for handle_status, row_status in asyncio.run(run()):
        assert handle_status == row_status


def test_negative_limits_return_nothing(adapter: SqliteStorageAdapter) -> None:
    async def run() -> None:
        tenant = tenant_primary()
        for index in range(3):
            claim = _make_claim(claim_text=f"c{index}", source_url=f"https://example.com/{index}")
            await adapter.upsert_claim(
                UpsertClaimInput(tenant=tenant, claim=claim, embedding=[1.0])
            )
        for vector in (None, [1.0]):
            found = await adapter.search_claims(
                ClaimSearchInput(tenant=tenant, query_vector=vector, limit=-1)
            )
            assert found == []

    asyncio.run(run())
