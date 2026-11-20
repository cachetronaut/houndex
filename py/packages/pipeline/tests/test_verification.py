from __future__ import annotations

import asyncio
from typing import Any

from houndex_core import compute_claim_id, tenant_primary, tenant_secondary
from houndex_core.schemas import Claim, OutputEnvelope, TraceEntry
from houndex_core.storage import UpsertClaimInput
from houndex_pipeline import verify_trace
from houndex_storage_local import LocalStorageAdapter


def _claim(text: str, tenant_id: str = "primary") -> Claim:
    source_url = f"https://example.com/{text}"
    data: dict[str, Any] = {
        "tenant_id": tenant_id,
        "claim_id": compute_claim_id(
            tenant_id=tenant_id, subject="Acme", claim_text=text, source_url=source_url
        ),
        "subject": "Acme",
        "category": "general",
        "polarity": "neutral",
        "scope": "scoped",
        "claim_text": text,
        "evidence_text": text,
        "confidence": "stated",
        "source_url": source_url,
        "source_tier": "authoritative",
        "extracted_at": 1,
    }
    return Claim.model_validate(data)


def _envelope(claim_ids: list[str], tenant_id: str = "primary") -> OutputEnvelope[object]:
    return OutputEnvelope[object](
        tenant_id=tenant_id,
        generated_at=1,
        trace=[TraceEntry(claim_id=claim_id, mechanism="lookup") for claim_id in claim_ids],
        payload={"answer": "x"},
    )


async def _store(*claims: Claim) -> LocalStorageAdapter:
    adapter = LocalStorageAdapter()
    for claim in claims:
        await adapter.upsert_claim(UpsertClaimInput(tenant=tenant_primary(), claim=claim))
    return adapter


def test_green_when_every_citation_resolves() -> None:
    async def run() -> None:
        first, second = _claim("one"), _claim("two")
        adapter = await _store(first, second)
        envelope = _envelope([first.claim_id, second.claim_id, first.claim_id])
        result = await verify_trace(adapter, tenant_primary(), envelope)
        assert result.grounded
        assert result.verdict.verdict == "green"
        assert [claim.claim_id for claim in result.resolved] == [first.claim_id, second.claim_id]
        assert result.unresolved == []
        assert result.verdict.evidence == [first.claim_id, second.claim_id]

    asyncio.run(run())


def test_yellow_lists_what_does_not_resolve() -> None:
    async def run() -> None:
        held, missing = _claim("held"), _claim("missing")
        adapter = await _store(held)
        result = await verify_trace(
            adapter, tenant_primary(), _envelope([held.claim_id, missing.claim_id])
        )
        assert result.verdict.verdict == "yellow"
        assert not result.grounded
        assert result.unresolved == [missing.claim_id]

    asyncio.run(run())


def test_red_when_nothing_resolves_or_nothing_is_cited() -> None:
    async def run() -> None:
        adapter = await _store()
        none = await verify_trace(adapter, tenant_primary(), _envelope([_claim("x").claim_id]))
        assert none.verdict.verdict == "red"
        empty = await verify_trace(adapter, tenant_primary(), _envelope([]))
        assert empty.verdict.verdict == "red"
        assert empty.unresolved == []

    asyncio.run(run())


def test_other_tenants_evidence_never_resolves() -> None:
    async def run() -> None:
        theirs = _claim("theirs", tenant_id="secondary")
        adapter = LocalStorageAdapter()
        await adapter.upsert_claim(UpsertClaimInput(tenant=tenant_secondary(), claim=theirs))
        crossed = await verify_trace(adapter, tenant_primary(), _envelope([theirs.claim_id]))
        assert crossed.verdict.verdict == "red"
        mismatched = await verify_trace(
            adapter, tenant_primary(), _envelope([theirs.claim_id], tenant_id="secondary")
        )
        assert mismatched.verdict.verdict == "red"
        assert mismatched.unresolved == [theirs.claim_id]

    asyncio.run(run())


def test_a_foreign_record_returned_by_an_adapter_does_not_resolve() -> None:
    async def run() -> None:
        theirs = _claim("theirs", tenant_id="secondary")
        adapter = LocalStorageAdapter()
        # The reference adapter files a record under the write's tenant as given.
        await adapter.upsert_claim(UpsertClaimInput(tenant=tenant_primary(), claim=theirs))
        result = await verify_trace(adapter, tenant_primary(), _envelope([theirs.claim_id]))
        assert result.verdict.verdict == "red"
        assert result.unresolved == [theirs.claim_id]

    asyncio.run(run())


def test_typed_payload_envelopes_are_accepted() -> None:
    async def run() -> None:
        held = _claim("held")
        adapter = await _store(held)
        envelope = OutputEnvelope[dict[str, str]](
            tenant_id="primary",
            generated_at=1,
            trace=[TraceEntry(claim_id=held.claim_id, mechanism="lookup")],
            payload={"answer": "x"},
        )
        assert (await verify_trace(adapter, tenant_primary(), envelope)).grounded

    asyncio.run(run())


def test_tenant_changed_during_a_lookup_does_not_widen_the_check() -> None:
    async def run() -> None:
        mine, theirs = _claim("mine"), _claim("theirs", tenant_id="secondary")
        tenant = tenant_primary()

        class Shifting(LocalStorageAdapter):
            async def get_claim(self, input):  # type: ignore[override]
                tenant.tenant_id = "secondary"
                return await super().get_claim(input)

        adapter = Shifting()
        await adapter.upsert_claim(UpsertClaimInput(tenant=tenant_primary(), claim=mine))
        await adapter.upsert_claim(UpsertClaimInput(tenant=tenant_secondary(), claim=theirs))
        result = await verify_trace(adapter, tenant, _envelope([mine.claim_id, theirs.claim_id]))
        assert [claim.claim_id for claim in result.resolved] == [mine.claim_id]
        assert result.unresolved == [theirs.claim_id]

    asyncio.run(run())
