"""Citation verification against an evidence store. An ``OutputEnvelope``'s
provenance ``trace`` names the claims an answer rests on; verifying it resolves
every cited claim id in the tenant's store, so an answer citing evidence the store
does not hold is caught before anyone relies on it. Pure over the
``StorageAdapter`` protocol, so any adapter works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from houndex_core.schemas import CitationVerdict, Claim, OutputEnvelope
from houndex_core.storage import GetClaimInput, StorageAdapter
from houndex_core.tenant import TenantContext

PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class TraceVerification:
    """The verdict on one envelope's citations, with what resolved and what did not.

    ``green``: every cited claim resolves. ``yellow``: some do. ``red``: none do,
    nothing is cited, or the envelope belongs to another tenant.
    """

    verdict: CitationVerdict
    resolved: list[Claim] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return self.verdict.verdict == "green"


def _cited_ids(envelope: OutputEnvelope[PayloadT]) -> list[str]:
    """Cited claim ids in trace order, each once."""
    return list(dict.fromkeys(entry.claim_id for entry in envelope.trace))


async def verify_trace(
    adapter: StorageAdapter, tenant: TenantContext, envelope: OutputEnvelope[PayloadT]
) -> TraceVerification:
    """Resolve every claim ``envelope.trace`` cites in ``tenant``'s store."""
    # One snapshot of the tenant and the citations for every lookup and check below.
    tenant = tenant.model_copy(deep=True)
    tenant_id = tenant.tenant_id
    if envelope.tenant_id != tenant_id:
        return TraceVerification(
            verdict=CitationVerdict(
                verdict="red", rationale="the envelope belongs to a different tenant"
            ),
            unresolved=_cited_ids(envelope),
        )
    cited = _cited_ids(envelope)
    if not cited:
        return TraceVerification(
            verdict=CitationVerdict(verdict="red", rationale="the envelope cites no evidence")
        )
    resolved: list[Claim] = []
    unresolved: list[str] = []
    for claim_id in cited:
        lookup = GetClaimInput(tenant=tenant.model_copy(), claim_id=claim_id)
        claim = await adapter.get_claim(lookup)
        # Only the tenant's own record of exactly this claim counts as resolving it.
        if claim is None or claim.tenant_id != tenant_id or claim.claim_id != claim_id:
            unresolved.append(claim_id)
        else:
            resolved.append(claim)
    if not unresolved:
        verdict = CitationVerdict(
            verdict="green",
            evidence=[claim.claim_id for claim in resolved],
            rationale=f"all {len(cited)} cited claims resolve in the evidence store",
        )
    else:
        verdict = CitationVerdict(
            verdict="yellow" if resolved else "red",
            evidence=[claim.claim_id for claim in resolved],
            rationale=f"{len(unresolved)} of {len(cited)} cited claims do not resolve",
        )
    return TraceVerification(verdict=verdict, resolved=resolved, unresolved=unresolved)


__all__ = ["TraceVerification", "verify_trace"]
