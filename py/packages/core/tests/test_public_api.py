from __future__ import annotations


def test_houndex_core_exports_schema_provider_and_storage_contracts() -> None:
    import houndex_core

    expected_names = {
        "CATEGORY_VALUES",
        "POLARITY_VALUES",
        "SCOPE_VALUES",
        "SOURCE_TIER_VALUES",
        "CONFIDENCE_VALUES",
        "RECONCILIATION_DECISION_VALUES",
        "CURATION_STATUS_VALUES",
        "KB_ACTION_VALUES",
        "NODE_KIND_VALUES",
        "EDGE_KIND_VALUES",
        "VERDICT_VALUES",
        "Claim",
        "Source",
        "Subject",
        "CategoryNode",
        "Run",
        "GraphNode",
        "Edge",
        "OutputEnvelope",
        "SearchProvider",
        "Scraper",
        "Embedder",
        "Reranker",
        "StorageAdapter",
        "EnsureTenantInput",
        "CreateRunInput",
        "UpsertClaimInput",
        "ClaimSearchInput",
        "canonicalize_url",
        "edge_idempotency_key",
    }

    missing = expected_names.difference(houndex_core.__all__)
    assert missing == set()
    for name in expected_names:
        assert hasattr(houndex_core, name)


def test_houndex_distribution_facade_matches_core_contract_exports() -> None:
    import houndex_core

    import houndex

    contract_names = {
        "CATEGORY_VALUES",
        "VERDICT_VALUES",
        "Claim",
        "Edge",
        "OutputEnvelope",
        "SearchProvider",
        "StorageAdapter",
        "canonicalize_url",
        "edge_idempotency_key",
    }

    for name in contract_names:
        assert getattr(houndex, name) is getattr(houndex_core, name)
        assert name in houndex.__all__
