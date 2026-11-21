from __future__ import annotations

import pytest
from houndex_core import DEFAULT_RRF_K, reciprocal_rank_fusion


def test_fuses_by_summed_reciprocal_rank_and_records_each_arms_rank() -> None:
    hits = reciprocal_rank_fusion({"lexical": ["a", "b"], "vector": ["b", "c"]}, k=1)
    assert [hit.id for hit in hits] == ["b", "a", "c"]
    by_id = {hit.id: hit for hit in hits}
    assert by_id["b"].score == pytest.approx(1 / 3 + 1 / 2)
    assert by_id["a"].ranks == {"lexical": 1, "vector": None}
    assert by_id["b"].ranks == {"lexical": 2, "vector": 1}


def test_counts_an_id_once_per_arm_at_its_best_rank() -> None:
    hits = reciprocal_rank_fusion({"only": ["a", "a", "b"]}, k=0)
    assert [(hit.id, hit.score, dict(hit.ranks)) for hit in hits] == [
        ("a", 1.0, {"only": 1}),
        ("b", 1 / 3, {"only": 3}),
    ]


def test_ties_break_by_id_and_empty_arms_are_fine() -> None:
    hits = reciprocal_rank_fusion({"x": ["b"], "y": ["a"], "z": []})
    assert [hit.id for hit in hits] == ["a", "b"]
    assert hits[0].score == 1 / (DEFAULT_RRF_K + 1)
    assert reciprocal_rank_fusion({}) == []


def test_rejects_a_negative_or_non_integer_k() -> None:
    for k in (-1, 1.5, True):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion({"x": ["a"]}, k=k)  # ty: ignore[invalid-argument-type]
