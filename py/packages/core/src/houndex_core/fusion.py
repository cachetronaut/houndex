"""Reciprocal rank fusion: merge ranked lists of ids into one ranking, keeping where each came from.

Each list (an "arm", such as a lexical and a vector search) contributes ``1 / (k + rank)`` for every
id it ranks, counting an id once per arm at its best rank. Ids are ordered by their summed score,
then by id, so the result is deterministic. Every hit records its 1-based rank in each arm, or
``None`` where that arm did not return it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: The rank constant from the original reciprocal rank fusion paper (Cormack et al., 2009).
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedHit:
    """One fused id: its summed score and its rank in each arm (``None`` if absent)."""

    id: str
    score: float
    ranks: Mapping[str, int | None]


def reciprocal_rank_fusion(
    arms: Mapping[str, Sequence[str]], *, k: int = DEFAULT_RRF_K
) -> list[FusedHit]:
    """Fuse ranked id lists, best first. Arms are summed in the order given."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int | None]] = {}
    for arm in arms.values():
        for identifier in arm:
            ranks.setdefault(identifier, dict.fromkeys(arms))
    for name, arm in arms.items():
        seen: set[str] = set()
        for rank, identifier in enumerate(arm, start=1):
            if identifier in seen:
                continue
            seen.add(identifier)
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank)
            ranks[identifier][name] = rank
    return [
        FusedHit(id=identifier, score=scores[identifier], ranks=ranks[identifier])
        for identifier in sorted(scores, key=lambda key: (-scores[key], key))
    ]
