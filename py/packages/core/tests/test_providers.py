from __future__ import annotations

import asyncio
from collections.abc import Sequence

from houndex_core import Embedder, VersionedEmbedder, embedder_version


class PlainEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _text in texts]


class NamedEmbedder(PlainEmbedder):
    async def version(self) -> str:
        return "model@sha256:abc"


def test_embedder_version_is_optional() -> None:
    plain, named = PlainEmbedder(), NamedEmbedder()

    assert isinstance(plain, Embedder)
    assert not isinstance(plain, VersionedEmbedder)
    assert isinstance(named, VersionedEmbedder)
    assert asyncio.run(embedder_version(plain)) is None
    assert asyncio.run(embedder_version(named)) == "model@sha256:abc"
