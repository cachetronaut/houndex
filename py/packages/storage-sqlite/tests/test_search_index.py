from __future__ import annotations

import math

import pytest
from houndex_storage_sqlite import (
    SearchDocument,
    SqliteSearchIndex,
    fuzzy_matches,
    match_expression,
    rank_by_similarity,
)


def _index(**kwargs: object) -> SqliteSearchIndex:
    index = SqliteSearchIndex(fields=("title", "body"), **kwargs)  # ty: ignore[invalid-argument-type]
    index.add(
        [
            SearchDocument("b", {"title": "Charizard", "body": "fire flying"}, order=2),
            SearchDocument("a", {"title": "Charmander", "body": "fire lizard"}, order=1),
            SearchDocument("c", {"title": "Squirtle", "body": "water turtle"}, order=3),
        ]
    )
    return index


def test_match_expression_quotes_each_word_as_a_prefix() -> None:
    assert match_expression("Fire, lizard!") == '"fire"* OR "lizard"*'
    assert match_expression("  ...  ") is None


def test_lexical_matches_word_prefixes_in_any_field() -> None:
    index = _index()
    assert set(index.lexical("char")) == {"a", "b"}
    assert index.lexical("turt") == ["c"]
    assert index.lexical("nothing here") == []


def test_lexical_ranks_by_bm25_then_breaks_ties_by_id_or_newest() -> None:
    index = _index()
    assert index.lexical("lizard fire")[0] == "a"  # both words beat one
    assert index.lexical("fire") == ["a", "b"]  # equal scores: by id
    assert index.lexical("fire", ties="newest") == ["b", "a"]


def test_lexical_pages_with_limit_and_offset() -> None:
    index = _index()
    assert index.lexical("fire", limit=1) == ["a"]
    assert index.lexical("fire", limit=1, offset=1) == ["b"]
    assert index.lexical("fire", limit=0) == []


def test_adding_an_id_again_replaces_it_and_remove_and_clear_forget() -> None:
    index = _index()
    index.add([SearchDocument("c", {"title": "Wartortle", "body": "fire"}, order=4)])
    assert index.lexical("turtle") == []
    assert set(index.lexical("fire")) == {"a", "b", "c"}
    index.remove(["a"])
    assert set(index.lexical("fire")) == {"b", "c"}
    index.clear()
    assert index.lexical("fire") == []


def test_fields_must_be_distinct_identifiers() -> None:
    for fields in (
        (),
        ("Title",),
        ("a", "a"),
        ("document_id",),
        ("x y",),
        ("rank",),
        ("rowid",),
        ("documents",),
    ):
        with pytest.raises(ValueError):
            SqliteSearchIndex(fields=fields)


def test_sqlite_uris_are_refused_so_files_stay_owner_only(tmp_path) -> None:
    with pytest.raises(ValueError):
        SqliteSearchIndex(f"file:{tmp_path / 'search.sqlite'}")
    assert not (tmp_path / "search.sqlite").exists()


def test_vectors_are_searchable_and_survive_reopening(tmp_path) -> None:
    path = str(tmp_path / "search.sqlite")
    index = SqliteSearchIndex(path)
    index.add(
        [
            SearchDocument("x", {"text": "one"}, vector=[1.0, 0.0]),
            SearchDocument("y", {"text": "two"}, vector=[0.6, 0.8]),
            SearchDocument("z", {"text": "three"}),
        ]
    )
    index.close()
    reopened = SqliteSearchIndex(path)
    ranked = reopened.similar([1.0, 0.0])
    assert [identifier for identifier, _ in ranked] == ["x", "y"]
    assert ranked[1][1] == pytest.approx(0.6)
    assert reopened.similar([1.0, 0.0], minimum=0.9) == [("x", pytest.approx(1.0))]
    assert reopened.similar([1.0, 0.0], limit=1)[0][0] == "x"
    assert reopened.lexical("two") == ["y"]
    reopened.close()
    assert (tmp_path / "search.sqlite").stat().st_mode & 0o777 == 0o600


def test_rank_by_similarity_dot_and_cosine() -> None:
    candidates = [("b", [2.0, 0.0]), ("a", [0.0, 1.0]), ("c", [1.0, 1.0])]
    by_dot = rank_by_similarity([1.0, 0.0], candidates, measure="dot")
    assert by_dot == [("b", 2.0), ("c", 1.0), ("a", 0.0)]
    by_cosine = rank_by_similarity([1.0, 0.0], candidates)
    assert by_cosine[0] == ("b", 1.0)
    assert by_cosine[1] == ("c", pytest.approx(1 / math.sqrt(2)))
    assert rank_by_similarity([1.0, 0.0], [("b", [1.0, 0.0]), ("a", [1.0, 0.0])]) == [
        ("a", 1.0),
        ("b", 1.0),
    ]
    with pytest.raises(ValueError):
        rank_by_similarity([1.0], [], measure="euclid")  # ty: ignore[invalid-argument-type]


def test_fuzzy_matches_tolerate_typos_and_respect_threshold_and_length() -> None:
    names = [("pikachu", "Pikachu"), ("raichu", "Raichu"), ("onix", "Onix")]
    assert fuzzy_matches("pikachoo", names, threshold=0.65) == ["pikachu"]
    assert fuzzy_matches("chu", names, threshold=0.0, minimum_length=4) == []
    assert fuzzy_matches("chu", names, threshold=0.0)[-1] == "onix"
