"""Tests for cascading context-aware generation."""

from __future__ import annotations

import pytest
import pytest_asyncio

from trimtab.generator import GenerationResult, Generator
from trimtab.grammar import Grammar


@pytest_asyncio.fixture
async def gen(mem_db, fake_embedder):
    grammar = Grammar.from_dict(
        {
            "origin": ["#mood# and #detail#."],
            "mood": ["dark and cold", "bright and warm", "eerie and still"],
            "detail": ["shadows move", "birds sing", "silence reigns"],
        }
    )
    await mem_db.upsert_grammar("test", grammar, fake_embedder)
    return Generator(mem_db, "test", fake_embedder)


@pytest.mark.asyncio
async def test_generate_returns_generation_result(gen):
    result = await gen.generate(context="spooky cave", temperature=0.0, seed=42)
    assert isinstance(result, GenerationResult)
    assert isinstance(result.text, str)
    assert len(result.text) > 5
    assert result.text.endswith(".")


@pytest.mark.asyncio
async def test_generate_deterministic(gen):
    r1 = await gen.generate(context="forest", temperature=0.0, seed=42)
    r2 = await gen.generate(context="forest", temperature=0.0, seed=42)
    assert r1.text == r2.text
    assert r1.ids == r2.ids


@pytest.mark.asyncio
async def test_generate_temperature_1_is_random(gen):
    results = set()
    for seed in range(20):
        result = await gen.generate(context="anything", temperature=1.0, seed=seed)
        results.add(result.text)
    assert len(results) > 1


@pytest.mark.asyncio
async def test_generate_no_rule_refs_in_output(gen):
    result = await gen.generate(context="test", temperature=0.3, seed=42)
    assert "#" not in result.text


@pytest.mark.asyncio
async def test_generate_empty_context(gen):
    result = await gen.generate(context="", temperature=1.0, seed=42)
    assert isinstance(result, GenerationResult)
    assert len(result.text) > 0


# ---- New behavior: ids returned from the cascading walk ---------------------


@pytest.mark.asyncio
async def test_generate_returns_ids_from_walked_expansions(gen):
    """Cascaded walks should surface the picked expansion ids so downstream
    agents can use them as center-node anchors for KG search."""
    result = await gen.generate(context="dark stormy night", temperature=0.0, seed=1)
    assert isinstance(result.ids, list)
    # The walk visits origin + mood + detail (3 rules with content picks), so
    # we expect at least those 3 ids when context drives deterministic picks.
    assert len(result.ids) >= 2
    # Auto-generated ids from add_expansion have the shape "{grammar}:{rule}:{hash}"
    # — by default none of the tests set custom ids so all walks produce auto ids.
    assert all(isinstance(i, str) and ":" in i for i in result.ids)


@pytest.mark.asyncio
async def test_generate_preserves_custom_ids_from_db(mem_db, fake_embedder):
    """When expansions are added with consumer-supplied ids (e.g. KG entity
    UUIDs or path-tagged strings), the walk surfaces those ids so the
    scoring agent can use them as delve center_node_uuids.

    Uses a grammar whose ``skill`` rule is populated entirely via
    ``add_expansion`` with custom ids — avoiding the auto-id rows
    ``upsert_grammar`` would otherwise insert for the same texts.
    """
    # Grammar only declares the origin rule. The skill rule will be
    # populated below via explicit add_expansion calls.
    grammar = Grammar.from_dict({"origin": ["applicant has #skill#"]})
    await mem_db.upsert_grammar("skills", grammar, fake_embedder)

    vec = await fake_embedder.create("audited defi protocols")
    mem_db.add_expansion(
        "skills",
        "skill",
        "audited defi protocols",
        vec,
        id="applicant/security/community_x/uuid-audit",
    )
    vec2 = await fake_embedder.create("found critical bugs")
    mem_db.add_expansion(
        "skills",
        "skill",
        "found critical bugs",
        vec2,
        id="applicant/security/community_x/uuid-bugs",
    )

    generator = Generator(mem_db, "skills", fake_embedder)
    result = await generator.generate(context="security audit work", temperature=0.0, seed=0)

    # The walk should contain at least one id that matches the custom path-tagged shape
    custom_ids = [i for i in result.ids if i.startswith("applicant/")]
    assert custom_ids, f"expected at least one custom id in {result.ids}"


@pytest.mark.asyncio
async def test_generation_result_is_namedtuple_unpackable(gen):
    """GenerationResult supports both tuple unpacking and field access."""
    result = await gen.generate(context="test", temperature=0.0, seed=7)
    text, ids = result  # tuple unpacking still works
    assert text == result.text
    assert ids == result.ids


@pytest.mark.asyncio
async def test_upsert_grammar_honors_explicit_expansion_ids(mem_db, fake_embedder):
    """Grammar dict entries of the shape ``{"text", "id"}`` should flow
    through ``upsert_grammar`` with the explicit id preserved — no two-step
    upsert + add_expansion dance required."""
    grammar = Grammar.from_dict(
        {
            "origin": ["applicant #skill#"],
            "skill": [
                {"text": "audited defi protocols", "id": "applicant/security/uuid-audit"},
                {"text": "found critical bugs", "id": "applicant/security/uuid-bugs"},
            ],
        }
    )
    await mem_db.upsert_grammar("skills", grammar, fake_embedder)

    generator = Generator(mem_db, "skills", fake_embedder)
    result = await generator.generate(context="audit work", temperature=0.0, seed=0)

    # The cascade walks origin → skill → one of the custom-id expansions.
    # At least one of the walked ids should be a custom one.
    custom_ids = [i for i in result.ids if i.startswith("applicant/")]
    assert custom_ids, f"expected a custom id in walk, got {result.ids}"
    # And none of the rows for the "skill" rule should have auto-generated ids.
    assert not any(i.startswith("skills:skill:") for i in result.ids)
