"""Tests for cascading context-aware generation."""

from __future__ import annotations

import pytest
import pytest_asyncio

from tests.conftest import add_rule, load_grammar_bulk
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
    await load_grammar_bulk(mem_db, "test", grammar, fake_embedder)
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
    # Auto-generated ids in v0.5 have the shape "r_{hex32}" — by default no
    # tests here set custom ids so all walks produce auto ids.
    assert all(isinstance(i, str) and i.startswith("r_") for i in result.ids)


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
    # populated below via explicit per-rule adds with custom ids.
    grammar = Grammar.from_dict({"origin": ["applicant has #skill#"]})
    await load_grammar_bulk(mem_db, "skills", grammar, fake_embedder)

    await add_rule(
        mem_db, "skills", "skill", "audited defi protocols",
        fake_embedder, id="applicant/security/community_x/uuid-audit",
    )
    await add_rule(
        mem_db, "skills", "skill", "found critical bugs",
        fake_embedder, id="applicant/security/community_x/uuid-bugs",
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
    text, ids, rules_used = result  # v0.5: 3-tuple (rules_used added)
    assert text == result.text
    assert ids == result.ids
    assert rules_used == result.rules_used


@pytest.mark.asyncio
async def test_bulk_load_honors_explicit_rule_ids(mem_db, fake_embedder):
    """Grammar dict entries of the shape ``{"text", "id"}`` should flow
    through the bulk loader with the explicit id preserved — no two-step
    load + per-rule-add dance required."""
    grammar = Grammar.from_dict(
        {
            "origin": ["applicant #skill#"],
            "skill": [
                {"text": "audited defi protocols", "id": "applicant/security/uuid-audit"},
                {"text": "found critical bugs", "id": "applicant/security/uuid-bugs"},
            ],
        }
    )
    await load_grammar_bulk(mem_db, "skills", grammar, fake_embedder)

    generator = Generator(mem_db, "skills", fake_embedder)
    result = await generator.generate(context="audit work", temperature=0.0, seed=0)

    # The cascade walks origin → skill → one of the custom-id entries.
    # At least one of the walked ids should be a custom one.
    custom_ids = [i for i in result.ids if i.startswith("applicant/")]
    assert custom_ids, f"expected a custom id in walk, got {result.ids}"
    # And none of the skill-symbol rows should fall back to the auto-id shape.
    assert not any(i.startswith("r_") for i in custom_ids)
