"""
Tests for OpeningsMatch.

The point of these tests is NOT to check that the similarity maths works.
It is to prove the SAFETY PROPERTY:

    The ranking layer can never cause a non-compliant product to be recommended.

In a life-safety domain, "the model usually gets it right" is not good enough.
The constraint layer must be provably inviolable. These tests are the proof.

Run:  python -m pytest src/test_recommender.py -v
"""

import pytest
from recommender import OpeningsMatch, Door


@pytest.fixture(scope="module")
def engine():
    return OpeningsMatch()


# ---------------------------------------------------------------------------
# SAFETY: fire-rated doors
# ---------------------------------------------------------------------------
def test_fire_door_never_gets_non_fire_rated_hardware(engine):
    """THE critical test. No component on a fire door may be non-fire-rated."""
    door = Door(fire_rated=True, security_level=3)
    result = engine.recommend_set(door)

    catalog = engine.catalog.set_index("product_id")
    for rec in result["recommendations"]:
        assert catalog.loc[rec.product_id, "fire_rated"], (
            f"SAFETY VIOLATION: {rec.name} is not fire-rated but was "
            f"recommended for a fire door."
        )


def test_fire_door_always_gets_a_closer(engine):
    """Fire doors must be self-closing. This is a legal requirement, not a
    preference - the recommender must never omit it."""
    door = Door(fire_rated=True)
    result = engine.recommend_set(door)
    categories = {r.category for r in result["recommendations"]}
    assert "closer" in categories, (
        "SAFETY VIOLATION: fire door specified without a self-closing device."
    )


def test_fire_door_always_gets_a_seal(engine):
    """Regression test for a real bug found during development.

    The security filter was originally applied to EVERY category. Because seals
    carry security_level=1, a high-security fire door filtered out all seals -
    silently dropping a legally-required fire component from the set.

    Seals are not security-bearing. This test locks that fix in place.
    """
    door = Door(fire_rated=True, security_level=5)   # max security
    result = engine.recommend_set(door)
    categories = {r.category for r in result["recommendations"]}
    assert "seal" in categories, (
        "REGRESSION: high-security fire door lost its intumescent seal."
    )


def test_non_fire_door_may_use_any_closer(engine):
    """The constraint should only BITE when it applies. A normal door should
    not be needlessly restricted to fire-rated stock."""
    fire = engine.filter_permissible(Door(fire_rated=True), "closer")
    normal = engine.filter_permissible(Door(fire_rated=False), "closer")
    assert len(normal) > len(fire), (
        "The fire constraint should widen the candidate pool when lifted."
    )


# ---------------------------------------------------------------------------
# SAFETY: egress routes
# ---------------------------------------------------------------------------
def test_egress_door_always_gets_an_exit_device(engine):
    """Doors on an escape route must have panic hardware."""
    door = Door(is_egress=True)
    result = engine.recommend_set(door)
    categories = {r.category for r in result["recommendations"]}
    assert "exit_device" in categories, (
        "SAFETY VIOLATION: egress door specified without panic hardware."
    )


def test_non_egress_door_gets_no_exit_device(engine):
    """Don't over-specify. A cupboard door does not need a panic bar."""
    door = Door(is_egress=False, fire_rated=False, security_level=1)
    result = engine.recommend_set(door)
    categories = {r.category for r in result["recommendations"]}
    assert "exit_device" not in categories


# ---------------------------------------------------------------------------
# CONSTRAINT: security levels
# ---------------------------------------------------------------------------
def test_high_security_door_rejects_weak_locks(engine):
    """A level-4 door must not be fitted with a level-1 passage latch."""
    door = Door(security_level=4)
    permissible = engine.filter_permissible(door, "lock")
    assert not permissible.empty
    assert (permissible["security_level"] >= 4).all(), (
        "A lock below the door's security level survived the filter."
    )


def test_security_filter_does_not_apply_to_seals(engine):
    """Seals are not security components. Filtering them by security level was
    the original bug - this asserts the fix directly."""
    door = Door(fire_rated=True, security_level=5)
    seals = engine.filter_permissible(door, "seal")
    assert not seals.empty, (
        "Security filter is wrongly applied to non-security-bearing seals."
    )


# ---------------------------------------------------------------------------
# CONSTRAINT: handing
# ---------------------------------------------------------------------------
def test_unhanded_door_rejects_handed_hardware(engine):
    door = Door(handed=False, fire_rated=False, security_level=1)
    for cat in ["lock", "hinge"]:
        permissible = engine.filter_permissible(door, cat)
        if not permissible.empty:
            assert not permissible["handed"].any(), (
                f"Handed {cat} survived the filter on an unhanded door."
            )


# ---------------------------------------------------------------------------
# EXPLAINABILITY
# ---------------------------------------------------------------------------
def test_every_recommendation_is_justified(engine):
    """A spec writer must be able to defend every choice to an inspector.
    An unexplained recommendation is unusable in this domain."""
    door = Door(fire_rated=True, is_egress=True, security_level=4)
    result = engine.recommend_set(door)
    for rec in result["recommendations"]:
        assert rec.reasons, f"{rec.name} was recommended with no justification."


def test_rule_only_recommendations_are_flagged_honestly(engine):
    """If similarity found NO signal and the item is present purely because a
    rule demanded it, that must be visible - not disguised as a confident match."""
    door = Door(fire_rated=True, is_egress=True, security_level=4)
    result = engine.recommend_set(door)
    for rec in result["recommendations"]:
        if rec.score == 0.0:
            assert rec.confidence == "rule-only", (
                f"{rec.name} has zero similarity but is not flagged as rule-only."
            )


# ---------------------------------------------------------------------------
# FAILURE HANDLING
# ---------------------------------------------------------------------------
def test_impossible_spec_warns_rather_than_silently_dropping(engine):
    """If the catalogue genuinely cannot satisfy a door, the system must SAY SO.
    Silently returning an incomplete set is the dangerous failure mode."""
    # An unhanded fire door at max security - deliberately hard to satisfy.
    door = Door(fire_rated=True, handed=False, security_level=5, is_egress=True)
    result = engine.recommend_set(door)

    required = set(engine.required_categories(door).keys())
    delivered = {r.category for r in result["recommendations"]}
    missing = required - delivered

    # Whatever it cannot fill, it must warn about. Never silently omit.
    assert len(result["warnings"]) == len(missing), (
        "A required category was dropped without a warning."
    )
