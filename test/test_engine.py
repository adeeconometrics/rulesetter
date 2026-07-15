"""Tests for rulesetter.engine -- rewriting engine, strategies, cycle detection."""

import pytest

from rulesetter.engine import (
    Redex,
    ReductionLimitExceededError,
    RewriteCycleDetectedError,
    RewriteEngine,
    Strategy,
    positions,
    replace_at,
)
from rulesetter.rule import rule
from rulesetter.term import const, op, var

# ---------------------------------------------------------------------------
# replace_at
# ---------------------------------------------------------------------------


class TestReplaceAt:
    def test_root(self) -> None:
        t = op("f", var("x"))
        assert replace_at(t, (), const("a")) == const("a")

    def test_first_child(self) -> None:
        t = op("f", var("x"), var("y"))
        result = replace_at(t, (0,), const("a"))
        assert result == op("f", const("a"), var("y"))

    def test_second_child(self) -> None:
        t = op("f", var("x"), var("y"))
        result = replace_at(t, (1,), const("b"))
        assert result == op("f", var("x"), const("b"))

    def test_nested(self) -> None:
        inner = op("g", var("x"), var("y"))
        outer = op("f", inner, var("z"))
        result = replace_at(outer, (0, 1), const("a"))
        expected = op("f", op("g", var("x"), const("a")), var("z"))
        assert result == expected

    def test_deeply_nested(self) -> None:
        t = op("f", op("g", op("h", var("x"))))
        result = replace_at(t, (0, 0, 0), const("a"))
        expected = op("f", op("g", op("h", const("a"))))
        assert result == expected

    def test_invalid_position(self) -> None:
        t = op("f", var("x"))
        with pytest.raises(IndexError):
            replace_at(t, (5,), const("a"))

    def test_descend_into_leaf(self) -> None:
        with pytest.raises(IndexError):
            replace_at(const("e"), (0,), const("a"))


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------


class TestPositions:
    def test_leaf(self) -> None:
        assert positions(var("x")) == [()]

    def test_const(self) -> None:
        assert positions(const("e")) == [()]

    def test_single_level(self) -> None:
        t = op("f", var("x"), var("y"))
        ps = positions(t)
        assert () in ps
        assert (0,) in ps
        assert (1,) in ps
        assert len(ps) == 3

    def test_nested(self) -> None:
        t = op("f", op("g", var("x")), var("y"))
        ps = positions(t)
        assert () in ps
        assert (0,) in ps
        assert (0, 0) in ps
        assert (1,) in ps
        assert len(ps) == 4


# ---------------------------------------------------------------------------
# find_redex
# ---------------------------------------------------------------------------


class TestFindRedex:
    def test_match_at_root(self) -> None:
        e = const("e")
        x = var("x")
        r = rule(op("mul", e, x), x, name="left_id")
        engine = RewriteEngine(rules=[r])

        t = op("mul", e, const("a"))
        redex = engine.find_redex(t)
        assert redex is not None
        assert redex.position == ()
        assert redex.rule.name == "left_id"

    def test_match_at_child(self) -> None:
        e = const("e")
        x = var("x")
        r = rule(op("mul", e, x), x, name="left_id")
        engine = RewriteEngine(rules=[r])

        # mul(a, mul(e, b)) -- redex is at position (1,)
        t = op("mul", const("a"), op("mul", e, const("b")))
        redex = engine.find_redex(t)
        assert redex is not None
        assert redex.position == (1,)

    def test_no_match(self) -> None:
        r = rule(op("mul", const("e"), var("x")), var("x"))
        engine = RewriteEngine(rules=[r])

        t = op("mul", const("a"), const("b"))
        assert engine.find_redex(t) is None

    def test_multiple_rules_first_wins(self) -> None:
        r1 = rule(op("f", var("x")), const("a"), name="r1")
        r2 = rule(op("f", var("x")), const("b"), name="r2")
        engine = RewriteEngine(rules=[r1, r2])

        t = op("f", const("x"))
        redex = engine.find_redex(t)
        assert redex is not None
        assert redex.rule.name == "r1"


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------


class TestStep:
    def test_rewrites_root(self) -> None:
        e = const("e")
        x = var("x")
        r = rule(op("mul", e, x), x)
        engine = RewriteEngine(rules=[r])

        t = op("mul", e, const("a"))
        result = engine.step(t)
        assert result == const("a")

    def test_normal_form(self) -> None:
        r = rule(op("f", var("x")), var("x"))
        engine = RewriteEngine(rules=[r])

        t = const("a")
        assert engine.step(t) is None


# ---------------------------------------------------------------------------
# reduce -- basic
# ---------------------------------------------------------------------------


class TestReduceBasic:
    def test_single_step(self) -> None:
        e = const("e")
        x = var("x")
        r = rule(op("mul", e, x), x)
        engine = RewriteEngine(rules=[r])

        t = op("mul", e, const("a"))
        assert engine.reduce(t) == const("a")

    def test_multiple_steps(self) -> None:
        e = const("e")
        x = var("x")
        r = rule(op("mul", e, x), x)
        engine = RewriteEngine(rules=[r])

        # mul(e, mul(e, a)) → mul(e, a) → a
        t = op("mul", e, op("mul", e, const("a")))
        assert engine.reduce(t) == const("a")

    def test_already_normal(self) -> None:
        r = rule(op("f", var("x")), var("x"))
        engine = RewriteEngine(rules=[r])

        t = const("a")
        assert engine.reduce(t) == const("a")

    def test_no_rules(self) -> None:
        engine = RewriteEngine(rules=[])
        t = op("f", const("a"))
        assert engine.reduce(t) == t


# ---------------------------------------------------------------------------
# reduce -- strategies
# ---------------------------------------------------------------------------


class TestStrategies:
    def test_innermost_normalizes_children_first(self) -> None:
        """With innermost strategy, children are normalized before root."""
        # Rule: f(x) → x
        r = rule(op("f", var("x")), var("x"))
        engine = RewriteEngine(rules=[r], strategy=Strategy.INNERMOST)

        # f(f(f(a))) → f(f(a)) → f(a) → a
        t = op("f", op("f", op("f", const("a"))))
        assert engine.reduce(t) == const("a")

    def test_outermost_applies_root_first(self) -> None:
        """With outermost strategy, root rules fire before descending."""
        # Rule: f(x) → x
        r = rule(op("f", var("x")), var("x"))
        engine = RewriteEngine(rules=[r], strategy=Strategy.OUTERMOST)

        t = op("f", op("f", op("f", const("a"))))
        assert engine.reduce(t) == const("a")

    def test_topdown(self) -> None:
        r = rule(op("f", var("x")), var("x"))
        engine = RewriteEngine(rules=[r], strategy=Strategy.TOPDOWN)

        t = op("f", op("f", op("f", const("a"))))
        assert engine.reduce(t) == const("a")

    def test_innermost_with_nested_redex(self) -> None:
        """Innermost: normalize the inner expression before applying outer rule."""
        # Rule: g(x) → x
        r = rule(op("g", var("x")), var("x"))
        engine = RewriteEngine(rules=[r], strategy=Strategy.INNERMOST)

        # f(g(a), g(b)) → f(a, b) (no rule on f, so stops here)
        t = op("f", op("g", const("a")), op("g", const("b")))
        result = engine.reduce(t)
        assert result == op("f", const("a"), const("b"))


# ---------------------------------------------------------------------------
# reduce -- cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    def test_cycle_detected(self) -> None:
        """Rules that create a cycle should be detected."""
        # Rule: a → b
        r1 = rule(const("a"), const("b"), name="a_to_b")
        # Rule: b → a
        r2 = rule(const("b"), const("a"), name="b_to_a")
        engine = RewriteEngine(rules=[r1, r2], track_visited=True)

        with pytest.raises(RewriteCycleDetectedError):
            engine.reduce(const("a"))

    def test_cycle_detection_disabled(self) -> None:
        """With track_visited=False, cycle detection is off."""
        r1 = rule(const("a"), const("b"))
        r2 = rule(const("b"), const("a"))
        engine = RewriteEngine(rules=[r1, r2], track_visited=False, max_steps=10)
        # Should hit max_steps instead of cycle detection
        with pytest.raises(ReductionLimitExceededError):
            engine.reduce(const("a"))

    def test_no_false_positive_on_different_terms(self) -> None:
        """Different terms with same hash shouldn't trigger false cycle."""
        # This test is mostly a sanity check -- distinct terms have distinct hashes
        # in our implementation since they're frozen dataclasses.
        r = rule(op("f", var("x")), var("x"))
        engine = RewriteEngine(rules=[r], track_visited=True)

        # f(a) → a, f(b) → b -- two different reductions
        t = op("tuple", op("f", const("a")), op("f", const("b")))
        # No rule on tuple, so engine just rewrites children
        # Actually with innermost, it tries children first.  tuple has no rule.
        result = engine.reduce(t)
        assert result == op("tuple", const("a"), const("b"))


# ---------------------------------------------------------------------------
# reduce -- max_steps
# ---------------------------------------------------------------------------


class TestMaxSteps:
    def test_exceeds_limit(self) -> None:
        """A non-terminating rule set should hit max_steps."""
        # Rule: f(x) → f(x)  (identity rewrite -- infinite loop)
        r = rule(op("f", var("x")), op("f", var("x")))
        engine = RewriteEngine(rules=[r], max_steps=5, track_visited=False)

        with pytest.raises(ReductionLimitExceededError) as exc_info:
            engine.reduce(op("f", const("a")))
        assert exc_info.value.steps == 5

    def test_custom_limit(self) -> None:
        r = rule(op("f", var("x")), op("f", var("x")))
        engine = RewriteEngine(rules=[r], max_steps=3, track_visited=False)

        with pytest.raises(ReductionLimitExceededError) as exc_info:
            engine.reduce(op("f", const("a")))
        assert exc_info.value.steps == 3

    def test_no_limit(self) -> None:
        """max_steps=0 means no limit.  Use with care."""
        # A terminating rule should complete fine with no limit
        r = rule(op("f", var("x")), var("x"))
        engine = RewriteEngine(rules=[r], max_steps=0)
        assert engine.reduce(op("f", const("a"))) == const("a")


# ---------------------------------------------------------------------------
# Integration: monoid rules
# ---------------------------------------------------------------------------


class TestMonoidIntegration:
    """End-to-end tests with a small monoid rule set."""

    @staticmethod
    def monoid_engine(strategy: Strategy = Strategy.INNERMOST) -> RewriteEngine:
        e = const("e")
        x = var("x")
        return RewriteEngine(
            rules=[
                rule(op("mul", e, x), x, name="left_id"),
                rule(op("mul", x, e), x, name="right_id"),
            ],
            strategy=strategy,
        )

    def test_left_identity(self) -> None:
        engine = self.monoid_engine()
        t = op("mul", const("e"), const("a"))
        assert engine.reduce(t) == const("a")

    def test_right_identity(self) -> None:
        engine = self.monoid_engine()
        t = op("mul", const("a"), const("e"))
        assert engine.reduce(t) == const("a")

    def test_nested_identity(self) -> None:
        engine = self.monoid_engine()
        # mul(e, mul(a, e)) → mul(e, a) → a
        t = op("mul", const("e"), op("mul", const("a"), const("e")))
        assert engine.reduce(t) == const("a")

    def test_deeply_nested(self) -> None:
        engine = self.monoid_engine()
        # mul(mul(e, a), mul(b, e)) → mul(a, mul(b, e)) → mul(a, b)
        t = op(
            "mul",
            op("mul", const("e"), const("a")),
            op("mul", const("b"), const("e")),
        )
        assert engine.reduce(t) == op("mul", const("a"), const("b"))

    def test_outermost_strategy(self) -> None:
        engine = self.monoid_engine(strategy=Strategy.OUTERMOST)
        t = op("mul", const("e"), op("mul", const("a"), const("e")))
        assert engine.reduce(t) == const("a")

    def test_term_already_simplified(self) -> None:
        engine = self.monoid_engine()
        t = op("mul", const("a"), const("b"))
        assert engine.reduce(t) == t


# ---------------------------------------------------------------------------
# Integration: group rules
# ---------------------------------------------------------------------------


class TestGroupIntegration:
    """End-to-end tests with group axioms (monoid + inverse)."""

    @staticmethod
    def group_engine() -> RewriteEngine:
        e = const("e")
        x = var("x")
        return RewriteEngine(
            rules=[
                rule(op("mul", e, x), x, name="left_id"),
                rule(op("mul", x, e), x, name="right_id"),
                rule(op("mul", op("inv", x), x), e, name="left_inv"),
                rule(op("mul", x, op("inv", x)), e, name="right_inv"),
            ],
            strategy=Strategy.INNERMOST,
        )

    def test_inverse_cancellation(self) -> None:
        engine = self.group_engine()
        t = op("mul", op("inv", const("a")), const("a"))
        assert engine.reduce(t) == const("e")

    def test_inverse_right(self) -> None:
        engine = self.group_engine()
        t = op("mul", const("a"), op("inv", const("a")))
        assert engine.reduce(t) == const("e")

    def test_nested_inverse(self) -> None:
        engine = self.group_engine()
        # mul(inv(a), a) → e  via left_inv
        # then mul(e, b) → b  via left_id
        t = op("mul", op("mul", op("inv", const("a")), const("a")), const("b"))
        assert engine.reduce(t) == const("b")

    def test_inverse_of_identity(self) -> None:
        engine = self.group_engine()
        # inv(e) -- no rule applies, stays as-is
        t = op("inv", const("e"))
        assert engine.reduce(t) == t


# ---------------------------------------------------------------------------
# Redex repr
# ---------------------------------------------------------------------------


class TestRedexRepr:
    def test_redex_fields(self) -> None:
        r = rule(op("f", var("x")), var("x"), name="simplify")
        subst = r.try_match(op("f", const("a")))
        assert subst is not None
        redex = Redex(rule=r, position=(1,), substitution=subst)
        assert redex.position == (1,)
        assert redex.rule.name == "simplify"
