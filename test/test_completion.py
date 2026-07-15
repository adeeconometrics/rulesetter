"""Tests for rulesetter.completion -- unification, critical pairs, confluence."""

from rulesetter.algebra.group import GroupSignature
from rulesetter.algebra.monoid import MonoidSignature
from rulesetter.completion import (
    ConfluenceResult,
    CriticalPair,
    all_critical_pairs,
    check_confluence,
    occurs_in,
    overlaps,
    unify,
)
from rulesetter.rule import rule
from rulesetter.term import const, op, var

# ---------------------------------------------------------------------------
# occurs_in
# ---------------------------------------------------------------------------


class TestOccursIn:
    def test_var_in_self(self) -> None:
        assert occurs_in(var("x"), var("x")) is True

    def test_var_not_in_other_var(self) -> None:
        assert occurs_in(var("x"), var("y")) is False

    def test_var_in_const(self) -> None:
        assert occurs_in(var("x"), const("a")) is False

    def test_var_in_op(self) -> None:
        t = op("f", var("x"), const("a"))
        assert occurs_in(var("x"), t) is True
        assert occurs_in(var("y"), t) is False

    def test_var_in_nested_op(self) -> None:
        t = op("f", op("g", var("x")))
        assert occurs_in(var("x"), t) is True
        assert occurs_in(var("y"), t) is False


# ---------------------------------------------------------------------------
# Unification
# ---------------------------------------------------------------------------


class TestUnify:
    def test_identical_terms(self) -> None:
        s = unify(const("a"), const("a"))
        assert s is not None
        assert s.is_empty()

    def test_different_constants(self) -> None:
        assert unify(const("a"), const("b")) is None

    def test_var_and_term(self) -> None:
        s = unify(var("x"), const("a"))
        assert s is not None
        assert s.apply(var("x")) == const("a")

    def test_symmetric_var(self) -> None:
        s = unify(const("a"), var("x"))
        assert s is not None
        assert s.apply(var("x")) == const("a")

    def test_two_vars(self) -> None:
        s = unify(var("x"), var("y"))
        assert s is not None
        # Both should map to the same thing
        assert s.apply(var("x")) == s.apply(var("y"))

    def test_same_var(self) -> None:
        s = unify(var("x"), var("x"))
        assert s is not None
        assert s.is_empty()

    def test_occurs_check(self) -> None:
        # x = f(x) has no unifier
        t = op("f", var("x"))
        assert unify(var("x"), t) is None

    def test_op_same_name_arity(self) -> None:
        s = unify(op("f", var("x")), op("f", const("a")))
        assert s is not None
        assert s.apply(var("x")) == const("a")

    def test_op_different_name(self) -> None:
        assert unify(op("f", var("x")), op("g", var("x"))) is None

    def test_op_different_arity(self) -> None:
        assert unify(op("f", var("x")), op("f", var("x"), var("y"))) is None

    def test_nested_unification(self) -> None:
        s = unify(op("f", var("x"), const("a")), op("f", const("b"), var("y")))
        assert s is not None
        assert s.apply(var("x")) == const("b")
        assert s.apply(var("y")) == const("a")

    def test_deeply_nested(self) -> None:
        s = unify(
            op("f", op("g", var("x"))),
            op("f", op("g", const("a"))),
        )
        assert s is not None
        assert s.apply(var("x")) == const("a")

    def test_const_vs_op(self) -> None:
        assert unify(const("f"), op("f", var("x"))) is None

    def test_complex_unification(self) -> None:
        # f(x, g(y)) = f(a, g(b))
        s = unify(
            op("f", var("x"), op("g", var("y"))),
            op("f", const("a"), op("g", const("b"))),
        )
        assert s is not None
        assert s.apply(var("x")) == const("a")
        assert s.apply(var("y")) == const("b")

    def test_shared_variable(self) -> None:
        # f(x, x) = f(a, a) → x = a
        s = unify(
            op("f", var("x"), var("x")),
            op("f", const("a"), const("a")),
        )
        assert s is not None
        assert s.apply(var("x")) == const("a")

    def test_shared_variable_conflict(self) -> None:
        # f(x, x) = f(a, b) → fail (x can't be both a and b)
        assert (
            unify(
                op("f", var("x"), var("x")),
                op("f", const("a"), const("b")),
            )
            is None
        )


# ---------------------------------------------------------------------------
# Overlaps / critical pairs
# ---------------------------------------------------------------------------


class TestOverlaps:
    def test_no_overlap_different_ops(self) -> None:
        r1 = rule(op("f", var("x")), const("a"))
        r2 = rule(op("g", var("x")), const("b"))
        assert overlaps(r1, r2) == []

    def test_no_overlap_no_unification(self) -> None:
        r1 = rule(op("f", const("a")), const("x"))
        r2 = rule(op("f", const("b")), const("y"))
        assert overlaps(r1, r2) == []

    def test_root_overlap(self) -> None:
        # Two rules with same LHS pattern → root overlap
        r1 = rule(op("f", var("x")), const("a"))
        r2 = rule(op("f", var("x")), const("b"))
        pairs = overlaps(r1, r2)
        assert len(pairs) == 1
        assert pairs[0].left == const("a")
        assert pairs[0].right == const("b")

    def test_subterm_overlap(self) -> None:
        # r1: f(g(x)) → a
        # r2: g(x) → b
        # Overlap: r1's subterm g(x) unifies with r2's LHS
        r1 = rule(op("f", op("g", var("x"))), const("a"))
        r2 = rule(op("g", var("x")), const("b"))
        pairs = overlaps(r1, r2)
        assert len(pairs) >= 1
        # One critical pair: f(b) vs a
        found = any(
            (p.left == const("a") and p.right == op("f", const("b")))
            or (p.left == op("f", const("b")) and p.right == const("a"))
            for p in pairs
        )
        assert found

    def test_self_overlap(self) -> None:
        # r: f(f(x)) → x
        # Root overlap with itself: f(f(x)) vs f(f(x)) → x vs x → trivial
        # Subterm overlap: f(x) at pos (0,) vs f(f(x)) → x vs f(x) → occurs check fails
        r = rule(op("f", op("f", var("x"))), var("x"))
        pairs = overlaps(r, r)
        # No non-trivial critical pairs (rule is orthogonal)
        assert len(pairs) == 0

    def test_self_overlap_non_trivial(self) -> None:
        # r: f(x, x) → a
        # Root self-overlap: f(x, x) vs f(x, x) → a vs a → trivial
        # But if we have a rule that overlaps with itself non-trivially:
        # r1: f(x, g(y)) → x
        # r2: g(z) → z
        # Subterm g(y) in r1.lhs unifies with r2.lhs g(z)
        r1 = rule(op("f", var("x"), op("g", var("y"))), var("x"))
        r2 = rule(op("g", var("z")), var("z"))
        pairs = overlaps(r1, r2)
        assert len(pairs) >= 1
        # Critical pair: f(x, z) vs x
        found = any(p.left == var("x") and p.right == op("f", var("x"), var("z")) for p in pairs)
        assert found


class TestAllCriticalPairs:
    def test_empty_rules(self) -> None:
        assert all_critical_pairs([]) == []

    def test_single_rule(self) -> None:
        r = rule(op("f", var("x")), const("a"))
        pairs = all_critical_pairs([r])
        # Self-overlap at root: f(x) vs f(x) → a vs a → trivial, no pair
        # Self-overlap at subterm: none (var has no subterms)
        assert len(pairs) == 0

    def test_two_rules(self) -> None:
        r1 = rule(op("f", var("x")), const("a"))
        r2 = rule(op("f", var("x")), const("b"))
        pairs = all_critical_pairs([r1, r2])
        assert len(pairs) >= 1


# ---------------------------------------------------------------------------
# Confluence check
# ---------------------------------------------------------------------------


class TestCheckConfluence:
    def test_empty_trs(self) -> None:
        result = check_confluence([])
        assert result.is_confluent is True
        assert result.pairs == []
        assert result.non_joinable == []

    def test_single_rule_confluent(self) -> None:
        r = rule(op("f", var("x")), var("x"))
        result = check_confluence([r])
        assert result.is_confluent is True

    def test_two_overlapping_rules_confluent(self) -> None:
        # f(x) → x and f(f(x)) → x
        # Critical pair: f(x) vs x — joinable (both reduce to x)
        r1 = rule(op("f", var("x")), var("x"))
        r2 = rule(op("f", op("f", var("x"))), var("x"))
        result = check_confluence([r1, r2])
        assert result.is_confluent is True

    def test_non_confluent_rules(self) -> None:
        # a → b and a → c (two rules with same LHS, different RHS)
        # Critical pair: b vs c — not joinable (no rules apply)
        r1 = rule(const("a"), const("b"))
        r2 = rule(const("a"), const("c"))
        result = check_confluence([r1, r2])
        assert result.is_confluent is False
        assert len(result.non_joinable) >= 1

    def test_confluence_result_fields(self) -> None:
        r1 = rule(const("a"), const("b"))
        r2 = rule(const("a"), const("c"))
        result = check_confluence([r1, r2])
        assert isinstance(result, ConfluenceResult)
        assert isinstance(result.pairs, list)
        assert isinstance(result.non_joinable, list)
        assert isinstance(result.is_confluent, bool)


# ---------------------------------------------------------------------------
# Confluence of monoid rules
# ---------------------------------------------------------------------------


class TestMonoidConfluence:
    """Check confluence of the standard monoid/group rule sets."""

    def test_monoid_rules_confluent(self) -> None:
        sig = MonoidSignature()
        result = check_confluence(sig.rules())
        # left_id and right_id have no critical pairs with each other
        # because mul(e, x) and mul(x, e) don't overlap
        assert result.is_confluent is True

    def test_group_rules_confluent(self) -> None:
        sig = GroupSignature()
        result = check_confluence(sig.rules())
        # Group rules (left_id, right_id, left_inv, right_inv) should be
        # locally confluent
        assert result.is_confluent is True


# ---------------------------------------------------------------------------
# CriticalPair.is_joinable
# ---------------------------------------------------------------------------


class TestCriticalPairJoinable:
    def test_joinable_pair(self) -> None:
        from rulesetter.engine import RewriteEngine

        r = rule(op("f", var("x")), var("x"))
        engine = RewriteEngine(rules=[r])
        cp = CriticalPair(
            left=op("f", const("a")),
            right=const("a"),
            rule1=r,
            rule2=r,
            position=(),
        )
        assert cp.is_joinable(engine) is True

    def test_non_joinable_pair(self) -> None:
        from rulesetter.engine import RewriteEngine

        engine = RewriteEngine(rules=[])
        cp = CriticalPair(
            left=const("a"),
            right=const("b"),
            rule1=rule(const("a"), const("b")),
            rule2=rule(const("a"), const("c")),
            position=(),
        )
        assert cp.is_joinable(engine) is False
