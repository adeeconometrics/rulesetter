"""Tests for rulesetter.completion -- unification, critical pairs, confluence, KB completion."""

from rulesetter.algebra.group import GroupSignature
from rulesetter.algebra.monoid import MonoidSignature
from rulesetter.completion import (
    ConfluenceResult,
    CriticalPair,
    Equation,
    KBResult,
    Orientation,
    all_critical_pairs,
    check_confluence,
    compare_kbo,
    knuth_bendix,
    occurs_in,
    orient,
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


# ---------------------------------------------------------------------------
# Term ordering (compare_kbo)
# ---------------------------------------------------------------------------


class TestCompareKBO:
    def test_same_term(self) -> None:
        assert compare_kbo(const("a"), const("a"), {}) == 0

    def test_different_constants(self) -> None:
        # Both weight 1, same precedence 0, but different names
        # Lexicographic on root: 'a' < 'b'
        assert compare_kbo(const("a"), const("b"), {}) == -1

    def test_weight_difference(self) -> None:
        # f(a) has weight 1+0+1+0 = 2, a has weight 1
        assert compare_kbo(op("f", const("a")), const("a"), {}) > 0

    def test_precedence_difference(self) -> None:
        prec = {"f": 10, "g": 0}
        # Both f(a) and g(a) have weight 2, but f has higher precedence
        assert compare_kbo(op("f", const("a")), op("g", const("a")), prec) > 0

    def test_subterm_property(self) -> None:
        # f(a) > a because a is a subterm
        assert compare_kbo(op("f", const("a")), const("a"), {}) > 0

    def test_nested_comparison(self) -> None:
        # f(a) vs f(b) -- same weight, same root, compare args
        assert compare_kbo(op("f", const("a")), op("f", const("b")), {}) == -1

    def test_arity_difference(self) -> None:
        # f(a, b) vs f(a) -- f(a,b) is larger
        assert compare_kbo(op("f", const("a"), const("b")), op("f", const("a")), {}) > 0

    def test_symmetry(self) -> None:
        s = op("f", const("a"))
        t = const("b")
        assert compare_kbo(s, t, {}) == -compare_kbo(t, s, {})

    def test_var_weight(self) -> None:
        # Var has weight 1, same as Const, but Var is "smaller" in the ordering
        # (variables are leaves that can be substituted, constants are fixed)
        assert compare_kbo(var("x"), const("a"), {}) < 0
        assert compare_kbo(const("a"), var("x"), {}) > 0

    def test_transitivity(self) -> None:
        a = const("a")
        b = const("b")
        c = const("c")
        prec = {"a": 1, "b": 2, "c": 3}
        assert compare_kbo(a, b, prec) < 0
        assert compare_kbo(b, c, prec) < 0
        assert compare_kbo(a, c, prec) < 0


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


class TestOrient:
    def test_orient_left_to_right(self) -> None:
        eq = Equation(op("f", const("a")), const("a"))
        prec = {"f": 10}
        result = orient(eq, prec)
        assert result is not None
        r, d = result
        assert r.lhs == op("f", const("a"))
        assert r.rhs == const("a")
        assert d == Orientation.LEFT_TO_RIGHT

    def test_orient_right_to_left(self) -> None:
        eq = Equation(const("a"), op("f", const("a")))
        prec = {"f": 10}
        result = orient(eq, prec)
        assert result is not None
        r, d = result
        # f(a) > a, so the rule should be f(a) → a
        assert r.lhs == op("f", const("a"))
        assert r.rhs == const("a")
        assert d == Orientation.RIGHT_TO_LEFT

    def test_unorientable(self) -> None:
        # a = a -- same term, can't orient
        eq = Equation(const("a"), const("a"))
        result = orient(eq, {})
        assert result is None

    def test_orient_with_precedence(self) -> None:
        # f(x) = g(x) with f > g
        eq = Equation(op("f", var("x")), op("g", var("x")))
        prec = {"f": 10, "g": 0}
        result = orient(eq, prec)
        assert result is not None
        r, d = result
        assert r.lhs == op("f", var("x"))
        assert r.rhs == op("g", var("x"))


# ---------------------------------------------------------------------------
# Knuth-Bendix completion
# ---------------------------------------------------------------------------


class TestKnuthBendix:
    def test_empty_equations(self) -> None:
        result = knuth_bendix([], {})
        assert result.success is True
        assert result.rules == []
        assert result.steps == 0

    def test_single_equation(self) -> None:
        # f(a) = a
        eq = Equation(op("f", const("a")), const("a"))
        result = knuth_bendix([eq], {"f": 10})
        assert result.success is True
        assert len(result.rules) == 1
        assert result.rules[0].lhs == op("f", const("a"))
        assert result.rules[0].rhs == const("a")

    def test_commutative_equations_diverge(self) -> None:
        # x * y = y * x -- this can't be oriented as a terminating rule
        eq = Equation(
            op("mul", var("x"), var("y")),
            op("mul", var("y"), var("x")),
            name="comm",
        )
        # With default precedence, mul has same weight both ways
        # This should fail or diverge
        result = knuth_bendix([eq], {"mul": 5})
        # The algorithm should either succeed with no rules or diverge
        # (commutativity is famously non-orientable)
        assert result.success is True or result.diverged is True

    def test_associative_equations(self) -> None:
        # mul(mul(x, y), z) = mul(x, mul(y, z))
        # This is orientable with the right precedence
        eq = Equation(
            op("mul", op("mul", var("x"), var("y")), var("z")),
            op("mul", var("x"), op("mul", var("y"), var("z"))),
            name="assoc",
        )
        result = knuth_bendix([eq], {"mul": 5})
        # Should complete successfully
        assert result.success is True

    def test_two_equations(self) -> None:
        # f(a) = b and g(a) = c
        eq1 = Equation(op("f", const("a")), const("b"))
        eq2 = Equation(op("g", const("a")), const("c"))
        result = knuth_bendix([eq1, eq2], {"f": 10, "g": 10})
        assert result.success is True
        assert len(result.rules) == 2

    def test_monoid_equations(self) -> None:
        # e * x = x and x * e = x
        eq1 = Equation(
            op("mul", const("e"), var("x")),
            var("x"),
            name="left_id",
        )
        eq2 = Equation(
            op("mul", var("x"), const("e")),
            var("x"),
            name="right_id",
        )
        result = knuth_bendix([eq1, eq2], {"mul": 5, "e": 0})
        assert result.success is True
        assert len(result.rules) >= 2

    def test_group_equations(self) -> None:
        # Group axioms: e*x=x, x*e=x, inv(x)*x=e, x*inv(x)=e
        eqs = [
            Equation(op("mul", const("e"), var("x")), var("x"), "left_id"),
            Equation(op("mul", var("x"), const("e")), var("x"), "right_id"),
            Equation(op("mul", op("inv", var("x")), var("x")), const("e"), "left_inv"),
            Equation(op("mul", var("x"), op("inv", var("x"))), const("e"), "right_inv"),
        ]
        result = knuth_bendix(eqs, {"mul": 5, "inv": 3, "e": 0})
        assert result.success is True
        # Should have at least the 4 group rules
        assert len(result.rules) >= 4

    def test_divergence_budget(self) -> None:
        # An equation that causes many critical pairs
        eq = Equation(op("f", var("x")), op("g", var("x")))
        result = knuth_bendix([eq], {"f": 10, "g": 5}, max_rules=5, max_eqs=5)
        # Should either succeed or diverge within budget
        assert result.success is True or result.diverged is True

    def test_kb_result_fields(self) -> None:
        eq = Equation(op("f", const("a")), const("a"))
        result = knuth_bendix([eq], {"f": 10})
        assert isinstance(result, KBResult)
        assert isinstance(result.rules, list)
        assert isinstance(result.steps, int)
        assert isinstance(result.success, bool)
        assert isinstance(result.diverged, bool)
