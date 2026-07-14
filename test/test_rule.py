"""Tests for rulesetter.rule -- substitution, matching, and rule application."""

from rulesetter.rule import Substitution, match, rule
from rulesetter.term import const, op, var

# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------


class TestSubstitution:
    def test_apply_var_bound(self) -> None:
        s = Substitution({var("x"): var("y")})
        assert s.apply(var("x")) == var("y")

    def test_apply_var_unbound(self) -> None:
        s = Substitution({var("x"): var("y")})
        assert s.apply(var("z")) == var("z")

    def test_apply_const(self) -> None:
        s = Substitution({var("x"): const("e")})
        assert s.apply(const("e")) == const("e")

    def test_apply_op(self) -> None:
        s = Substitution({var("x"): const("a"), var("y"): const("b")})
        t = op("f", var("x"), var("y"))
        assert s.apply(t) == op("f", const("a"), const("b"))

    def test_apply_partial(self) -> None:
        s = Substitution({var("x"): const("a")})
        t = op("f", var("x"), var("y"))
        result = s.apply(t)
        assert result == op("f", const("a"), var("y"))

    def test_apply_nested(self) -> None:
        s = Substitution({var("x"): const("a")})
        t = op("f", op("g", var("x"), var("y")))
        result = s.apply(t)
        expected = op("f", op("g", const("a"), var("y")))
        assert result == expected

    def test_compose(self) -> None:
        s1 = Substitution({var("x"): var("y")})
        s2 = Substitution({var("y"): const("a")})
        s3 = s1.compose(s2)
        assert s3.apply(var("x")) == const("a")
        assert s3.apply(var("y")) == const("a")

    def test_compose_preserves_unbound(self) -> None:
        s1 = Substitution({var("x"): var("y")})
        s2 = Substitution({var("y"): const("a")})
        s3 = s1.compose(s2)
        # z was never bound
        assert s3.apply(var("z")) == var("z")

    def test_is_empty(self) -> None:
        assert Substitution({}).is_empty() is True
        assert Substitution({var("x"): const("a")}).is_empty() is False

    def test_vars(self) -> None:
        s = Substitution({var("x"): const("a"), var("y"): const("b")})
        assert s.vars() == {var("x"), var("y")}

    def test_repr(self) -> None:
        s = Substitution({var("x"): const("e")})
        assert "x" in repr(s)
        assert "e" in repr(s)


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


class TestMatch:
    def test_var_matches_any(self) -> None:
        s = match(var("x"), const("e"))
        assert s is not None
        assert s.apply(var("x")) == const("e")

    def test_var_matches_op(self) -> None:
        t = op("f", const("a"))
        s = match(var("x"), t)
        assert s is not None
        assert s.apply(var("x")) == t

    def test_const_matches_same(self) -> None:
        s = match(const("e"), const("e"))
        assert s is not None

    def test_const_no_match(self) -> None:
        assert match(const("e"), const("f")) is None
        assert match(const("e"), var("x")) is None

    def test_op_matches_same(self) -> None:
        p = op("f", var("x"), const("a"))
        t = op("f", const("b"), const("a"))
        s = match(p, t)
        assert s is not None
        assert s.apply(var("x")) == const("b")

    def test_op_name_mismatch(self) -> None:
        assert match(op("f", var("x")), op("g", var("x"))) is None

    def test_op_arity_mismatch(self) -> None:
        assert match(op("f", var("x")), op("f", var("x"), var("y"))) is None

    def test_same_var_consistent(self) -> None:
        # Pattern: f(x, x) -- both args must match the same term
        p = op("f", var("x"), var("x"))
        s = match(p, op("f", const("a"), const("a")))
        assert s is not None
        assert s.apply(var("x")) == const("a")

    def test_same_var_inconsistent(self) -> None:
        p = op("f", var("x"), var("x"))
        s = match(p, op("f", const("a"), const("b")))
        assert s is None

    def test_distinct_vars(self) -> None:
        p = op("f", var("x"), var("y"))
        s = match(p, op("f", const("a"), const("b")))
        assert s is not None
        assert s.apply(var("x")) == const("a")
        assert s.apply(var("y")) == const("b")

    def test_nested_pattern(self) -> None:
        p = op("f", op("g", var("x")))
        t = op("f", op("g", const("a")))
        s = match(p, t)
        assert s is not None
        assert s.apply(var("x")) == const("a")

    def test_deeply_nested(self) -> None:
        p = op("f", op("g", op("h", var("x"))))
        t = op("f", op("g", op("h", const("a"))))
        s = match(p, t)
        assert s is not None
        assert s.apply(var("x")) == const("a")

    def test_match_root_only(self) -> None:
        """Matching succeeds only if the whole pattern matches the whole term."""
        p = op("f", var("x"))
        t = op("g", op("f", const("a")))
        # Pattern f(x) should NOT match g(f(a)) -- the roots differ
        assert match(p, t) is None

    def test_empty_op_args_differ(self) -> None:
        """Arity mismatch prevents matching."""
        p = op("f", var("x"), var("y"))
        t = op("f", const("a"))
        assert match(p, t) is None


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------


class TestRule:
    def test_apply_at_root(self) -> None:
        # Rule: mul(e, x) → x
        e = const("e")
        x = var("x")
        r = rule(op("mul", e, x), x, name="left_id")
        t = op("mul", e, const("a"))
        result = r.apply(t)
        assert result == const("a")

    def test_apply_no_match(self) -> None:
        r = rule(op("f", var("x")), var("x"))
        t = op("g", const("a"))
        assert r.apply(t) is None

    def test_apply_substitutes_rhs(self) -> None:
        # Rule: f(x) → g(x, x)
        r = rule(op("f", var("x")), op("g", var("x"), var("x")))
        t = op("f", const("a"))
        result = r.apply(t)
        assert result == op("g", const("a"), const("a"))

    def test_apply_complex_rhs(self) -> None:
        # Rule: f(x, y) → g(y, f(x, y))
        r = rule(
            op("f", var("x"), var("y")),
            op("g", var("y"), op("f", var("x"), var("y"))),
        )
        t = op("f", const("a"), const("b"))
        result = r.apply(t)
        expected = op("g", const("b"), op("f", const("a"), const("b")))
        assert result == expected

    def test_rule_with_name(self) -> None:
        r = rule(const("e"), var("x"), name="test")
        assert "test" in repr(r)

    def test_rule_repr(self) -> None:
        r = rule(op("f", var("x")), var("x"))
        s = repr(r)
        assert "f(x)" in s
        assert "→" in s

    def test_try_match(self) -> None:
        r = rule(op("f", var("x")), var("x"))
        t = op("f", const("a"))
        subst = r.try_match(t)
        assert subst is not None
        assert subst.apply(var("x")) == const("a")


# ---------------------------------------------------------------------------
# End-to-end: monoid left identity
# ---------------------------------------------------------------------------


class TestMonoidLeftId:
    """Integration test: mul(e, x) → x as the first monoid rule."""

    def test_simplification(self) -> None:
        e = const("e")
        x = var("x")
        mul = "mul"

        left_id = rule(op(mul, e, x), x, name="left_id")

        # mul(e, a) → a
        term = op(mul, e, const("a"))
        assert left_id.apply(term) == const("a")

        # mul(e, mul(e, a)) → mul(e, a) → a (first step only)
        nested = op(mul, e, op(mul, e, const("a")))
        assert left_id.apply(nested) == op(mul, e, const("a"))

    def test_no_match_without_identity(self) -> None:
        e = const("e")
        x = var("x")
        left_id = rule(op("mul", e, x), x)

        # mul(a, b) -- no identity in first position
        term = op("mul", const("a"), const("b"))
        assert left_id.apply(term) is None
