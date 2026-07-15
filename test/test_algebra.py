"""Tests for rulesetter.algebra -- Monoid and Group signatures."""

import pytest

from rulesetter.algebra.group import GroupSignature
from rulesetter.algebra.monoid import MonoidSignature
from rulesetter.term import const, op, var

# ---------------------------------------------------------------------------
# MonoidSignature
# ---------------------------------------------------------------------------


class TestMonoidSignature:
    def test_default_names(self) -> None:
        sig = MonoidSignature()
        assert sig.e == const("e")
        assert sig.mul(var("x"), var("y")) == op("mul", var("x"), var("y"))

    def test_custom_names(self) -> None:
        sig = MonoidSignature(identity_name="zero", mul_name="add")
        assert sig.e == const("zero")
        assert sig.mul(var("x"), var("y")) == op("add", var("x"), var("y"))

    def test_rules_count(self) -> None:
        sig = MonoidSignature()
        assert len(sig.rules()) == 2

    def test_rule_names(self) -> None:
        sig = MonoidSignature()
        names = {r.name for r in sig.rules()}
        assert names == {"left_id", "right_id"}

    def test_left_id_rule(self) -> None:
        sig = MonoidSignature()
        r = sig.rules()[0]
        # mul(e, x) → x
        assert r.apply(sig.mul(sig.e, const("a"))) == const("a")

    def test_right_id_rule(self) -> None:
        sig = MonoidSignature()
        r = sig.rules()[1]
        # mul(x, e) → x
        assert r.apply(sig.mul(const("a"), sig.e)) == const("a")

    def test_with_associativity_left(self) -> None:
        sig = MonoidSignature()
        rules = sig.with_associativity("left")
        assert len(rules) == 3
        names = {r.name for r in rules}
        assert "assoc_left" in names

    def test_with_associativity_right(self) -> None:
        sig = MonoidSignature()
        rules = sig.with_associativity("right")
        assert len(rules) == 3
        names = {r.name for r in rules}
        assert "assoc_right" in names

    def test_with_associativity_invalid(self) -> None:
        sig = MonoidSignature()
        with pytest.raises(ValueError, match="direction"):
            sig.with_associativity("diagonal")

    def test_engine_factory(self) -> None:
        sig = MonoidSignature()
        engine = sig.engine()
        t = sig.mul(sig.e, const("a"))
        assert engine.reduce(t) == const("a")

    def test_engine_with_assoc(self) -> None:
        sig = MonoidSignature()
        engine = sig.engine(include_associativity=True)
        assert len(engine.rules) == 3

    def test_var_convenience(self) -> None:
        sig = MonoidSignature()
        assert sig.var("x") == var("x")


# ---------------------------------------------------------------------------
# GroupSignature
# ---------------------------------------------------------------------------


class TestGroupSignature:
    def test_default_names(self) -> None:
        sig = GroupSignature()
        assert sig.e == const("e")
        assert sig.mul(var("x"), var("y")) == op("mul", var("x"), var("y"))
        assert sig.inv(var("x")) == op("inv", var("x"))

    def test_custom_names(self) -> None:
        sig = GroupSignature(identity_name="zero", mul_name="add", inv_name="neg")
        assert sig.e == const("zero")
        assert sig.inv(var("x")) == op("neg", var("x"))

    def test_rules_count(self) -> None:
        sig = GroupSignature()
        assert len(sig.rules()) == 4

    def test_rule_names(self) -> None:
        sig = GroupSignature()
        names = {r.name for r in sig.rules()}
        assert names == {"left_id", "right_id", "left_inv", "right_inv"}

    def test_left_inv_rule(self) -> None:
        sig = GroupSignature()
        r = [r for r in sig.rules() if r.name == "left_inv"][0]
        # mul(inv(x), x) → e
        assert r.apply(sig.mul(sig.inv(const("a")), const("a"))) == sig.e

    def test_right_inv_rule(self) -> None:
        sig = GroupSignature()
        r = [r for r in sig.rules() if r.name == "right_inv"][0]
        # mul(x, inv(x)) → e
        assert r.apply(sig.mul(const("a"), sig.inv(const("a")))) == sig.e

    def test_with_associativity(self) -> None:
        sig = GroupSignature()
        rules = sig.with_associativity("left")
        assert len(rules) == 5

    def test_engine_factory(self) -> None:
        sig = GroupSignature()
        engine = sig.engine()
        # mul(inv(a), a) → e
        t = sig.mul(sig.inv(const("a")), const("a"))
        assert engine.reduce(t) == sig.e

    def test_to_monoid(self) -> None:
        sig = GroupSignature()
        mon = sig.to_monoid()
        assert isinstance(mon, MonoidSignature)
        assert mon.identity_name == sig.identity_name
        assert mon.mul_name == sig.mul_name


# ---------------------------------------------------------------------------
# Integration: group simplification chains
# ---------------------------------------------------------------------------


class TestGroupSimplification:
    """End-to-end: group engine simplifying nested expressions."""

    @staticmethod
    def _engine() -> GroupSignature:
        return GroupSignature()

    def test_cancel_inverse_pair(self) -> None:
        sig = self._engine()
        engine = sig.engine()
        # mul(a, mul(inv(a), b)) -- innermost: mul(inv(a), a) won't match,
        # but mul(a, inv(a)) matches right_inv
        # Actually: innermost first normalizes children.
        #   mul(inv(a), b) has no rule → stays
        #   mul(a, mul(inv(a), b)) has no rule at root → stays
        # Let's use a term that actually works:
        # mul(mul(a, inv(a)), b) → mul(e, b) → b
        t = sig.mul(sig.mul(const("a"), sig.inv(const("a"))), const("b"))
        assert engine.reduce(t) == const("b")

    def test_inverse_times_inverse(self) -> None:
        sig = self._engine()
        engine = sig.engine()
        # inv(a) has no simplification rule -- stays as-is
        t = sig.inv(const("a"))
        assert engine.reduce(t) == t

    def test_identity_cancels(self) -> None:
        sig = self._engine()
        engine = sig.engine()
        t = sig.mul(sig.e, sig.mul(const("a"), sig.e))
        assert engine.reduce(t) == const("a")
