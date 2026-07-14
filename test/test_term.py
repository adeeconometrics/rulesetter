"""Tests for rulesetter.term."""

import pytest

from rulesetter.term import Op, const, op, pformat, var

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestVar:
    def test_frozen(self) -> None:
        v = var("x")
        with pytest.raises(AttributeError):
            v.name = "y"  # type: ignore[misc]

    def test_hash(self) -> None:
        assert hash(var("x")) == hash(var("x"))
        assert hash(var("x")) != hash(var("y"))
        # Usable in sets
        s = {var("x"), var("x"), var("y")}
        assert len(s) == 2

    def test_equality(self) -> None:
        assert var("x") == var("x")
        assert var("x") != var("y")
        assert var("x") != const("x")


class TestConst:
    def test_frozen(self) -> None:
        c = const("e")
        with pytest.raises(AttributeError):
            c.name = "f"  # type: ignore[misc]

    def test_equality(self) -> None:
        assert const("e") == const("e")
        assert const("e") != const("0")

    def test_children(self) -> None:
        assert const("e").children() == []


class TestOp:
    def test_frozen(self) -> None:
        t = op("mul", var("x"), var("y"))
        with pytest.raises(AttributeError):
            t.name = "add"  # type: ignore[misc]

    def test_no_args_raises(self) -> None:
        with pytest.raises(ValueError, match="no arguments"):
            Op("f", ())

    def test_arity(self) -> None:
        t = op("add", var("x"), var("y"))
        assert len(t.args) == 2
        assert t.name == "add"

    def test_children(self) -> None:
        x, y = var("x"), var("y")
        t = op("mul", x, y)
        assert t.children() == [x, y]

    def test_equality(self) -> None:
        a = op("f", var("x"))
        b = op("f", var("x"))
        c = op("f", var("y"))
        assert a == b
        assert a != c
        assert a != op("g", var("x"))


# ---------------------------------------------------------------------------
# Tree queries
# ---------------------------------------------------------------------------


class TestSize:
    def test_leaf(self) -> None:
        assert var("x").size() == 1
        assert const("e").size() == 1

    def test_nested(self) -> None:
        # mul(x, inv(x)) has 4 nodes
        t = op("mul", var("x"), op("inv", var("x")))
        assert t.size() == 4

    def test_deep(self) -> None:
        # f(f(f(x))) has 4 nodes
        t = op("f", op("f", op("f", var("x"))))
        assert t.size() == 4


class TestDepth:
    def test_leaf(self) -> None:
        assert var("x").depth() == 0
        assert const("e").depth() == 0

    def test_single_level(self) -> None:
        t = op("f", var("x"), var("y"))
        assert t.depth() == 1

    def test_nested(self) -> None:
        t = op("f", op("g", var("x")))
        assert t.depth() == 2


class TestSubterm:
    def test_root(self) -> None:
        t = op("f", var("x"))
        assert t.subterm(()) is t

    def test_first_child(self) -> None:
        x, y = var("x"), var("y")
        t = op("f", x, y)
        assert t.subterm((0,)) is x
        assert t.subterm((1,)) is y

    def test_nested_path(self) -> None:
        inner = op("g", var("x"))
        outer = op("f", inner, var("y"))
        assert outer.subterm((0, 0)) == var("x")

    def test_out_of_range(self) -> None:
        t = op("f", var("x"))
        with pytest.raises(IndexError):
            t.subterm((5,))


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


class TestPformat:
    def test_var(self) -> None:
        assert pformat(var("x")) == "x"

    def test_const(self) -> None:
        assert pformat(const("e")) == "e"

    def test_op_binary(self) -> None:
        t = op("mul", var("x"), var("y"))
        assert pformat(t) == "mul(x, y)"

    def test_op_nested(self) -> None:
        t = op("mul", op("add", var("x"), var("y")), var("z"))
        assert pformat(t) == "mul(add(x, y), z)"

    def test_op_unary(self) -> None:
        t = op("inv", var("x"))
        assert pformat(t) == "inv(x)"
