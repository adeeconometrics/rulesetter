from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Position: path from root to a subterm
# ---------------------------------------------------------------------------

Position = tuple[int, ...]
"""A path of child indices from the root of a term to a subterm.

``()`` is the root. ``(0,)`` is the first child. ``(1, 2)`` is the third child
of the second child, etc.
"""


# ---------------------------------------------------------------------------
# Term AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Term:
    """Base class for all terms in the rewrite system.

    Terms are immutable, hashable, and form finite trees.  Three concrete
    subtypes exist:

    * :class:`Var`  -- a variable (pattern wildcard)
    * :class:`Const` -- a named constant (e.g. identity element)
    * :class:`Op`   -- an n-ary operation applied to arguments
    """

    def children(self) -> list[Term]:
        """Return the direct subterms of this term."""
        return []

    def size(self) -> int:
        """Number of nodes in the term tree."""
        return 1

    def depth(self) -> int:
        """Height of the term tree (leaf = 0)."""
        return 0

    def subterm(self, pos: Position) -> Term:
        """Return the subterm at *pos*, or raise ``IndexError``."""
        if pos == ():
            return self
        raise IndexError(f"position {pos} out of range for term {self!r}")

    def subterm_size(self, pos: Position) -> int:
        """Size of the subterm rooted at *pos*."""
        return self.subterm(pos).size()

    def _repr_args(self) -> str:
        return ""


@dataclass(frozen=True, slots=True)
class Var(Term):
    """A pattern variable.

    Variables match any term during pattern matching.  Two variables with the
    same name in a single rule refer to the same matched subterm.
    """

    name: str

    def _repr_args(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Const(Term):
    """A named constant (zero-ary function symbol).

    Examples: ``Const("e")`` for a monoid identity, ``Const("0")`` for zero.
    """

    name: str

    def _repr_args(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Op(Term):
    """An operation applied to a fixed number of argument terms.

    ``Op("mul", (x, y))`` represents the application of the binary operation
    ``mul`` to subterms ``x`` and ``y``.  The arity is implicit in the length
    of ``args``.
    """

    name: str
    args: tuple[Term, ...]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.args:
            raise ValueError(
                f"Op {self.name!r} has no arguments; use Const({self.name!r}) for a zero-ary symbol"
            )

    # ------------------------------------------------------------------
    # Tree queries
    # ------------------------------------------------------------------

    def children(self) -> list[Term]:
        return list(self.args)

    def size(self) -> int:
        return 1 + sum(a.size() for a in self.args)

    def depth(self) -> int:
        if not self.args:
            return 0
        return 1 + max(a.depth() for a in self.args)

    def subterm(self, pos: Position) -> Term:
        if pos == ():
            return self
        head, *rest = pos
        if head < 0 or head >= len(self.args):
            raise IndexError(
                f"child index {head} out of range for {self.name!r} (arity {len(self.args)})"
            )
        return self.args[head].subterm(tuple(rest))

    def _repr_args(self) -> str:
        arg_strs = ", ".join(repr(a) for a in self.args)
        return f"{self.name}, ({arg_strs})"


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def var(name: str) -> Var:
    """Shorthand for ``Var(name)``."""
    return Var(name)


def const(name: str) -> Const:
    """Shorthand for ``Const(name)``."""
    return Const(name)


def op(name: str, *args: Term) -> Op:
    """Shorthand for ``Op(name, tuple(args))``."""
    return Op(name, tuple(args))


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _fmt(term: Term, prec: int = 0) -> str:
    """Pretty-print a term with minimal parentheses."""
    if isinstance(term, Var):
        return term.name
    if isinstance(term, Const):
        return term.name
    if isinstance(term, Op):
        inner = ", ".join(_fmt(a) for a in term.args)
        s = f"{term.name}({inner})"
        if prec > 0:
            s = f"({s})"
        return s
    raise TypeError(f"unknown term type: {type(term)}")


def pformat(term: Term) -> str:
    """Return a human-readable string for *term*."""
    return _fmt(term)


def pprint(term: Term) -> None:
    """Print *term* to stdout."""
    print(pformat(term))
