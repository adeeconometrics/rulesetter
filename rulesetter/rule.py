from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rulesetter.term import (
    Const,
    Op,
    Term,
    Var,
    pformat,
)

# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Substitution:
    """A mapping from variables to terms.

    Substitutions are applied to terms to produce new terms.  They are built
    during pattern matching and used when applying rewrite rules.
    """

    bindings: dict[Var, Term]

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(self, term: Term) -> Term:
        """Apply this substitution to *term*, replacing each bound variable.

        The replacement is structural -- it descends into subterms and replaces
        variables wherever they appear.  Unbound variables are left unchanged.
        """
        if isinstance(term, Var):
            return self.bindings.get(term, term)
        if isinstance(term, Const):
            return term
        if isinstance(term, Op):
            new_args = tuple(self.apply(a) for a in term.args)
            if new_args == term.args:
                return term  # structural sharing
            return Op(term.name, new_args)
        raise TypeError(f"unknown term type: {type(term)}")

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self, other: Substitution) -> Substitution:
        """Return the composition ``other · self``.

        The resulting substitution, when applied to a term *t*, is equivalent
        to first applying ``self`` then ``other``::

            (other.compose(self)).apply(t) == other.apply(self.apply(t))
        """
        new_bindings: dict[Var, Term] = {}
        for v, t in self.bindings.items():
            new_bindings[v] = other.apply(t)
        for v, t in other.bindings.items():
            if v not in self.bindings:
                new_bindings[v] = t
        return Substitution(new_bindings)

    def __or__(self, other: Substitution) -> Substitution:
        """Shorthand for ``self.compose(other)``."""
        return self.compose(other)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        return len(self.bindings) == 0

    def vars(self) -> set[Var]:
        """Return the set of variables bound by this substitution."""
        return set(self.bindings.keys())

    def __repr__(self) -> str:
        if not self.bindings:
            return "Substitution({})"
        items = ", ".join(f"{v.name} → {pformat(t)}" for v, t in self.bindings.items())
        return f"Substitution({{{items}}})"


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


def match(pattern: Term, term: Term) -> Optional[Substitution]:
    """Match *pattern* against *term*, returning a substitution or ``None``.

    A pattern is a term that may contain :class:`Var` nodes.  Variables act
    as wildcards: they match any subterm.  The same variable appearing
    multiple times in the pattern must match the *same* subterm (consistency
    check).

    Returns ``None`` if the pattern does not match.
    """
    return _match(pattern, term, Substitution({}))


def _match(pattern: Term, term: Term, subst: Substitution) -> Optional[Substitution]:
    """Internal recursive matching with accumulated substitution."""
    if isinstance(pattern, Var):
        # Variable: bind if unbound, else check consistency
        if pattern in subst.bindings:
            if subst.bindings[pattern] == term:
                return subst
            return None  # inconsistency: same var, different terms
        return Substitution({**subst.bindings, pattern: term})

    if isinstance(pattern, Const):
        if isinstance(term, Const) and pattern.name == term.name:
            return subst
        return None

    if isinstance(pattern, Op):
        if not isinstance(term, Op):
            return None
        if pattern.name != term.name:
            return None
        if len(pattern.args) != len(term.args):
            return None
        # Match argument-by-argument, threading the substitution
        current = subst
        for p_arg, t_arg in zip(pattern.args, term.args, strict=True):
            result = _match(p_arg, t_arg, current)
            if result is None:
                return None
            current = result
        return current

    raise TypeError(f"unknown term type: {type(pattern)}")


# ---------------------------------------------------------------------------
# Rewrite rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rule:
    """A directed rewrite rule ``lhs → rhs``.

    *lhs* (left-hand side) is a pattern that may contain variables.  When the
    rule is applied at a position where *lhs* matches the local subterm, the
    matched subterm is replaced by *rhs* with the variable bindings applied.

    *name* is an optional human-readable label (e.g. ``"left_id"``).
    """

    lhs: Term
    rhs: Term
    name: str = ""

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def try_match(self, term: Term) -> Optional[Substitution]:
        """Try to match the LHS against *term*."""
        return match(self.lhs, term)

    def apply(self, term: Term) -> Optional[Term]:
        """Apply this rule at the root of *term*.

        Returns the rewritten term, or ``None`` if the LHS does not match.
        """
        subst = self.try_match(term)
        if subst is None:
            return None
        return subst.apply(self.rhs)

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lhs_str = pformat(self.lhs)
        rhs_str = pformat(self.rhs)
        label = f" [{self.name}]" if self.name else ""
        return f"Rule({lhs_str} → {rhs_str}{label})"


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def rule(lhs: Term, rhs: Term, name: str = "") -> Rule:
    """Shorthand for ``Rule(lhs, rhs, name)``."""
    return Rule(lhs, rhs, name)
