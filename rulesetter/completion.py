from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rulesetter.engine import RewriteEngine, positions, replace_at
from rulesetter.rule import Rule, Substitution
from rulesetter.term import Const, Op, Position, Term, Var

# ---------------------------------------------------------------------------
# Unification
# ---------------------------------------------------------------------------


def occurs_in(v: Var, term: Term) -> bool:
    """Return ``True`` if variable *v* appears anywhere in *term*."""
    if isinstance(term, Var):
        return v == term
    if isinstance(term, Const):
        return False
    if isinstance(term, Op):
        return any(occurs_in(v, a) for a in term.args)
    raise TypeError(f"unknown term type: {type(term)}")


def unify(s: Term, t: Term) -> Optional[Substitution]:
    """Find a most general unifier (mgu) for terms *s* and *t*.

    Returns a :class:`Substitution` ``σ`` such that ``σ(s) == σ(t)``, or
    ``None`` if no unifier exists (terms are not unifiable).

    Uses the classic Robinson unification algorithm with occurs check.
    """
    return _unify(s, t, Substitution({}))


def _unify(s: Term, t: Term, subst: Substitution) -> Optional[Substitution]:
    """Internal recursive unification with accumulated substitution."""
    # Apply current bindings
    s = subst.apply(s)
    t = subst.apply(t)

    # Identical terms
    if s == t:
        return subst

    # Variable cases
    if isinstance(s, Var):
        if occurs_in(s, t):
            return None  # occurs check
        return Substitution({**subst.bindings, s: t})

    if isinstance(t, Var):
        if occurs_in(t, s):
            return None  # occurs check
        return Substitution({**subst.bindings, t: s})

    # Constant case
    if isinstance(s, Const) and isinstance(t, Const):
        return subst if s.name == t.name else None

    # Op case
    if isinstance(s, Op) and isinstance(t, Op):
        if s.name != t.name or len(s.args) != len(t.args):
            return None
        current = subst
        for a, b in zip(s.args, t.args, strict=True):
            result = _unify(a, b, current)
            if result is None:
                return None
            current = result
        return current

    # Mixed Const/Op or unknown types
    return None


# ---------------------------------------------------------------------------
# Critical pair
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CriticalPair:
    """A critical pair arising from an overlap between two rules.

    *left* and *right* are the two terms that result from reducing the
    overlapped term in two different ways.  The pair is *joinable* if both
    sides reduce to the same normal form.
    """

    left: Term
    right: Term
    rule1: Rule
    rule2: Rule
    position: Position

    def is_joinable(self, engine: RewriteEngine) -> bool:
        """Return ``True`` if both sides reduce to the same normal form."""
        try:
            nl = engine.reduce(self.left)
            nr = engine.reduce(self.right)
            return nl == nr
        except Exception:
            # If reduction fails (cycle, limit), treat as not joinable
            return False


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------


def overlaps(rule1: Rule, rule2: Rule) -> list[CriticalPair]:
    """Find all critical pairs between *rule1* and *rule2*.

    This includes:
    1. **Non-trivial overlaps**: a subterm of ``rule1.lhs`` at a non-root
       position unifies with ``rule2.lhs``.
    2. **Root overlap**: ``rule1.lhs`` itself unifies with ``rule2.lhs``
       (only when the two rules are distinct).

    Each overlap produces a :class:`CriticalPair` with the two divergent
    reducts.
    """
    pairs: list[CriticalPair] = []
    lhs1 = rule1.lhs

    for pos in positions(lhs1):
        subterm = lhs1.subterm(pos)

        # Root overlap: only between distinct rules
        if pos == () and rule1 is not rule2:
            sigma = unify(lhs1, rule2.lhs)
            if sigma is not None:
                # Left reduct: apply rule1 at root
                left = sigma.apply(rule1.rhs)
                # Right reduct: apply rule2 at root
                right = sigma.apply(rule2.rhs)
                if left != right:
                    pairs.append(
                        CriticalPair(
                            left=left,
                            right=right,
                            rule1=rule1,
                            rule2=rule2,
                            position=pos,
                        )
                    )
            continue

        # Non-root overlap: unify subterm with rule2.lhs
        if pos != ():
            sigma = unify(subterm, rule2.lhs)
            if sigma is not None:
                # Left reduct: apply rule1 at root
                left = sigma.apply(rule1.rhs)
                # Right reduct: replace subterm with rule2's rhs
                new_sub = sigma.apply(rule2.rhs)
                right = sigma.apply(replace_at(lhs1, pos, new_sub))
                if left != right:
                    pairs.append(
                        CriticalPair(
                            left=left,
                            right=right,
                            rule1=rule1,
                            rule2=rule2,
                            position=pos,
                        )
                    )

    return pairs


def all_critical_pairs(rules: list[Rule]) -> list[CriticalPair]:
    """Compute all critical pairs among the given rules.

    Returns pairs from all ordered pairs ``(rule_i, rule_j)`` including
    self-overlaps (where ``rule_i is rule_j``).
    """
    pairs: list[CriticalPair] = []
    for r1 in rules:
        for r2 in rules:
            pairs.extend(overlaps(r1, r2))
    return pairs


# ---------------------------------------------------------------------------
# Confluence check
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfluenceResult:
    """Result of a confluence check on a term rewriting system.

    *is_confluent* is ``True`` if all critical pairs are joinable.
    *pairs* lists every critical pair found.
    *non_joinable* lists only the pairs that failed to join.
    """

    is_confluent: bool
    pairs: list[CriticalPair]
    non_joinable: list[CriticalPair]


def check_confluence(
    rules: list[Rule],
    *,
    max_steps: int = 1000,
    track_visited: bool = True,
) -> ConfluenceResult:
    """Check whether *rules* form a locally confluent TRS.

    Computes all critical pairs and attempts to join each one by reducing
    both sides to normal form.  Returns a :class:`ConfluenceResult` with
    the outcome.

    Parameters
    ----------
    rules:
        The rewrite rules to check.
    max_steps:
        Maximum reduction steps per critical pair join attempt.
    track_visited:
        Enable cycle detection during reduction.
    """
    pairs = all_critical_pairs(rules)
    engine = RewriteEngine(
        rules=list(rules),
        max_steps=max_steps,
        track_visited=track_visited,
    )

    non_joinable: list[CriticalPair] = []
    for cp in pairs:
        if not cp.is_joinable(engine):
            non_joinable.append(cp)

    return ConfluenceResult(
        is_confluent=len(non_joinable) == 0,
        pairs=pairs,
        non_joinable=non_joinable,
    )
