from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from rulesetter.engine import RewriteEngine, positions, replace_at
from rulesetter.rule import Rule, Substitution
from rulesetter.term import Const, Op, Position, Term, Var

# ---------------------------------------------------------------------------
# Term ordering (simplified KBO)
# ---------------------------------------------------------------------------

SymbolPrecedence = dict[str, int]
"""Mapping from function symbol names to integer precedences.
Higher = greater.  Symbols not in the mapping default to 0."""


def _weight(term: Term, prec: SymbolPrecedence) -> int:
    """Compute the weight (node count + symbol precedences) of *term*."""
    if isinstance(term, Var):
        return 1
    if isinstance(term, Const):
        return 1 + prec.get(term.name, 0)
    if isinstance(term, Op):
        return 1 + prec.get(term.name, 0) + sum(_weight(a, prec) for a in term.args)
    raise TypeError(f"unknown term type: {type(term)}")


def _lex_compare(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Lexicographic comparison: -1, 0, or 1."""
    for x, y in zip(a, b, strict=False):
        if x < y:
            return -1
        if x > y:
            return 1
    if len(a) < len(b):
        return -1
    if len(a) > len(b):
        return 1
    return 0


def compare_kbo(s: Term, t: Term, prec: SymbolPrecedence) -> int:
    """Compare terms using a simplified Knuth-Bendix ordering (KBO).

    Returns -1 if s < t, 0 if s == t, 1 if s > t.

    The ordering is:
    1. Compare by weight (sum of node counts + symbol precedences)
    2. If equal, compare by root symbol precedence
    3. If still equal, compare arguments lexicographically
    4. Subterm property: if t is a subterm of s (and s != t), then s > t
    """
    if s == t:
        return 0

    # Weight comparison
    ws = _weight(s, prec)
    wt = _weight(t, prec)
    if ws < wt:
        return -1
    if ws > wt:
        return 1

    # Same weight: compare root symbol precedence
    def _root_prec(term: Term) -> int:
        if isinstance(term, Var):
            return 0
        if isinstance(term, Const):
            return prec.get(term.name, 0)
        if isinstance(term, Op):
            return prec.get(term.name, 0)
        raise TypeError(f"unknown term type: {type(term)}")

    ps = _root_prec(s)
    pt = _root_prec(t)
    if ps < pt:
        return -1
    if ps > pt:
        return 1

    # Same precedence: compare by symbol name
    def _root_name(term: Term) -> str:
        if isinstance(term, Var):
            return ""
        if isinstance(term, Const):
            return term.name
        if isinstance(term, Op):
            return term.name
        raise TypeError(f"unknown term type: {type(term)}")

    ns = _root_name(s)
    nt = _root_name(t)
    if ns < nt:
        return -1
    if ns > nt:
        return 1

    # Same root: compare arguments lexicographically
    def _args_key(term: Term) -> tuple[int, ...]:
        if isinstance(term, Var):
            return ()
        if isinstance(term, Const):
            return ()
        if isinstance(term, Op):
            return tuple(_weight(a, prec) for a in term.args)
        raise TypeError(f"unknown term type: {type(term)}")

    cmp = _lex_compare(_args_key(s), _args_key(t))
    if cmp != 0:
        return cmp

    # Fall back to structural comparison on arguments
    def _struct_args(term: Term) -> tuple[Term, ...]:
        if isinstance(term, Op):
            return term.args
        return ()

    sa = _struct_args(s)
    ta = _struct_args(t)
    for a, b in zip(sa, ta, strict=False):
        cmp = compare_kbo(a, b, prec)
        if cmp != 0:
            return cmp
    if len(sa) < len(ta):
        return -1
    if len(sa) > len(ta):
        return 1
    return 0


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
# Orientation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Equation:
    """An undirected equation ``lhs = rhs``."""

    lhs: Term
    rhs: Term
    name: str = ""


class Orientation(Enum):
    """Result of orienting an equation."""

    LEFT_TO_RIGHT = "ltr"
    RIGHT_TO_LEFT = "rtl"
    UNORIENTABLE = "unorientable"


def orient(eq: Equation, ordering: SymbolPrecedence) -> tuple[Rule, Orientation] | None:
    """Orient an equation into a directed rule.

    Returns ``(rule, direction)`` where *direction* indicates which way the
    equation was oriented.  Returns ``None`` if the equation is unorientable
    (neither direction is strictly greater in the ordering).
    """
    cmp = compare_kbo(eq.lhs, eq.rhs, ordering)
    if cmp > 0:
        return Rule(eq.lhs, eq.rhs, name=eq.name), Orientation.LEFT_TO_RIGHT
    if cmp < 0:
        return Rule(eq.rhs, eq.lhs, name=eq.name), Orientation.RIGHT_TO_LEFT
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

    This includes non-trivial overlaps (a subterm of ``rule1.lhs`` at a
    non-root position unifies with ``rule2.lhs``) and root overlaps
    (``rule1.lhs`` itself unifies with ``rule2.lhs``, only when the two
    rules are distinct).

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
# Simplification helpers
# ---------------------------------------------------------------------------


def _simplify_term(term: Term, rules: list[Rule], max_steps: int = 100) -> Term:
    """Fully reduce *term* using the given rules."""
    engine = RewriteEngine(rules=rules, max_steps=max_steps, track_visited=False)
    return engine.reduce(term)


def _simplify_rhs(rule: Rule, rules: list[Rule]) -> Rule:
    """Simplify the RHS of a rule using all other rules."""
    new_rhs = _simplify_term(rule.rhs, rules)
    if new_rhs == rule.rhs:
        return rule
    return Rule(rule.lhs, new_rhs, name=rule.name)


def _simplify_lhs(rule: Rule, rules: list[Rule]) -> Rule | Equation | None:
    """Try to simplify the LHS of a rule using other rules.

    If the LHS reduces to a smaller term, returns a new rule.
    If it reduces but can't be re-oriented, returns an Equation.
    If it doesn't reduce, returns the original rule.
    """
    reduced = _simplify_term(rule.lhs, rules)
    if reduced == rule.lhs:
        return rule
    # LHS was reducible -- this rule is now an equation
    return Equation(reduced, rule.rhs, name=rule.name)


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


# ---------------------------------------------------------------------------
# Knuth-Bendix completion
# ---------------------------------------------------------------------------

# Maximum number of rules before declaring divergence
_MAX_RULES = 200

# Maximum number of pending equations before declaring divergence
_MAX_EQS = 200


@dataclass
class KBResult:
    """Result of a Knuth-Bendix completion attempt.

    *success* is ``True`` if the algorithm converged to a confluent TRS.
    *rules* is the resulting rule set (may be partial if not successful).
    *steps* is the number of iterations of the completion loop.
    *diverged* indicates whether the algorithm ran out of budget.
    *unorientable* lists equations that could not be oriented.
    """

    success: bool
    rules: list[Rule]
    steps: int
    diverged: bool = False
    unorientable: list[Equation] = field(default_factory=list)


def knuth_bendix(
    equations: list[Equation],
    ordering: SymbolPrecedence,
    *,
    max_rules: int = _MAX_RULES,
    max_eqs: int = _MAX_EQS,
) -> KBResult:
    """Run the Knuth-Bendix completion algorithm.

    Given a set of *equations* over a term algebra with signature described
    by *ordering* (a mapping from symbol names to integer precedences),
    attempt to construct a confluent and terminating term rewriting system.

    Parameters
    ----------
    equations:
        The initial set of equations to complete.
    ordering:
        Symbol precedences for the reduction ordering.
    max_rules:
        Maximum number of rules before aborting.
    max_eqs:
        Maximum number of pending equations before aborting.

    Returns
    -------
    KBResult
        The completion result.  ``result.success`` is ``True`` if the
        algorithm converged.
    """
    # Phase 1: Orient initial equations
    rules: list[Rule] = []
    pending: list[Equation] = list(equations)

    # Orient all initial equations
    oriented: list[Equation] = []
    for eq in pending:
        result = orient(eq, ordering)
        if result is not None:
            rule, _dir = result
            rules.append(rule)
        else:
            oriented.append(eq)
    pending = oriented  # equations that couldn't be oriented

    # Phase 2: Completion loop
    steps = 0
    unorientable: list[Equation] = []
    while pending:
        steps += 1

        # Budget check
        if len(rules) > max_rules:
            return KBResult(success=False, rules=rules, steps=steps, diverged=True)
        if len(pending) > max_eqs:
            return KBResult(success=False, rules=rules, steps=steps, diverged=True)

        # Deduce: pick one equation from pending
        eq = pending.pop(0)

        # Orient the equation
        result = orient(eq, ordering)
        if result is None:
            # Unorientable equation -- track and skip
            # (In a full KB implementation, we'd use unfailing completion)
            unorientable.append(eq)
            continue

        new_rule, _dir = result

        # Simplify the new rule's RHS using existing rules
        new_rule = _simplify_rhs(new_rule, rules)

        # Simplify existing rules' LHS using the new rule
        simplified_rules: list[Rule] = []
        for r in rules:
            simp = _simplify_lhs(r, [new_rule])
            if simp is None:
                # LHS was reducible but can't be re-oriented
                continue
            if isinstance(simp, Equation):
                # LHS reduced to something else -- re-orient
                re = orient(simp, ordering)
                if re is not None:
                    simplified_rules.append(re[0])
                else:
                    # Can't re-orient -- skip
                    continue
            else:
                simplified_rules.append(simp)

        rules = simplified_rules
        rules.append(new_rule)

        # Compute critical pairs between new rule and all existing rules
        all_rules = list(rules)
        for r in all_rules:
            if r is new_rule:
                continue
            pairs = overlaps(new_rule, r)
            for cp in pairs:
                # Simplify both sides
                left = _simplify_term(cp.left, rules)
                right = _simplify_term(cp.right, rules)
                if left != right:
                    pending.append(Equation(left, right))

        # Also compute self-overlaps for the new rule
        self_pairs = overlaps(new_rule, new_rule)
        for cp in self_pairs:
            left = _simplify_term(cp.left, rules)
            right = _simplify_term(cp.right, rules)
            if left != right:
                pending.append(Equation(left, right))

    # Phase 3: Verify confluence
    conf_result = check_confluence(rules)

    return KBResult(
        success=conf_result.is_confluent,
        rules=rules,
        steps=steps,
        unorientable=unorientable,
    )
