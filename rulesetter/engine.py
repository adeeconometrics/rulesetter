from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rulesetter.rule import Rule, Substitution
from rulesetter.term import Op, Position, Term, pformat

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ReductionLimitExceededError(Exception):
    """Raised when the rewrite engine exceeds ``max_steps``."""

    def __init__(self, term: Term, steps: int) -> None:
        self.term = term
        self.steps = steps
        super().__init__(f"reduction exceeded {steps} steps; last term: {pformat(term)}")


class RewriteCycleDetectedError(Exception):
    """Raised when the rewrite engine detects a cycle (term revisited)."""

    def __init__(self, term: Term) -> None:
        self.term = term
        super().__init__(f"cycle detected; term revisited: {pformat(term)}")


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class Strategy(Enum):
    """Rule application strategy for the rewrite engine.

    * ``INNERMOST`` -- normalize subterms (bottom-up) before applying rules at
      the root.  Guarantees that arguments are in normal form when a root rule
      fires.
    * ``OUTERMOST`` -- apply rules at the root first, then descend into
      subterms.  Useful when a root rule can eliminate the need to rewrite
      children.
    * ``TOPDOWN`` -- at each node, try rules; if none match, descend into the
      first child that can be rewritten.  Breadth-first traversal.
    """

    INNERMOST = "innermost"
    OUTERMOST = "outermost"
    TOPDOWN = "topdown"


# ---------------------------------------------------------------------------
# Redex (a match site)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Redex:
    """A reducible expression: a rule matched at a specific position."""

    rule: Rule
    position: Position
    substitution: Substitution


# ---------------------------------------------------------------------------
# Term replacement
# ---------------------------------------------------------------------------


def replace_at(term: Term, pos: Position, replacement: Term) -> Term:
    """Return *term* with the subterm at *pos* replaced by *replacement*.

    Raises ``IndexError`` if *pos* is invalid.
    """
    if pos == ():
        return replacement
    if not isinstance(term, Op):
        raise IndexError(f"cannot descend into non-Op term at position {pos}")
    head, *rest = pos
    if head < 0 or head >= len(term.args):
        raise IndexError(
            f"child index {head} out of range for {term.name!r} (arity {len(term.args)})"
        )
    new_child = replace_at(term.args[head], tuple(rest), replacement)
    new_args = term.args[:head] + (new_child,) + term.args[head + 1 :]
    return Op(term.name, new_args)


# ---------------------------------------------------------------------------
# Positions enumeration
# ---------------------------------------------------------------------------


def positions(term: Term) -> list[Position]:
    """Return all positions in *term* in pre-order (root first, left-to-right)."""
    result: list[Position] = [()]
    if isinstance(term, Op):
        for i, child in enumerate(term.args):
            for sub_pos in positions(child):
                result.append((i,) + sub_pos)
    return result


# ---------------------------------------------------------------------------
# Rewrite engine
# ---------------------------------------------------------------------------


@dataclass
class RewriteEngine:
    """A rule-based rewrite engine with configurable strategy.

    Parameters
    ----------
    rules:
        The rewrite rules to apply (tried in order; first match wins).
    strategy:
        How to traverse the term tree when applying rules.
    max_steps:
        Hard limit on the number of rewrite steps.  ``0`` means no limit.
    track_visited:
        If ``True``, maintain a set of visited term hashes and raise
        :class:`RewriteCycleDetected` if a term is seen twice.
    """

    rules: list[Rule]
    strategy: Strategy = Strategy.INNERMOST
    max_steps: int = 1000
    track_visited: bool = True

    # ------------------------------------------------------------------
    # Single-step: find and apply one rewrite
    # ------------------------------------------------------------------

    def find_redex(self, term: Term) -> Optional[Redex]:
        """Return the first redex found according to the current strategy, or ``None``."""
        if self.strategy is Strategy.INNERMOST:
            return self._find_innermost(term, ())
        if self.strategy is Strategy.OUTERMOST:
            return self._find_outermost(term, ())
        if self.strategy is Strategy.TOPDOWN:
            return self._find_topdown(term, ())
        raise ValueError(f"unknown strategy: {self.strategy}")

    def apply_at(self, term: Term, redex: Redex) -> Term:
        """Apply *redex* to *term*, returning the rewritten term."""
        new_subterm = redex.substitution.apply(redex.rule.rhs)
        return replace_at(term, redex.position, new_subterm)

    def step(self, term: Term) -> Optional[Term]:
        """Perform one rewrite step.

        Returns the rewritten term, or ``None`` if *term* is in normal form.
        """
        redex = self.find_redex(term)
        if redex is None:
            return None
        return self.apply_at(term, redex)

    # ------------------------------------------------------------------
    # Multi-step: reduce to normal form
    # ------------------------------------------------------------------

    def reduce(self, term: Term) -> Term:
        """Repeatedly apply rules until *term* is in normal form.

        Raises
        ------
        ReductionLimitExceededError
            If *max_steps* is reached.
        RewriteCycleDetectedError
            If *track_visited* is ``True`` and a term is seen twice.
        """
        visited: set[int] = set()
        current = term
        steps = 0

        while True:
            if self.track_visited:
                h = hash(current)
                if h in visited:
                    raise RewriteCycleDetectedError(current)
                visited.add(h)

            next_term = self.step(current)
            if next_term is None:
                return current  # normal form reached

            current = next_term
            steps += 1

            if self.max_steps > 0 and steps >= self.max_steps:
                raise ReductionLimitExceededError(current, steps)

    # ------------------------------------------------------------------
    # Strategy: innermost
    # ------------------------------------------------------------------

    def _find_innermost(self, term: Term, path: Position) -> Optional[Redex]:
        """Bottom-up: normalize children first, then try root."""
        if isinstance(term, Op):
            for i, child in enumerate(term.args):
                redex = self._find_innermost(child, path + (i,))
                if redex is not None:
                    return redex
        # All children are in normal form -- try root
        return self._try_rules_at(term, path)

    # ------------------------------------------------------------------
    # Strategy: outermost
    # ------------------------------------------------------------------

    def _find_outermost(self, term: Term, path: Position) -> Optional[Redex]:
        """Top-down: try root first, then descend."""
        redex = self._try_rules_at(term, path)
        if redex is not None:
            return redex
        if isinstance(term, Op):
            for i, child in enumerate(term.args):
                redex = self._find_outermost(child, path + (i,))
                if redex is not None:
                    return redex
        return None

    # ------------------------------------------------------------------
    # Strategy: topdown
    # ------------------------------------------------------------------

    def _find_topdown(self, term: Term, path: Position) -> Optional[Redex]:
        """Breadth-first: try current level, then first matching child."""
        redex = self._try_rules_at(term, path)
        if redex is not None:
            return redex
        if isinstance(term, Op):
            for i, child in enumerate(term.args):
                redex = self._find_topdown(child, path + (i,))
                if redex is not None:
                    return redex
        return None

    # ------------------------------------------------------------------
    # Helper: try all rules at a given position
    # ------------------------------------------------------------------

    def _try_rules_at(self, term: Term, path: Position) -> Optional[Redex]:
        """Try each rule against *term* at the given *path*."""
        for rule in self.rules:
            subst = rule.try_match(term)
            if subst is not None:
                return Redex(rule=rule, position=path, substitution=subst)
        return None
