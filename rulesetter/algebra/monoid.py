from __future__ import annotations

from dataclasses import dataclass

from rulesetter.engine import RewriteEngine, Strategy
from rulesetter.rule import Rule
from rulesetter.term import Const, Op, Term, Var, const, op, var

# ---------------------------------------------------------------------------
# Monoid signature
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MonoidSignature:
    """The theory of monoids: a set with an associative binary operation and
    an identity element.

    **Operations**

    * ``e`` -- identity element (constant)
    * ``mul(x, y)`` -- binary multiplication (associative)

    **Axioms (oriented as rewrite rules)**

    * ``mul(e, x) → x``  (left identity)
    * ``mul(x, e) → x``  (right identity)

    Associativity ``mul(mul(x,y),z) = mul(x,mul(y,z))`` is *not* included
    as a rewrite rule by default because it interacts badly with the other
    rules and can cause non-termination.  Use
    :meth:`with_associativity` to add it in either orientation.
    """

    # Symbol names (configurable for multi-sorted or custom signatures)
    identity_name: str = "e"
    mul_name: str = "mul"

    # ------------------------------------------------------------------
    # Term constructors
    # ------------------------------------------------------------------

    @property
    def e(self) -> Const:
        """The identity constant."""
        return const(self.identity_name)

    def mul(self, x: Term, y: Term) -> Op:
        """Build ``mul(x, y)``."""
        return op(self.mul_name, x, y)

    def var(self, name: str) -> Var:
        """Convenience: create a variable."""
        return var(name)

    # ------------------------------------------------------------------
    # Rewrite rules
    # ------------------------------------------------------------------

    def rules(self) -> list[Rule]:
        """Return the standard monoid rewrite rules (without associativity).

        * ``mul(e, x) → x``  (left identity)
        * ``mul(x, e) → x``  (right identity)
        """
        x = self.var("x")
        return [
            Rule(self.mul(self.e, x), x, name="left_id"),
            Rule(self.mul(x, self.e), x, name="right_id"),
        ]

    def with_associativity(self, direction: str = "left") -> list[Rule]:
        """Return rules including an oriented associativity law.

        Parameters
        ----------
        direction:
            ``"left"``  → ``mul(mul(x,y),z) → mul(x,mul(y,z))`` (flatten left)
            ``"right"`` → ``mul(x,mul(y,z)) → mul(mul(x,y),z)`` (flatten right)
        """
        x, y, z = self.var("x"), self.var("y"), self.var("z")
        lhs = self.mul(self.mul(x, y), z)
        rhs = self.mul(x, self.mul(y, z))
        if direction == "left":
            assoc = Rule(lhs, rhs, name="assoc_left")
        elif direction == "right":
            assoc = Rule(rhs, lhs, name="assoc_right")
        else:
            raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")
        return self.rules() + [assoc]

    # ------------------------------------------------------------------
    # Engine factory
    # ------------------------------------------------------------------

    def engine(
        self,
        *,
        include_associativity: bool = False,
        assoc_direction: str = "left",
        strategy: Strategy = Strategy.INNERMOST,
        max_steps: int = 1000,
        track_visited: bool = True,
    ) -> RewriteEngine:
        """Build a :class:`RewriteEngine` pre-loaded with monoid rules.

        Parameters
        ----------
        include_associativity:
            If ``True``, add the associativity rule.
        assoc_direction:
            Orientation for the associativity rule (``"left"`` or ``"right"``).
        strategy:
            Rewrite strategy.
        max_steps:
            Maximum rewrite steps before raising.
        track_visited:
            Enable cycle detection.
        """
        if include_associativity:
            rules = self.with_associativity(assoc_direction)
        else:
            rules = self.rules()
        return RewriteEngine(
            rules=rules,
            strategy=strategy,
            max_steps=max_steps,
            track_visited=track_visited,
        )
