from __future__ import annotations

from dataclasses import dataclass

from rulesetter.algebra.monoid import MonoidSignature
from rulesetter.engine import RewriteEngine, Strategy
from rulesetter.rule import Rule
from rulesetter.term import Const, Op, Term, Var, const, op, var

# ---------------------------------------------------------------------------
# Group signature
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroupSignature:
    """The theory of groups: a monoid with inverses.

    **Operations**

    * ``e`` -- identity element (constant)
    * ``mul(x, y)`` -- binary multiplication (associative)
    * ``inv(x)`` -- multiplicative inverse

    **Axioms (oriented as rewrite rules)**

    * ``mul(e, x) → x``          (left identity)
    * ``mul(x, e) → x``          (right identity)
    * ``mul(inv(x), x) → e``     (left inverse)
    * ``mul(x, inv(x)) → e``     (right inverse)

    Associativity is not included by default (same rationale as
    :class:`MonoidSignature`).
    """

    identity_name: str = "e"
    mul_name: str = "mul"
    inv_name: str = "inv"

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

    def inv(self, x: Term) -> Op:
        """Build ``inv(x)``."""
        return op(self.inv_name, x)

    def var(self, name: str) -> Var:
        """Convenience: create a variable."""
        return var(name)

    # ------------------------------------------------------------------
    # Rewrite rules
    # ------------------------------------------------------------------

    def rules(self) -> list[Rule]:
        """Return the standard group rewrite rules.

        * ``mul(e, x) → x``          (left identity)
        * ``mul(x, e) → x``          (right identity)
        * ``mul(inv(x), x) → e``     (left inverse)
        * ``mul(x, inv(x)) → e``     (right inverse)
        """
        x = self.var("x")
        return [
            Rule(self.mul(self.e, x), x, name="left_id"),
            Rule(self.mul(x, self.e), x, name="right_id"),
            Rule(self.mul(self.inv(x), x), self.e, name="left_inv"),
            Rule(self.mul(x, self.inv(x)), self.e, name="right_inv"),
        ]

    def with_associativity(self, direction: str = "left") -> list[Rule]:
        """Return rules including an oriented associativity law.

        Parameters
        ----------
        direction:
            ``"left"``  → ``mul(mul(x,y),z) → mul(x,mul(y,z))``
            ``"right"`` → ``mul(x,mul(y,z)) → mul(mul(x,y),z)``
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
        """Build a :class:`RewriteEngine` pre-loaded with group rules."""
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

    # ------------------------------------------------------------------
    # Convenience: lift monoid terms
    # ------------------------------------------------------------------

    def to_monoid(self) -> MonoidSignature:
        """Return the underlying :class:`MonoidSignature`."""
        return MonoidSignature(
            identity_name=self.identity_name,
            mul_name=self.mul_name,
        )
