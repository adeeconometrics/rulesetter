# rulesetter

A symbolic term rewrite system in Python.

## What is this?

rulesetter is a library for defining and executing term rewriting systems. It provides:

- **Terms** -- immutable, hashable trees built from variables (`Var`), constants (`Const`), and n-ary operations (`Op`)
- **Rules** -- directed rewrite rules `lhs → rhs` where the LHS may contain variables that act as pattern wildcards
- **Rewrite engine** -- applies rules to terms using configurable strategies (innermost, outermost, top-down), with cycle detection and step limits
- **Knuth-Bendix completion** -- automatically derives a confluent and terminating rewrite system from a set of equations, using a simplified Knuth-Bendix ordering (KBO) for term orientation
- **Algebra signatures** -- pre-built theory modules for monoids and groups, with their axioms oriented as rewrite rules

The library is useful for exploring equational reasoning, automated theorem proving, and simplification of algebraic expressions.

## Quick start

```python
from rulesetter import Term, Var, Const, Op, Rule, RewriteEngine, Strategy

# Build terms
x, y = Var("x"), Var("y")
a = Const("a")

# Define a rule: mul(a, x) → x  (left identity)
identity_rule = Rule(Op("mul", (a, x)), x, name="left_id")

# Create an engine and reduce
engine = RewriteEngine(rules=[identity_rule], strategy=Strategy.INNERMOST)
term = Op("mul", (a, Op("mul", (a, Var("b")))))
result = engine.reduce(term)
print(result)  # b
```

Or use the algebra signatures:

```python
from rulesetter.algebra import MonoidSignature, GroupSignature

# Monoid
mono = MonoidSignature()
engine = mono.engine()
x = mono.var("x")
term = mono.mul(mono.e, x)
print(engine.reduce(term))  # x

# Group
grp = GroupSignature()
engine = grp.engine()
term = grp.mul(grp.inv(x), x)
print(engine.reduce(term))  # e
```

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone <repo-url>
cd rulesetter
uv sync
```

## Running

```sh
uv run python main.py
```

## Testing

```sh
uv run pytest
```

To run with coverage or verbose output:

```sh
uv run pytest -v
```

## Linting and type checking

```sh
uv run ruff check . --statistics    # lint
uv run ruff format --check .        # format check (use `ruff format .` to fix)
uv run mypy src/ --strict           # typecheck
```

## Building docs

```sh
uv run sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.

## How it works

### Terms

The term algebra has three constructors:

| Type   | Example                     | Description                        |
| ------ | --------------------------- | ---------------------------------- |
| `Var`  | `Var("x")`                 | Pattern variable (matches anything) |
| `Const`| `Const("e")`               | Named constant (zero-ary symbol)    |
| `Op`   | `Op("mul", (x, y))`        | N-ary operation applied to arguments|

Terms are immutable, hashable, and form finite trees. Convenience constructors `var()`, `const()`, and `op()` are available:

```python
from rulesetter.term import var, const, op

x = var("x")
e = const("e")
t = op("mul", e, x)  # Op("mul", (Const("e"), Var("x")))
```

### Rules and pattern matching

A `Rule` is a directed equation `lhs → rhs`. The LHS may contain variables that match any subterm. The same variable must match consistently across the pattern.

```python
from rulesetter.rule import Rule, match

x = var("x")
rule = Rule(op("mul", const("e"), x), x, name="left_id")

# Match against a term
term = op("mul", const("e"), const("a"))
subst = match(rule.lhs, term)
print(subst)  # Substitution({x → a})
```

### Rewrite engine

`RewriteEngine` applies rules to terms using a chosen strategy:

- **`Strategy.INNERMOST`** -- bottom-up: normalize children first, then try rules at the root
- **`Strategy.OUTERMOST`** -- top-down: try rules at the root first, then descend
- **`Strategy.TOPDOWN`** -- breadth-first: try current level, then first matching child

```python
from rulesetter import RewriteEngine, Strategy

rules = [
    Rule(op("mul", const("e"), var("x")), var("x"), name="left_id"),
    Rule(op("mul", var("x"), const("e")), var("x"), name="right_id"),
]

engine = RewriteEngine(rules=rules, strategy=Strategy.INNERMOST, max_steps=100)
term = op("mul", op("mul", const("e"), const("a")), const("e"))
print(engine.reduce(term))  # a
```

The engine raises `ReductionLimitExceededError` if `max_steps` is exceeded, and `RewriteCycleDetectedError` if a term is revisited (when `track_visited=True`).

### Knuth-Bendix completion

Given a set of equations and a symbol precedence, the Knuth-Bendix algorithm attempts to automatically derive a confluent and terminating rewrite system:

```python
from rulesetter.completion import Equation, knuth_bendix
from rulesetter.term import var, const, op

x, y, z = var("x"), var("y"), var("z")
e = const("e")

equations = [
    Equation(op("mul", e, x), x, name="left_id"),
    Equation(op("mul", x, e), x, name="right_id"),
]

# Symbol precedences: higher = greater in the ordering
precedence = {"e": 1, "mul": 2}

result = knuth_bendix(equations, precedence)
print(result.success)   # True if a confluent TRS was found
print(result.rules)     # The resulting rewrite rules
```

### Confluence checking

You can check whether an existing set of rules is locally confluent by computing all critical pairs and attempting to join them:

```python
from rulesetter.completion import check_confluence

result = check_confluence(rules)
print(result.is_confluent)      # True if all critical pairs join
print(result.non_joinable)      # Pairs that failed to join
```

## Extending the library

### Adding new algebra signatures

Create a new module under `rulesetter/algebra/` following the pattern in `monoid.py` or `group.py`:

```python
# rulesetter/algebra/ring.py
from dataclasses import dataclass
from rulesetter.engine import RewriteEngine, Strategy
from rulesetter.rule import Rule
from rulesetter.term import Const, Op, Term, Var, const, op, var

@dataclass(frozen=True, slots=True)
class RingSignature:
    """A commutative ring with identity."""

    zero_name: str = "0"
    one_name: str = "1"
    add_name: str = "add"
    mul_name: str = "mul"
    neg_name: str = "neg"

    @property
    def zero(self) -> Const:
        return const(self.zero_name)

    @property
    def one(self) -> Const:
        return const(self.one_name)

    def add(self, x: Term, y: Term) -> Op:
        return op(self.add_name, x, y)

    def mul(self, x: Term, y: Term) -> Op:
        return op(self.mul_name, x, y)

    def neg(self, x: Term) -> Op:
        return op(self.neg_name, x)

    def rules(self) -> list[Rule]:
        x = var("x")
        return [
            Rule(self.add(self.zero, x), x, name="add_zero_l"),
            Rule(self.add(x, self.zero), x, name="add_zero_r"),
            Rule(self.add(self.neg(x), x), self.zero, name="add_neg_l"),
            Rule(self.add(x, self.neg(x)), self.zero, name="add_neg_r"),
            Rule(self.mul(self.one, x), x, name="mul_one_l"),
            Rule(self.mul(x, self.one), x, name="mul_one_r"),
            Rule(self.mul(self.zero, x), self.zero, name="mul_zero_l"),
            Rule(self.mul(x, self.zero), self.zero, name="mul_zero_r"),
        ]

    def engine(self, **kwargs) -> RewriteEngine:
        return RewriteEngine(rules=self.rules(), **kwargs)
```

Register it in `rulesetter/algebra/__init__.py`:

```python
from rulesetter.algebra.ring import RingSignature

__all__ = ["MonoidSignature", "GroupSignature", "RingSignature"]
```

### Adding custom rewrite rules

Define rules using the `Rule` constructor or the `rule()` shorthand:

```python
from rulesetter.rule import Rule, rule
from rulesetter.term import var, const, op

x, y = var("x"), var("y")

# Commutativity
comm = rule(op("add", x, y), op("add", y, x), name="comm")

# Distributivity
dist_l = rule(
    op("mul", x, op("add", y, var("z"))),
    op("add", op("mul", x, y), op("mul", x, var("z"))),
    name="dist_l",
)
```

### Implementing a custom strategy

Subclass `RewriteEngine` and override the `find_redex` method:

```python
from rulesetter.engine import RewriteEngine, Redex

class RandomStrategyEngine(RewriteEngine):
    """Pick a random redex instead of the first one found."""

    def find_redex(self, term):
        import random
        all_redexes = self._collect_all_redexes(term, ())
        if not all_redexes:
            return None
        return random.choice(all_redexes)

    def _collect_all_redexes(self, term, path):
        redexes = []
        redex = self._try_rules_at(term, path)
        if redex is not None:
            redexes.append(redex)
        if hasattr(term, 'args'):
            for i, child in enumerate(term.args):
                redexes.extend(self._collect_all_redexes(child, path + (i,)))
        return redexes
```

### Using term ordering for completion

The `compare_kbo` function implements a simplified Knuth-Bendix ordering. Control symbol precedences to influence which direction equations are oriented:

```python
from rulesetter.completion import compare_kbo, orient, Equation
from rulesetter.term import var, const, op

x, y = var("x"), var("y")
e = const("e")

# Higher precedence → tends to be on the LHS (larger term)
precedence = {"e": 1, "mul": 2, "add": 2}

eq = Equation(op("mul", e, x), x)
rule, direction = orient(eq, precedence)
print(rule)        # Rule(mul(e, x) → x [left_id])
print(direction)   # Orientation.LEFT_TO_RIGHT
```

## Project structure

```
rulesetter/
├── rulesetter/
│   ├── __init__.py          # Public API re-exports
│   ├── term.py              # Term AST (Var, Const, Op)
│   ├── rule.py              # Rules, substitutions, pattern matching
│   ├── engine.py            # Rewrite engine with strategies
│   ├── completion.py        # Knuth-Bendix completion, unification, KBO
│   └── algebra/
│       ├── __init__.py
│       ├── monoid.py        # Monoid signature
│       └── group.py         # Group signature
├── test/
│   ├── conftest.py
│   ├── test_term.py
│   ├── test_rule.py
│   ├── test_engine.py
│   ├── test_completion.py
│   └── test_algebra.py
├── docs/                    # Sphinx documentation
├── pyproject.toml
├── ruff.toml
└── main.py
```

## License

See [LICENSE](LICENSE).
