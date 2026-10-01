# CENIC-Bench

A robotics benchmark built on [convex error-controlled
simulation](https://arxiv.org/abs/2511.08771).

> [!WARNING]
> This benchmark is in the early stages of development. Expect rough edges and
> breaking changes.

## Usage

Install with [uv](https://docs.astral.sh/uv/):

```
uv sync
```

Test a simple pendulum swingup task:
```
uv run examples/pendulum_swingup.py
```

## Development

Run unit tests with [pytest](https://docs.pytest.org/):
```
uv run pytest
```

Lint and format with [ruff](https://docs.astral.sh/ruff/):
```
uv run ruff check
uv run ruff format
```
