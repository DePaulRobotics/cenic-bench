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

CENIC-Bench is designed to evaluate any control policy, whether it's based on
classical control, reinforcement learning, a large AI model, or anything else.
The API is simply a python function mapping observations to actions:
```python
policy: Callable[[Observation], Action]
```
Given such a policy and a compatible task, CENIC-Bench sets randomized initial
conditions, simulates the system with error-controlled integration, and
evaluates whether or not the task was completed successfully.

See [`examples/pendulum_swingup.py`](examples/pendulum_swingup.py) for a simple
example.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for developer details.
