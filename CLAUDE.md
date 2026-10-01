# CLAUDE.md

CENIC-Bench is a robotics benchmark built on Drake's CENIC (convex
error-controlled) integrator. A policy is any `Callable[[Observation], Action]`;
a task randomizes initial conditions, simulates, and reports success. Early
stage: breaking changes are fine.

## Commands

```
uv sync                                         # install (incl. dev group)
uv run pytest                                   # all tests
uv run pytest tests/test_pendulum_swingup.py -k success   # single test
uv run ruff check && uv run ruff format         # lint + format (line length 80)
uv run examples/pendulum_swingup.py --headless --policy energy_shaping
```

Python >= 3.13, Drake >= 1.57. Always go through `uv run`; don't pip install.

## Layout

- `cenic_bench/tasks/base.py` — `Task` ABC. Its `__init__` builds the diagram
  via `create_scene()`, configures the simulator for CENIC with error control,
  and installs a monitor that terminates the sim when `success()` is true.
  `run_episode(seed)` = reset, `AdvanceTo(timeout)`, success iff terminated.
- `cenic_bench/tasks/pendulum_swingup.py` — reference task implementation. Copy
  its structure for new tasks (policy wrapped in a `LeafSystem`, `visualize`
  flag toggling meshcat + realtime).
- `cenic_bench/parsing.py` — `make_parser(plant)` registers the `cenic_bench`
  package so models load via `package://cenic_bench/models/...`.
- `cenic_bench/models/` — SDF/URDF models and meshes.
- `tests/` — lightweight pytest sanity/regression checks.

## Conventions

Follow [CONTRIBUTING.md](CONTRIBUTING.md). The points most often relevant:

- Keep code as simple as possible; avoid new abstractions and global variables.
- Keep PRs small (aim ~100 lines, hard cap ~1000 excluding meshes).
- Every change gets targeted, fast pytest tests. Tests must construct tasks
  with `visualize=False` so they run headless.
- Explain *why* a change was made in the PR description, not in code comments.

### Adding a task

- Subclass `Task` and implement every abstract member, including the natural
  language `task_description`, `action_description`, `observation_description`
  and the `action_example` / `observation_example` values.
- Export it from `cenic_bench/tasks/__init__.py`.
- Docstring must state goal, observations, actions, and success conditions.
- Plants must be continuous-time (`time_step=0.0`) — CENIC requires it.
- `reset()` must be deterministic per seed (use `np.random.default_rng(seed)`).
- `success()` runs every integration step; keep it cheap.
- Support both visualized (realtime) and headless (as fast as possible) runs.
- Add tests mirroring `tests/test_pendulum_swingup.py`: properties, reset
  determinism, success edge cases, observation mapping.

### Adding a model

- Load it through `make_parser`, not a raw `Parser`; `tests/test_parsing.py`
  automatically checks every `.sdf` under `models/` parses.
- Check license and add attribution / a `LICENSE.md` next to the model if
  needed. Sanity-check masses, inertias, collision geometry, and hydroelastic
  support (see CONTRIBUTING.md for the Drake model visualizer workflow).
