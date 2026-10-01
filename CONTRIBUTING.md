# Contributing Guidelines

Contribute new tasks, models, or capabilities by forking this repository,
developing on feature branches, and opening pull requests on GitHub.

## General Guidelines

**Each PR gets two reviews**, one from a human and one from AI (copilot,
coderabbit, etc.). Code is not merged into `main` until all reviewers have
approved.

**Keep PRs short**. Each pull request should have under 1000 lines of code,
except in exceptional circumstances. Aim for closer to 100 lines if possible.
Break larger contributions into a "PR train" if needed. Object meshes don't
count toward this limit.

**Avoid global variables** unless absolutely necessary.

Aim for code that is **as simple as possible, but no simpler**. Avoid
unnecessary abstractions, especially when using AI coding tools.

**Include unit tests**. Each PR should be backed by `pytest` unit tests. Keep
these lightweight and targeted. We are aiming for quick sanity and regression
checks rather than comprehensive code coverage.

**Lint and unit tests must pass** before any PR is merged into `main`. Run
`pytest` and `ruff` regularly during development.

**Prefer "squash and merge"** when merging, so each commit is linked to a PR.
Use clear, descriptive names in PRs to keep the commit history as clean as
possible.

## Adding New Models

Models of robots and objects go in `cenic_bench/models`. Model files from the
internet almost always need some level of curation, even if they come directly
from a manufacturer. When adding new models, be sure to:

- **Check the license.** Often robot models come with specific terms and
  conditions, which may require a `LICENSE.md` to be distributed with the model.
  Make sure it's okay for us to distribute the model, and add the appropriate
  license file if needed.
- **Provide attribution.** If the model is adapted from some external source,
  document where it is from.
- **Check object masses.** Make sure the mass of each object or robot link is
  reasonable.
- **Check inertias.** Use the [Drake model
  visualizer](https://drake.mit.edu/pydrake/pydrake.visualization.model_visualizer.html)
  (`uv run -m pydrake.visualization.model_visualizer --help`) to view inertia
  ellipses. These should roughly coincide with object geometries. 
- **Check collision geometries.** Collision meshes should typically be
  simpler/coarser than their visual counterparts. Use the [Drake model
  visualizer](https://drake.mit.edu/pydrake/pydrake.visualization.model_visualizer.html)
  as above to check that all relevant components have reasonable collision
  geometries.
- **Make an informed choice about hydroelastic support.** Most models should be
  simulated with [hydroelastic
  contact](https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html).
  Make sure your model is configured to support hydroelastics by default, unless
  there is good reason not to.
- **Run an interactive simulation.** Use the mouse [CTRL + drag] to interact
  with the scene in meshcat, run a simple control policy if possible, and fix
  any obvious modeling errors.

## Adding New Tasks

Task definitions go in `cenic_bench/tasks`, and implement the
[`Task`](cenic_bench/tasks/base.py) abstract base class. Be sure to:

- Add the task to `cenic_bench/tasks/__init__.py` for easy importing. 
- Clearly document the goal, inputs, outputs, and success conditions in the
  docstring.
- Think carefully about what frequency the controller should run at.
- Whenever possible, include the ability to run both with a visualizer (in real
  time) and headless (as fast as possible).
- Keep the `success()` check as computationally light as possible, because this
  check runs at each integration step.
- Make sure the `reset()` map is deterministic: the same seed should produce the
  same initial state. 

## Updating Existing Tasks/Models

When contributing updates, changes and bug fixes, be sure to:

- Document the reasons for the change in the PR description, not in the source
  code.
- Make sure docstrings and descriptions remain up to date, particularly if task
  success metrics change.