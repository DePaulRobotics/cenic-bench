"""Run a policy on the Push-T task."""

import argparse

import numpy as np

from cenic_bench.tasks import PushT


def dummy_policy(obs):
    """A simple policy example that just returns zero force."""
    return np.zeros(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless, as fast as possible, without meshcat.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for the episode."
    )
    args = parser.parse_args()

    # Define the task.
    task = PushT(
        policy=dummy_policy,
        visualize=not args.headless,
    )

    # Simulate from a random initial state, and evaluate whether the policy was
    # successful or not.
    success = task.run_episode(seed=args.seed)
    print("Success:", success)
