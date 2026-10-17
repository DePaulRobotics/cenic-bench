"""Run a policy on the cart-pole with walls task."""

import argparse

import numpy as np
from pydrake.systems.controllers import LinearQuadraticRegulator

from cenic_bench.tasks import CartPoleWalls


def dummy_policy(obs):
    """A simple policy example that just returns zero force."""
    return np.array([0.0])


def make_lqr_policy():
    """Build a saturated LQR policy that balances the pole upright.

    The controller is designed on the model linearized about the upright
    equilibrium, ignoring the walls, and its force is clipped to the cart's
    force limit.
    """
    # Estimated physical parameters
    cart_mass = 1.0
    pole_mass = 0.5
    length = 0.5
    gravity = 9.81
    max_force = 10.0

    # Linearized dynamics about upright, with state [x, error, x_dot,
    # error_dot], where error is the pole's angle from upright.
    a = pole_mass * gravity / cart_mass
    b = gravity * (cart_mass + pole_mass) / (cart_mass * length)
    A = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -a, 0.0, 0.0],
            [0.0, b, 0.0, 0.0],
        ]
    )
    B = np.array([[0.0], [0.0], [1.0], [-1.0 / length]]) / cart_mass
    Q = 10.0 * np.eye(4)
    R = np.array([[1.0]])
    K, _ = LinearQuadraticRegulator(A, B, Q, R)

    def policy(obs):
        x, cos_theta, sin_theta, x_dot, theta_dot = obs
        # Angle from upright (theta = pi), wrapped to [-pi, pi].
        error = np.arctan2(-sin_theta, -cos_theta)
        u = -K @ np.array([x, error, x_dot, theta_dot])
        return np.clip(u, -max_force, max_force)

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices={"dummy", "lqr"},
        default="dummy",
        help="Which policy to run.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless, as fast as possible, without meshcat.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for the episode."
    )
    args = parser.parse_args()

    # Create a policy function mapping observations to actions.
    policy_fn = make_lqr_policy() if args.policy == "lqr" else dummy_policy

    # Define the task.
    task = CartPoleWalls(
        policy=policy_fn,
        visualize=not args.headless,
    )

    # Simulate from a random initial state, and evaluate whether the policy was
    # successful or not.
    success = task.run_episode(seed=args.seed)
    print("Success:", success)
