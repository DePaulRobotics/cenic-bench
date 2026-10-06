"""Run a policy on the cart-pole swing-up task."""

import argparse

import numpy as np
from pydrake.systems.controllers import LinearQuadraticRegulator

from cenic_bench.tasks import CartPoleSwingup


def dummy_policy(obs):
    """A simple policy example that just returns zero force."""
    return np.array([0.0])


def make_energy_shaping_policy():
    """Build a policy that swings the pole up, then balances it.

    Far from upright, push the cart so as to drive the pole's mechanical
    energy toward that of the upright equilibrium, with a weak spring pulling
    the cart back to the center. Near upright, switch to an LQR controller
    designed on the model linearized about the upright equilibrium.
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
    Q = np.diag([10.0, 10.0, 1.0, 1.0])
    R = np.array([[0.1]])
    K, _ = LinearQuadraticRegulator(A, B, Q, R)

    def policy(obs):
        x, cos_theta, sin_theta, x_dot, theta_dot = obs

        # Angle from upright (theta = pi), wrapped to [-pi, pi].
        error = np.arctan2(-sin_theta, -cos_theta)

        # Local LQR controller near upright.
        if abs(error) < 0.4:
            u = -K @ np.array([x, error, x_dot, theta_dot])
            return np.clip(u, -max_force, max_force)

        # The pole's energy changes at a rate proportional to
        # theta_dot * cos(theta) * cart acceleration, so choose the cart
        # acceleration to push the energy toward the upright value.
        energy = (
            0.5 * pole_mass * length**2 * theta_dot**2
            - pole_mass * gravity * length * cos_theta
        )
        desired_energy = pole_mass * gravity * length
        k_energy, k_x, k_v = 5.0, 1.0, 1.0
        accel = (
            k_energy * (desired_energy - energy) * theta_dot * cos_theta
            - k_x * x
            - k_v * x_dot
        )

        # The force that produces this cart acceleration, accounting for the
        # pole pushing back on the cart.
        u = (cart_mass + pole_mass * sin_theta**2) * accel + (
            pole_mass
            * sin_theta
            * (gravity * cos_theta + length * theta_dot**2)
        )
        return np.clip(np.array([u]), -max_force, max_force)

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices={"dummy", "energy_shaping"},
        default="dummy",
        help="Which policy to run.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless, as fast as possible, without meshcat.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for the episode."
    )
    args = parser.parse_args()

    # Create a policy function mapping observations to actions.
    policy_fn = (
        make_energy_shaping_policy()
        if args.policy == "energy_shaping"
        else dummy_policy
    )

    # Define the task.
    task = CartPoleSwingup(
        policy=policy_fn,
        visualize=not args.headless,
    )

    # Simulate from a random initial state, and evaluate whether the policy was
    # successful or not.
    success = task.run_episode(seed=args.seed)
    print("Success:", success)
