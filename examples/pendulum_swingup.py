"""Run a policy on the pendulum swing-up task."""

import argparse

import numpy as np

from cenic_bench.tasks import PendulumSwingup


def dummy_policy(obs):
    """A simple policy example that just returns zero torque."""
    return np.array([0.0])


def energy_shaping_policy(obs):
    """Pump energy into the pendulum to swing it up, then stabilize upright.

    Far from upright, drive the total mechanical energy toward that of the
    upright equilibrium with u = k * theta_dot * (E_desired - E). Near
    upright, switch to a PD controller to catch and hold the pendulum.
    """
    # Estimated physical parameters
    mass = 0.5
    length = 0.5
    damping = 0.1
    gravity = 9.81

    cos_theta, sin_theta, theta_dot = obs

    # Angle from upright (theta = pi), wrapped to [-pi, pi].
    error = np.arctan2(-sin_theta, -cos_theta)

    # Local PD controller near upright.
    if abs(error) < 0.5:
        kp, kd = 20.0, 3.0
        return np.array([-kp * error - kd * theta_dot])

    inertia = mass * length**2
    energy = 0.5 * inertia * theta_dot**2 - mass * gravity * length * cos_theta
    desired_energy = mass * gravity * length

    # Energy pumping plus compensation for joint damping.
    k = 0.1
    u = k * theta_dot * (desired_energy - energy) + damping * theta_dot
    return np.array([u])


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

    task = PendulumSwingup(
        energy_shaping_policy
        if args.policy == "energy_shaping"
        else dummy_policy,
        visualize=not args.headless,
    )
    success = task.run_episode(seed=args.seed)
    print("Success:", success)
