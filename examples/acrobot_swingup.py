"""Run a policy on the acrobot swing-up task."""

import argparse

import numpy as np
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.controllers import LinearQuadraticRegulator
from pydrake.systems.framework import OutputPortSelection
from pydrake.systems.primitives import Linearize

from cenic_bench.parsing import make_parser
from cenic_bench.tasks import AcrobotSwingup


def dummy_policy(obs):
    """A simple policy example that just returns zero torque."""
    return np.array([0.0])


def make_energy_shaping_policy():
    """Build a policy that swings the acrobot up, then balances it.

    Far from upright, use partial feedback linearization to drive the elbow
    straight with a PD law, plus a torque that pumps energy toward that of the
    upright equilibrium. Once x'Sx < threshold, switch to an LQR controller
    designed on the model linearized about upright.
    """
    plant = MultibodyPlant(time_step=0.0)
    make_parser(plant).AddModelsFromUrl(
        "package://cenic_bench/models/acrobot/acrobot.sdf"
    )
    plant.Finalize()
    context = plant.CreateDefaultContext()
    actuation = plant.get_actuation_input_port()
    actuation.FixValue(context, [0.0])
    max_torque = plant.GetJointActuatorByName("elbow").effort_limit()
    k_p, k_e, threshold = 10.0, 5.0, 100.0
    k_d = 2 * np.sqrt(k_p)

    plant.SetPositions(context, [np.pi, 0.0])
    up_energy = plant.CalcPotentialEnergy(context)
    linear = Linearize(
        plant,
        context,
        input_port_index=actuation.get_index(),
        output_port_index=OutputPortSelection.kNoOutput,
    )
    Q = np.diag([10.0, 10.0, 1.0, 1.0])
    R = np.array([[0.1]])
    K, S = LinearQuadraticRegulator(linear.A(), linear.B(), Q, R)

    def policy(obs):
        cos_q1, sin_q1, cos_q2, sin_q2, q1_dot, q2_dot = obs
        q1 = np.arctan2(sin_q1, cos_q1)
        q2 = np.arctan2(sin_q2, cos_q2)

        x = np.array([np.arctan2(-sin_q1, -cos_q1), q2, q1_dot, q2_dot])
        if x @ S @ x < threshold:
            return np.clip(-K @ x, -max_torque, max_torque)

        plant.SetPositions(context, [q1, q2])
        plant.SetVelocities(context, [q1_dot, q2_dot])
        M = plant.CalcMassMatrix(context)
        h = plant.CalcBiasTerm(context) - plant.CalcGravityGeneralizedForces(
            context
        )
        energy = plant.CalcKineticEnergy(context) + plant.CalcPotentialEnergy(
            context
        )
        y = -k_p * q2 - k_d * q2_dot
        u = (
            (M[1, 1] - M[1, 0] * M[0, 1] / M[0, 0]) * y
            + h[1]
            - M[1, 0] * h[0] / M[0, 0]
            + k_e * (up_energy - energy) * q2_dot
        )
        return np.clip(np.array([u]), -max_torque, max_torque)

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
    task = AcrobotSwingup(
        policy=policy_fn,
        visualize=not args.headless,
    )

    # Simulate from a random initial state, and evaluate whether the policy was
    # successful or not.
    success = task.run_episode(seed=args.seed)
    print("Success:", success)
