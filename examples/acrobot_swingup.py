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

    Far from upright, apply the elbow torque that drives the total energy
    toward that of the upright equilibrium while a PD term straightens the
    elbow. Near upright, switch to an LQR controller designed on the model
    linearized about the upright equilibrium.
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

    plant.SetPositions(context, [0.0, 0.0])
    down_energy = plant.CalcPotentialEnergy(context)
    elbow_inertia = plant.CalcMassMatrix(context)[1, 1]
    plant.SetPositions(context, [0.0, np.pi / 2])
    max_gravity_torque = abs(plant.CalcGravityGeneralizedForces(context)[1])
    plant.SetPositions(context, [np.pi, 0.0])
    up_energy = plant.CalcPotentialEnergy(context)
    energy_range = up_energy - down_energy

    k_d = 2 * energy_range * elbow_inertia
    k_p = 2 * energy_range * max_gravity_torque
    k_v = 2 * np.sqrt(k_p * k_d)

    linear = Linearize(
        plant,
        context,
        input_port_index=actuation.get_index(),
        output_port_index=OutputPortSelection.kNoOutput,
    )
    angle_tol = AcrobotSwingup.angle_tolerance
    velocity_tol = AcrobotSwingup.velocity_tolerance
    Q = np.diag(
        [1 / angle_tol**2, 1 / angle_tol**2]
        + [1 / velocity_tol**2, 1 / velocity_tol**2]
    )
    R = np.array([[1 / max_torque**2]])
    K, S = LinearQuadraticRegulator(linear.A(), linear.B(), Q, R)
    threshold = max_torque**2 / (K @ np.linalg.solve(S, K.T)).item()

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
        det = np.linalg.det(M)
        u = (
            -(k_v * q2_dot + k_p * q2) * det
            + k_d * (M[0, 0] * h[1] - M[1, 0] * h[0])
        ) / ((energy - up_energy) * det + k_d * M[0, 0])
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
