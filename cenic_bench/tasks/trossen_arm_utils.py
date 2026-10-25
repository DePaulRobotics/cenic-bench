"""Shared helpers for driving one arm of the Trossen Stationary AI.

Used by every task built on this robot, since each arm's actuator layout and
the mimic-to-independent-actuator gripper workaround are the same regardless
of which task is being run.
"""

import numpy as np

NUM_ARM_JOINTS = 6
NUM_ARM_ACTUATORS = (
    NUM_ARM_JOINTS + 2
)  # 6 joints + 2 independent gripper fingers
# All-zero joint angles put the gripper clear of the table and frame; a
# convenient, verified-safe default pose for either arm.
HOME_ARM_POSITION = np.zeros(NUM_ARM_JOINTS)
HOME_GRIPPER_POSITION = 0  # close


def desired_positions(
    plant, model_instance, prefix, arm_position, gripper_position
):
    """Build the 8-actuator desired-position vector for one arm.

    `gripper_position` is sent to both carriage joints, replacing the URDF's
    mimic constraint (removed because CENIC's continuous-time plants don't
    support constraints). Indexed by JointActuator.input_start(), which is
    NOT the same ordering as the plant's positions.
    """
    u = np.zeros(plant.num_actuators(model_instance))
    for i, q in enumerate(arm_position):
        actuator = plant.GetJointActuatorByName(f"{prefix}_motor_{i}")
        u[actuator.input_start()] = q
    for name in (
        f"{prefix}_left_carriage_motor",
        f"{prefix}_right_carriage_motor",
    ):
        actuator = plant.GetJointActuatorByName(name)
        u[actuator.input_start()] = gripper_position
    return u


def set_arm_configuration(
    plant, plant_context, prefix, arm_position, gripper_position
):
    """Set one arm's joints directly on a context, at rest.

    Operates per-joint via Joint.set_angle()/set_translation(), which sidesteps
    any question of position-vector ordering entirely.
    """
    for i, q in enumerate(arm_position):
        joint = plant.GetJointByName(f"{prefix}_joint_{i}")
        joint.set_angle(plant_context, q)
        joint.set_angular_rate(plant_context, 0.0)
    for name in (
        f"{prefix}_left_carriage_joint",
        f"{prefix}_right_carriage_joint",
    ):
        joint = plant.GetJointByName(name)
        joint.set_translation(plant_context, gripper_position)
        joint.set_translation_rate(plant_context, 0.0)
