import numpy as np
from pydrake.geometry import (
    AddRigidHydroelasticProperties,
    Box,
    ProximityProperties,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.analysis import (
    ApplySimulatorConfig,
    Simulator,
    SimulatorConfig,
)
from pydrake.systems.framework import DiagramBuilder

from cenic_bench.parsing import make_parser


def _drive(left_torque, right_torque):
    """Apply constant wheel torques on flat ground for 0.5 s with CENIC.

    Returns the chassis pose in the world at the end.
    """
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    make_parser(plant).AddModelsFromUrl(
        "package://cenic_bench/models/diff_drive/diff_drive.sdf"
    )
    props = ProximityProperties()
    AddRigidHydroelasticProperties(props)
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        RigidTransform([0, 0, -0.05]),
        Box(2, 2, 0.1),
        "ground",
        props,
    )
    plant.Finalize()
    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    u = np.zeros(plant.num_actuated_dofs())
    for name, torque in [("left", left_torque), ("right", right_torque)]:
        plant.GetJointActuatorByName(name).set_actuation_vector([torque], u)
    plant.get_actuation_input_port().FixValue(plant_context, u)

    simulator = Simulator(diagram, context)
    config = SimulatorConfig(
        integration_scheme="cenic", accuracy=1e-3, use_error_control=True
    )
    ApplySimulatorConfig(config, simulator)
    simulator.AdvanceTo(0.5)
    return plant.EvalBodyPoseInWorld(
        plant_context, plant.GetBodyByName("chassis")
    )


def test_only_wheels_are_actuated():
    """The model has exactly two actuators, one per wheel joint."""
    plant = MultibodyPlant(time_step=0.0)
    make_parser(plant).AddModelsFromUrl(
        "package://cenic_bench/models/diff_drive/diff_drive.sdf"
    )
    plant.Finalize()
    actuators = [
        plant.get_joint_actuator(i) for i in plant.GetJointActuatorIndices()
    ]
    assert sorted(a.joint().name() for a in actuators) == ["left", "right"]


def test_equal_torques_drive_forward():
    """Equal positive wheel torques move the chassis forward."""
    assert _drive(0.05, 0.05).translation()[0] > 0


def test_opposite_torques_turn_left():
    """A faster right wheel turns the robot left."""
    X_WC = _drive(-0.05, 0.05)
    assert RollPitchYaw(X_WC.rotation()).yaw_angle() > 0
