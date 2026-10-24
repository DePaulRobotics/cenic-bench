import numpy as np
from pydrake.geometry import (
    AddRigidHydroelasticProperties,
    Box,
    ProximityProperties,
)
from pydrake.math import RigidTransform
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.analysis import (
    ApplySimulatorConfig,
    Simulator,
    SimulatorConfig,
)
from pydrake.systems.framework import DiagramBuilder

from cenic_bench.parsing import make_parser


def test_only_x_and_y_are_actuated():
    """The pusher has exactly two actuators, driving x and y."""
    plant = MultibodyPlant(time_step=0.0)
    make_parser(plant).AddModelsFromUrl(
        "package://cenic_bench/models/pusher/pusher.sdf"
    )
    plant.Finalize()
    actuators = [
        plant.get_joint_actuator(i) for i in plant.GetJointActuatorIndices()
    ]
    assert [a.joint().name() for a in actuators] == ["x", "y"]


def test_pushing_the_stem_moves_the_t_forward():
    """Driving the pusher into the stem pushes the T along +y."""
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = make_parser(plant)
    parser.AddModelsFromUrl("package://cenic_bench/models/t_block/t_block.sdf")
    parser.AddModelsFromUrl("package://cenic_bench/models/pusher/pusher.sdf")
    props = ProximityProperties()
    AddRigidHydroelasticProperties(properties=props)
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        RigidTransform([0, 0, -0.05]),
        Box(2, 2, 0.1),
        "floor",
        props,
    )
    plant.Finalize()
    diagram = builder.Build()

    simulator = Simulator(diagram)
    root_context = simulator.get_mutable_context()
    context = plant.GetMyMutableContextFromRoot(root_context)
    plant.GetJointByName("y").set_translation(context, -0.15)
    y_actuator = plant.GetJointActuatorByName("y")
    u = np.zeros(plant.num_actuated_dofs())
    u[y_actuator.input_start()] = y_actuator.effort_limit()
    plant.get_actuation_input_port().FixValue(context, u)

    config = SimulatorConfig(
        integration_scheme="cenic", accuracy=1e-3, use_error_control=True
    )
    ApplySimulatorConfig(config, simulator)
    simulator.AdvanceTo(1.0)

    t = plant.GetBodyByName("t")
    assert plant.EvalBodyPoseInWorld(context, t).translation()[1] > 0
