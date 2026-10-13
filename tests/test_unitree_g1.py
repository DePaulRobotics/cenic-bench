import numpy as np
import pytest
from pydrake.geometry import Capsule, Sphere
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import RevoluteJoint
from pydrake.systems.framework import DiagramBuilder

from cenic_bench.parsing import make_parser


def make_g1():
    """Load the G1 into a continuous-time plant with a scene graph."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    make_parser(plant).AddModelsFromUrl(
        "package://cenic_bench/models/unitree_g1/g1.sdf"
    )
    plant.Finalize()
    diagram = builder.Build()
    return plant, scene_graph, diagram.CreateDefaultContext()


@pytest.fixture(scope="module")
def g1():
    """Share one loaded G1 across this module's tests."""
    return make_g1()


def test_joints(g1):
    """The robot has a floating pelvis and the expected revolute joints."""
    plant, _, _ = g1
    joints = [plant.get_joint(i) for i in plant.GetJointIndices()]
    revolute = [j for j in joints if isinstance(j, RevoluteJoint)]
    assert len(revolute) == 29
    assert plant.GetBodyByName("pelvis").is_floating_base_body()
    assert plant.num_positions() == 7 + 29
    assert plant.num_velocities() == 6 + 29


def test_actuators(g1):
    """Every revolute joint, and only those, has a finite effort limit."""
    plant, _, _ = g1
    actuators = [
        plant.get_joint_actuator(i) for i in plant.GetJointActuatorIndices()
    ]
    revolute = {
        plant.get_joint(i).name()
        for i in plant.GetJointIndices()
        if isinstance(plant.get_joint(i), RevoluteJoint)
    }
    assert len(actuators) == 29
    assert {a.joint().name() for a in actuators} == revolute
    for a in actuators:
        assert 0 < a.effort_limit() < np.inf


@pytest.mark.parametrize(
    "joint, effort",
    [
        ("left_shoulder_pitch_joint", 25.0),  # 5020
        ("left_hip_pitch_joint", 88.0),  # 7520-14
        ("left_knee_joint", 139.0),  # 7520-22
        ("left_wrist_yaw_joint", 5.0),  # 4010
        ("waist_roll_joint", 50.0),  # 2 x 5020 linkage
        ("left_ankle_pitch_joint", 50.0),  # 2 x 5020 linkage
    ],
)
def test_effort_limits(g1, joint, effort):
    """One joint per motor group has mjlab's effort limit."""
    plant, _, _ = g1
    assert plant.GetJointActuatorByName(joint).effort_limit() == effort


def test_mass(g1):
    """Total mass matches the sum of mjlab's link masses."""
    plant, _, context = g1
    plant_context = plant.GetMyContextFromRoot(context)
    assert plant.CalcTotalMass(plant_context) == pytest.approx(33.341142)


def test_collision_geometry(g1):
    """Collision shapes are compliant hydroelastic primitives with friction."""
    _, scene_graph, _ = g1
    inspector = scene_graph.model_inspector()
    ids = [
        i
        for i in inspector.GetAllGeometryIds()
        if inspector.GetProximityProperties(i) is not None
    ]
    assert len(ids) == 33
    for i in ids:
        assert isinstance(inspector.GetShape(i), (Capsule, Sphere))
        props = inspector.GetProximityProperties(i)
        assert props.HasProperty("hydroelastic", "hydroelastic_modulus")
        assert props.HasProperty("hydroelastic", "resolution_hint")
        friction = props.GetProperty("material", "coulomb_friction")
        assert friction.static_friction() > 0
        assert friction.dynamic_friction() > 0


def test_no_self_collision_at_default_pose(g1):
    """No collision geometries overlap at the default pose."""
    _, scene_graph, context = g1
    query = scene_graph.get_query_output_port().Eval(
        scene_graph.GetMyContextFromRoot(context)
    )
    assert not query.HasCollisions()
