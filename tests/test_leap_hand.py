import numpy as np
import pytest
from pydrake.geometry import Convex, Mesh, Role
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder

from cenic_bench.parsing import make_parser


def make_leap_hand(side):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    (model,) = make_parser(plant).AddModelsFromUrl(
        f"package://cenic_bench/models/leap_hand/leap_hand_{side}.urdf"
    )
    plant.Finalize()
    return plant, scene_graph, model


@pytest.mark.parametrize("side", ["left", "right"])
def test_leap_hand_parses_and_finalizes(side):
    plant, _, model = make_leap_hand(side)
    assert plant.num_positions() == 23
    assert plant.num_velocities() == 22
    assert plant.num_actuators() == 16
    for i in plant.GetJointActuatorIndices(model):
        assert plant.get_joint_actuator(i).effort_limit() == 0.95

    mass = sum(
        plant.get_body(i).default_mass() for i in plant.GetBodyIndices(model)
    )
    assert mass == pytest.approx(0.745, abs=0.01)


@pytest.mark.parametrize("side", ["left", "right"])
def test_leap_hand_collisions_are_compliant_hydroelastic(side):
    _, scene_graph, _ = make_leap_hand(side)
    inspector = scene_graph.model_inspector()
    geometry_ids = inspector.GetAllGeometryIds(Role.kProximity)
    assert len(geometry_ids) == 71

    for geometry_id in geometry_ids:
        props = inspector.GetProximityProperties(geometry_id)
        assert props.HasProperty("hydroelastic", "compliance_type")
        assert props.GetProperty("hydroelastic", "hydroelastic_modulus") == 1e6
        assert (
            props.GetProperty("material", "coulomb_friction").static_friction()
            == 1.0
        )


def mesh_bounds(inspector, geometry_id):
    hull = inspector.GetShape(geometry_id).GetConvexHull()
    X = inspector.GetPoseInFrame(geometry_id)
    vertices = np.array(
        [X @ hull.vertex(i) for i in range(hull.num_vertices())]
    )
    return vertices.min(axis=0), vertices.max(axis=0)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("link", ["fingertip", "thumb_fingertip"])
def test_leap_hand_fingertip_visual_contains_collision(side, link):
    plant, scene_graph, _ = make_leap_hand(side)
    inspector = scene_graph.model_inspector()
    frame_id = plant.GetBodyFrameIdOrThrow(plant.GetBodyByName(link).index())
    (visual,) = [
        g
        for g in inspector.GetGeometries(frame_id, Role.kIllustration)
        if isinstance(inspector.GetShape(g), Mesh)
    ]
    (collision,) = [
        g
        for g in inspector.GetGeometries(frame_id, Role.kProximity)
        if isinstance(inspector.GetShape(g), Convex)
    ]

    visual_min, visual_max = mesh_bounds(inspector, visual)
    collision_min, collision_max = mesh_bounds(inspector, collision)
    assert np.all(collision_min >= visual_min - 1e-3)
    assert np.all(collision_max <= visual_max + 1e-3)
