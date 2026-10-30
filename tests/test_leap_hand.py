from itertools import product

import numpy as np
import pytest
from pydrake.geometry import Box, Convex, Mesh, Role
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder

from cenic_bench.parsing import make_parser


def make_leap_hand(side):
    """Load the `side` Leap Hand into a finalized continuous-time plant.

    Returns the plant, its scene graph, and the hand's model instance.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    (model,) = make_parser(plant).AddModelsFromUrl(
        f"package://cenic_bench/models/leap_hand/leap_hand_{side}.urdf"
    )
    plant.Finalize()
    return plant, scene_graph, model


@pytest.mark.parametrize("side", ["left", "right"])
def test_leap_hand_parses_and_finalizes(side):
    """The hand has the expected DOFs, 16 actuators limited to 0.95 N·m,
    and a total mass of about 0.745 kg."""
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
    """Every collision geometry is compliant hydroelastic with modulus
    1e6 Pa and friction 1.0."""
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


def geometry_vertices(inspector, geometry_id):
    """Return a box's corners or a mesh's convex hull vertices, expressed in
    the geometry's parent frame."""
    shape = inspector.GetShape(geometry_id)
    if isinstance(shape, Box):
        points = np.array(list(product([-0.5, 0.5], repeat=3))) * shape.size()
    else:
        hull = shape.GetConvexHull()
        points = [hull.vertex(i) for i in range(hull.num_vertices())]
    X = inspector.GetPoseInFrame(geometry_id)
    return np.array([X @ p for p in points])


def mesh_bounds(inspector, geometry_id):
    """Return the axis-aligned (min, max) corners of a geometry's convex
    hull, expressed in its parent frame."""
    vertices = geometry_vertices(inspector, geometry_id)
    return vertices.min(axis=0), vertices.max(axis=0)


@pytest.mark.parametrize("side", ["left", "right"])
def test_leap_hand_visual_and_collision_centers_agree(side):
    """Each link's visual and collision geometry have bounding box centers
    within 3 mm of each other."""
    plant, scene_graph, model = make_leap_hand(side)
    inspector = scene_graph.model_inspector()
    for i in plant.GetBodyIndices(model):
        frame_id = plant.GetBodyFrameIdIfExists(i)
        if frame_id is None:
            continue
        centers = []
        for role in [Role.kIllustration, Role.kProximity]:
            geometry_ids = inspector.GetGeometries(frame_id, role)
            if not geometry_ids:
                break
            vertices = np.vstack(
                [geometry_vertices(inspector, g) for g in geometry_ids]
            )
            centers.append((vertices.min(axis=0) + vertices.max(axis=0)) / 2)
        if len(centers) == 2:
            distance = np.linalg.norm(centers[0] - centers[1])
            assert distance < 3e-3, plant.get_body(i).name()


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("link", ["fingertip", "thumb_fingertip"])
def test_leap_hand_fingertip_visual_contains_collision(side, link):
    """The fingertip's convex collision mesh fits within the bounding box
    of its visual mesh, within 1 mm."""
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


@pytest.mark.parametrize("side", ["left", "right"])
def test_leap_hand_com_inside_visual(side):
    """Each link's center of mass lies within the bounding box of its visual
    meshes."""
    plant, scene_graph, model = make_leap_hand(side)
    inspector = scene_graph.model_inspector()
    for i in plant.GetBodyIndices(model):
        frame_id = plant.GetBodyFrameIdIfExists(i)
        if frame_id is None:
            continue
        bounds = [
            mesh_bounds(inspector, g)
            for g in inspector.GetGeometries(frame_id, Role.kIllustration)
        ]
        if not bounds:
            continue
        visual_min = np.min([lo for lo, _ in bounds], axis=0)
        visual_max = np.max([hi for _, hi in bounds], axis=0)
        com = plant.get_body(i).default_com()
        name = plant.get_body(i).name()
        assert np.all((visual_min <= com) & (com <= visual_max)), name
