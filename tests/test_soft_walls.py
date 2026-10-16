import numpy as np
import pytest
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, ContactModel
from pydrake.systems.framework import DiagramBuilder

from cenic_bench.parsing import make_parser

WALLS = [("left_wall", -1.0), ("right_wall", 1.0)]


def make_scene(contact_model=None):
    """Load the cart-pole and soft walls into a continuous-time plant,
    optionally choosing the contact model."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = make_parser(plant)
    parser.AddModelsFromUrl(
        "package://cenic_bench/models/cart_pole/cart_pole.sdf"
    )
    parser.AddModelsFromUrl(
        "package://cenic_bench/models/cart_pole/soft_walls.sdf"
    )
    if contact_model is not None:
        plant.set_contact_model(contact_model)
    plant.Finalize()
    diagram = builder.Build()
    context = plant.GetMyMutableContextFromRoot(diagram.CreateDefaultContext())
    return plant, scene_graph, context


def tilt_pole(plant, context, bob_x):
    """Center the cart and tilt the pole from upright so the bob's center is
    at x = bob_x."""
    plant.SetPositions(context, [0.0, np.pi + np.arcsin(bob_x / 0.5)])


def force_on_bob(plant, context, wall):
    """Return the world-frame point contact force on the bob, checking that
    the only contact is between the bob and the given wall."""
    results = plant.get_contact_results_output_port().Eval(context)
    assert results.num_point_pair_contacts() == 1
    info = results.point_pair_contact_info(0)
    pole = plant.GetBodyByName("pole")
    bob_id = plant.GetCollisionGeometriesForBody(pole)[0]
    (wall_id,) = plant.GetCollisionGeometriesForBody(plant.GetBodyByName(wall))
    pair = info.point_pair()
    assert {pair.id_A, pair.id_B} == {bob_id, wall_id}
    # contact_force() acts on body B.
    if info.bodyB_index() == pole.index():
        return info.contact_force()
    return -info.contact_force()


@pytest.mark.parametrize("wall", ["left_wall", "right_wall"])
def test_wall_stiffness(wall):
    """Each wall declares a positive point contact stiffness."""
    plant, scene_graph, _ = make_scene()
    inspector = scene_graph.model_inspector()
    (wall_id,) = plant.GetCollisionGeometriesForBody(plant.GetBodyByName(wall))
    props = inspector.GetProximityProperties(wall_id)
    assert props.HasProperty("material", "point_contact_stiffness")
    assert props.GetProperty("material", "point_contact_stiffness") > 0


@pytest.mark.parametrize("wall, side", WALLS)
def test_wall_pushes_bob_back(wall, side):
    """Under point contact, a bob just past a wall's contact line touches only
    that wall and is pushed back toward the center."""
    plant, _, context = make_scene(ContactModel.kPoint)
    tilt_pole(plant, context, side * 0.12)
    force = force_on_bob(plant, context, wall)
    assert side * force[0] < 0


@pytest.mark.parametrize("wall, side", WALLS)
def test_deep_contact_pushes_back(wall, side):
    """A bob 20 cm past a wall's contact line is still pushed back toward the
    center."""
    plant, _, context = make_scene(ContactModel.kPoint)
    tilt_pole(plant, context, side * 0.30)
    force = force_on_bob(plant, context, wall)
    assert side * force[0] < 0


def test_default_contact_model():
    """With Drake's default contact model, the bob-wall pair falls back to the
    same point contact, pushing the bob back toward the center."""
    plant, _, context = make_scene()
    tilt_pole(plant, context, 0.12)
    force = force_on_bob(plant, context, "right_wall")
    assert force[0] < 0
    results = plant.get_contact_results_output_port().Eval(context)
    assert results.num_hydroelastic_contacts() == 0


def test_no_contact_upright():
    """Upright and centered, the bob touches neither wall."""
    plant, _, context = make_scene()
    tilt_pole(plant, context, 0.0)
    results = plant.get_contact_results_output_port().Eval(context)
    assert results.num_point_pair_contacts() == 0
    assert results.num_hydroelastic_contacts() == 0
