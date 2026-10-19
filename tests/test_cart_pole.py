from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder

from cenic_bench.parsing import make_parser


def test_bob_has_collision_geometry():
    """The pole's bob has a collision geometry."""
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    make_parser(plant).AddModelsFromUrl(
        "package://cenic_bench/models/cart_pole/cart_pole.sdf"
    )
    plant.Finalize()
    pole = plant.GetBodyByName("pole")
    assert len(plant.GetCollisionGeometriesForBody(pole)) == 1
