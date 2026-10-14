from pydrake.multibody.plant import MultibodyPlant

from cenic_bench.parsing import make_parser


def test_cube_parses_and_finalizes():
    plant = MultibodyPlant(time_step=0.0)
    make_parser(plant).AddModelsFromUrl(
        "package://cenic_bench/models/cube/cube.urdf"
    )
    plant.Finalize()
    assert plant.num_positions() == 7
    assert plant.num_velocities() == 6
    assert plant.num_bodies() == 2  # world + cube_link
