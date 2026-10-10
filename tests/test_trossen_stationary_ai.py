import numpy as np
from pydrake.multibody.plant import MultibodyPlant

from cenic_bench.parsing import make_parser

MODEL_URL = "package://cenic_bench/models/trossen_stationary_ai/stationary_ai.urdf"


def _load_plant():
    plant = MultibodyPlant(time_step=0.0)
    make_parser(plant).AddModelsFromUrl(MODEL_URL)
    plant.Finalize()
    return plant


def test_degrees_of_freedom():
    plant = _load_plant()
    assert plant.num_positions() == 16
    assert plant.num_velocities() == 16
    # Each gripper's mimicked finger isn't separately actuated.
    assert plant.num_actuators() == 14


def test_frame_sits_above_tabletop():
    # Regression test: world_joint/frame_joint previously both claimed
    # frame_link as their child, silently misplacing the whole assembly.
    plant = _load_plant()
    context = plant.CreateDefaultContext()
    tabletop = plant.GetFrameByName("tabletop_link").CalcPoseInWorld(context)
    frame = plant.GetFrameByName("frame_link").CalcPoseInWorld(context)
    np.testing.assert_allclose(tabletop.translation(), [0, 0, 0])
    np.testing.assert_allclose(frame.translation(), [0, 0, 0.02])
