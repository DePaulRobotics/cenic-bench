import numpy as np
from pydrake.multibody.plant import MultibodyPlant

from cenic_bench.parsing import make_parser


def _load_plant():
    """Load the acrobot into a finalized continuous-time plant."""
    plant = MultibodyPlant(time_step=0.0)
    make_parser(plant).AddModelsFromUrl(
        "package://cenic_bench/models/acrobot/acrobot.sdf"
    )
    plant.Finalize()
    return plant


def test_only_elbow_is_actuated():
    """The model has exactly one actuator, and it drives the elbow."""
    plant = _load_plant()
    actuators = [
        plant.get_joint_actuator(i) for i in plant.GetJointActuatorIndices()
    ]
    assert [a.joint().name() for a in actuators] == ["elbow"]


def test_upright_lower_link_is_above_pivot():
    """The lower link's COM is above the pivot when upright."""
    plant = _load_plant()
    context = plant.CreateDefaultContext()
    plant.GetJointByName("shoulder").set_angle(context, np.pi)
    plant.GetJointByName("elbow").set_angle(context, 0.0)
    lower = plant.GetBodyByName("lower_link")
    com = plant.EvalBodyPoseInWorld(context, lower) @ (
        lower.CalcCenterOfMassInBodyFrame(context)
    )
    pivot = plant.GetFrameByName("base").CalcPoseInWorld(context)
    assert com[2] > pivot.translation()[2]
