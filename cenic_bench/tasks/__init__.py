from cenic_bench.tasks.base import Task
from cenic_bench.tasks.cart_pole_swingup import CartPoleSwingup
from cenic_bench.tasks.cart_pole_walls import CartPoleWalls
from cenic_bench.tasks.pendulum_swingup import PendulumSwingup
from cenic_bench.tasks.trossen_cube_pick_place import TrossenCubePickPlace

__all__ = [
    "Task",
    "CartPoleSwingup",
    "CartPoleWalls",
    "PendulumSwingup",
    "TrossenCubePickPlace",
]
