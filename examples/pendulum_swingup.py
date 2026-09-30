import numpy as np
from cenic_bench.tasks import PendulumSwingup

def my_dummy_policy(obs):
    """A simple policy example that just returns zero torque."""
    return np.array([0.0])

if __name__ == "__main__":
    task = PendulumSwingup(my_dummy_policy, visualize=True)
    success = task.run_episode(seed=1)
    print("Success:", success)