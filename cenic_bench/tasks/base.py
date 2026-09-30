from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydrake.systems.analysis import (
    ApplySimulatorConfig,
    Simulator,
    SimulatorConfig,
    SimulatorStatus,
)
from pydrake.systems.framework import Context, Diagram, EventStatus

Observation = Any
Action = Any


class Task(ABC):
    """An abstract base class defining a task."""

    def __init__(
        self,
        policy: Callable[[Observation], Action],
        accuracy: float = 1e-3,
        realtime: bool = False,
    ):
        """Set up a simulation environment with CENIC.

        Args:
            policy: A function that maps observations to actions. This is what
                    we are evaluating.
            accuracy: The accuracy tolerance for error-controlled simulation.
            realtime: If True, run in approximately real time. Otherwise the
                      simulation will run as fast as possible.
        """
        self.diagram = self.create_scene(policy)
        self.context = self.diagram.CreateDefaultContext()

        self.simulator = Simulator(self.diagram, self.context)

        # Use continuous-time integration with CENIC.
        config = SimulatorConfig(
            integration_scheme="cenic",
            accuracy=accuracy,
            use_error_control=True,
            target_realtime_rate=1.0 if realtime else 0.0,
        )
        ApplySimulatorConfig(config, self.simulator)

        # Create a monitor for the success condition.
        def monitor(root_context: Context) -> EventStatus:
            if self.success(root_context):
                return EventStatus.ReachedTermination(
                    self.diagram, "Task completed successfully."
                )
            return EventStatus.Succeeded()

        self.simulator.set_monitor(monitor)

    @abstractmethod
    def create_scene(self, policy: Callable[[Observation], Action]) -> Diagram:
        """Create a drake system diagram representing the simulated scenario.

        Args:
            policy: The controller to be evaluated. This method typically wraps
                    the policy in a Drake LeafSystem for simulation.

        Returns:
            A Drake Diagram representing the simulated scenario.
        """

    @abstractmethod
    def reset(self, context: Context, seed: int = 0) -> None:
        """Set a fresh, valid initial state for the task.

        Args:
            context: The simulated system state (changed on output.)
            seed: A random seed defining the initial state.
        """
        pass

    @abstractmethod
    def success(self, context: Context) -> bool:
        """Check if the task has been successfully completed.

        Args:
            context: The simulated system state.

        Returns:
            True if the task is successful, False otherwise.

        Note:
            This method is called at each simulation step, so it should be
            as efficient as possible.
        """
        pass

    @property
    @abstractmethod
    def timeout(self) -> float:
        """How long (in seconds) the task can run before timing out."""
        pass

    @property
    @abstractmethod
    def task_description(self) -> str:
        """A natural language description of the task."""
        pass

    def run_episode(self, seed: int = 0) -> bool:
        """Run a single episode of the task.

        Returns True if the success condition was reached before the timeout.
        """
        self.context.SetTime(0.0)
        self.reset(self.context, seed)
        self.simulator.Initialize()

        status = self.simulator.AdvanceTo(self.timeout)

        # Reaching the termination condition indicates the success condition
        # was met during the simulation.
        return (
            status.reason()
            == SimulatorStatus.ReturnReason.kReachedTerminationCondition
        )
