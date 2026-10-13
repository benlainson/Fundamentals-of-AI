
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple  # noqa: UP035

from actions import Action
from percept import Percept


def default_performance_measure(dirt: List[bool]) -> int:
    """AIMA's performance measure (p. 38): one point for every clean
    square, awarded at every time step."""
    return sum(1 for is_dirty in dirt if not is_dirty)


@dataclass
class VacuumEnvironment:
    """A configurable, deterministic vacuum-cleaner world.

    Attributes
    ----------
    size:
        Number of locations in the (1-D) world. Default 2, matching
        the textbook's two-square A/B world.
    dirt:
        Initial dirt configuration, one bool per location (True =
        dirty). Defaults to "every square dirty" if omitted.
    agent_location:
        Index of the agent's starting location.
    performance_measure:
        Callable(dirt_list) -> int, applied once per time step and
        accumulated into ``performance``. Swappable so a different
        cost/reward structure (e.g. penalize moves) can be plugged in
        without touching the simulation loop.
    """

    size: int = 2
    dirt: List[bool] = field(default_factory=list)
    agent_location: int = 0
    performance_measure: Callable[[List[bool]], int] = default_performance_measure

    performance: int = field(default=0, init=False)
    time_step: int = field(default=0, init=False)
    history: List[Tuple[int, Tuple[bool, ...], Action]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("size must be >= 1")
        if not self.dirt:
            self.dirt = [True] * self.size
        if len(self.dirt) != self.size:
            raise ValueError(f"len(dirt)={len(self.dirt)} must equal size={self.size}")
        if not (0 <= self.agent_location < self.size):
            raise ValueError(f"agent_location {self.agent_location} out of range [0, {self.size})")
        # copy so callers' lists aren't mutated behind their back
        self.dirt = list(self.dirt)

    # -- sensors -----------------------------------------------------
    def percept(self) -> Percept:
        """Return what the agent currently senses."""
        status = "Dirty" if self.dirt[self.agent_location] else "Clean"
        return Percept(location=self.agent_location, status=status)

    # -- actuators -----------------------------------------------------
    def execute(self, action: Action) -> None:
        """Apply one action, update the world, and score this time step."""
        if action == Action.SUCK:
            self.dirt[self.agent_location] = False
        elif action == Action.MOVE_LEFT:
            self.agent_location = max(0, self.agent_location - 1)
        elif action == Action.MOVE_RIGHT:
            self.agent_location = min(self.size - 1, self.agent_location + 1)
        elif action == Action.NO_OP:
            pass
        else:
            raise ValueError(f"Unknown action: {action!r}")

        self.performance += self.performance_measure(self.dirt)
        self.time_step += 1
        self.history.append((self.agent_location, tuple(self.dirt), action))

    def is_clean(self) -> bool:
        return not any(self.dirt)

    # -- simulation loop -----------------------------------------------------
    def run(self, agent, max_steps: int = 1000, stop_when_clean: bool = False) -> int:
        """Run the sense-think-act loop for up to ``max_steps`` steps.

        Returns the final accumulated performance score. This is the
        top-level "environment simulator": it wires the agent's
        percept -> action decision to the environment's actuators and
        keeps score, without either side needing to know about the
        other's internals.
        """
        for _ in range(max_steps):
            if stop_when_clean and self.is_clean():
                break
            p = self.percept()
            a = agent.act(p)
            self.execute(a)
        return self.performance