from __future__ import annotations

from abc import ABC, abstractmethod

from actions import Action
from percept import Percept


class Agent(ABC):
    """Minimal agent interface: map a percept to an action."""

    @abstractmethod
    def act(self, percept: Percept) -> Action:
        raise NotImplementedError


class SimpleReflexVacuumAgent(Agent):
    """Table 2.1 / Fig. 2.8 simple reflex agent for the vacuum world."""

    def __init__(self, size: int = 2):
        self.size = size

    def act(self, percept: Percept) -> Action:
        location, status = percept
        if status == "Dirty":
            return Action.SUCK
        if location < self.size - 1:
            return Action.MOVE_RIGHT
        return Action.MOVE_LEFT
