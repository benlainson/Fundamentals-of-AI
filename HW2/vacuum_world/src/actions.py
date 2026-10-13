from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    SUCK = "Suck"
    MOVE_LEFT = "Left"
    MOVE_RIGHT = "Right"
    NO_OP = "NoOp"

    def __str__(self) -> str:  # nicer printing than "Action.SUCK"
        return self.value
