from __future__ import annotations

from typing import NamedTuple


class Percept(NamedTuple):
    """What the agent senses at a given time step.

    location: the agent's current location, an int in [0, size).
    status:   "Dirty" or "Clean" -- the dirt sensor reading for the
    square the agent currently occupies.
    """

    location: int
    status: str  # "Dirty" | "Clean"
