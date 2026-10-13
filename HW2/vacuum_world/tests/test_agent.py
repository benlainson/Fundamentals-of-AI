import pytest

from actions import Action
from agent import SimpleReflexVacuumAgent
from percept import Percept


@pytest.fixture
def agent():
    return SimpleReflexVacuumAgent(size=2)


def test_sucks_when_dirty_regardless_of_location(agent):
    assert agent.act(Percept(location=0, status="Dirty")) == Action.SUCK
    assert agent.act(Percept(location=1, status="Dirty")) == Action.SUCK


def test_matches_table_2_1_at_A(agent):
    # (A, Clean) -> Right
    assert agent.act(Percept(location=0, status="Clean")) == Action.MOVE_RIGHT


def test_matches_table_2_1_at_B(agent):
    # (B, Clean) -> Left
    assert agent.act(Percept(location=1, status="Clean")) == Action.MOVE_LEFT


def test_action_depends_only_on_current_percept():
    """Simple reflex agents are stateless: the same percept must always
    yield the same action, independent of call history."""
    agent = SimpleReflexVacuumAgent(size=2)
    p = Percept(location=0, status="Clean")
    first = agent.act(p)
    agent.act(Percept(location=1, status="Dirty"))
    agent.act(Percept(location=1, status="Clean"))
    second = agent.act(p)
    assert first == second == Action.MOVE_RIGHT


def test_generalizes_to_larger_world():
    agent = SimpleReflexVacuumAgent(size=4)
    assert agent.act(Percept(location=0, status="Clean")) == Action.MOVE_RIGHT
    assert agent.act(Percept(location=2, status="Clean")) == Action.MOVE_RIGHT
    assert agent.act(Percept(location=3, status="Clean")) == Action.MOVE_LEFT
    assert agent.act(Percept(location=1, status="Dirty")) == Action.SUCK