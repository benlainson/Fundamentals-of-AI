import pytest

from actions import Action
from agent import SimpleReflexVacuumAgent
from environment import VacuumEnvironment
from percept import Percept


def test_default_construction_all_dirty():
    env = VacuumEnvironment(size=2)
    assert env.dirt == [True, True]
    assert env.agent_location == 0
    assert env.performance == 0
    assert env.time_step == 0


def test_rejects_mismatched_dirt_length():
    with pytest.raises(ValueError):
        VacuumEnvironment(size=2, dirt=[True, True, True])


def test_rejects_out_of_range_agent_location():
    with pytest.raises(ValueError):
        VacuumEnvironment(size=2, agent_location=5)


def test_rejects_size_less_than_one():
    with pytest.raises(ValueError):
        VacuumEnvironment(size=0)


def test_percept_reflects_current_location_and_dirt():
    env = VacuumEnvironment(size=2, dirt=[True, False], agent_location=0)
    assert env.percept() == Percept(location=0, status="Dirty")

    env = VacuumEnvironment(size=2, dirt=[True, False], agent_location=1)
    assert env.percept() == Percept(location=1, status="Clean")


def test_suck_cleans_current_square_only():
    env = VacuumEnvironment(size=2, dirt=[True, True], agent_location=0)
    env.execute(Action.SUCK)
    assert env.dirt == [False, True]


def test_move_right_and_left_within_bounds():
    env = VacuumEnvironment(size=3, dirt=[False, False, False], agent_location=0)
    env.execute(Action.MOVE_RIGHT)
    assert env.agent_location == 1
    env.execute(Action.MOVE_RIGHT)
    assert env.agent_location == 2
    env.execute(Action.MOVE_LEFT)
    assert env.agent_location == 1


def test_move_does_not_go_out_of_bounds():
    env = VacuumEnvironment(size=2, dirt=[False, False], agent_location=0)
    env.execute(Action.MOVE_LEFT)  # already at leftmost
    assert env.agent_location == 0

    env = VacuumEnvironment(size=2, dirt=[False, False], agent_location=1)
    env.execute(Action.MOVE_RIGHT)  # already at rightmost
    assert env.agent_location == 1


def test_no_op_changes_nothing_but_still_ticks():
    env = VacuumEnvironment(size=2, dirt=[True, False], agent_location=0)
    env.execute(Action.NO_OP)
    assert env.dirt == [True, False]
    assert env.agent_location == 0
    assert env.time_step == 1


def test_unknown_action_raises():
    env = VacuumEnvironment(size=2, dirt=[True, False], agent_location=0)
    with pytest.raises(ValueError):
        env.execute("bogus")  # type: ignore[arg-type]


def test_performance_awards_one_point_per_clean_square_per_step():
    env = VacuumEnvironment(size=2, dirt=[True, False], agent_location=0)
    env.execute(Action.NO_OP)  # 1 clean square (B) this step
    assert env.performance == 1
    env.execute(Action.SUCK)  # A now also clean -> 2 clean squares
    assert env.performance == 1 + 2


def test_is_clean():
    env = VacuumEnvironment(size=2, dirt=[True, False], agent_location=0)
    assert not env.is_clean()
    env.execute(Action.SUCK)
    assert env.is_clean()


def test_history_records_each_step():
    env = VacuumEnvironment(size=2, dirt=[True, False], agent_location=0)
    env.execute(Action.SUCK)
    env.execute(Action.MOVE_RIGHT)
    assert len(env.history) == 2
    loc0, dirt0, action0 = env.history[0]
    assert (loc0, dirt0, action0) == (0, (False, False), Action.SUCK)
    loc1, dirt1, action1 = env.history[1]
    assert (loc1, dirt1, action1) == (1, (False, False), Action.MOVE_RIGHT)


def test_run_matches_hand_computed_trace():
    """start=A(0), both dirty, 5 steps -- traced by hand against Table 2.1:
    t1 Suck (A->clean, perf+=1=1)
    t2 Right (perf+=1=2)
    t3 Suck (B->clean, perf+=2=4)
    t4 Left (perf+=2=6)
    t5 Right (perf+=2=8)
    """
    env = VacuumEnvironment(size=2, dirt=[True, True], agent_location=0)
    agent = SimpleReflexVacuumAgent(size=2)
    score = env.run(agent, max_steps=5)
    assert score == 8
    assert env.time_step == 5
    assert env.dirt == [False, False]


def test_run_respects_max_steps():
    env = VacuumEnvironment(size=2, dirt=[True, True], agent_location=0)
    agent = SimpleReflexVacuumAgent(size=2)
    env.run(agent, max_steps=3)
    assert env.time_step == 3


def test_run_stop_when_clean_exits_early():
    env = VacuumEnvironment(size=2, dirt=[False, False], agent_location=0)
    agent = SimpleReflexVacuumAgent(size=2)
    env.run(agent, max_steps=1000, stop_when_clean=True)
    # World starts clean, so the loop should never execute an action.
    assert env.time_step == 0
    assert env.performance == 0


def test_custom_performance_measure_is_used():
    def flat_one(dirt):
        return 1

    env = VacuumEnvironment(
        size=2, dirt=[True, True], agent_location=0, performance_measure=flat_one
    )
    env.execute(Action.SUCK)
    assert env.performance == 1
    env.execute(Action.MOVE_RIGHT)
    assert env.performance == 2
