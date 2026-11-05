"""
test_planning.py
================
Automated tests for Homework 7, Exercise 5.  Run from inside src/:

    python -m pytest -v

Test groups:
  1. Gridworld model: state set, terminals, transition probabilities,
     the required UP-at-(1, 1) case, merged outcomes, walls, rewards.
  2. Toy MDP A -> B -> T: hand-checked synchronous value-iteration values.
  3. Shared helpers: tie-breaking order.
  4. Solvers on the gridworld: stopping rules, residuals, policy stability,
     and agreement between value iteration and policy iteration.
"""

import os
import sys

# Make the modules in src/ importable no matter where pytest is launched from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from mdp import GridWorldMDP, TabularMDP, make_toy_chain_mdp, ACTIONS
from planning_utils import greedy_action, greedy_policy, q_values, zero_values
from value_iteration import value_iteration, value_iteration_sweep
from policy_iteration import policy_iteration, policy_evaluation

PROB_TOL = 1e-12


# ---------------------------------------------------------------------------
# Fixtures: solve the gridworld once and share the results across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid():
    return GridWorldMDP()


@pytest.fixture(scope="module")
def vi_result(grid):
    return value_iteration(grid)


@pytest.fixture(scope="module")
def pi_result(grid):
    return policy_iteration(grid)


# ---------------------------------------------------------------------------
# Group 1: Gridworld model
# ---------------------------------------------------------------------------

def test_state_set_has_11_states_and_excludes_wall(grid):
    assert len(grid.states) == 11
    assert (2, 2) not in grid.states
    assert (3, 4) in grid.states and (2, 4) in grid.states


def test_terminals_have_no_actions_and_nonterminals_have_four(grid):
    assert grid.terminals == {(3, 4), (2, 4)}
    for s in grid.states:
        if grid.is_terminal(s):
            assert grid.actions(s) == []
        else:
            assert grid.actions(s) == list(ACTIONS)   # also checks tie-break order


def test_transition_probabilities_valid(grid):
    """Required test: nonnegative and summing to 1 for every (s, a)."""
    for s in grid.nonterminal_states:
        for a in grid.actions(s):
            outcomes = grid.transitions(s, a)
            probs = [p for p, _, _ in outcomes]
            assert all(p >= 0 for p in probs), (s, a)
            assert abs(sum(probs) - 1.0) < PROB_TOL, (s, a)
            successors = [s_next for _, s_next, _ in outcomes]
            assert len(successors) == len(set(successors)), "outcomes not merged"
            assert all(s_next in grid.states for s_next in successors)


def test_up_at_1_1(grid):
    """Required test: UP at (1, 1) -> (2,1): 0.8, (1,1): 0.1, (1,2): 0.1."""
    dist = grid.successor_distribution((1, 1), "UP")
    assert set(dist) == {(2, 1), (1, 1), (1, 2)}
    assert dist[(2, 1)] == pytest.approx(0.8, abs=PROB_TOL)
    assert dist[(1, 1)] == pytest.approx(0.1, abs=PROB_TOL)
    assert dist[(1, 2)] == pytest.approx(0.1, abs=PROB_TOL)


def test_outcomes_landing_in_same_cell_are_combined(grid):
    """LEFT at (1,1): intended LEFT and DOWN-slip both stay put -> 0.9 merged."""
    dist = grid.successor_distribution((1, 1), "LEFT")
    assert dist == pytest.approx({(1, 1): 0.9, (2, 1): 0.1})


def test_wall_blocks_movement(grid):
    """RIGHT at (2, 1) hits the wall at (2, 2), so the agent stays with 0.8."""
    dist = grid.successor_distribution((2, 1), "RIGHT")
    assert dist == pytest.approx({(2, 1): 0.8, (3, 1): 0.1, (1, 1): 0.1})
    assert (2, 2) not in dist


def test_rewards_depend_on_entered_cell(grid):
    """+1 entering (3,4), -1 entering (2,4), -0.04 otherwise, no extra step cost."""
    rewards = {s_next: r for _, s_next, r in grid.transitions((3, 3), "RIGHT")}
    assert rewards[(3, 4)] == 1.0
    assert rewards[(2, 3)] == pytest.approx(-0.04)
    assert rewards[(3, 3)] == pytest.approx(-0.04)      # bumped the boundary
    rewards = {s_next: r for _, s_next, r in grid.transitions((2, 3), "RIGHT")}
    assert rewards[(2, 4)] == -1.0


# ---------------------------------------------------------------------------
# Group 2: Toy MDP A -> B -> T
# ---------------------------------------------------------------------------

def test_toy_mdp_synchronous_value_iteration():
    """Required test: V_1(A)=0, V_1(B)=1, V_2(A)=0.9, V_2(B)=1."""
    toy = make_toy_chain_mdp(gamma=0.9)
    V0 = zero_values(toy)
    V1, _ = value_iteration_sweep(toy, V0)
    V2, _ = value_iteration_sweep(toy, V1)
    assert V1["A"] == pytest.approx(0.0) and V1["B"] == pytest.approx(1.0)
    assert V2["A"] == pytest.approx(0.9) and V2["B"] == pytest.approx(1.0)
    assert V1["T"] == 0.0 and V2["T"] == 0.0
    assert V0 == {"A": 0.0, "B": 0.0, "T": 0.0}        # old values not mutated


def test_toy_mdp_value_iteration_and_policy_evaluation_exact():
    """Both solvers' machinery recovers the exact values V(A)=0.9, V(B)=1."""
    toy = make_toy_chain_mdp(gamma=0.9)
    vi = value_iteration(toy)
    V_pi, _ = policy_evaluation(toy, {"A": "GO", "B": "GO"})
    for V in (vi["V"], V_pi):
        assert V["A"] == pytest.approx(0.9, abs=1e-8)
        assert V["B"] == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Group 3: Tie-breaking
# ---------------------------------------------------------------------------

def test_ties_broken_in_up_right_down_left_order():
    """Equal-valued actions: the earliest in UP, RIGHT, DOWN, LEFT wins."""
    P = {"S": {a: [(1.0, "T", 0.5)] for a in ACTIONS}, "T": {}}
    tied = TabularMDP(["S", "T"], ["T"], P, gamma=0.9)
    assert greedy_action(tied, zero_values(tied), "S") == "UP"

    # RIGHT and LEFT tied for best (UP and DOWN worse): RIGHT must win.
    P2 = {"S": {"UP": [(1.0, "T", 0.0)], "RIGHT": [(1.0, "T", 1.0)],
                "DOWN": [(1.0, "T", 0.0)], "LEFT": [(1.0, "T", 1.0)]}, "T": {}}
    tied2 = TabularMDP(["S", "T"], ["T"], P2, gamma=0.9)
    assert greedy_action(tied2, zero_values(tied2), "S") == "RIGHT"


def test_differences_below_tie_tolerance_count_as_ties():
    P = {"S": {"UP": [(1.0, "T", 1.0)], "RIGHT": [(1.0, "T", 1.0 + 1e-13)],
               "DOWN": [(1.0, "T", 0.0)], "LEFT": [(1.0, "T", 0.0)]}, "T": {}}
    near_tie = TabularMDP(["S", "T"], ["T"], P, gamma=0.9)
    assert greedy_action(near_tie, zero_values(near_tie), "S") == "UP"


# ---------------------------------------------------------------------------
# Group 4: Solvers on the gridworld
# ---------------------------------------------------------------------------

def test_value_iteration_stopping_rule(vi_result):
    deltas = vi_result["deltas"]
    assert len(deltas) == vi_result["sweeps"]
    assert deltas[-1] < 1e-8                     # stopped because change < 1e-8
    assert all(d >= 1e-8 for d in deltas[:-1])   # ... and not any earlier


def test_terminal_values_are_zero(grid, vi_result, pi_result):
    for result in (vi_result, pi_result):
        for t in grid.terminals:
            assert result["V"][t] == 0.0


def test_bellman_residuals_are_small(vi_result, pi_result):
    assert vi_result["residual"] < 1e-8
    assert pi_result["residual"] < 1e-8


def test_policy_iteration_final_policy_is_stable(grid, pi_result):
    assert greedy_policy(grid, pi_result["V"]) == pi_result["policy"]
    assert pi_result["changes_per_round"][-1] == 0


def test_algorithms_agree_on_values(grid, vi_result, pi_result):
    """Required test: final values agree within 1e-6 at every nonterminal state."""
    for s in grid.nonterminal_states:
        assert abs(vi_result["V"][s] - pi_result["V"][s]) < 1e-6, s


def test_algorithms_agree_on_actions_or_are_tied(grid, vi_result, pi_result):
    """Any differing greedy action must be effectively tied under VI's Q-values."""
    for s in grid.nonterminal_states:
        a_vi, a_pi = vi_result["policy"][s], pi_result["policy"][s]
        if a_vi != a_pi:
            qs = q_values(grid, vi_result["V"], s)
            assert abs(qs[a_vi] - qs[a_pi]) < 1e-6, (s, a_vi, a_pi)


def test_risky_route_avoided_at_1_4(grid, vi_result):
    """At (1,4), UP heads straight into -1; the optimal action goes LEFT instead."""
    assert vi_result["policy"][(1, 4)] == "LEFT"
    assert vi_result["policy"][(2, 3)] == "UP"   # never RIGHT into the -1 cell