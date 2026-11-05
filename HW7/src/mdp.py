"""
mdp.py
======
MDP representation for Homework 7, Exercise 5.

This module contains:
  * TabularMDP         - a small generic container for a finite MDP. Both
                         planning algorithms only talk to this interface, so
                         they work on the gridworld AND on the toy test MDP.
  * GridWorldMDP       - the 3x4 gridworld from Exercise 2:
                           - 11 states (12 cells minus the wall at (2, 2))
                           - terminals (3, 4) = +1 and (2, 4) = -1
                           - 0.8 intended / 0.1 / 0.1 perpendicular slips
                           - -0.04 for every non-terminal-entering transition
  * make_toy_chain_mdp - the deterministic A -> B -> terminal MDP used to
                         check value iteration by hand.

Coordinate convention: (row, col), 1-indexed, row 1 is the BOTTOM row.
So UP increases the row number and RIGHT increases the column number.
"""

# ---------------------------------------------------------------------------
# Section 1: Actions, movement deltas, and slip directions
# ---------------------------------------------------------------------------

UP, RIGHT, DOWN, LEFT = "UP", "RIGHT", "DOWN", "LEFT"

# The ORDER of this tuple matters: it is the tie-breaking priority required
# by the assignment (UP, RIGHT, DOWN, LEFT). Every place that loops over
# actions uses this order, so ties are always broken the same way.
ACTIONS = (UP, RIGHT, DOWN, LEFT)

# (change in row, change in column) for each direction.
DELTAS = {
    UP:    (1, 0),
    RIGHT: (0, 1),
    DOWN:  (-1, 0),
    LEFT:  (0, -1),
}

# The two directions at right angles to each intended direction.
# With probability 0.1 each, the agent slips into one of these instead.
PERPENDICULAR = {
    UP:    (LEFT, RIGHT),
    DOWN:  (RIGHT, LEFT),
    RIGHT: (UP, DOWN),
    LEFT:  (DOWN, UP),
}


# ---------------------------------------------------------------------------
# Section 2: Generic tabular MDP
# ---------------------------------------------------------------------------

class TabularMDP:
    """
    A finite MDP stored as an explicit transition table.

    transitions[s][a] is a list of (probability, next_state, reward) tuples.
    Terminal states map to an empty dict: they have no actions, and their
    value is fixed at zero by the solvers.
    """

    def __init__(self, states, terminals, transitions, gamma):
        self.states = list(states)          # every state, including terminals
        self.terminals = set(terminals)     # states where the episode ends
        self.P = transitions                # P[s][a] -> [(p, s_next, r), ...]
        self.gamma = gamma                  # discount factor

    def is_terminal(self, s):
        """True if s is a terminal (absorbing, value 0) state."""
        return s in self.terminals

    def actions(self, s):
        """
        Actions available in state s, in tie-breaking order.
        Python dicts preserve insertion order, and the gridworld inserts
        actions in ACTIONS order, so this list is UP, RIGHT, DOWN, LEFT.
        Terminal states return an empty list.
        """
        if self.is_terminal(s):
            return []
        return list(self.P[s].keys())

    def transitions(self, s, a):
        """List of (probability, next_state, reward) for taking a in s."""
        return self.P[s][a]

    def successor_distribution(self, s, a):
        """
        {next_state: probability} for (s, a). Convenient for tests, e.g.
        checking that UP at (1, 1) gives {(2,1): 0.8, (1,1): 0.1, (1,2): 0.1}.
        """
        return {s_next: p for p, s_next, _ in self.P[s][a]}

    @property
    def nonterminal_states(self):
        """All states that have actions (used for reporting and tests)."""
        return [s for s in self.states if not self.is_terminal(s)]


# ---------------------------------------------------------------------------
# Section 3: The Exercise 2 gridworld
# ---------------------------------------------------------------------------

class GridWorldMDP(TabularMDP):
    """
    The 3x4 stochastic gridworld.

    Movement model: the intended move succeeds with probability p_intended
    (0.8); with probability p_slip (0.1) each, the agent moves in one of the
    two perpendicular directions instead. Any move that would leave the grid
    or enter the wall leaves the agent where it is.

    Reward model: the reward depends only on the cell being ENTERED:
      +1 for entering (3, 4), -1 for entering (2, 4), -0.04 otherwise
      (including "bumping" into a wall and staying put). No extra step cost
      is added on top of the terminal-entry reward.
    """

    def __init__(self,
                 n_rows=3,
                 n_cols=4,
                 walls=((2, 2),),
                 terminal_rewards=None,
                 step_reward=-0.04,
                 gamma=0.9,
                 p_intended=0.8,
                 p_slip=0.1):
        if terminal_rewards is None:
            terminal_rewards = {(3, 4): 1.0, (2, 4): -1.0}

        # Store the grid geometry and model parameters. These must be set
        # BEFORE building the transition table, because the helper methods
        # below use them.
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.walls = set(walls)
        self.terminal_rewards = dict(terminal_rewards)
        self.step_reward = step_reward
        self.p_intended = p_intended
        self.p_slip = p_slip

        # State set: every cell in the grid except walls -> 11 states.
        states = [(r, c)
                  for r in range(1, n_rows + 1)
                  for c in range(1, n_cols + 1)
                  if (r, c) not in self.walls]

        # Build the full transition table once, up front.
        P = {}
        for s in states:
            if s in self.terminal_rewards:
                P[s] = {}                      # terminals have no actions
            else:
                P[s] = {a: self._build_outcomes(s, a) for a in ACTIONS}

        super().__init__(states, self.terminal_rewards.keys(), P, gamma)

    # -- helpers ------------------------------------------------------------

    def _in_bounds(self, cell):
        """True if the cell is inside the 3x4 grid."""
        r, c = cell
        return 1 <= r <= self.n_rows and 1 <= c <= self.n_cols

    def _move(self, s, direction):
        """
        Deterministic result of physically moving one step in `direction`.
        Hitting the boundary or the wall means staying in place.
        """
        dr, dc = DELTAS[direction]
        target = (s[0] + dr, s[1] + dc)
        if not self._in_bounds(target) or target in self.walls:
            return s
        return target

    def reward(self, s_next):
        """Reward for a transition that ENTERS s_next."""
        return self.terminal_rewards.get(s_next, self.step_reward)

    def _build_outcomes(self, s, a):
        """
        Compute the (probability, next_state, reward) list for (s, a).

        Three physical outcomes are possible (intended, slip one way, slip
        the other way). If two of them land in the same cell -- e.g. in a
        corner, where two different moves both bounce back -- their
        probabilities are COMBINED into a single entry, as the assignment
        requires.
        """
        slip_1, slip_2 = PERPENDICULAR[a]
        attempts = [(a, self.p_intended),
                    (slip_1, self.p_slip),
                    (slip_2, self.p_slip)]

        combined = {}                           # next_state -> total prob
        for direction, p in attempts:
            s_next = self._move(s, direction)
            combined[s_next] = combined.get(s_next, 0.0) + p

        return [(p, s_next, self.reward(s_next))
                for s_next, p in combined.items()]


# ---------------------------------------------------------------------------
# Section 4: Deterministic toy MDP for hand-checkable tests
# ---------------------------------------------------------------------------

def make_toy_chain_mdp(gamma=0.9):
    """
    A -> B -> T (terminal), one action "GO" in each nonterminal state.
      A --GO--> B with reward 0
      B --GO--> T with reward +1
    Expected synchronous VI from zeros (gamma = 0.9):
      V_1(A) = 0,   V_1(B) = 1
      V_2(A) = 0.9, V_2(B) = 1
    """
    GO = "GO"
    P = {
        "A": {GO: [(1.0, "B", 0.0)]},
        "B": {GO: [(1.0, "T", 1.0)]},
        "T": {},
    }
    return TabularMDP(states=["A", "B", "T"], terminals=["T"],
                      transitions=P, gamma=gamma)