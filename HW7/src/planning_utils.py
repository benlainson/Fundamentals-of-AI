"""
planning_utils.py
=================
Shared Bellman machinery used by BOTH value iteration and policy iteration.

Putting these functions in one place guarantees that the two algorithms
compute action values, break ties, and measure convergence in exactly the
same way. Any difference in their results then comes from the algorithms
themselves, not from slightly different helper code.

Contents:
  * zero_values               - V(s) = 0 for every state (initialization)
  * q_value / q_values        - one-step lookahead action values Q(s, a)
  * greedy_action             - argmax over actions with UP/RIGHT/DOWN/LEFT ties
  * greedy_policy             - greedy action for every nonterminal state
  * bellman_optimality_backup - max_a Q(s, a)  (the Bellman update)
  * bellman_residual          - max_s |(T V)(s) - V(s)|
"""

# Two action values within this distance are treated as equal, and the
# tie goes to whichever action comes first in the MDP's action order
# (UP, RIGHT, DOWN, LEFT for the gridworld).
TIE_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# Section 1: Initialization
# ---------------------------------------------------------------------------

def zero_values(mdp):
    """Return a fresh value dictionary with V(s) = 0 for every state.

    Terminal states are included and stay at 0 forever, which implements
    the "terminal-state values are zero" convention.
    """
    return {s: 0.0 for s in mdp.states}


# ---------------------------------------------------------------------------
# Section 2: Action values (one-step lookahead)
# ---------------------------------------------------------------------------

def q_value(mdp, V, s, a):
    """
    Q(s, a) = sum over s' of  P(s' | s, a) * [ R(s, a, s') + gamma * V(s') ]

    V(s') is 0 whenever s' is terminal, because terminal values are never
    changed from their initial 0. So entering (3, 4) contributes exactly
    p * (+1 + gamma * 0) = p * 1: the episode ends and nothing is added
    after the terminal reward.
    """
    total = 0.0
    for p, s_next, r in mdp.transitions(s, a):
        total += p * (r + mdp.gamma * V[s_next])
    return total


def q_values(mdp, V, s):
    """All action values for state s, as an ordered {action: Q} dict.

    The dict keeps the MDP's action order, which is also the tie-break order.
    """
    return {a: q_value(mdp, V, s, a) for a in mdp.actions(s)}


# ---------------------------------------------------------------------------
# Section 3: Greedy action selection with deterministic tie-breaking
# ---------------------------------------------------------------------------

def greedy_action(mdp, V, s, tol=TIE_TOLERANCE):
    """
    Return the action maximizing Q(s, a).

    Ties are broken by action order: first compute the best value, then
    return the FIRST action (in UP, RIGHT, DOWN, LEFT order) whose value is
    within `tol` of that best value. Comparing every action against the
    true maximum (instead of against a running "best so far") means the
    result never depends on the order in which near-equal values appear.
    """
    qs = q_values(mdp, V, s)
    best = max(qs.values())
    for a, q in qs.items():          # iterates in tie-break order
        if q >= best - tol:
            return a
    # Unreachable: the maximizing action always satisfies the test above.
    raise RuntimeError("greedy_action found no action")


def greedy_policy(mdp, V, tol=TIE_TOLERANCE):
    """Greedy (deterministic) policy for every nonterminal state."""
    return {s: greedy_action(mdp, V, s, tol) for s in mdp.nonterminal_states}


# ---------------------------------------------------------------------------
# Section 4: Bellman optimality backup and residual
# ---------------------------------------------------------------------------

def bellman_optimality_backup(mdp, V, s):
    """
    (T V)(s) = max_a Q(s, a)   for nonterminal s
             = 0               for terminal s
    This is the Bellman update that value iteration applies at every state.
    """
    if mdp.is_terminal(s):
        return 0.0
    return max(q_values(mdp, V, s).values())


def bellman_residual(mdp, V):
    """
    Maximum Bellman optimality residual (maximum Bellman update error):

        max over s of | (T V)(s) - V(s) |

    It measures how far V is from satisfying the Bellman optimality
    equation V = T V. The true optimal V* has residual exactly 0. For a
    discounted MDP, a small residual eps guarantees
        || V - V* ||_inf  <=  eps / (1 - gamma),
    so it is a meaningful accuracy check for EITHER algorithm's output.
    """
    return max(abs(bellman_optimality_backup(mdp, V, s) - V[s])
               for s in mdp.states)