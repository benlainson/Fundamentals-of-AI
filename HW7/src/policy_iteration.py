"""
policy_iteration.py
===================
Policy iteration (Exercise 5).

Algorithm:
    pi_0(s) = UP for every nonterminal s
    repeat:
        1. Policy evaluation: compute V^pi for the FIXED current policy by
           iterating the expectation backup
               V(s) <- sum_{s'} P(s'|s,pi(s)) [ R + gamma * V(s') ]
           until the max value change is below 1e-10.
        2. Policy improvement: pi'(s) = argmax_a Q^pi(s, a), with ties
           broken UP, RIGHT, DOWN, LEFT.
        3. If pi' == pi, stop (the policy is stable). Otherwise pi <- pi'.

Policy evaluation here is iterative (not a linear-system solve), so we can
also report the total number of evaluation sweeps across all rounds.
"""

from planning_utils import (
    zero_values,
    q_value,
    greedy_policy,
    bellman_residual,
)

EVAL_TOLERANCE = 1e-10        # evaluation stopping threshold (assignment)
DEFAULT_MAX_EVAL_SWEEPS = 100_000
DEFAULT_MAX_ROUNDS = 1_000


# ---------------------------------------------------------------------------
# Section 1: Initial policy
# ---------------------------------------------------------------------------

def initial_policy(mdp, action="UP"):
    """Deterministic starting policy: UP in every nonterminal state."""
    return {s: action for s in mdp.nonterminal_states}


# ---------------------------------------------------------------------------
# Section 2: Iterative policy evaluation (expectation backups)
# ---------------------------------------------------------------------------

def policy_evaluation(mdp, policy, V_init=None, tol=EVAL_TOLERANCE,
                      max_sweeps=DEFAULT_MAX_EVAL_SWEEPS):
    """
    Evaluate a FIXED deterministic policy.

    Each sweep is synchronous (new values go into a separate dict), just
    like value iteration, but uses the EXPECTATION backup Q(s, pi(s))
    instead of the max over actions. Repeats until the max change < tol.

    V_init lets policy iteration "warm start" from the previous round's
    values. Any starting point converges to the same V^pi (the expectation
    backup is a gamma-contraction); warm starting only reduces the sweeps.

    Returns (V, sweeps).
    """
    V = dict(V_init) if V_init is not None else zero_values(mdp)
    for sweep in range(1, max_sweeps + 1):
        V_new = {}
        max_change = 0.0
        for s in mdp.states:
            if mdp.is_terminal(s):
                V_new[s] = 0.0                              # terminals fixed at 0
            else:
                V_new[s] = q_value(mdp, V, s, policy[s])    # expectation backup
            max_change = max(max_change, abs(V_new[s] - V[s]))
        V = V_new
        if max_change < tol:
            return V, sweep
    raise RuntimeError(f"Policy evaluation did not converge in {max_sweeps} sweeps")


# ---------------------------------------------------------------------------
# Section 3: Policy iteration (evaluate -> improve -> check stability)
# ---------------------------------------------------------------------------

def policy_iteration(mdp, tol=EVAL_TOLERANCE, warm_start=True,
                     max_rounds=DEFAULT_MAX_ROUNDS):
    """
    Run policy iteration starting from the all-UP policy.

    Returns a dict with:
        "V"               : value of the final (stable) policy
        "policy"          : final policy {nonterminal state: action}
        "rounds"          : number of policy-improvement rounds, including
                            the last round, which found no change
        "eval_sweeps"     : total policy-evaluation sweeps over all rounds
        "sweeps_per_round": evaluation sweeps used in each round
        "changes_per_round": number of states whose action changed each round
        "residual"        : max Bellman optimality residual of the final V
    """
    policy = initial_policy(mdp)
    V = zero_values(mdp)
    total_eval_sweeps = 0
    sweeps_per_round = []
    changes_per_round = []

    for rounds in range(1, max_rounds + 1):
        # --- 1. Policy evaluation for the current fixed policy -------------
        V, sweeps = policy_evaluation(mdp, policy,
                                      V_init=V if warm_start else None,
                                      tol=tol)
        total_eval_sweeps += sweeps
        sweeps_per_round.append(sweeps)

        # --- 2. Greedy policy improvement (shared tie-breaking rule) -------
        new_policy = greedy_policy(mdp, V)

        # --- 3. Stability check --------------------------------------------
        changed = sum(1 for s in policy if new_policy[s] != policy[s])
        changes_per_round.append(changed)
        if changed == 0:
            break
        policy = new_policy
    else:
        raise RuntimeError(f"Policy iteration did not stabilize in {max_rounds} rounds")

    return {
        "V": V,
        "policy": policy,
        "rounds": rounds,
        "eval_sweeps": total_eval_sweeps,
        "sweeps_per_round": sweeps_per_round,
        "changes_per_round": changes_per_round,
        "residual": bellman_residual(mdp, V),
    }