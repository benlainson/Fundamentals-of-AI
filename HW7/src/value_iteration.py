"""
value_iteration.py
==================
Synchronous value iteration (Exercise 5).

Algorithm:
    V_0(s) = 0 for all s
    repeat for k = 0, 1, 2, ...
        V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V_k(s') ]
                     (V_{k+1}(s) = 0 for terminal s)
    until  max_s |V_{k+1}(s) - V_k(s)| < tol   (tol = 1e-8)
    then extract the greedy policy from the final values.

"Synchronous" means every V_{k+1}(s) is computed from the OLD array V_k.
The new values are written into a SEPARATE dictionary, and only after the
whole sweep finishes does that dictionary replace the old one.
"""

from planning_utils import (
    zero_values,
    bellman_optimality_backup,
    greedy_policy,
    bellman_residual,
)

VI_TOLERANCE = 1e-8          # stopping threshold required by the assignment
DEFAULT_MAX_SWEEPS = 100_000  # safety cap so a bug can never loop forever


# ---------------------------------------------------------------------------
# Section 1: One synchronous sweep
# ---------------------------------------------------------------------------

def value_iteration_sweep(mdp, V_old):
    """
    Perform ONE full synchronous sweep and return (V_new, max_change).

    V_old is only READ, never modified. Every backup uses V_k values,
    including states that were already updated earlier in this sweep.
    That is the difference from in-place (Gauss-Seidel) updates.
    """
    V_new = {}
    max_change = 0.0
    for s in mdp.states:
        V_new[s] = bellman_optimality_backup(mdp, V_old, s)   # 0 if terminal
        max_change = max(max_change, abs(V_new[s] - V_old[s]))
    return V_new, max_change


# ---------------------------------------------------------------------------
# Section 2: The full algorithm
# ---------------------------------------------------------------------------

def value_iteration(mdp, tol=VI_TOLERANCE, max_sweeps=DEFAULT_MAX_SWEEPS,
                    record_history=False):
    """
    Run synchronous value iteration from V_0 = 0.

    Returns a dict with:
        "V"        : final value function {state: value}
        "policy"   : greedy policy {nonterminal state: action}
        "sweeps"   : number of full sweeps performed (including the final
                     sweep whose change fell below tol)
        "residual" : max Bellman optimality residual of the final V
        "deltas"   : max change recorded after each sweep (convergence trace)
        "history"  : [V_0, V_1, ..., V_K] if record_history=True, else None
    """
    V = zero_values(mdp)                       # V_0 = 0 everywhere
    history = [dict(V)] if record_history else None
    deltas = []
    sweeps = 0

    while sweeps < max_sweeps:
        V_next, delta = value_iteration_sweep(mdp, V)   # V_{k+1} from V_k
        sweeps += 1
        deltas.append(delta)
        V = V_next                                      # swap only after the sweep
        if record_history:
            history.append(dict(V))
        if delta < tol:                                 # stopping rule
            break
    else:
        raise RuntimeError(f"Value iteration did not converge in {max_sweeps} sweeps")

    return {
        "V": V,
        "policy": greedy_policy(mdp, V),       # greedy extraction, tie-broken
        "sweeps": sweeps,
        "residual": bellman_residual(mdp, V),
        "deltas": deltas,
        "history": history,
    }