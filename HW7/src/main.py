"""
main.py
=======
Entry point for Homework 7, Exercise 5.

Runs value iteration (VI) and policy iteration (PI) on the Exercise 2
gridworld and prints everything the assignment asks to report:
  * a 3x4 grid of final state values for each algorithm
  * a 3x4 grid of greedy actions for each algorithm
    (wall shown as "#####", terminals shown as "+1 T" / "-1 T")
  * the maximum Bellman optimality residual of each returned value function
  * iteration counts (VI sweeps; PI improvement rounds and evaluation sweeps)
  * an agreement check between the two algorithms
  * per-state action values and the gap between the best and second-best
    action, used to explain any (near-)ties and the "risky route" discussion

Run from inside the src/ folder:   python main.py
"""

from mdp import GridWorldMDP, ACTIONS
from planning_utils import q_values
from value_iteration import value_iteration
from policy_iteration import policy_iteration

AGREEMENT_TOLERANCE = 1e-6     # required agreement between VI and PI values
CELL_WIDTH = 9                 # width of each printed grid cell

# Short arrow-style labels make the policy grid easy to read.
ARROWS = {"UP": "^ UP", "RIGHT": "> RIGHT", "DOWN": "v DOWN", "LEFT": "< LEFT"}


# ---------------------------------------------------------------------------
# Section 1: Grid formatting helpers
# ---------------------------------------------------------------------------

def special_cell_label(mdp, cell):
    """Label for wall/terminal cells, or None for an ordinary state."""
    if cell in mdp.walls:
        return "#####"                                  # wall: not a state
    if mdp.is_terminal(cell):
        reward = mdp.terminal_rewards[cell]
        return f"{reward:+.0f} T"                       # "+1 T" or "-1 T"
    return None


def format_grid(mdp, cell_text):
    """
    Build a printable 3x4 grid. Row 3 (the top row) is printed first so the
    picture matches the usual drawing of the gridworld; row 1 is the bottom.
    `cell_text(cell)` returns the string for an ordinary (nonterminal) state.
    """
    header = "       " + "".join(f"c{c}".center(CELL_WIDTH) for c in range(1, mdp.n_cols + 1))
    lines = [header]
    for r in range(mdp.n_rows, 0, -1):
        cells = []
        for c in range(1, mdp.n_cols + 1):
            label = special_cell_label(mdp, (r, c))
            text = label if label is not None else cell_text((r, c))
            cells.append(text.center(CELL_WIDTH))
        lines.append(f"  r{r} | " + "".join(cells))
    return "\n".join(lines)


def value_grid(mdp, V):
    """3x4 grid of state values (4 decimal places)."""
    return format_grid(mdp, lambda s: f"{V[s]:.4f}")


def policy_grid(mdp, policy):
    """3x4 grid of greedy actions."""
    return format_grid(mdp, lambda s: ARROWS[policy[s]])


# ---------------------------------------------------------------------------
# Section 2: Reporting one algorithm's results
# ---------------------------------------------------------------------------

def print_header(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def report_algorithm(mdp, name, result, counts_lines):
    """Print the value grid, policy grid, residual, and iteration counts."""
    print_header(name)
    print("\nFinal state values V(s):")
    print(value_grid(mdp, result["V"]))
    print("\nGreedy actions:")
    print(policy_grid(mdp, result["policy"]))
    print(f"\nMax Bellman optimality residual: {result['residual']:.3e}")
    for line in counts_lines:
        print(line)


# ---------------------------------------------------------------------------
# Section 3: Agreement check and action-value (tie) analysis
# ---------------------------------------------------------------------------

def compare_algorithms(mdp, vi, pi):
    """
    Check that VI and PI values agree to within 1e-6 at every nonterminal
    state, and list any states where their greedy actions differ.
    Returns (max_value_difference, list_of_differing_states).
    """
    max_diff = max(abs(vi["V"][s] - pi["V"][s]) for s in mdp.nonterminal_states)
    differing = [s for s in mdp.nonterminal_states if vi["policy"][s] != pi["policy"][s]]
    return max_diff, differing


def print_action_values(mdp, V):
    """
    For every nonterminal state, print Q(s, a) for all four actions and the
    margin between the best and second-best action. A margin far above
    1e-12 means the greedy choice is NOT a tie; a margin near 0 means the
    two actions are effectively tied and tie-breaking decides.
    """
    print("\nAction values Q(s, a) under the VI values (best marked with *):")
    print("  state   " + "".join(a.rjust(10) for a in ACTIONS) + "   margin")
    for s in mdp.nonterminal_states:
        qs = q_values(mdp, V, s)
        ordered = sorted(qs.values(), reverse=True)
        margin = ordered[0] - ordered[1]
        best = max(qs.values())
        row = ""
        for a in ACTIONS:
            mark = "*" if qs[a] >= best - 1e-12 else " "
            row += f"{qs[a]:9.4f}{mark}"
        print(f"  {str(s):8}{row}   {margin:.4f}")


# ---------------------------------------------------------------------------
# Section 4: Main program
# ---------------------------------------------------------------------------

def main():
    mdp = GridWorldMDP()   # gamma = 0.9, 0.8/0.1/0.1 slips, -0.04 step reward

    print("Gridworld: 3x4, wall at (2, 2), terminals (3, 4) = +1, (2, 4) = -1")
    print(f"States: {len(mdp.states)} (incl. 2 terminals), gamma = {mdp.gamma}, "
          f"step reward = {mdp.step_reward}")

    # --- Run both algorithms -------------------------------------------------
    vi = value_iteration(mdp)
    pi = policy_iteration(mdp)                         # warm-started evaluation
    pi_cold = policy_iteration(mdp, warm_start=False)  # for the work comparison

    # --- Per-algorithm reports ----------------------------------------------
    report_algorithm(mdp, "VALUE ITERATION (synchronous, tol = 1e-8)", vi, [
        f"Value-iteration sweeps: {vi['sweeps']}",
    ])

    report_algorithm(mdp, "POLICY ITERATION (iterative evaluation, tol = 1e-10)", pi, [
        f"Policy-improvement rounds: {pi['rounds']} (includes final no-change round)",
        f"Total policy-evaluation sweeps (warm start): {pi['eval_sweeps']}",
        f"  sweeps per round:  {pi['sweeps_per_round']}",
        f"  actions changed per round: {pi['changes_per_round']}",
        f"Total policy-evaluation sweeps (cold start, for comparison): "
        f"{pi_cold['eval_sweeps']} over {pi_cold['rounds']} rounds",
    ])

    # --- Agreement ------------------------------------------------------------
    print_header("AGREEMENT BETWEEN VALUE ITERATION AND POLICY ITERATION")
    max_diff, differing = compare_algorithms(mdp, vi, pi)
    status = "PASS" if max_diff < AGREEMENT_TOLERANCE else "FAIL"
    print(f"Max |V_VI(s) - V_PI(s)| over nonterminal states: {max_diff:.3e}  "
          f"(< {AGREEMENT_TOLERANCE:g}: {status})")
    if differing:
        print(f"Greedy actions differ at: {differing} (see margins below)")
    else:
        print("Greedy actions are identical at every nonterminal state.")

    # --- Action values / tie and risk analysis ------------------------------
    print_action_values(mdp, vi["V"])

    # --- Work comparison (rough operation counts) -----------------------------
    print_header("COMPUTATIONAL WORK (rough backup counts)")
    n = len(mdp.nonterminal_states)
    n_actions = len(ACTIONS)
    vi_backups = vi["sweeps"] * n * n_actions            # every action per state
    pi_eval_backups = pi["eval_sweeps"] * n              # one action per state
    pi_improve_backups = pi["rounds"] * n * n_actions    # greedy improvement
    print(f"VI : {vi['sweeps']} sweeps x {n} states x {n_actions} actions "
          f"= {vi_backups} action-value computations")
    print(f"PI : {pi['eval_sweeps']} eval sweeps x {n} states x 1 action "
          f"+ {pi['rounds']} improvements x {n} x {n_actions} "
          f"= {pi_eval_backups + pi_improve_backups} action-value computations")


if __name__ == "__main__":
    main()