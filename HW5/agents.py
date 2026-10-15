"""Time-limited Connect Four agents built on the AIMA (4th ed.) code in aima/games.py.

* ``AlphaBetaAgent`` adapts ``alpha_beta_cutoff_search``: the fixed depth ``d`` is replaced
  by iterative deepening that stops at a wall-clock deadline, with a heuristic evaluation
  function for non-terminal states and centre-first move ordering.
* ``MCTSAgent`` adapts ``monte_carlo_tree_search``: the fixed iteration count ``N`` is
  replaced by "run select/expand/simulate/backpropagate until the deadline", then return
  the move of the most-visited root child.

Both agents expose ``choose(state) -> (move, info)`` and always return a legal move.

Garbage collection: the MCTS tree is a huge graph of parent<->child reference cycles, and
Python's cyclic collector then spends 20-25% of a 15 s move scanning it, with single pauses
of up to ~1.4 s that push a move past its deadline.  tournament.play_game therefore turns the
collector off for the whole game and calls gc.collect() between moves, outside the timed
window.  If you call the agents yourself with the default collector on, expect such overruns.
"""

import math
import os
import random
import sys
from collections import Counter
from time import perf_counter

try:
    from aima.games import ConnectFour
    from aima.utils import MCT_Node, ucb
except ImportError:  # allow running from a clone that was not pip-installed
    for _p in (os.environ.get("AIMA_PATH"),
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "aima-python"),
               os.path.expanduser("~/aima-python")):
        if _p and os.path.isdir(os.path.join(_p, "aima")):
            sys.path.insert(0, os.path.abspath(_p))
            break
    from aima.games import ConnectFour
    from aima.utils import MCT_Node, ucb

ROWS, COLS, K = 6, 7, 4          # standard Connect Four: 6 rows x 7 columns, 4 in a row
WIN = 1_000_000                  # magnitude of a proven win/loss score
SAFETY_MARGIN = 0.05             # seconds held back so a move never exceeds the limit


def make_game(ordered=False):
    """Standard 6x7 Connect Four.

    NOTE: ``ConnectFour()`` in aima/games.py defaults to h=7, v=6, i.e. 7 rows by 6
    columns (x is the row index, y the column index).  We pass h=6, v=7 to get the
    traditional 7-column board.  ``ordered=True`` returns a subclass whose ``actions``
    lists the centre columns first (move ordering for alpha-beta).
    """
    return OrderedConnectFour(ROWS, COLS, K) if ordered else ConnectFour(ROWS, COLS, K)


class OrderedConnectFour(ConnectFour):
    """Same rules as ConnectFour; legal moves are returned centre-column first."""

    def __init__(self, h=ROWS, v=COLS, k=K):
        super().__init__(h, v, k)
        self.center = (v + 1) / 2

    def actions(self, state):
        board, h = state.board, self.h
        moves = [m for m in state.moves if m[0] == h or (m[0] + 1, m[1]) in board]
        moves.sort(key=lambda m: abs(m[1] - self.center))
        return moves


# ----------------------------------------------------------------------------------
# Heuristic evaluation for non-terminal states
# ----------------------------------------------------------------------------------

class Evaluator:
    """Heuristic value of a non-terminal Connect Four state for ``player``.

    Features (all computed as own minus opponent):
      1. Positional value: each piece is worth the number of 4-in-a-row windows that
         contain its square (3 in a corner ... 13 near the centre).  Centre squares
         take part in more potential fours, so they are worth more.
      2. Open twos: windows holding exactly two of the player's pieces and two empty
         squares (an alignment that can still grow into a four).
      3. Open threes: windows holding exactly three of the player's pieces and one empty
         square (one move from winning, forces a reply).  Weighted most heavily; the
         opponent's threes count slightly more (W3_OPP > W3) so the agent is not careless
         about defence.
    A window containing pieces of both colours can never become a four and scores 0.
    """

    W_POS, W2, W3, W3_OPP = 1, 4, 25, 28

    def __init__(self, rows=ROWS, cols=COLS, k=K):
        windows = []
        for x in range(1, rows + 1):
            for y in range(1, cols + 1):
                for dx, dy in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = tuple((x + i * dx, y + i * dy) for i in range(k))
                    if all(1 <= cx <= rows and 1 <= cy <= cols for cx, cy in cells):
                        windows.append(cells)
        self.windows = windows
        self.cell_weight = Counter(c for w in windows for c in w)

    def __call__(self, state, player):
        board = state.board
        get = board.get
        cw = self.cell_weight
        pos = 0
        for cell, who in board.items():
            pos += cw[cell] if who == player else -cw[cell]
        own2 = opp2 = own3 = opp3 = 0
        for w in self.windows:
            p = o = 0
            for c in w:
                v = get(c)
                if v is None:
                    continue
                if v == player:
                    p += 1
                else:
                    o += 1
            if p and o:
                continue
            if p == 3:
                own3 += 1
            elif o == 3:
                opp3 += 1
            elif p == 2:
                own2 += 1
            elif o == 2:
                opp2 += 1
        return (self.W_POS * pos + self.W2 * (own2 - opp2)
                + self.W3 * own3 - self.W3_OPP * opp3)


# ----------------------------------------------------------------------------------
# Alpha-beta with iterative deepening and a wall-clock deadline
# ----------------------------------------------------------------------------------

class TimeUp(Exception):
    """Raised inside the search when the deadline passes."""


class AlphaBetaAgent:
    name = "alpha-beta"

    def __init__(self, time_limit=15.0, evaluator=None, margin=SAFETY_MARGIN, max_depth=None):
        self.game = make_game(ordered=True)
        self.time_limit = time_limit
        self.margin = margin
        self.max_depth = max_depth
        self.evaluate = evaluator or Evaluator()

    def choose(self, state):
        t0 = perf_counter()
        deadline = t0 + self.time_limit - self.margin
        game = self.game
        player = game.to_move(state)
        actions = game.actions(state)
        best = actions[0]                       # legal fallback: centre-most column
        info = {"depth": 0, "nodes": 0, "score": None, "forced": len(actions) == 1,
                "aborted_depth": None}
        if len(actions) == 1:
            return best, info

        limit = len(state.moves) if self.max_depth is None else min(self.max_depth, len(state.moves))
        for D in range(1, limit + 1):
            try:
                move, score, nodes = self._root_search(state, player, D, actions, deadline)
            except TimeUp as stop:
                info["nodes"] += stop.args[0] if stop.args else 0
                info["aborted_depth"] = D
                break
            info["nodes"] += nodes
            best, info["depth"], info["score"] = move, D, score
            actions = [move] + [a for a in actions if a != move]   # best move first next time
            if abs(score) >= WIN - 1000:         # forced win/loss inside the horizon
                break
        return best, info

    def _root_search(self, state, player, D, actions, deadline):
        """One fixed-depth pass; adapted from aima.games.alpha_beta_cutoff_search."""
        game, evaluate = self.game, self.evaluate
        nodes = 0

        def leaf(s, depth):
            if game.terminal_test(s):
                u = game.utility(s, player)       # +1 / -1 / 0
                return u * (WIN - depth)          # prefer quicker wins, slower losses
            return evaluate(s, player)

        def max_value(s, alpha, beta, depth):
            nonlocal nodes
            nodes += 1
            if perf_counter() >= deadline:
                raise TimeUp(nodes)
            if depth >= D or game.terminal_test(s):
                return leaf(s, depth)
            v = -math.inf
            for a in game.actions(s):
                v = max(v, min_value(game.result(s, a), alpha, beta, depth + 1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(s, alpha, beta, depth):
            nonlocal nodes
            nodes += 1
            if perf_counter() >= deadline:
                raise TimeUp(nodes)
            if depth >= D or game.terminal_test(s):
                return leaf(s, depth)
            v = math.inf
            for a in game.actions(s):
                v = min(v, max_value(game.result(s, a), alpha, beta, depth + 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        best_score, beta, best_action = -math.inf, math.inf, None
        for a in actions:
            v = min_value(game.result(state, a), best_score, beta, 1)
            if v > best_score:
                best_score, best_action = v, a
        return best_action, best_score, nodes


# ----------------------------------------------------------------------------------
# Monte Carlo tree search that runs until a wall-clock deadline
# ----------------------------------------------------------------------------------

class MCTSAgent:
    name = "mcts"

    def __init__(self, time_limit=15.0, seed=None, C=1.4, margin=SAFETY_MARGIN):
        self.game = make_game(ordered=False)
        self.time_limit = time_limit
        self.margin = margin
        self.C = C
        self.rng = random.Random(seed)          # all playout randomness comes from this seed

    def choose(self, state):
        t0 = perf_counter()
        deadline = t0 + self.time_limit - self.margin
        game, rng, C = self.game, self.rng, self.C
        actions = game.actions(state)
        info = {"sims": 0, "tree_depth": 0, "forced": len(actions) == 1,
                "best_visits": None, "best_winrate": None}
        if len(actions) == 1:
            return actions[0], info

        def select(n):
            """Walk down the tree by UCB1 to a leaf (adapted from aima's recursive version)."""
            d = 0
            while n.children:
                n = max(n.children.keys(), key=lambda c: ucb(c, C))
                d += 1
            return n, d

        def expand(n):
            """Add all children of a leaf, then pick one of them by UCB (as in AIMA)."""
            if not n.children and not game.terminal_test(n.state):
                n.children = {MCT_Node(state=game.result(n.state, a), parent=n): a
                              for a in game.actions(n.state)}
            return select(n)

        def simulate(s):
            """Uniformly random playout; value is from the view of the player who moved into s."""
            player = game.to_move(s)
            while not game.terminal_test(s):
                s = game.result(s, rng.choice(game.actions(s)))
            return -game.utility(s, player)

        def backprop(n, utility):
            while n is not None:
                if utility > 0:                  # AIMA scoring: win = 1, draw = loss = 0
                    n.U += utility
                n.N += 1
                n = n.parent
                utility = -utility

        root = MCT_Node(state=state)
        sims = 0
        while sims == 0 or perf_counter() < deadline:
            leaf, d1 = select(root)
            child, d2 = expand(leaf)
            backprop(child, simulate(child.state))
            sims += 1
            if d1 + d2 > info["tree_depth"]:
                info["tree_depth"] = d1 + d2

        best_child = max(root.children, key=lambda c: c.N)   # most-visited child
        info.update(sims=sims, best_visits=best_child.N,
                    best_winrate=best_child.U / best_child.N if best_child.N else None)
        return root.children[best_child], info