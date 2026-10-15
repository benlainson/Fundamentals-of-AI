"""Play alpha-beta vs. MCTS Connect Four games and summarise the results."""

import gc
import json
import math
import os
import random
import statistics as st
import time
from multiprocessing import Pool
from time import perf_counter

from agents import AlphaBetaAgent, MCTSAgent, make_game, ROWS, COLS

AB, MC = "alpha-beta", "mcts"


def board_text(game, state):
    """6 rows x 7 columns, row 1 at the top (AIMA's x is the row, y the column)."""
    rows = []
    for x in range(1, game.h + 1):
        rows.append(" ".join(state.board.get((x, y), ".") for y in range(1, game.v + 1)))
    return "\n".join(rows)


def play_game(task):
    """Play one game. ``task`` = dict(index, ab_first, seed, time_limit, out_dir)."""
    idx, ab_first, seed, limit = task["index"], task["ab_first"], task["seed"], task["time_limit"]
    random.seed(seed)
    gc.disable()   # see agents.py docstring; collected explicitly between moves
    game = make_game()
    agents = {AB: AlphaBetaAgent(time_limit=limit), MC: MCTSAgent(time_limit=limit, seed=seed)}
    side = {"X": AB if ab_first else MC, "O": MC if ab_first else AB}   # X moves first
    state, moves, illegal = game.initial, [], 0
    t_game = perf_counter()
    while not game.terminal_test(state):
        who = side[state.to_move]
        t0 = perf_counter()
        move, info = agents[who].choose(state)
        elapsed = perf_counter() - t0
        gc.collect()                                # free the search tree outside the timed window
        legal = game.actions(state)
        if move not in legal:                       # should never happen; keep the game legal
            illegal += 1
            move = legal[0]
        rec = {"ply": len(moves) + 1, "player": who, "mark": state.to_move,
               "col": move[1], "time": elapsed}
        rec.update(info)
        moves.append(rec)
        state = game.result(state, move)
    u = game.utility(state, "X")
    winner = "draw" if u == 0 else (side["X"] if u > 0 else side["O"])
    gc.enable()
    out = {"game": idx, "seed": seed, "first": AB if ab_first else MC, "winner": winner,
           "plies": len(moves), "illegal_moves": illegal, "time_limit": limit,
           "wall_seconds": perf_counter() - t_game, "final_board": board_text(game, state),
           "moves": moves}
    path = os.path.join(task["out_dir"], f"game_{idx:02d}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(path + ".tmp", path)
    return idx, winner, ("AB" if ab_first else "MCTS"), len(moves), out["wall_seconds"]


def run_tournament(n_games=30, time_limit=15.0, seed_base=5000, out_dir="results",
                   workers=1, resume=False):
    """Alpha-beta moves first in even-numbered games (0,2,...), MCTS in odd-numbered games."""
    games_dir = os.path.join(out_dir, "games")
    os.makedirs(games_dir, exist_ok=True)
    tasks = []
    for i in range(n_games):
        if resume and os.path.exists(os.path.join(games_dir, f"game_{i:02d}.json")):
            continue
        tasks.append({"index": i, "ab_first": i % 2 == 0, "seed": seed_base + i,
                      "time_limit": time_limit, "out_dir": games_dir})
    print(f"running {len(tasks)} games, {time_limit}s per move, {workers} worker(s)", flush=True)
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as pool:
            for r in pool.imap_unordered(play_game, tasks):
                print("finished game %02d: winner=%s first=%s plies=%d (%.0fs)" % (*r,), flush=True)
    else:
        for t in tasks:
            r = play_game(t)
            print("finished game %02d: winner=%s first=%s plies=%d (%.0fs)" % (*r,), flush=True)
    print("total wall time %.1f min" % ((time.time() - t0) / 60), flush=True)


# ----------------------------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------------------------

def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial test p-value."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(x for x in pmf if x <= pmf[k] + 1e-12))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_games(out_dir):
    gdir = os.path.join(out_dir, "games")
    files = sorted(f for f in os.listdir(gdir) if f.endswith(".json"))
    return [json.load(open(os.path.join(gdir, f))) for f in files]


def _stats(xs):
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "min": float("nan"),
                "max": float("nan"), "sd": float("nan")}
    return {"n": len(xs), "mean": st.mean(xs), "median": st.median(xs), "min": min(xs),
            "max": max(xs), "sd": st.pstdev(xs) if len(xs) > 1 else 0.0}


def summarize(out_dir="results"):
    games = load_games(out_dir)
    n = len(games)
    wins = {AB: 0, MC: 0, "draw": 0}
    for g in games:
        wins[g["winner"]] += 1
    by_first = {}
    for first in (AB, MC):
        sub = [g for g in games if g["first"] == first]
        by_first[first] = {"games": len(sub),
                           AB: sum(g["winner"] == AB for g in sub),
                           MC: sum(g["winner"] == MC for g in sub),
                           "draw": sum(g["winner"] == "draw" for g in sub)}
    decisive = wins[AB] + wins[MC]
    pval = binom_two_sided(wins[AB], decisive)
    lo, hi = wilson(wins[AB], decisive)

    moves = [m for g in games for m in g["moves"]]
    effort = {}
    for who in (AB, MC):
        ms = [m for m in moves if m["player"] == who]
        real = [m for m in ms if not m["forced"]]
        effort[who] = {"moves": len(ms), "forced_moves": len(ms) - len(real),
                       "time": _stats([m["time"] for m in real])}
        if who == AB:
            effort[who]["depth"] = _stats([m["depth"] for m in real])
            effort[who]["nodes"] = _stats([m["nodes"] for m in real])
            effort[who]["nodes_per_s"] = (sum(m["nodes"] for m in real)
                                          / max(1e-9, sum(m["time"] for m in real)))
            effort[who]["depth_hist"] = {d: sum(m["depth"] == d for m in real)
                                         for d in sorted({m["depth"] for m in real})}
        else:
            effort[who]["sims"] = _stats([m["sims"] for m in real])
            effort[who]["sims_per_s"] = (sum(m["sims"] for m in real)
                                         / max(1e-9, sum(m["time"] for m in real)))
            effort[who]["tree_depth"] = _stats([m["tree_depth"] for m in real])

    # effort by game phase (early = plies 1-10, mid = 11-24, late = 25+)
    phases = {"plies 1-10": (1, 10), "plies 11-24": (11, 24), "plies 25+": (25, 99)}
    phase_effort = {}
    for label, (a, b) in phases.items():
        ab_ms = [m for m in moves if m["player"] == AB and not m["forced"] and a <= m["ply"] <= b]
        mc_ms = [m for m in moves if m["player"] == MC and not m["forced"] and a <= m["ply"] <= b]
        phase_effort[label] = {
            "ab_moves": len(ab_ms),
            "ab_depth": st.mean([m["depth"] for m in ab_ms]) if ab_ms else float("nan"),
            "mcts_moves": len(mc_ms),
            "mcts_sims": st.mean([m["sims"] for m in mc_ms]) if mc_ms else float("nan")}

    summary = {"games": n, "wins": wins, "by_first": by_first,
               "decisive": decisive, "binomial_p": pval, "ab_win_share_ci95": [lo, hi],
               "effort": effort, "phase_effort": phase_effort,
               "plies": _stats([g["plies"] for g in games]),
               "max_move_time": max((m["time"] for m in moves), default=float("nan")),
               "moves_over_limit": sum(m["time"] > g["time_limit"] for g in games for m in g["moves"]),
               "illegal_moves": sum(g["illegal_moves"] for g in games),
               "time_limit": games[0]["time_limit"] if games else None}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(out_dir, "games_table.csv"), "w") as f:
        f.write("game,seed,first_player,winner,plies,ab_mean_depth,mcts_mean_sims,wall_seconds\n")
        for g in games:
            a = [m["depth"] for m in g["moves"] if m["player"] == AB and not m["forced"]]
            s = [m["sims"] for m in g["moves"] if m["player"] == MC and not m["forced"]]
            f.write("%d,%d,%s,%s,%d,%.2f,%.0f,%.0f\n" % (
                g["game"], g["seed"], g["first"], g["winner"], g["plies"],
                st.mean(a) if a else float("nan"), st.mean(s) if s else float("nan"),
                g["wall_seconds"]))
    return summary


def print_summary(s):
    w, bf = s["wins"], s["by_first"]
    print(f"\n=== {s['games']} games, {s['time_limit']}s per move ===")
    print(f"alpha-beta wins: {w[AB]}   MCTS wins: {w[MC]}   draws: {w['draw']}")
    print("\nfirst mover   games  AB wins  MCTS wins  draws")
    for first in (AB, MC):
        r = bf[first]
        print(f"{first:<12} {r['games']:>6} {r[AB]:>8} {r[MC]:>10} {r['draw']:>6}")
    print(f"\nexact binomial test on {s['decisive']} decisive games: p = {s['binomial_p']:.4f}; "
          f"95% Wilson CI for AB share of decisive games: "
          f"[{s['ab_win_share_ci95'][0]:.2f}, {s['ab_win_share_ci95'][1]:.2f}]")
    for who, e in s["effort"].items():
        t = e["time"]
        print(f"\n{who}: {e['moves']} moves ({e['forced_moves']} forced, excluded below)")
        print(f"  time/move: mean {t['mean']:.2f}s  median {t['median']:.2f}s  max {t['max']:.2f}s")
        if who == AB:
            print(f"  deepest completed depth: mean {e['depth']['mean']:.2f} "
                  f"(min {e['depth']['min']:.0f}, max {e['depth']['max']:.0f}); "
                  f"nodes/move mean {e['nodes']['mean']:,.0f}; {e['nodes_per_s']:,.0f} nodes/s")
        else:
            print(f"  simulations/move: mean {e['sims']['mean']:,.0f} (min {e['sims']['min']:,.0f}, "
                  f"max {e['sims']['max']:,.0f}); {e['sims_per_s']:,.0f} sims/s; "
                  f"mean deepest tree node {e['tree_depth']['mean']:.1f} plies")
    print(f"\nmoves over time limit: {s['moves_over_limit']}, illegal moves: {s['illegal_moves']}, "
          f"slowest move: {s['max_move_time']:.2f}s")