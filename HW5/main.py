"""Alpha-beta pruning vs. Monte Carlo tree search on Connect Four.

Usage (from this directory):
    python main.py                       # full experiment: 30 games, 15 s per move
    python main.py --workers 2           # run two games at a time (faster, see README)
    python main.py --resume              # skip games whose result file already exists
    python main.py --summarize           # only recompute the summary from results/
    python main.py --selftest            # 4 quick games at 0.3 s per move (smoke test)

Requires the AIMA code (pip install -e . in the aima-python clone, or set AIMA_PATH to the
clone's directory) and numpy.
"""

import argparse
import os

from tournament import run_tournament, summarize, print_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--time-limit", type=float, default=15.0, help="seconds per move")
    ap.add_argument("--seed-base", type=int, default=5000, help="game i uses seed seed-base + i")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None, help="output directory (default: results)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if a.selftest:
        out = a.out or os.path.join(here, "results_selftest")
        run_tournament(n_games=4, time_limit=0.3, seed_base=a.seed_base, out_dir=out,
                       workers=a.workers, resume=False)
    else:
        out = a.out or os.path.join(here, "results")
        if not a.summarize:
            run_tournament(n_games=a.games, time_limit=a.time_limit, seed_base=a.seed_base,
                           out_dir=out, workers=a.workers, resume=a.resume)
    print_summary(summarize(out))


if __name__ == "__main__":
    main()