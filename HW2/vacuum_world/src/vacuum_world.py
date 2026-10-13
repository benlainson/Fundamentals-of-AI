#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
from typing import Dict, List

from agent import SimpleReflexVacuumAgent
from environment import VacuumEnvironment


def location_name(index: int, size: int) -> str:
    """Render a location index as a letter (A, B, C, ...) for size<=26,
    matching the textbook's A/B naming for the size=2 case."""
    if size <= 26:
        return chr(ord("A") + index)
    return str(index)


def run_all_configurations(
    size: int, max_steps: int, stop_when_clean: bool, verbose: bool
) -> List[Dict]:
    """Run the agent over every (agent start location) x (initial dirt
    configuration) combination -- exactly what the exercise asks for."""
    results = []
    for agent_loc in range(size):
        for dirt_bits in itertools.product([True, False], repeat=size):
            env = VacuumEnvironment(
                size=size, dirt=list(dirt_bits), agent_location=agent_loc
            )
            agent = SimpleReflexVacuumAgent(size=size)
            score = env.run(agent, max_steps=max_steps, stop_when_clean=stop_when_clean)
            results.append(
                {
                    "agent_start": location_name(agent_loc, size),
                    "initial_dirt": "".join("D" if d else "C" for d in dirt_bits),
                    "steps": env.time_step,
                    "performance": score,
                }
            )
            if verbose:
                print(
                    f"start={location_name(agent_loc, size)} "
                    f"dirt={''.join('D' if d else 'C' for d in dirt_bits)} "
                    f"-> steps={env.time_step} performance={score}"
                )
                for t, (loc, dirt_state, action) in enumerate(env.history, start=1):
                    dirt_str = "".join("D" if d else "C" for d in dirt_state)
                    print(
                        f"    t={t}: action={action} -> loc={location_name(loc, size)} dirt={dirt_str}"
                    )
    return results


def print_results_table(results: List[Dict]) -> None:
    header = f"{'Agent Start':<12}{'Initial Dirt':<15}{'Steps':<8}{'Performance':<12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['agent_start']:<12}{r['initial_dirt']:<15}{r['steps']:<8}{r['performance']:<12}"
        )
    print("-" * len(header))
    total = sum(r["performance"] for r in results)
    avg = total / len(results) if results else 0.0
    print(f"Configurations run: {len(results)}")
    print(f"Total performance:  {total}")
    print(f"Average performance:{avg:>8.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AIMA vacuum-cleaner world environment simulator and simple reflex agent."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2,
        help="Number of locations in the linear vacuum world (default: 2, "
        "the textbook's A/B world).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Number of time steps to run each configuration for "
        "(default: 1000, per AIMA's performance-measure convention, p. 38).",
    )
    parser.add_argument(
        "--stop-when-clean",
        action="store_true",
        help="Stop a run early once the whole world is clean, instead of "
        "running the full --max-steps (off by default).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a step-by-step trace for each configuration run.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Run only ONE configuration: the agent's starting location "
        "index (0-based). Must be used together with --dirt.",
    )
    parser.add_argument(
        "--dirt",
        type=str,
        default=None,
        help="Run only ONE configuration: initial dirt string of length "
        "--size, e.g. 'DC' = location 0 dirty, location 1 clean. Must be "
        "used together with --start.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.size < 1:
        parser.error("--size must be >= 1")

    single_run_requested = args.start is not None or args.dirt is not None
    if single_run_requested:
        if args.start is None or args.dirt is None:
            parser.error("--start and --dirt must be given together")
        if not (0 <= args.start < args.size):
            parser.error(f"--start must be in [0, {args.size})")
        if len(args.dirt) != args.size:
            parser.error(f"--dirt must have length {args.size}")
        if any(c.upper() not in ("D", "C") for c in args.dirt):
            parser.error("--dirt must contain only 'D' (dirty) and 'C' (clean)")

        dirt = [c.upper() == "D" for c in args.dirt]
        env = VacuumEnvironment(size=args.size, dirt=dirt, agent_location=args.start)
        agent = SimpleReflexVacuumAgent(size=args.size)
        score = env.run(
            agent, max_steps=args.max_steps, stop_when_clean=args.stop_when_clean
        )

        print(
            f"start={location_name(args.start, args.size)} dirt={args.dirt.upper()} "
            f"-> steps={env.time_step} performance={score}"
        )
        if args.verbose:
            for t, (loc, dirt_state, action) in enumerate(env.history, start=1):
                dirt_str = "".join("D" if d else "C" for d in dirt_state)
                print(
                    f"  t={t}: action={action} -> loc={location_name(loc, args.size)} dirt={dirt_str}"
                )
        return

    results = run_all_configurations(
        size=args.size,
        max_steps=args.max_steps,
        stop_when_clean=args.stop_when_clean,
        verbose=args.verbose,
    )
    print_results_table(results)


if __name__ == "__main__":
    main()
