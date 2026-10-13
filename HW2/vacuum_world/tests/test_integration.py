import itertools
import subprocess
import sys
import os

from agent import SimpleReflexVacuumAgent
from environment import VacuumEnvironment
from vacuum_world import run_all_configurations, location_name

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
VACUUM_WORLD_PY = os.path.join(SRC_DIR, "vacuum_world.py")


def test_location_name_size_2_uses_A_B():
    assert location_name(0, 2) == "A"
    assert location_name(1, 2) == "B"


def test_run_all_configurations_covers_every_combination_size_2():
    results = run_all_configurations(
        size=2, max_steps=10, stop_when_clean=False, verbose=False
    )
    # 2 starting locations x 2^2 dirt configurations = 8
    assert len(results) == 8
    seen = {(r["agent_start"], r["initial_dirt"]) for r in results}
    expected_dirt_strings = {
        "".join(bits) for bits in itertools.product("DC", repeat=2)
    }
    expected = {(start, dirt) for start in ("A", "B") for dirt in expected_dirt_strings}
    assert seen == expected


def test_run_all_configurations_covers_every_combination_size_3():
    results = run_all_configurations(
        size=3, max_steps=5, stop_when_clean=False, verbose=False
    )
    # 3 starting locations x 2^3 dirt configurations = 24
    assert len(results) == 24


def test_every_configuration_produces_nonnegative_performance():
    results = run_all_configurations(
        size=2, max_steps=100, stop_when_clean=False, verbose=False
    )
    for r in results:
        assert r["performance"] >= 0
        assert r["steps"] == 100


def test_all_clean_start_yields_max_possible_score():
    """If the world starts fully clean, every one of max_steps steps
    scores 'size' points (all squares clean), so total == size * max_steps."""
    env = VacuumEnvironment(size=2, dirt=[False, False], agent_location=0)
    agent = SimpleReflexVacuumAgent(size=2)
    score = env.run(agent, max_steps=50)
    assert score == 2 * 50


def test_cli_runs_default_sweep_without_error():
    result = subprocess.run(
        [sys.executable, VACUUM_WORLD_PY, "--max-steps", "10"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Configurations run: 8" in result.stdout
    assert "Total performance:" in result.stdout


def test_cli_single_configuration_mode():
    result = subprocess.run(
        [
            sys.executable,
            VACUUM_WORLD_PY,
            "--start",
            "0",
            "--dirt",
            "DD",
            "--max-steps",
            "5",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "performance=8" in result.stdout


def test_cli_rejects_mismatched_dirt_length():
    result = subprocess.run(
        [sys.executable, VACUUM_WORLD_PY, "--start", "0", "--dirt", "DDD"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "length" in result.stderr


def test_cli_rejects_start_without_dirt():
    result = subprocess.run(
        [sys.executable, VACUUM_WORLD_PY, "--start", "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
