"""
main.py

Entry point. Runs every test case through the resolution algorithm,
prints the result and the number of resolution steps, and checks that
the result matches the expected answer.

Usage (from inside the src folder):
    python main.py             # summary output
    python main.py --verbose   # also print every resolution step
"""

import sys

from resolution import pl_resolution, pl_resolution_with_stats, clause_to_str
from test_cases import TEST_CASES


def run_test(case, verbose=False):
    """Run one test case, print its results, and return True if it passed."""
    print("=" * 70)
    print(case["name"])
    print("=" * 70)

    # Show the KB in readable form
    print("Knowledge base:")
    for clause in case["kb"]:
        print("   ", clause_to_str(frozenset(clause)))
    print(f"Query: {case['query']}")
    print()

    # The required interface function (bool only)
    result = pl_resolution(case["kb"], case["query"])

    # The stats version gives the step count for Requirement 7.
    # It runs the same deterministic algorithm, so the answer is identical.
    _, steps = pl_resolution_with_stats(case["kb"], case["query"],
                                        verbose=verbose)

    passed = (result == case["expected"])

    print()
    print(f"Result:            KB {'entails' if result else 'does NOT entail'} "
          f"{case['query']}  ({result})")
    print(f"Expected:          {case['expected']}")
    print(f"Resolution steps:  {steps}")
    print(f"Why:               {case['explanation']}")
    print(f"Status:            {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def main():
    # Turn on step-by-step tracing if the user passed --verbose
    verbose = "--verbose" in sys.argv

    print("Resolution step = one resolve() call on a distinct pair of clauses")
    print("that contains at least one complementary pair of literals.\n")

    results = [run_test(case, verbose) for case in TEST_CASES]

    # Final summary line
    print("=" * 70)
    print(f"{sum(results)}/{len(results)} test cases passed.")


# Only run main() when this file is executed directly,
# not when it is imported by another file.
if __name__ == "__main__":
    main()