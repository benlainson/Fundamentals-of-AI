"""
resolution.py

Core implementation of propositional logic resolution by refutation.

REPRESENTATION
--------------
- A LITERAL is a string. A positive literal is just the symbol name ("P").
  A negative literal has a "~" in front of it ("~P").
- A CLAUSE is a frozenset of literals, read as a disjunction (OR).
      frozenset({"~P", "Q"})  means  (~P OR Q)
  We use frozenset because:
    * sets ignore order and duplicates, so {"P","Q"} == {"Q","P"}
    * frozensets are immutable and hashable, so a clause can be stored
      inside another set. That is how we prevent duplicate clauses.
- The EMPTY CLAUSE is frozenset(). It represents a contradiction (False).
- A KNOWLEDGE BASE is a collection of clauses, read as a conjunction (AND).
"""

from itertools import combinations


# ---------------------------------------------------------------------------
# Helper functions for literals and clauses
# ---------------------------------------------------------------------------

def negate(literal):
    """
    Return the complement of a literal.
        negate("P")  -> "~P"
        negate("~P") -> "P"
    """
    if literal.startswith("~"):
        return literal[1:]      # strip the "~" to make it positive
    return "~" + literal        # add a "~" to make it negative


def make_clause(literals):
    """
    Convert any iterable of literals (list, tuple, set) into a frozenset clause.
    This lets users write clauses as plain lists like ["~P", "Q"].
    """
    return frozenset(literals)


def is_tautology(clause):
    """
    A clause is a tautology if it contains a literal AND its complement,
    e.g. (P OR ~P OR Q). Such a clause is always True, so it can never help
    derive the empty clause. Discarding these keeps the search smaller
    without affecting correctness.
    """
    return any(negate(lit) in clause for lit in clause)


def clause_to_str(clause):
    """Pretty-print a clause, e.g. frozenset({'~P','Q'}) -> '(~P v Q)'."""
    if not clause:
        return "{}  (empty clause)"
    # sort the literals by symbol name so output is consistent between runs
    return "(" + " v ".join(sorted(clause, key=lambda l: l.lstrip("~"))) + ")"


# ---------------------------------------------------------------------------
# Requirement 2: resolve two clauses
# ---------------------------------------------------------------------------

def resolve(ci, cj):
    """
    Return the set of ALL resolvents of clauses ci and cj.

    For every literal L in ci whose complement ~L is in cj, we form:
        resolvent = (ci without L)  UNION  (cj without ~L)

    Example:
        ci = (P v Q),  cj = (~P v R)
        P in ci and ~P in cj  ->  resolvent = (Q v R)

    If the two clauses share more than one complementary pair, we produce
    one resolvent per pair (requirement: "all possible resolvent clauses").
    Note that we only cancel ONE pair at a time, as resolution requires.
    """
    resolvents = set()
    for literal in ci:
        complement = negate(literal)
        if complement in cj:
            # remove the clashing pair and merge what is left
            new_clause = (ci - {literal}) | (cj - {complement})
            resolvents.add(frozenset(new_clause))
    return resolvents


# ---------------------------------------------------------------------------
# Requirement 3: resolution-based inference
# ---------------------------------------------------------------------------
def clause_sort_key(clause):
    """
    Give clauses a fixed ordering: shorter clauses first, then alphabetical.
    Python sets have no guaranteed order, so without this the step count
    could change between runs. Short clauses first also tends to reach
    the empty clause sooner.
    """
    return (len(clause), sorted(clause))

def pl_resolution_with_stats(kb, query, verbose=False):
    """
    Run resolution by refutation and return (entailed, steps).

        entailed: True if kb entails query, else False
        steps:    number of resolution steps performed

    DEFINITION OF A RESOLUTION STEP:
        One call to resolve(ci, cj) on a distinct pair of clauses that
        contain at least one complementary pair of literals (i.e. the pair
        can actually be resolved). Pairs with nothing to cancel are skipped
        and not counted, and each pair is only ever resolved once.
    """
    # --- Step A: load the KB as a set of frozenset clauses -----------------
    # Using a set means duplicate clauses in the input are merged right away.
    clauses = {make_clause(c) for c in kb}

    # --- Step B: negate the query and add it to the KB ---------------------
    # Proof by contradiction: if KB AND ~query is unsatisfiable,
    # then KB entails query.
    negated_query = make_clause([negate(query)])
    clauses.add(negated_query)

    if verbose:
        print("Starting clauses (KB + negated query):")
        for c in clauses:
            print("   ", clause_to_str(c))

    # Remembers every pair already resolved so we never redo the work.
    # Each entry is a frozenset of two clauses, so (A,B) and (B,A) match.
    resolved_pairs = set()
    steps = 0

    # --- Step C: repeatedly resolve pairs ----------------------------------
    while True:
        new = set()   # clauses generated during this round

        # combinations() yields every unordered pair of distinct clauses
        for ci, cj in combinations(sorted(clauses, key=clause_sort_key), 2):
            pair = frozenset([ci, cj])
            if pair in resolved_pairs:
                continue                      # already done in a past round
            resolved_pairs.add(pair)

            # Skip pairs with no complementary literals (nothing to resolve)
            if not any(negate(lit) in cj for lit in ci):
                continue

            steps += 1                        # this counts as one step
            resolvents = resolve(ci, cj)

            for r in resolvents:
                # --- Success: empty clause means contradiction found ------
                if len(r) == 0:
                    if verbose:
                        print(f"  Step {steps}: {clause_to_str(ci)} + "
                              f"{clause_to_str(cj)} -> EMPTY CLAUSE")
                    return True, steps

                # Ignore always-true clauses; they cannot lead to a proof
                if is_tautology(r):
                    continue

                # Requirement 4: only keep clauses we have never seen
                if r not in clauses and r not in new:
                    new.add(r)
                    if verbose:
                        print(f"  Step {steps}: {clause_to_str(ci)} + "
                              f"{clause_to_str(cj)} -> {clause_to_str(r)}")

        # --- Failure: nothing new was produced ----------------------------
        # Every possible resolvent is already in the KB, so the search
        # cannot go any further. KB AND ~query is satisfiable.
        if not new:
            return False, steps

        # Add this round's new clauses and go around again
        clauses |= new


def pl_resolution(kb, query) -> bool:
    """
    Determine whether kb entails query
    using propositional resolution.

    Parameters:
        kb: A collection of clauses in CNF. Each clause is a collection of
            literal strings, e.g. [["~P", "Q"], ["P"]].
        query: A propositional symbol, e.g. "Q".

    Returns:
        True if kb entails query.
        False otherwise.
    """
    # Required interface: just return the boolean part of the result.
    entailed, _ = pl_resolution_with_stats(kb, query)
    return entailed