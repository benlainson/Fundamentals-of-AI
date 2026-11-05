"""
test_cases.py

Knowledge bases and queries used to demonstrate pl_resolution.

Every KB is a list of clauses, and every clause is a list of literals.
Literals are strings: "P" is positive, "~P" is negated.
All KBs are already in CNF (Requirement 1), so no conversion is needed.

Each test case is a dictionary with:
    name        - short label for printing
    kb          - list of clauses (CNF)
    query       - a single propositional symbol
    expected    - the answer we expect (True = entailed)
    explanation - why that answer is correct
"""

# ---------------------------------------------------------------------------
# Test 1: Knowledge base from Task 3  (Requirement 5)
# ---------------------------------------------------------------------------
# Original sentences and their CNF forms (P => Q becomes ~P v Q,
# and ~(A AND B) becomes ~A v ~B by De Morgan's law):
#     X => Z              ==   ~X v Z
#     (J AND K) => X      ==   ~J v ~K v X
#     (C AND J) => K      ==   ~C v ~J v K
#     (D AND X) => J      ==   ~D v ~X v J
#     (D AND C) => J      ==   ~D v ~C v J
#     D
#     C
# NOTE: ~Z is NOT listed here. It is the negated query, and
# pl_resolution adds it automatically.
TASK3_KB = [
    ["~X", "Z"],              # X => Z
    ["~J", "~K", "X"],        # (J AND K) => X
    ["~C", "~J", "K"],        # (C AND J) => K
    ["~D", "~X", "J"],        # (D AND X) => J
    ["~D", "~C", "J"],        # (D AND C) => J
    ["D"],                    # fact
    ["C"],                    # fact
]

# ---------------------------------------------------------------------------
# Test 2: a KB that DOES entail the query  (Requirement 6a)
# ---------------------------------------------------------------------------
# Proof by cases:  P v Q,  P => R,  Q => R.   Query: R
# Whichever of P or Q is true, R follows. So R must be true.
# This is a good test because neither P nor Q is known on its own,
# so simple forward chaining on facts cannot prove R. Resolution can.
ENTAILED_KB = [
    ["P", "Q"],            # P v Q
    ["~P", "R"],           # P => R   ==   ~P v R
    ["~Q", "R"],           # Q => R   ==   ~Q v R
]

# ---------------------------------------------------------------------------
# Test 3: a KB that does NOT entail the query  (Requirement 6b)
# ---------------------------------------------------------------------------
# Same as Test 2 but WITHOUT Q => R.   Query: R
# Now there is a model where the KB is true but R is false:
#     P = False, Q = True, R = False
#   (P v Q) = True,  (~P v R) = True,  but R = False.
# Because a counter-model exists, the KB cannot entail R.
NOT_ENTAILED_KB = [
    ["P", "Q"],            # P v Q
    ["~P", "R"],           # P => R   ==   ~P v R
]

# ---------------------------------------------------------------------------
# All test cases collected in one list so main.py can loop over them
# ---------------------------------------------------------------------------
TEST_CASES = [
        {
        "name": "Test 1: Task 3 knowledge base",
        "kb": TASK3_KB,
        "query": "Z",
        "expected": True,
        "explanation": (
            "D and C are facts, so (D AND C) => J gives J. Then (C AND J) => K "
            "gives K, (J AND K) => X gives X, and X => Z gives Z. Adding ~Z "
            "contradicts this chain, so the empty clause is derived. "
            "(D AND X) => J is never needed for the proof."
        ),
    },
    {
        "name": "Test 2: proof by cases (entailed)",
        "kb": ENTAILED_KB,
        "query": "R",
        "expected": True,
        "explanation": (
            "In every model, either P or Q is true, and each of them implies R. "
            "So R is true in every model of the KB. Resolving ~R with the two "
            "implications gives ~P and ~Q, which contradict P v Q."
        ),
    },
    {
        "name": "Test 3: missing rule (not entailed)",
        "kb": NOT_ENTAILED_KB,
        "query": "R",
        "expected": False,
        "explanation": (
            "The model P=False, Q=True, R=False satisfies every clause of the KB "
            "and ~R. Since KB AND ~R is satisfiable, no contradiction exists, so "
            "resolution eventually runs out of new clauses."
        ),
    },
]