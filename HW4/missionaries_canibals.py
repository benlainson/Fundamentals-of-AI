"""Missionaries and cannibals solved with breadth-first search.
 
State: (m, c, b) = missionaries and cannibals on the LEFT bank, and the boat
side (1 = left, 0 = right). Everything on the right bank is implied (3 - m, 3 - c).
Start (3, 3, 1); goal (0, 0, 0).
 
Usage: python missionaries_cannibals.py
"""
from collections import deque
 
N = 3
START, GOAL = (N, N, 1), (0, 0, 0)
BOAT_LOADS = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]  # (missionaries, cannibals)
 
 
def safe(m, c):
    """No bank may have missionaries outnumbered (unless it has none)."""
    left_ok = m == 0 or m >= c
    right_ok = (N - m) == 0 or (N - m) >= (N - c)
    return left_ok and right_ok
 
 
def successors(state):
    m, c, b = state
    sign = -1 if b == 1 else 1          # boat leaving left removes people from left
    for dm, dc in BOAT_LOADS:
        nm, nc = m + sign * dm, c + sign * dc
        if 0 <= nm <= N and 0 <= nc <= N and safe(nm, nc):
            yield (dm, dc), (nm, nc, 1 - b)
 
 
def bfs(check_repeats=True, node_cap=5_000_000):
    """Return (path_of_states, moves, nodes_generated). BFS => fewest crossings."""
    frontier = deque([(START, [START], [])])
    reached = {START}
    generated = 1
    while frontier:
        state, path, moves = frontier.popleft()
        if state == GOAL:
            return path, moves, generated
        for load, nxt in successors(state):
            if check_repeats:
                if nxt in reached:
                    continue
                reached.add(nxt)
            generated += 1
            if generated > node_cap:
                return None, None, generated
            frontier.append((nxt, path + [nxt], moves + [load]))
    return None, None, generated
 
 
def all_optimal(depth):
    """Count every shortest solution (depth crossings) by exhaustive DFS."""
    count = 0
    def dfs(state, d):
        nonlocal count
        if state == GOAL:
            count += d == depth
            return
        if d >= depth:
            return
        for _, nxt in successors(state):
            dfs(nxt, d + 1)
    dfs(START, 0)
    return count
 
 
def reachable():
    seen, stack = {START}, [START]
    while stack:
        for _, nxt in successors(stack.pop()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen
 
 
if __name__ == "__main__":
    valid = [(m, c, b) for m in range(N + 1) for c in range(N + 1)
             for b in (0, 1) if safe(m, c)]
    print(f"safe states: {len(valid)} of {2 * (N + 1) ** 2}; "
          f"reachable from start: {len(reachable())}")
 
    path, moves, gen = bfs(check_repeats=True)
    print(f"\nBFS with repeated-state check: {len(moves)} crossings, "
          f"{gen} nodes generated")
    for i, ((dm, dc), s) in enumerate(zip(moves, path[1:]), 1):
        side = "left->right" if path[i - 1][2] == 1 else "right->left"
        print(f"  {i:2d}. {dm}M {dc}C {side:11s} -> left bank (M,C)=({s[0]},{s[1]}), "
              f"boat {'left' if s[2] else 'right'}")
 
    _, m2, gen2 = bfs(check_repeats=False)
    print(f"\nBFS WITHOUT repeated-state check: {len(m2)} crossings, "
          f"{gen2} nodes generated")
    print(f"number of distinct optimal solutions: {all_optimal(len(moves))}")
 
    print("\nedges (state graph):")
    for s in sorted(reachable(), reverse=True):
        print(" ", s, "->", [n for _, n in successors(s)])