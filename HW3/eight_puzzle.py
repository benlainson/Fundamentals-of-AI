import heapq
import random
import time

GOAL = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)                     # 0 is the blank
 
# Where each tile belongs: tile -> (row, col)
GOAL_POS = {tile: divmod(i, 3) for i, tile in enumerate(GOAL)}
 
 
def neighbors(state):
    """Return every state reachable by sliding one tile into the blank."""
    blank = state.index(0)
    row, col = divmod(blank, 3)
    result = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:   # up, down, left, right
        r, c = row + dr, col + dc
        if 0 <= r < 3 and 0 <= c < 3:                    # stay on the board
            swap = r * 3 + c
            new = list(state)
            new[blank], new[swap] = new[swap], new[blank]
            result.append(tuple(new))
    return result
 
 
def is_solvable(state):
    """On a 3x3 board, a state is solvable iff its inversion count is even."""
    tiles = [t for t in state if t != 0]
    inversions = sum(1 for i in range(len(tiles))
                     for j in range(i + 1, len(tiles))
                     if tiles[i] > tiles[j])
    return inversions % 2 == 0

def astar(start, heuristic):
    """Return (path, nodes_expanded). `heuristic` is any function state -> number."""
    counter = 0                                   # tie-breaker for equal f values
    frontier = [(heuristic(start), 0, counter, start)]   # (f, g, tiebreak, state)
    best_g = {start: 0}                           # cheapest known cost to each state
    parent = {start: None}                        # for rebuilding the path
    expanded = 0
 
    while frontier:
        f, g, _, state = heapq.heappop(frontier)
 
        if g > best_g[state]:                     # stale entry: cheaper route found later
            continue
 
        if state == GOAL:
            path = []
            while state is not None:
                path.append(state)
                state = parent[state]
            return path[::-1], expanded
 
        expanded += 1
        for nxt in neighbors(state):
            new_g = g + 1                         # every move costs 1
            if nxt not in best_g or new_g < best_g[nxt]:
                best_g[nxt] = new_g
                parent[nxt] = state
                counter += 1
                heapq.heappush(frontier,
                               (new_g + heuristic(nxt), new_g, counter, nxt))
 
    return None, expanded                         # frontier empty: unsolvable

def zero_h(state):
    """Baseline (no guidance). Makes A* behave like BFS. Not used in the table."""
    return 0
 
 
def misplaced(state):
    """Number of tiles (not counting the blank) that aren't on their goal square."""
    return sum(1 for i, tile in enumerate(state) if tile != 0 and tile != GOAL[i])
 
 
def manhattan(state):
    """Sum over tiles of (row distance + column distance) to the goal square."""
    total = 0
    for i, tile in enumerate(state):
        if tile == 0:
            continue                              # the blank doesn't count
        r, c = divmod(i, 3)
        goal_r, goal_c = GOAL_POS[tile]
        total += abs(r - goal_r) + abs(c - goal_c)
    return total
 
 
def max_swap(state):
    board = list(state)
    blank_home = GOAL.index(0)
    swaps = 0
    while True:
        blank = board.index(0)
        if blank != blank_home:
            wanted = GOAL[blank]                  # tile that belongs where the blank is
            j = board.index(wanted)
            board[blank], board[j] = board[j], board[blank]
        else:
            for i, tile in enumerate(board):
                if tile != GOAL[i]:               # any misplaced tile
                    board[blank], board[i] = board[i], board[blank]
                    break
            else:
                return swaps                      # nothing misplaced: solved
        swaps += 1
 

def row_col_misplaced(state):
    """(# tiles in the wrong row) + (# tiles in the wrong column), blank excluded."""
    wrong_rows = wrong_cols = 0
    for i, tile in enumerate(state):
        if tile == 0:
            continue
        r, c = divmod(i, 3)
        goal_r, goal_c = GOAL_POS[tile]
        wrong_rows += (r != goal_r)
        wrong_cols += (c != goal_c)
    return wrong_rows + wrong_cols

HEURISTICS = [
    ("Misplaced tiles", misplaced),
    ("Manhattan", manhattan),
    ("n-MaxSwap", max_swap),
    ("Row/col misplaced", row_col_misplaced),
]
 
 
def random_solvable_puzzle(rng):
    """Shuffle until we get a solvable, non-goal state."""
    while True:
        tiles = list(GOAL)
        rng.shuffle(tiles)
        state = tuple(tiles)
        if state != GOAL and is_solvable(state):
            return state
 
 
def compare(num_puzzles=30, seed=42):
    rng = random.Random(seed)                     # fixed seed => repeatable results
    puzzles = [random_solvable_puzzle(rng) for _ in range(num_puzzles)]
 
    print(f"Comparing heuristics on {num_puzzles} random solvable puzzles (seed={seed})\n")
    header = f"{'Heuristic':<20}{'Avg h(start)':>14}{'Avg expanded':>15}{'Avg time (ms)':>16}{'Avg moves':>12}"
    print(header)
    print("-" * len(header))
 
    optimal_lengths = None
    for name, h in HEURISTICS:
        total_h = total_expanded = total_time = 0
        lengths = []
        for start in puzzles:
            t0 = time.perf_counter()
            path, expanded = astar(start, h)
            total_time += time.perf_counter() - t0
            total_h += h(start)
            total_expanded += expanded
            lengths.append(len(path) - 1)
 
        # All four heuristics are admissible, so solution lengths must agree.
        if optimal_lengths is None:
            optimal_lengths = lengths
        assert lengths == optimal_lengths, f"{name} returned a non-optimal solution!"
 
        n = num_puzzles
        print(f"{name:<20}{total_h / n:>14.2f}{total_expanded / n:>15.1f}"
              f"{total_time / n * 1000:>16.1f}{sum(lengths) / n:>12.2f}")
 
 
if __name__ == "__main__":
    compare()