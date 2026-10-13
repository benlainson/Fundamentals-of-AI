"""Tic-tac-toe agent using minimax with alpha-beta pruning.

Board: tuple of 9 cells, each 'X', 'O' or ' ', indexed 0-8 (row-major).
Utility is always from X's point of view: +1 X wins, -1 O wins, 0 draw.

Usage:
    python tictactoe.py selfplay [n_games] [seed]   # agent vs. agent
    python tictactoe.py test                         # run the test suite
"""
import random
import sys
import unittest

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6)]
EMPTY_BOARD = (' ',) * 9


def winner(board):
    for a, b, c in LINES:
        if board[a] != ' ' and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_terminal(board):
    return winner(board) is not None or ' ' not in board


def utility(board):
    """+1 if X won, -1 if O won, 0 for a draw. Only valid on terminal boards."""
    if not is_terminal(board):
        raise ValueError("utility() called on a non-terminal board")
    w = winner(board)
    return 1 if w == 'X' else -1 if w == 'O' else 0


def to_move(board):
    """X moves when counts are equal, otherwise O."""
    return 'X' if board.count('X') == board.count('O') else 'O'


def legal_moves(board):
    return [i for i, c in enumerate(board) if c == ' ']


def result(board, move):
    b = list(board)
    b[move] = to_move(board)
    return tuple(b)


def alphabeta(board, alpha=-2, beta=2):
    """Minimax value of `board` (from X's view) with alpha-beta pruning."""
    if is_terminal(board):
        return utility(board)
    if to_move(board) == 'X':
        v = -2
        for m in legal_moves(board):
            v = max(v, alphabeta(result(board, m), alpha, beta))
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return v
    v = 2
    for m in legal_moves(board):
        v = min(v, alphabeta(result(board, m), alpha, beta))
        beta = min(beta, v)
        if alpha >= beta:
            break
    return v


def best_moves(board):
    """All moves achieving the optimal minimax value for the player to move."""
    if is_terminal(board):
        raise ValueError("no move available: board is terminal")
    player = to_move(board)
    # Full window per child so every move gets its exact value (no pruning ties).
    scored = [(alphabeta(result(board, m)), m) for m in legal_moves(board)]
    target = max(s for s, _ in scored) if player == 'X' else min(s for s, _ in scored)
    return [m for s, m in scored if s == target]


def choose_move(board, rng=None):
    """Pick an optimal move; break ties randomly if rng is given, else lowest index."""
    moves = best_moves(board)
    return rng.choice(moves) if rng else moves[0]


def play_game(agent_x, agent_o):
    """Play one game; agents are functions board -> move. Returns utility."""
    board = EMPTY_BOARD
    while not is_terminal(board):
        agent = agent_x if to_move(board) == 'X' else agent_o
        move = agent(board)
        assert board[move] == ' ', "agent chose an illegal move"
        board = result(board, move)
    return utility(board)


def selfplay(n_games=20, seed=0):
    rng = random.Random(seed)
    # Two separate copies of the agent (each with its own RNG for tie-breaking).
    agent_a = lambda b, r=random.Random(rng.random()): choose_move(b, r)
    agent_b = lambda b, r=random.Random(rng.random()): choose_move(b, r)
    x_wins = o_wins = draws = 0
    for g in range(n_games):
        # Alternate which copy plays X.
        ax, ao = (agent_a, agent_b) if g % 2 == 0 else (agent_b, agent_a)
        u = play_game(ax, ao)
        x_wins += u == 1
        o_wins += u == -1
        draws += u == 0
        print(f"game {g + 1:2d}: X={'A' if g % 2 == 0 else 'B'} "
              f"O={'B' if g % 2 == 0 else 'A'} -> "
              f"{'X wins' if u == 1 else 'O wins' if u == -1 else 'draw'}")
    print(f"\n{n_games} games: X wins={x_wins}, O wins={o_wins}, draws={draws}")


def B(s):
    """Build a board from a 9-char string, using '.' or '_' for empty."""
    return tuple(' ' if c in '._' else c for c in s)


class Tests(unittest.TestCase):
    def test_win_detection(self):
        self.assertEqual(winner(B("XXX......")), 'X')
        self.assertEqual(winner(B("O..O..O..")), 'O')
        self.assertEqual(winner(B("X...X...X")), 'X')
        self.assertEqual(winner(B("..O.O.O..")), 'O')
        self.assertIsNone(winner(B("XO.......")))

    def test_utilities_and_terminal(self):
        self.assertEqual(utility(B("XXXOO....")), 1)
        self.assertEqual(utility(B("OOOXX.X..")), -1)
        draw = B("XOXXOOOXX")
        self.assertTrue(is_terminal(draw))
        self.assertEqual(utility(draw), 0)
        with self.assertRaises(ValueError):
            utility(EMPTY_BOARD)

    def test_terminal_board_refuses_to_move(self):
        with self.assertRaises(ValueError):
            choose_move(B("XXXOO...."))
        with self.assertRaises(ValueError):
            choose_move(B("XOXXOOOXX"))

    def test_take_immediate_win_x(self):
        # X X . / O O . / . . .  -> X to move must play 2 and win.
        self.assertEqual(best_moves(B("XX.OO....")), [2])

    def test_take_immediate_win_o(self):
        # X X . / O O . / X . .  -> O to move (X has 3, O has 2); O wins at 5.
        self.assertEqual(best_moves(B("XX.OO.X..")), [5])

    def test_block_opponent_win(self):
        # X X . / . O . / . . .  -> O to move must block at 2.
        self.assertEqual(best_moves(B("XX..O....")), [2])

    def test_block_opponent_win_for_x(self):
        # O O . / . X . / . . X  -> X to move (2 each) must block at 2.
        self.assertEqual(best_moves(B("OO..X...X")), [2])

    def test_assignment_position(self):
        # The position from the minimax homework: X wins with index 6.
        self.assertEqual(best_moves(B("XOXOX...O")), [6])

    def test_corner_opening_reply(self):
        # X in corner: the only non-losing reply for O is the center.
        self.assertEqual(best_moves(B("X........")), [4])

    def test_empty_board_is_draw(self):
        self.assertEqual(alphabeta(EMPTY_BOARD), 0)

    def test_always_legal_on_all_reachable_boards(self):
        seen, stack = set(), [EMPTY_BOARD]
        while stack:
            b = stack.pop()
            if b in seen:
                continue
            seen.add(b)
            if is_terminal(b):
                continue
            m = choose_move(b)
            self.assertEqual(b[m], ' ')
            for mv in legal_moves(b):
                stack.append(result(b, mv))
        self.assertEqual(len(seen), 5478)  # known count of reachable positions

    def test_agent_never_loses(self):
        # Optimal agent as O or X vs. every possible opponent line must never lose.
        def check(board, agent_player):
            if is_terminal(board):
                u = utility(board)
                self.assertFalse((agent_player == 'X' and u == -1) or
                                 (agent_player == 'O' and u == 1))
                return
            if to_move(board) == agent_player:
                check(result(board, choose_move(board)), agent_player)
            else:
                for m in legal_moves(board):
                    check(result(board, m), agent_player)
        check(EMPTY_BOARD, 'X')
        check(EMPTY_BOARD, 'O')


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selfplay"
    if cmd == "selfplay":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        s = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        selfplay(n, s)
    elif cmd == "test":
        unittest.main(argv=[sys.argv[0], "-v"])