"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy 

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board: list):
    """
    Returns player who has the next turn on a board.
    In initial state, X makes first move.
    Then, player alternates between O and X for subsequent moves
    """
    num_empty = sum(row.count(EMPTY) for row in board)
    num_moves_made = 9 - num_empty

    if num_moves_made % 2 == 0:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    Basically just all the row and column index of all EMPTY cells on the board.
    """
    possible_actions = set()

    for i in range(len(board)):
        for j in range(len(board[i])):
            curr_tile = board[i][j]
            if curr_tile is EMPTY:
                possible_actions.add((i, j))

    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action

    is_valid_action = i < 3 and j < 3 and board[i][j] is EMPTY

    if is_valid_action:
        curr_player = player(board)
        res_board = deepcopy(board)
        res_board[i][j] = curr_player
        return res_board
    else:
        raise Exception("Invalid action")


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for player in (X,O):
        # Check Horizontal
        for row in board:
            if(row.count(player) == 3):
                return player

        # Check Vertical
        for j in range(len(board)):
            verticals = [board[i][j] for i in range(len(board))]
            if verticals.count(player) == 3:
                return player

        # Check Diagonals
        first_diagonal = [board[i][i] for i in range(len(board))]
        second_diagonal = [board[i][~i] for i in range(len(board))]
        if first_diagonal.count(player) == 3 or second_diagonal.count(player) == 3:
            return player
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Game is over when a winner is found or if there are no more possible actions
    game_is_over = (winner(board) != None) or (len(actions(board)) == 0)
    return game_is_over


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    util = {X: 1, None: 0, O: -1}
    return util.get(winner(board))


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    if player(board) == X:
        return optimize_score(board, maximize=True)[1]
    else:
        return optimize_score(board, maximize=False)[1]


def optimize_score(board, maximize=True):
    '''
    Initially, X & O start with their worst scores:
        1. If player is MAX(X), initial score is -INF
        2. If player is MIN(O), initial score is INF
    MAX(X) wants to maximize the score (get 1)
    MIN(O) wants to minimize the score (get -1)
    '''
    optimal_move = None

    if terminal(board):
        return (utility(board), optimal_move)

    curr_score = -math.inf if maximize else math.inf
    possible_actions = actions(board)

    '''
    For every possible action
        1. Check the resultant state of the board 
        2. Subsequent state of the board made in response by the opponent
           trying to optimize their score
    Make the move that optimizes my score (maximize if I am X, minimize if I am O)
    '''
    for action in possible_actions:
        resultant_state = result(board, action)

        if maximize: 
            # Calculate resultant score when opponent tries to minimize their score
            resultant_score = optimize_score(resultant_state, maximize=False)[0]
            # If the resultant score is higher than my current score, make the move
            if resultant_score > curr_score:
                curr_score, optimal_move = resultant_score, action
        else: 
            # Calculate resultant score when opponent tries to maximize their score
            resultant_score = optimize_score(resultant_state, maximize=True)[0]
            # If the resultant score is lower than my current score, make the move
            if resultant_score < curr_score: 
                curr_score, optimal_move = resultant_score, action

    return (curr_score, optimal_move)




