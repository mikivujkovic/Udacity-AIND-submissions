"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math

infinity = float('inf')  # to avoid calling float function all the time


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def heuristic_1(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Score is calculated with the following formula:

    score = (a - 3b)*c

    where:

    a : number of legal moves for player 1
    b : number of legal moves for the opponent
    c : number of already taken fields

    Parameters
    ----------
    game : `isolation.Board`
    An instance of `isolation.Board` encoding the current state of the
    game (e.g., player locations and blocked cells).

    player : object
    A player instance in the current game (i.e., an object corresponding to
    one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
    The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return -infinity

    if game.is_winner(player):
        return infinity

    # number of legal moves for the player
    my_moves_no = len(game.get_legal_moves(player))
    # number of opponents legal moves
    op_moves_no = len(game.get_legal_moves(game.get_opponent(player)))
    # number of fields already taken
    taken = 49 - len(game.get_blank_spaces())

    score = (my_moves_no - 3 * op_moves_no) * taken

    return score


def heuristic_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Score is calculated in two phases:

    First we check if board is divided between players by already taken fields,
    which would mean that players are playing on two separate smaller boards.
    In that case, player with more legal moves will certainly win regardless of
    other players strategy.

    If board is not divided, we just calculate the score by comparing number of
    legal moves for both players.


    Parameters
    ----------
    game : `isolation.Board`
    An instance of `isolation.Board` encoding the current state of the
    game (e.g., player locations and blocked cells).

    player : object
    A player instance in the current game (i.e., an object corresponding to
    one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
    The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return -infinity

    if game.is_winner(player):
        return infinity

    divided = False

    # legal moves for player
    my_moves = game.get_legal_moves(player)
    # legal moves for the opponent
    op_moves = game.get_legal_moves(game.get_opponent(player))

    my_moves_no = len(my_moves)  # number of legal moves for the player
    op_moves_no = len(op_moves)  # number of legal moves for the opponent

    # make a set of mutual legal moves
    mutual = frozenset(my_moves).intersection(op_moves)

    # if there are no mutual legal moves, the board is divided
    if len(mutual) == 0:
        divided = True

    # if board is divided player with more legal moves win
    if divided:
        if my_moves_no > op_moves_no:
            score = infinity
        elif my_moves_no < op_moves_no:
            score = -infinity
        else:
            score = 0
    else:
        # if board is not divided we calculate the score as
        # a difference between number of legal moves
        score = my_moves_no - op_moves_no

    return score


def heuristic_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Score is calculated as a difference in number of players legal moves
    corrected for the distance from the board center.

    Distance is calculated as square root of the sum of sqared distances in
    rows and columns.

    It is presumed that the player closer to the center has more chances to win
    the game.


    Parameters
    ----------
    game : `isolation.Board`
    An instance of `isolation.Board` encoding the current state of the
    game (e.g., player locations and blocked cells).

    player : object
    A player instance in the current game (i.e., an object corresponding to
    one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
    The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return -infinity

    if game.is_winner(player):
        return infinity

    # number of legal moves for the player
    my_moves_no = len(game.get_legal_moves(player))
    # number of legal moves for the opponent
    op_moves_no = len(game.get_legal_moves(game.get_opponent(player)))
    diff = float(my_moves_no - op_moves_no)

    center = (3, 3)
    # location of the player
    me = game.get_player_location(player)
    # location of the opponent
    him = game.get_player_location(game.get_opponent(player))

    # players distance from the center
    my_distance = math.sqrt(math.pow((center[0] - me[0]), 2) +
                            math.pow((center[1] - me[1]), 2))
    # opponents distance from the center
    his_distance = math.sqrt(math.pow((center[0] - him[0]), 2) +
                             math.pow((center[1] - him[1]), 2))

    score = diff - my_distance + his_distance

    return score


def heuristic_4(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Score is calculated as a difference in number of players legal moves
    corrected for the number of mutual legal moves and distance from the
    center of the board.

    This should promote taking one of the mutual moves in order to decrease
    available moves for the opponent and playing closer to the center.

    Parameters
    ----------
    game : `isolation.Board`
    An instance of `isolation.Board` encoding the current state of the
    game (e.g., player locations and blocked cells).

    player : object
    A player instance in the current game (i.e., an object corresponding to
    one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
    The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return -infinity

    if game.is_winner(player):
        return infinity

    # legal moves for player
    my_moves = game.get_legal_moves(player)
    # legal moves for the opponent
    op_moves = game.get_legal_moves(game.get_opponent(player))

    # number of legal moves for the player
    my_moves_no = len(my_moves)
    # number of opponents legal moves
    op_moves_no = len(op_moves)

    # make a set of mutual legal moves
    mutual = frozenset(my_moves).intersection(op_moves)

    center = (3, 3)
    # location of the player
    me = game.get_player_location(player)
    # location of the opponent
    him = game.get_player_location(game.get_opponent(player))

    # players distance from the center
    my_distance = math.sqrt(math.pow((center[0] - me[0]), 2) +
                            math.pow((center[1] - me[1]), 2))
    # opponents distance from the center
    his_distance = math.sqrt(math.pow((center[0] - him[0]), 2) +
                             math.pow((center[1] - him[1]), 2))

    score = my_moves_no - op_moves_no + len(mutual) - my_distance + \
        his_distance

    return score


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    return heuristic_4(game, player)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            depth = 0
            while True:
                # get the move based on the selected searach algorithm
                if self.method == "minimax":
                    _, move = self.minimax(game, depth + 1)
                elif self.method == "alphabeta":
                    _, move = self.alphabeta(game, depth + 1)
                if move == (-1, -1) or self.search_depth != -1:
                    return move

                if self.iterative:
                    # if iterative search, increase depth
                    depth += 1

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return move

        # Return the best move from the last completed search iteration
        # raise NotImplementedError

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # calculate max value for moves
        def max_value(game, depth):
            if depth == 0:
                return self.score(game, self)
            value = -infinity
            for move in game.get_legal_moves():
                new_board = game.forecast_move(move)
                current_value = max_value(new_board, depth - 1)
                if value < current_value:
                    value = current_value
            return value

        # calculate best score and best move
        best_score = -infinity
        best_move = (-1, -1)
        if depth == 0:
            return self.score(game, self), game.get_player_location(self)
        for move in game.get_legal_moves():
            new_board = game.forecast_move(move)
            current_score = max_value(new_board, depth - 1)
            if best_score < current_score:
                best_score = current_score
                best_move = move
        return best_score, best_move

    def alphabeta(self, game, depth, alpha=-infinity, beta=infinity,
                  maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # calculate max value for moves
        def max_value(game, depth, alpha=-infinity, beta=infinity):
            if depth == 0:
                return self.score(game, self)
            value = -infinity
            legal_moves = game.get_legal_moves()
            for move in legal_moves:
                new_board = game.forecast_move(move)
                current_value = min_value(new_board, depth - 1, alpha, beta)
                if value < current_value:
                    value = current_value
                if value >= beta:
                    return value
                if value > alpha:
                    alpha = value
            return value

        # calculate min value for moves
        def min_value(game, depth, alpha=-infinity, beta=infinity):
            if depth == 0:
                return self.score(game, self)
            value = infinity
            legal_moves = game.get_legal_moves()
            for move in legal_moves:
                new_board = game.forecast_move(move)
                current_value = max_value(new_board, depth - 1, alpha, beta)
                if value > current_value:
                    value = current_value
                if value <= alpha:
                    return value
                if value < beta:
                    beta = value
            return value

        # calculate best score and best move
        best_score = -infinity
        best_move = (-1, -1)
        for move in game.get_legal_moves():
            new_board = game.forecast_move(move)
            current_score = min_value(new_board, depth - 1, alpha, beta)
            if best_score < current_score:
                best_score = current_score
                best_move = move
            if best_score >= beta:
                return best_score, best_move
            if best_score > alpha:
                alpha = best_score
        return best_score, best_move
