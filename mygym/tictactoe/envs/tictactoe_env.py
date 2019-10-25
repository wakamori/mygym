import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


######################
# TicTacToeEnv class
######################


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    BOARD_ROWS = 3
    BOARD_COLS = 3

    def __init__(self):
        super().__init__()
        board = Board(self.BOARD_ROWS, self.BOARD_COLS)
        self.action_space = spaces.Discrete(board.board.size)
        self.observation_space = spaces.Box(
            low=0,
            high=len(board.STONE_TYPES),
            shape=board.board.shape
        )
        p1 = HumanPlayer(is_p1=True)
        p2 = RandomPlayer(is_p1=False)
        #p2 = AIPlayer(is_p1=False, rows=self.BOARD_ROWS, cols=self.BOARD_COLS)
        self.env = TicTacToe(board, p1, p2)
        self.reward_range = [-100., 1.]
        self.np_random = None

    def step(self, action):
        x, y = divmod(action, 3)
        x, y, symbol = self.env.p1.act(self.env.board.board, x=x, y=y)
        self.env.board.put(x, y, symbol)
        result = self.env.board.judge()
        done = np.any(result == True)
        if done:
            if result[0]:
                # p1 won
                reward = 1
            elif result[1]:
                # p2 won
                reward = 0
            else:
                # draw
                reward = 0.5
            info = {}
            return self.env.board.board, reward, done, info
        x, y, symbol = self.env.p2.act(self.env.board.board)
        self.env.board.put(x, y, symbol)
        result = self.env.board.judge()
        done = np.any(result == True)
        if result[0]:
            # p1 won
            reward = 1
        elif result[1]:
            # p2 won
            reward = 0
        else:
            # draw
            reward = 0.5
        info = {}
        return self.env.board.board, reward, done, info

    def reset(self):
        #self.env.p2.backup()
        #self.env.p2.reset()
        self.env.board.reset()
        return self.env.board.board

    def render(self, mode='human', close=False):
        if mode == 'human':
            self.env.board.show()
        elif mode == 'ansi':
            return self.env.board.hash()
        else:
            super().render(mode=mode)

    def close(self):
        pass

    def seed(self, seed=None):
        return self.env.p2.seed(seed)


###############
# Board class
###############


class Board:
    STONE_TYPES = [
        '.',  # 0: empty
        'x',  # 1: stone 1
        'o',  # 2: stone 2
    ]

    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.board = np.zeros((nrows, ncols), dtype=np.int32)

    def reset(self):
        self.board = np.zeros((self.nrows, self.ncols), dtype=np.int32)

    def show(self):
        print('------')
        for i in range(self.nrows):
            for j in range(self.ncols):
                print(self.STONE_TYPES[self.board[i, j]], end=' ')
            print()
        print('------')

    def hash(self):
        return str(self.board)

    def is_end(self):
        return np.any(self.judge() == True)

    def check(self, x, y):
        if x < 0 or x >= self.nrows:
            #print('invalid x')
            return False
        if y < 0 or y >= self.ncols:
            #print('invalid y')
            return False
        if self.board[x, y] != 0:
            #print('cannot put onto ({}, {})'.format(x, y))
            return False
        return True

    def put(self, x, y, symbol):
        if not self.check(x, y):
            return False
        self.board[x, y] = symbol
        return True

    def next_state(self, x, y, symbol):
        new_board = self.new_board()
        new_board.board[x, y] = symbol
        return new_board

    def new_board(self):
        new_board = Board(self.nrows, self.ncols)
        new_board.board = np.copy(self.board)
        return new_board

    def judge(self):
        p1_won = False
        p2_won = False

        # check vertical
        p1_won = np.any(np.all(self.board == 1, axis=0))
        p2_won = np.any(np.all(self.board == 2, axis=0))

        # check horizontal
        p1_won = np.any(np.all(self.board == 1, axis=1)) or p1_won
        p2_won = np.any(np.all(self.board == 2, axis=1)) or p2_won

        # check diagonal
        p1_won = np.all(np.diag(self.board) == 1) or p1_won
        p2_won = np.all(np.diag(self.board) == 2) or p2_won
        p1_won = np.all(np.diag(np.fliplr(self.board)) == 1) or p1_won
        p2_won = np.all(np.diag(np.fliplr(self.board)) == 2) or p2_won

        draw = not np.any(self.board == 0)

        return np.array([p1_won, p2_won, draw])


################
# Player class
################

class PlayerBase:
    def __init__(self, is_p1=True):
        self.symbol = 1 if is_p1 else 2

    def act(self, board):
        pass
        # return x, y, symbol


class HumanPlayer(PlayerBase):
    def __init__(self, is_p1=True):
        super().__init__(is_p1)

    def act(self, board, x=None, y=None):
        if x is None or y is None:
            print('player {}: input x, y'.format('1' if self.is_p1 else '2'))
            while True:
                x = input('x=')
                y = input('y=')
                try:
                    x = int(x)
                    y = int(y)
                except:
                    print('invalid literal')
                else:
                    return x, y, self.symbol
        else:
            return x, y, self.symbol


class RandomPlayer(PlayerBase):
    def __init__(self, is_p1=True, seed=None):
        super().__init__(is_p1)
        self.np_random = None
        self.seed(seed)

    def act(self, board):
        empty_list = np.argwhere(board == 0)
        if empty_list.shape[0] > 0:
            n = self.np_random.choice(empty_list.shape[0], 1)
            x, y = empty_list[n][0]
            return x, y, self.symbol

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class AIPlayer(PlayerBase):
    def __init__(self, is_p1=True, seed=None, eps=1e-2, step_size=1e-1, rows=None, cols=None):
        super().__init__(is_p1)
        self.V = {}
        self.states = []
        self.greedy = []
        self.np_random = None
        self.eps = eps
        self.step_size = step_size
        self.seed(seed)
        self._init_all_states(rows, cols)

    def _init_all_states(self, rows, cols):
        self.all_states = {}
        board = Board(rows, cols)
        symbol = 1
        self.all_states[board.hash()] = (board, board.is_end())
        self._init_states_impl(board, symbol)
        for hash_val in self.all_states:
            state, is_end = self.all_states[hash_val]
            if is_end:
                p1_won, p2_won, draw = state.judge()
                if (self.symbol == 1 and p1_won) or (self.symbol == 2 and p2_won):
                    self.V[hash_val] = 1.0
                elif (self.symbol == 1 and p2_won) or (self.symbol == 2 and p1_won):
                    self.V[hash_val] = 0.0
                elif draw:
                    self.V[hash_val] = 0.5
            else:
                self.V[hash_val] = 0.5

    def _init_states_impl(self, board, symbol):
        for i in range(board.ncols):
            for j in range(board.nrows):
                if board.check(i, j):
                    new_board = board.next_state(i, j, symbol)
                    if new_board.hash() not in self.all_states:
                        self.all_states[new_board.hash()] = (
                            new_board, new_board.is_end())
                        if not new_board.is_end():
                            self._init_states_impl(
                                new_board, 2 if symbol == 1 else 1)

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def act(self, board):
        values = []
        new_board = Board(board.shape[0], board.shape[1])
        new_board.board = board
        for i in range(new_board.ncols):
            for j in range(new_board.nrows):
                if new_board.board[i, j] == 0:
                    _state = new_board.next_state(i, j, self.symbol).hash()
                    _action = [i, j]
                    _value = self.V[_state]
                    values.append([_state, _action, _value])

        if self.np_random.rand() < self.eps:
            idx = self.np_random.choice(len(values))
            state, action, _ = values[idx]
            self.states.append(state)
            self.greedy.append(False)
            return action[0], action[1], self.symbol

        self.np_random.shuffle(values)
        values.sort(key=lambda x: x[2], reverse=True)
        state, action, _ = values[0]
        self.states.append(state)
        self.greedy.append(True)
        return action[0], action[1], self.symbol

    def reset(self):
        self.states = []
        self.greedy = []

    def backup(self):
        for i in reversed(range(len(self.states) - 1)):
            state = self.states[i]
            td_error = self.greedy[i] * \
                (self.V[self.states[i + 1]] - self.V[state])
            self.V[state] += self.step_size * td_error


###################
# TicTacToe class
###################


class TicTacToe:
    def __init__(self, board, p1, p2):
        self.board = board
        self.p1 = p1
        self.p2 = p2

    def next_p(self):
        while True:
            yield self.p1
            yield self.p2

    def start(self, show=False):
        alt = self.next_p()
        while not np.any(self.board.judge() == True):
            player = next(alt)
            x, y, symbol = player.act(self.board)
            self.board.put(x, y, symbol)
            if show:
                self.board.show()
