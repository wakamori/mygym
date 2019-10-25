import unittest

import gym
import mygym


class TestTicTacToe(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_make(self):
        self.assertTrue(isinstance(gym.make('TicTacToe-v0'),
                                   mygym.tictactoe.envs.tictactoe_env.TicTacToeEnv))


if __name__ == "__main__":
    unittest.main()
