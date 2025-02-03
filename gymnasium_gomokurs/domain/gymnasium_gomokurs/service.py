from typing import Optional
import numpy as np
import gymnasium
from .ports import ManagerInterface
from .models.game import *
import asyncio

class GomokursEnv(gymnasium.Env):
    def __init__(self, manager_interface: ManagerInterface, size: int = 20):
        self.size = size
        self._manager_interface = manager_interface
        self._board = np.zeros((size, size))
        self.observation_space = gymnasium.spaces.Dict(
            {
                "availables":   gymnasium.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "player":       gymnasium.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "opponent":     gymnasium.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        ) 
        self.action_space = gymnasium.spaces.Discrete(size ** 2)
        self.loop = asyncio.get_event_loop()

    def _action_to_move(self, action_idx: int) -> Move:
        x = action_idx / self.size
        y = action_idx % self.size

        return Move(x, y)

    def _get_obs(self):
        return {
            "availables":   (self._board == CellStatus.AVAILABLE).astype(int),
            "player":       (self._board == CellStatus.PLAYER).astype(int),
            "opponent":     (self._board == CellStatus.OPPONENT).astype(int),
        }

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.board = np.zeros((self.size, self.size))

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        move = self._action_to_move(action)
        if self._board[move.x][move.y] != CellStatus.AVAILABLE:
            raise Exception(f"player move invalid: cell at position {move} is not available")
        self._board[move.x][move.y] = CellStatus.PLAYER

        self._manager_interface.notify_move(move)

        opponent_move = self._manager_interface.get_opponent_turn()
        if self._board[opponent_move.x][opponent_move.y] != CellStatus.AVAILABLE:
            raise Exception(f"opponent move invalid: cell at position {opponent_move} is not available")
        self._board[move.x][move.y] = CellStatus.OPPONENT

        terminated = False
        truncated = False
        reward = 0
        observation = self._get_obs()
        info = self._get_info()

        return terminated, truncated, reward, observation, info
