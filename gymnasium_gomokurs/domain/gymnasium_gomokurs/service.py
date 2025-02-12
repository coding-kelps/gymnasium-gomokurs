from typing import Optional
import gymnasium
from .ports import ManagerInterface
from .models.game import *
import logging

logger = logging.getLogger("gymnasium-gomokurs")

class GomokursEnv(gymnasium.Env):
    def __init__(self, manager_interface: ManagerInterface):
        self._manager_interface = manager_interface

    def _action_to_move(self, action_idx: int) -> Move:
        x = int((action_idx - action_idx % self.size) / self.size)
        y = int(action_idx % self.size)

        return Move(x, y)

    def _get_obs(self):
        return {
            "availables":   (self.state == CellStatus.AVAILABLE.value).astype(int),
            "player":       (self.state == CellStatus.PLAYER.value).astype(int),
            "opponent":     (self.state == CellStatus.OPPONENT.value).astype(int),
        }

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.size, self.state = self._manager_interface.get_init_state()
        self.observation_space = gymnasium.spaces.Dict(
            {
                "availables":   gymnasium.spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "player":       gymnasium.spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "opponent":     gymnasium.spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = gymnasium.spaces.Discrete(self.size ** 2)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        move = self._action_to_move(action)

        if self.state[move.x][move.y] != CellStatus.AVAILABLE.value:
            raise Exception(f"player move invalid: cell at position {move} is not available")
        self.state[move.x][move.y] = CellStatus.PLAYER.value
        self._manager_interface.notify_move(move)

        opponent_move, result, end = self._manager_interface.get_opponent_turn()
        if opponent_move:
            if self.state[opponent_move.x][opponent_move.y] != CellStatus.AVAILABLE.value:
                raise Exception(f"opponent move invalid: cell at position {opponent_move} is not available")
            self.state[opponent_move.x][opponent_move.y] = CellStatus.OPPONENT.value

        terminated = True if result else False
        truncated = True if end else False

        if result == GameEnd.WIN:
            reward = 1.0
        elif result == GameEnd.DRAW:
            reward = 0.5
        elif result == GameEnd.LOOSE:
            reward = 0.0
        else:
            reward = 0.0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        pass
