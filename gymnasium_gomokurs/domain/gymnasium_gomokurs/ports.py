from typing import Dict
from .models.game import Move
import numpy as np

class ManagerInterface:
    def get_init_state(self) -> tuple[int, np.array]:
        pass

    def get_opponent_turn(self) -> tuple[Move | None, bool | None]:
        pass
    
    def notify_move(self, move: Move) -> None:
        pass

    def notify_error(self, error_msg: str) -> None:
        pass
