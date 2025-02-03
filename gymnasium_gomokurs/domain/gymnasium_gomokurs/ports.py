from typing import Dict
from .models.game import Move

class ManagerInterface:
    def get_opponent_turn(self) -> Move:
        pass

    def is_game_finished(self) -> bool:
        pass
    
    def notify_move(self, move) -> None:
        pass
