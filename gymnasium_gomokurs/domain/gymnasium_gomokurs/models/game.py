from enum import Enum

class CellStatus(Enum):
    AVAILABLE   = 0
    PLAYER      = 1
    OPPONENT    = 2


class Move:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"{{ x: {self.x}, y: {self.y} }}"

class RelativeField(Enum):
    OWN_STONE       = 0
    OPPONENT_STONE  = 1

class RelativeTurn:
    def __init__(self, x: int, y: int, field: RelativeField):
        self.move = Move(x, y)
        self.field = field

class GameEnd(Enum):
    DRAW    = 0
    WIN     = 1
    LOOSE   = 2
