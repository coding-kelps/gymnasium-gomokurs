from enum import Enum

class Move:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class RelativeField(Enum):
    OWN_STONE       = 0
    OPPONENT_STONE  = 1

class RelativeTurn:
    def __init__(self, x: int, y: int, field: RelativeField):
        self.move = Move(x, y)
        self.field = field
