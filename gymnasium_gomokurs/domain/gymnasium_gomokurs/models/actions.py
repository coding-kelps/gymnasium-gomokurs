from .game import Move, RelativeTurn

class StartAction:
    def __init__(self, size: int):
        self.size = size

class TurnAction:
    def __init__(self, move: Move):
        self.move = move

class BeginAction:
    def __init__(self):
        pass

class BoardAction:
    def __init__(self, turns: list[RelativeTurn]):
        self.turns = turns

class InfoAction:
    def __init__(self, info: str):
        self.info = info

class EndAction:
    def __init__(self):
        pass

class AboutAction:
    def __init__(self):
        pass

class UnknownAction:
    def __init__(self, msg: str):
        self.msg = msg

class ErrorAction:
    def __init__(self, msg: str):
        self.msg = msg
