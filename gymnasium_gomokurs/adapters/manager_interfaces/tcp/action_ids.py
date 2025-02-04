from enum import Enum

class ActionID(Enum):
    # Action that can be send from the manager to the player.

    MANAGER_PROTOCOL_COMPATIBLE = 0x00.to_bytes(1, 'big')
    MANAGER_START               = 0x01.to_bytes(1, 'big')
    MANAGER_TURN                = 0x02.to_bytes(1, 'big')
    MANAGER_BEGIN               = 0x03.to_bytes(1, 'big')
    MANAGER_BOARD               = 0x04.to_bytes(1, 'big')
    MANAGER_INFO                = 0x05.to_bytes(1, 'big')
    MANAGER_END                 = 0x06.to_bytes(1, 'big')
    MANAGER_ABOUT               = 0x07.to_bytes(1, 'big')
    MANAGER_UNKNOWN             = 0x08.to_bytes(1, 'big')
    MANAGER_ERROR               = 0x09.to_bytes(1, 'big')

    # Actions that can be send from the player to the manager.

    PLAYER_PROTOCOL_VERSION     = 0x0A.to_bytes(1, 'big')
    PLAYER_READY                = 0x0B.to_bytes(1, 'big')
    PLAYER_PLAY                 = 0x0C.to_bytes(1, 'big')
    PLAYER_METADATA             = 0x0D.to_bytes(1, 'big')
    PLAYER_UNKNOWN              = 0x0E.to_bytes(1, 'big')
    PLAYER_ERROR                = 0x0F.to_bytes(1, 'big')
    PLAYER_MESSAGE              = 0x10.to_bytes(1, 'big')
    PLAYER_DEBUG                = 0x11.to_bytes(1, 'big')
    PLAYER_SUGGESTION           = 0x12.to_bytes(1, 'big')
