from enum import Enum

class ActionID(Enum):
    # Action that can be send from the manager to the player.

    MANAGER_PROTOCOL_COMPATIBLE = 0x00
    MANAGER_START               = 0x01
    MANAGER_TURN                = 0x02
    MANAGER_BEGIN               = 0x03
    MANAGER_BOARD               = 0x04
    MANAGER_INFO                = 0x05
    MANAGER_END                 = 0x06
    MANAGER_ABOUT               = 0x07
    MANAGER_UNKNOWN             = 0x08
    MANAGER_ERROR               = 0x09

    # Actions that can be send from the player to the manager.

    PLAYER_PROTOCOL_VERSION     = 0x0A
    PLAYER_READY                = 0x0B
    PLAYER_PLAY                 = 0x0C
    PLAYER_METADATA             = 0x0D
    PLAYER_UNKNOWN              = 0x0E
    PLAYER_ERROR                = 0x0F
    PLAYER_MESSAGE              = 0x10
    PLAYER_DEBUG                = 0x11
    PLAYER_SUGGESTION           = 0x12
