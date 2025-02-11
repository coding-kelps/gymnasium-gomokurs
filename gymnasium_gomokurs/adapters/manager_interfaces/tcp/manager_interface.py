from .action_ids import ActionID
from ....domain.gymnasium_gomokurs.ports import ManagerInterface
from ....domain.gymnasium_gomokurs.models.game import *
from typing import Dict
import numpy as np
import logging
import socket

logger = logging.getLogger(__name__)

def create_tcp_manager_interface_from_active_connection(host: str = "localhost", port: int = 49912):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    tcp = TcpManagerInterface(s)

    return tcp
    
def create_tcp_manager_interface_from_passive_connection(host: str = "localhost", port: int = 49912):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen()

    conn, addr = s.accept()
    logging.debug(f"accepted tcp connection with manager of address: {addr}")
    tcp = TcpManagerInterface(conn)

    return tcp

class TcpManagerInterface(ManagerInterface):
    PROTOCOL_VERSION = "0.2.0"

    def __init__(self, conn: socket.socket):
        self.conn = conn

        try:
            self._check_protocol_compatibility()
        except Exception as e:
            raise Exception(f"protocol compability check failed: {e}")
    
    def get_init_state(self) -> tuple[int, np.ndarray]:
        return self._receive_init_state()
    
    def get_opponent_turn(self) -> tuple[Move | None, bool | None]:
        return self._receive_opponent_turn()
    
    def notify_move(self, move: Move) -> None:
        return self._send_move(move)
    
    def notify_error(self, error_msg: str) -> None:
        return self._send_error(error_msg)

    def _check_protocol_compatibility(self):
        self.conn.sendall(ActionID.PLAYER_PROTOCOL_VERSION.value)

        encoded_version = self.PROTOCOL_VERSION.encode("utf-8")

        self.conn.sendall(len(encoded_version).to_bytes(4, 'big'))
        self.conn.sendall(encoded_version)

        data = self.conn.recv(1)
        if not data:
            raise Exception("connection to manager closed before protocol validation")
        
        if data == ActionID.MANAGER_PROTOCOL_COMPATIBLE.value:
            logging.debug("manager recognized protocol as compatible")
            return
        elif data == ActionID.MANAGER_UNKNOWN.value:
            raise Exception("manager does not know protocol compatibility check action")
        elif data == ActionID.MANAGER_ERROR.value:
            data = self.conn.recv(4)
            payload_len = int.from_bytes(data, 'big')

            data = self.conn.recv(payload_len)
            error_msg = data.decode("utf-8")

            raise Exception(f"manager error: {error_msg}")
        else:
            raise Exception(f"unexpected manager action with ID {data} at protocol compatibility validation")
    
    def _receive_init_state(self) -> tuple[int, np.ndarray]:
        board_initialized = False

        while True:        
            data = self.conn.recv(1)
            if not data:
                raise Exception("connection closed by the remote host")

            if data == ActionID.MANAGER_START.value:
                size = self._start_handler()
                board = np.zeros((size, size))
                board_initialized = True
                self.size = size

                logging.debug(f"initialized board following START action")

                self._send_readiness()
            elif data == ActionID.MANAGER_RESTART.value:
                if not self.size:
                    err = "manager send RESTART action but board was never initialized"
                    self._send_error(err)
                    raise Exception(err)
                
                board = np.zeros((self.size, self.size))
                board_initialized = True

                self._send_readiness()
            elif data == ActionID.MANAGER_TURN.value:
                move = self._turn_handler()

                if not board_initialized:
                    err = "unexpected TURN action before game initialization"
                    self._send_error(err)
                    raise Exception(err)
                    
                board[move.x][move.y] = CellStatus.OPPONENT.value

                logging.debug(f"initialized state following TURN action")
                return (self.size, board)
            elif data == ActionID.MANAGER_BEGIN.value:
                if not board_initialized:
                    err = "unexpected BEGIN action before game initialization"
                    self._send_error(err)
                    raise Exception(err)
                
                logging.debug(f"initialized state following BEGIN action")
                return (self.size, board)
            elif data == ActionID.MANAGER_BOARD.value:
                if not board_initialized:
                    err = "unexpected BOARD action before game initialization"
                    self._send_error(err)
                    raise Exception(err)

                turns = self._board_handler()

                for turn in turns:
                    board[turn.move.x][turn.move.y] = CellStatus.PLAYER.value if turn.field == RelativeField.OWN_STONE.value else CellStatus.OPPONENT.value
                
                logging.debug(f"initialized state following BOARD action")
                return (self.size, board)
            else:
                err = f"unexpected action with id {data} before game initialization"
                self._send_error(err)
                raise Exception(err)
    
    def _receive_opponent_turn(self) -> tuple[Move | None, GameEnd | None, bool | None]:
        while True:
            data = self.conn.recv(1)
            if not data:
                raise Exception("connection closed by the remote host")
            
            if data == ActionID.MANAGER_TURN.value:
                move = self._turn_handler()

                return (move, None, None)
            elif data == ActionID.MANAGER_RESULT.value:
                game_end = self._result_handler()

                return None, game_end, None
            elif data == ActionID.MANAGER_END.value:
                logging.debug("manager requested session termination")

                return None, None, True
            elif data == ActionID.MANAGER_INFO.value:
                info = self._info_handler()

                logging.info(f"received info: {info}")
                continue
            elif data == ActionID.MANAGER_UNKNOWN.value:
                unknown = self._unknown_handler()
                raise Exception(f"manager error: {unknown}")
            elif data == ActionID.MANAGER_ERROR.value:
                error = self._error_handler()
                raise Exception(f"manager error: {error}")
            elif data == ActionID.MANAGER_ABOUT.value:
                self._about_handler()

                logging.debug(f"send metadata following ABOUT action")
                continue
            else:
                err = f"unexpected action with id {data} after game initialization"
                self._send_error(err)
                raise Exception(err)

    def _start_handler(self) -> int:
        data = self.conn.recv(1)

        size = int.from_bytes(data, 'big')

        return size

    def _turn_handler(self) -> Move:
        data = self.conn.recv(2)

        move = Move(int.from_bytes(data[:1], 'big'), int.from_bytes(data[1:], 'big'))

        return move

    def _board_handler(self)-> list[RelativeTurn]:
        TURN_PACKET_SIZE = 3

        data = self.conn.recv(4)

        nb_turn = int.from_bytes(data, 'big')

        data = self.conn.recv(nb_turn * TURN_PACKET_SIZE)

        turns = []
        for i in range(nb_turn, step = 3):
            x       = int.from_bytes(data[i:i+1],   'big')
            y       = int.from_bytes(data[i+1:i+2], 'big')
            field   = int.from_bytes(data[i+2:i+3], 'big')

            turns.append((x, y, field))

        return turns

    def _info_handler(self) -> str:
        data = self.conn.recv(4)
        payload_len = int.from_bytes(data, 'big')

        data = self.conn.recv(payload_len)
        info = data.decode("utf-8")

        return info
    
    def _result_handler(self) -> GameEnd:
        data = self.conn.recv(1)
        result_value = int.from_bytes(data, 'big')

        if result_value == GameEnd.DRAW.value:
            return GameEnd.DRAW
        elif result_value == GameEnd.WIN.value:
            return GameEnd.WIN
        elif result_value == GameEnd.LOOSE.value:
            return GameEnd.LOOSE
        else:
            Exception("invalid result value")

    def _about_handler(self) -> None:
        self._send_metadata({"name": "gymnasium-gomokurs"})
    
    def _unknown_handler(self)-> str:
        data = self.conn.recv(4)
        payload_len = int.from_bytes(data, 'big')

        data = self.conn.recv(payload_len)
        unknown_msg = data.decode("utf-8")

        return unknown_msg

    def _error_handler(self)-> str:
        data = self.conn.recv(4)
        payload_len = int.from_bytes(data, 'big')

        data = self.conn.recv(payload_len)
        error_msg = data.decode("utf-8")

        return error_msg

    def _send_readiness(self) -> None:
        self.conn.sendall(ActionID.PLAYER_READY.value)

    def _send_move(self, move) -> None:
        data = (
            ActionID.PLAYER_PLAY.value +
            move.x.to_bytes(1, 'big') +
            move.y.to_bytes(1, 'big')
        )
        self.conn.sendall(data)


    def _send_metadata(self, metadata: Dict[str, str]) -> None:
        fmt_metadata = ",".join([f"{k}={v}" for k, v in metadata.items()])
        encoded_fmt_metadata = fmt_metadata.encode("utf-8")
        data = (
            ActionID.PLAYER_METADATA.value +
            len(encoded_fmt_metadata).to_bytes(4, 'big') +
            encoded_fmt_metadata
        )
        self.conn.sendall(data)


    def _send_unknown(self, msg: str) -> None:
        encoded_msg = msg.encode("utf-8")
        data = (
            ActionID.PLAYER_UNKNOWN.value +
            len(encoded_msg).to_bytes(4, 'big') +
            encoded_msg
        )
        self.conn.sendall(data)


    def _send_error(self, msg: str) -> None:
        encoded_msg = msg.encode("utf-8")
        data = (
            ActionID.PLAYER_MESSAGE.value +
            len(encoded_msg).to_bytes(4, 'big') +
            encoded_msg
        )
        self.conn.sendall(data)
