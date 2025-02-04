import asyncio
from .action_ids import ActionID
from ....domain.gymnasium_gomokurs.ports import ManagerInterface
from ....domain.gymnasium_gomokurs.models.game import *
from typing import Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TCPManagerInterface(ManagerInterface):
    PROTOCOL_VERSION = "0.1.0"

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self.loop = asyncio.get_event_loop()

        try:
            self.loop.run_until_complete(self._check_protocol_compatibility())
        except Exception as e:
            raise Exception(f"protocol compability check failed: {e}")
    
    def get_init_state(self) -> tuple[int, np.ndarray]:
        return self.loop.run_until_complete(self._receive_init_state())
    
    def get_opponent_turn(self) -> tuple[Move | None, bool | None]:
        return self.loop.run_until_complete(self._receive_opponent_turn())
    
    def notify_move(self, move: Move) -> None:
        return self.loop.run_until_complete(self._send_move(move))
    
    def notify_error(self, error_msg: str) -> None:
        return self.loop.run_until_complete(self._send_error(error_msg))

    async def _check_protocol_compatibility(self):
        data = bytearray(ActionID.PLAYER_PROTOCOL_VERSION)

        encoded_version = self.PROTOCOL_VERSION.encode("utf-8")

        data.append(len(encoded_version).to_bytes(4, 'big'))
        data.append(encoded_version)

        self.writer.write(data)
        await self.writer.drain()

        data = await self.reader.read(1)

        if not data:
            raise Exception("connection to manager closed before protocol validation")
        elif data[0] == ActionID.MANAGER_PROTOCOL_COMPATIBLE:
            logging.debug("manager recognized protocol as compatible")
            return
        elif data[0] == ActionID.MANAGER_UNKNOWN:
            raise Exception("manager does not know protocol compatibility check action")
        elif data[0] == ActionID.MANAGER_ERROR:
            data = self.socket.recv(4)
            payload_len = int.from_bytes(data, 'big')

            data = self.socket.recv(payload_len)
            error_msg = data.decode("utf-8")

            raise Exception(f"manager error: {error_msg}")
        else:
            raise Exception(f"unexpected manager action with ID {data[0]} at protocol compatibility validation")
    
    async def _receive_init_state(self) -> tuple[int, np.ndarray]:
        while True:        
            data = await self.reader.read(1)
            if not data:
                raise Exception("connection closed by the remote host")
            
            if data[0] == ActionID.MANAGER_START:
                size = self.loop.run_until_complete(self._start_handler())
                board = np.zeros((size, size))

                logging.debug(f"initialized board following START action")

                await self._send_readiness()
            elif data[0] == ActionID.MANAGER_TURN:
                move = self.loop.run_until_complete(self._turn_handler())

                if not board:
                    err = "unexpected TURN action before game initialization"
                    await self._send_error(err)
                    raise Exception(err)
                    
                board[move.x][move.y] = CellStatus.OPPONENT

                logging.debug(f"initialized state following TURN action")
                return (size, board)
            elif data[0] == ActionID.MANAGER_BEGIN:
                if not board:
                    err = "unexpected BEGIN action before game initialization"
                    await self._send_error(err)
                    raise Exception(err)
                
                logging.debug(f"initialized state following BEGIN action")
                return (size, board)
            elif data[0] == ActionID.MANAGER_BOARD:
                if not board:
                    err = "unexpected BOARD action before game initialization"
                    await self._send_error(err)
                    raise Exception(err)

                turns = await self._board_handler()

                for turn in turns:
                    board[turn.move.x][turn.move.y] = CellStatus.PLAYER if turn.field == RelativeField.OWN_STONE else CellStatus.OPPONENT
                
                logging.debug(f"initialized state following BOARD action")
                return (size, board)
            else:
                err = f"unexpected action with id {data[0]} before game initialization"
                await self._send_error(err)
                raise Exception(err)
    
    async def _receive_opponent_turn(self) -> tuple[Move | None, bool | None]:
        while True:        
            data = await self.reader.read(1)
            if not data:
                raise Exception("connection closed by the remote host")
            
            if data[0] == ActionID.MANAGER_TURN:
                move = await self._turn_handler()

                return (move, None)
            elif data[0] == ActionID.MANAGER_END:
                return None, True
            elif data[0] == ActionID.MANAGER_INFO:
                info = await self._info_handler()

                logging.info(f"received info: {info}")
                continue
            elif data[0] == ActionID.MANAGER_UNKNOWN:
                unknown = await self._unknown_handler()
                raise Exception(f"manager error: {unknown}")
            elif data[0] == ActionID.MANAGER_ERROR:
                error = await self._error_handler()
                raise Exception(f"manager error: {error}")
            elif data[0] == ActionID.MANAGER_ABOUT:
                await self._about_handler()

                logging.debug(f"send metadata following ABOUT action")
                continue
            else:
                err = f"unexpected action with id {data[0]} after game initialization"
                await self._send_error(err)
                raise Exception(err)

    async def _start_handler(self) -> int:
        data = await self.reader.read(1)

        size = int.from_bytes(data, 'big')

        return size

    async def _turn_handler(self) -> Move:
        data = await self.reader.read(2)

        move = Move(int.from_bytes(data[0], 'big'), int.from_bytes(data[1], 'big'))

        await self.opponent_turn_queue.put(move)

    async def _board_handler(self)-> list[RelativeTurn]:
        TURN_PACKET_SIZE = 3

        data = await self.reader.read(4)

        nb_turn = int.from_bytes(data, 'big')

        data = await self.reader.read(nb_turn * TURN_PACKET_SIZE)

        turns = [(x, y, field) for x, y, field in zip(data[0::3], data[1::3], data[2::3])]

        return turns

    async def _info_handler(self)-> str:
        data = self.socket.recv(4)
        payload_len = int.from_bytes(data, 'big')

        data = self.socket.recv(payload_len)
        info = data.decode("utf-8")

        return info
    
    async def _about_handler(self)-> None:
        await self._send_metadata({"name": "gymnasium-gomokurs"})
    
    async def _unknown_handler(self)-> str:
        data = self.socket.recv(4)
        payload_len = int.from_bytes(data, 'big')

        data = self.socket.recv(payload_len)
        unknown_msg = data.decode("utf-8")

        return unknown_msg

    async def _error_handler(self)-> str:
        data = self.socket.recv(4)
        payload_len = int.from_bytes(data, 'big')

        data = self.socket.recv(payload_len)
        error_msg = data.decode("utf-8")

        return error_msg

    async def _send_readiness(self) -> None:
        data = bytearray(ActionID.PLAYER_READY)

        self.writer.write(data)
        await self.writer.drain()

    async def _send_move(self, move) -> None:
        data = bytearray(ActionID.PLAYER_PLAY)

        data.append(move.x.to_bytes(1, 'big'))
        data.append(move.y.to_bytes(1, 'big'))

        self.writer.write(data)
        await self.writer.drain()

    async def _send_metadata(self, metadata: Dict[str, str]):
        data = bytearray(ActionID.PLAYER_METADATA)

        fmt_metadata = ",".join([f"{k}={v}" for k, v in metadata.items()])
        encoded_fmt_metadata = fmt_metadata.encode("utf-8")

        data.append(len(encoded_fmt_metadata).to_bytes(4, 'big'))
        data.append(encoded_fmt_metadata)

        self.writer.write(data)
        await self.writer.drain()

    async def _send_unknown(self, msg: str) -> None:
        data = bytearray(ActionID.PLAYER_UNKNOWN)

        encoded_msg = msg.encode("utf-8")

        data.append(len(encoded_msg).to_bytes(4, 'big'))
        data.append(encoded_msg)

        self.writer.write(data)
        await self.writer.drain()

    async def _send_error(self, msg: str) -> None:
        data = bytearray(ActionID.PLAYER_MESSAGE)

        encoded_msg = msg.encode("utf-8")

        data.append(len(encoded_msg).to_bytes(4, 'big'))
        data.append(encoded_msg)

        self.writer.write(data)
        await self.writer.drain()
