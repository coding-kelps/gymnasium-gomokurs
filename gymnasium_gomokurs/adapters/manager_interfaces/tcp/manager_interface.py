import asyncio
from .action_ids import ActionID
from ....domain.gymnasium_gomokurs.ports import ManagerInterface
from ....domain.gymnasium_gomokurs.models.game import *
from typing import Dict, NoReturn

class TCPManagerInterface(ManagerInterface):
    PROTOCOL_VERSION = "0.1.0"

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self.opponent_turn_queue = asyncio.Queue()
        self.game_finished_queue = asyncio.Queue(1)
        self.manager_error_queue = asyncio.Queue(1)
        self.loop = asyncio.get_event_loop()

        try:
            self.loop.run_until_complete(self._check_protocol_compatibility())
        except Exception as e:
            raise Exception(f"protocol compability check failed: {e}")

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

    def get_opponent_turn(self) -> Move:
        opponent_turn = self.loop.run_until_complete(self.opponent_turn_queue.get())

        return opponent_turn
    
    def is_game_finished(self) -> bool:
        try:
            _ = self.opponent_turn_queue.get_nowait()

            return True
        except asyncio.QueueEmpty:
            return False

    async def _listen(self) -> NoReturn:
        handlers = {
            ActionID.MANAGER_START:     self._start_handler,
            ActionID.MANAGER_TURN:      self._turn_handler,
            ActionID.MANAGER_BEGIN:     self._begin_handler,
            ActionID.MANAGER_BOARD:     self._board_handler,
            ActionID.MANAGER_INFO:      self._info_handler,
            ActionID.MANAGER_END:       self._end_handler,
            ActionID.MANAGER_ABOUT:     self._about_handler,
            ActionID.MANAGER_UNKNOWN:   self._unknown_handler,
            ActionID.MANAGER_ERROR:     self._error_handler,
        }

        try:
            while True:
                if not self.reader:
                    raise Exception("tcp reader not initialized")
                
                data = await self.reader.read(1)
                if not data:
                    raise Exception("connection closed by the remote host")
                
                handler = handlers.get(data[0])
                if not handler:
                    await self.notify_unknown(f"unknown action id {data[0]}")
                    raise Exception(f"received unknown action id {data[0]}")

                await handler()
        except asyncio.CancelledError:
            pass
    
    async def _start_handler(self)-> None:
        data = await self.reader.read(1)

        size = int.from_bytes(data, 'big')

        pass


    async def _turn_handler(self)-> None:
        data = await self.reader.read(2)

        move = Move(int.from_bytes(data[0], 'big'), int.from_bytes(data[1], 'big'))

        await self.opponent_turn_queue.put(move)

    async def _begin_handler(self)-> None:
        pass
    
    async def _board_handler(self)-> None:
        TURN_PACKET_SIZE = 3

        data = await self.reader.read(4)

        nb_turn = int.from_bytes(data, 'big')

        data = await self.reader.read(nb_turn * TURN_PACKET_SIZE)

        turns = [(x, y, field) for x, y, field in zip(data[0::3], data[1::3], data[2::3])]

        pass

    async def _info_handler(self)-> None:
        data = self.socket.recv(4)
        payload_len = int.from_bytes(data, 'big')

        data = self.socket.recv(payload_len)
        _ = data.decode("utf-8")

        pass

    async def _end_handler(self)-> None:
        await self.opponent_turn_queue.put(True)
    
    async def _about_handler(self)-> None:
        await self._notify_metadata({"name": "gymnasium-gomokurs"})
    
    async def _unknown_handler(self)-> None:
        data = self.socket.recv(4)
        payload_len = int.from_bytes(data, 'big')

        data = self.socket.recv(payload_len)
        unknown_msg = data.decode("utf-8")

        await self.manager_error_queue.put(unknown_msg)

    async def _error_handler(self)-> None:
        data = self.socket.recv(4)
        payload_len = int.from_bytes(data, 'big')

        data = self.socket.recv(payload_len)
        error_msg = data.decode("utf-8")

        await self.manager_error_queue.put(error_msg)

    async def _notify_readiness(self) -> None:
        data = bytearray(ActionID.PLAYER_READY)

        self.writer.write(data)
        await self.writer.drain()

    def notify_move(self, move) -> None:
        data = bytearray(ActionID.PLAYER_PLAY)

        data.append(move.x.to_bytes(1, 'big'))
        data.append(move.y.to_bytes(1, 'big'))

        self.writer.write(data)
        self.loop.run_until_complete(self.writer.drain())

    async def _notify_metadata(self, metadata: Dict[str, str]):
        data = bytearray(ActionID.PLAYER_METADATA)

        fmt_metadata = ",".join([f"{k}={v}" for k, v in metadata.items()])
        encoded_fmt_metadata = fmt_metadata.encode("utf-8")

        data.append(len(encoded_fmt_metadata).to_bytes(4, 'big'))
        data.append(encoded_fmt_metadata)

        self.writer.write(data)
        await self.writer.drain()

    async def _notify_unknown(self, msg: str) -> None:
        data = bytearray(ActionID.PLAYER_UNKNOWN)

        encoded_msg = msg.encode("utf-8")

        data.append(len(encoded_msg).to_bytes(4, 'big'))
        data.append(encoded_msg)

        self.writer.write(data)
        await self.writer.drain()

    async def notify_error(self, msg: str) -> None:
        data = bytearray(ActionID.PLAYER_MESSAGE)

        encoded_msg = msg.encode("utf-8")

        data.append(len(encoded_msg).to_bytes(4, 'big'))
        data.append(encoded_msg)

        self.writer.write(data)
        self.loop.run_until_complete(self.writer.drain())
