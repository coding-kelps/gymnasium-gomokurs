import asyncio
from .action_ids import ActionID
from typing import Dict

class Interface():
    PROTOCOL_VERSION = "0.1.0"

    def __init__(self,
            manager_host: str = "localhost",
            manager_port: int = 49912):
        self.manager_host = manager_host
        self.manager_port = manager_port
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.manager_host, self.manager_port)

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

    async def listen(self):
        handlers = {
            ActionID.MANAGER_START:     self.__start_handler,
            ActionID.MANAGER_TURN:      self.__turn_handler,
            ActionID.MANAGER_BEGIN:     self.__begin_handler,
            ActionID.MANAGER_BOARD:     self.__board_handler,
            ActionID.MANAGER_INFO:      self.__info_handler,
            ActionID.MANAGER_END:       self.__end_handler,
            ActionID.MANAGER_ABOUT:     self.__about_handler,
            ActionID.MANAGER_UNKNOWN:   self.__unknown_handler,
            ActionID.MANAGER_ERROR:     self.__error_handler,
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
    
    async def __start_handler(self):
        data = await self.reader.read(1)

        board_size = int.from_bytes(data, 'big')

    async def __turn_handler(self):
        data = await self.reader.read(2)

        x = int.from_bytes(data[0], 'big')
        y = int.from_bytes(data[1], 'big')

    async def __begin_handler(self):
        return
    
    async def __board_handler(self):
        TURN_PACKET_SIZE = 3

        data = await self.reader.read(4)

        nb_turn = int.from_bytes(data, 'big')

        data = await self.reader.read(nb_turn * TURN_PACKET_SIZE)

        turns = [(x, y, field) for x, y, field in zip(data[0::3], data[1::3], data[2::3])]

    async def __info_handler(self):
        data = self.socket.recv(4)
        payload_len = int.from_bytes(data, 'big')

        data = self.socket.recv(payload_len)
        info = data.decode("utf-8")

    async def __end_handler(self):
        return
    
    async def __about_handler(self):
        return
    
    async def __unknown_handler(self):
        data = self.socket.recv(4)
        payload_len = int.from_bytes(data, 'big')

        data = self.socket.recv(payload_len)
        unknown_msg = data.decode("utf-8")

    async def __error_handler(self):
        data = self.socket.recv(4)
        payload_len = int.from_bytes(data, 'big')

        data = self.socket.recv(payload_len)
        error_msg = data.decode("utf-8")

    async def notify_readiness(self):
        data = bytearray(ActionID.PLAYER_READY)

        self.writer.write(data)
        await self.writer.drain()

    async def notify_move(self, move):
        data = bytearray(ActionID.PLAYER_PLAY)

        data.append(move.x.to_bytes(1, 'big'))
        data.append(move.y.to_bytes(1, 'big'))

        self.writer.write(data)
        await self.writer.drain()

    async def notify_metadata(self, metadata: Dict[str, str]):
        data = bytearray(ActionID.PLAYER_METADATA)

        fmt_metadata = ",".join([f"{k}={v}" for k, v in metadata.items()])
        encoded_fmt_metadata = fmt_metadata.encode("utf-8")

        data.append(len(encoded_fmt_metadata).to_bytes(4, 'big'))
        data.append(encoded_fmt_metadata)

        self.writer.write(data)
        await self.writer.drain()

    async def notify_unknown(self, msg: str):
        data = bytearray(ActionID.PLAYER_UNKNOWN)

        encoded_msg = msg.encode("utf-8")

        data.append(len(encoded_msg).to_bytes(4, 'big'))
        data.append(encoded_msg)

        self.writer.write(data)
        await self.writer.drain()

    async def notify_error(self, msg: str):
        data = bytearray(ActionID.PLAYER_ERROR)

        encoded_msg = msg.encode("utf-8")

        data.append(len(encoded_msg).to_bytes(4, 'big'))
        data.append(encoded_msg)

        self.writer.write(data)
        await self.writer.drain()

    async def notify_message(self, msg: str):
        data = bytearray(ActionID.PLAYER_MESSAGE)

        encoded_msg = msg.encode("utf-8")

        data.append(len(encoded_msg).to_bytes(4, 'big'))
        data.append(encoded_msg)

        self.writer.write(data)
        await self.writer.drain()

    async def notify_debug(self, msg: str):
        data = bytearray(ActionID.PLAYER_DEBUG)

        encoded_msg = msg.encode("utf-8")

        data.append(len(encoded_msg).to_bytes(4, 'big'))
        data.append(encoded_msg)

        self.writer.write(data)
        await self.writer.drain()

    async def notify_suggestion(self, move):
        data = bytearray(ActionID.PLAYER_SUGGESTION)

        data.append(move.x.to_bytes(1, 'big'))
        data.append(move.y.to_bytes(1, 'big'))

        self.writer.write(data)
        await self.writer.drain()
