from typing import Dict

class ManagerInterface:
    async def listen(self):
        pass

    async def notify_readiness(self):
        pass

    async def notify_move(self, move):
        pass

    async def notify_metadata(self, metadata: Dict[str, str]):
        pass

    async def notify_unknown(self, msg: str):
        pass

    async def notify_error(self, msg: str):
        pass

    async def notify_error(self, msg: str):
        pass

    async def notify_debug(self, msg: str):
        pass

    async def notify_suggestion(self, move):
        pass
