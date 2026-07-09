# jarvis/components/automation/controllers/blinds_controller.py

from typing import Any, Dict
import aiohttp
from .base_controller import BaseController
from ....utils.logging_utils import get_logger

logger = get_logger(__name__)


class BlindsController(BaseController):
    """Controller for motorized blinds/shades.

    Position is expressed as 0 (fully closed) .. 100 (fully open). If no
    ip_address is configured the controller behaves as a virtual device so
    scenes that reference blinds succeed gracefully instead of being skipped.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ip_address = config.get("ip_address")
        self.port = config.get("port", 80)
        self.api_key = config.get("api_key")
        self.position = 100  # assume open at startup
        self.session = None

    async def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        if self.session:
            await self.session.close()

    async def execute(self, action: str, value: Any = None) -> bool:
        """Supported actions: power_on/open, power_off/close, set_position."""
        try:
            if action in ("power_on", "open"):
                return await self._set_position(100)
            if action in ("power_off", "close"):
                return await self._set_position(0)
            if action == "set_position":
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    return await self._set_position(int(value))
                return False
            logger.warning(f"Unsupported blinds action: {action}")
            return False
        except Exception as e:
            logger.error(f"Error executing blinds command: {e}")
            return False

    async def _set_position(self, position: int) -> bool:
        self.position = position
        if not self.ip_address:
            return True  # virtual device: nothing to talk to, treat as success
        try:
            url = f"http://{self.ip_address}:{self.port}/api/position/{position}"
            headers = {"Authorization": self.api_key} if self.api_key else {}
            async with self.session.post(url, headers=headers) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Error setting blinds position: {e}")
            return False

    async def get_state(self) -> Dict[str, Any]:
        return {"position": self.position, "open": self.position > 0}
