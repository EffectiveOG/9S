# jarvis/components/automation/controllers/tv_controller.py

import asyncio
from typing import Dict, Any, Optional
import aiohttp
from .base_controller import BaseController
from ....utils.logging_utils import get_logger

logger = get_logger(__name__)

class TVController(BaseController):
    """Controller for smart TVs supporting HDMI-CEC or network control."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ip_address = config.get("ip_address")
        self.port = config.get("port", 8001)  # Default Samsung TV port
        self.auth_key = config.get("auth_key")
        self.input_sources = config.get("input_sources", {})
        self.volume_step = config.get("volume_step", 5)
        self.session = None
    
    async def start(self):
        """Initialize TV connection."""
        self.session = aiohttp.ClientSession()
        try:
            state = await self.get_state()
            logger.info(f"TV controller started: {state}")
        except Exception as e:
            logger.error(f"Error initializing TV controller: {e}")
    
    async def stop(self):
        """Cleanup TV connection."""
        if self.session:
            await self.session.close()
    
    async def execute(self, action: str, value: Any = None) -> bool:
        """
        Execute TV command.
        
        Supported actions:
        - power_on
        - power_off
        - power_toggle
        - set_input
        - set_volume
        - increase_volume
        - decrease_volume
        - mute
        - unmute
        """
        try:
            if action == "power_on":
                return await self._send_command("power/on")
            elif action == "power_off":
                return await self._send_command("power/off")
            elif action == "power_toggle":
                state = await self.get_state()
                return await self.execute("power_on" if not state["power"] else "power_off")
            elif action == "set_input":
                if value in self.input_sources:
                    return await self._send_command(f"source/{self.input_sources[value]}")
                return False
            elif action == "set_volume":
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    return await self._send_command(f"volume/{int(value)}")
                return False
            elif action == "increase_volume":
                state = await self.get_state()
                new_volume = min(100, state["volume"] + self.volume_step)
                return await self.execute("set_volume", new_volume)
            elif action == "decrease_volume":
                state = await self.get_state()
                new_volume = max(0, state["volume"] - self.volume_step)
                return await self.execute("set_volume", new_volume)
            elif action == "mute":
                return await self._send_command("mute/on")
            elif action == "unmute":
                return await self._send_command("mute/off")
            else:
                logger.warning(f"Unsupported TV action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing TV command: {e}")
            return False
    
    async def _send_command(self, endpoint: str) -> bool:
        """Send command to TV."""
        if not self.ip_address:
            return False
            
        try:
            url = f"http://{self.ip_address}:{self.port}/api/{endpoint}"
            headers = {"Authorization": self.auth_key} if self.auth_key else {}
            
            async with self.session.post(url, headers=headers) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Error sending TV command: {e}")
            return False
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current TV state."""
        if not self.ip_address:
            return {"power": False, "volume": 0, "muted": False, "source": None}
            
        try:
            url = f"http://{self.ip_address}:{self.port}/api/state"
            headers = {"Authorization": self.auth_key} if self.auth_key else {}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                return {"power": False, "volume": 0, "muted": False, "source": None}
                
        except Exception as e:
            logger.error(f"Error getting TV state: {e}")
            return {"power": False, "volume": 0, "muted": False, "source": None}

