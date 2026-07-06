# jarvis/components/automation/controllers/light_controller.py

from typing import Dict, Any, Optional, Tuple
import asyncio
from .base_controller import BaseController
import aiohttp
from ....utils.logging_utils import get_logger

logger = get_logger(__name__)

class LightController(BaseController):
    """Controller for smart lights (supports Philips Hue, LIFX, etc.)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bridge_ip = config.get("bridge_ip")
        self.api_key = config.get("api_key")
        self.light_id = config.get("light_id")
        self.protocol = config.get("protocol", "hue")  # or "lifx"
        self.transition_time = config.get("transition_time", 400)  # ms
        self.session = None
        
        # Define scenes
        self.scenes = {
            "movie": {"brightness": 30, "color_temp": 2700},
            "reading": {"brightness": 100, "color_temp": 4000},
            "relaxing": {"brightness": 50, "color_temp": 2200},
            "energizing": {"brightness": 100, "color_temp": 6500}
        }
    
    async def start(self):
        """Initialize light connection."""
        self.session = aiohttp.ClientSession()
        try:
            state = await self.get_state()
            logger.info(f"Light controller started: {state}")
        except Exception as e:
            logger.error(f"Error initializing light controller: {e}")
    
    async def stop(self):
        """Cleanup light connection."""
        if self.session:
            await self.session.close()
    
    async def execute(self, action: str, value: Any = None) -> bool:
        """
        Execute light command.
        
        Supported actions:
        - power_on
        - power_off
        - power_toggle
        - set_brightness
        - set_color
        - set_color_temp
        - set_scene
        """
        try:
            if action == "power_on":
                return await self._set_state({"on": True})
            elif action == "power_off":
                return await self._set_state({"on": False})
            elif action == "power_toggle":
                state = await self.get_state()
                return await self.execute("power_on" if not state["on"] else "power_off")
            elif action == "set_brightness":
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    return await self._set_state({"brightness": int(value * 2.54)})
                return False
            elif action == "set_color":
                if isinstance(value, (tuple, list)) and len(value) == 3:
                    return await self._set_state({"rgb": value})
                return False
            elif action == "set_color_temp":
                if isinstance(value, (int, float)) and 2000 <= value <= 6500:
                    return await self._set_state({"ct": int((value - 2000) * 347/4500)})
                return False
            elif action == "set_scene":
                if isinstance(value, str) and value in self.scenes:
                    scene = self.scenes[value]
                    return await self._set_state({
                        "on": True,
                        "brightness": int(scene["brightness"] * 2.54),
                        "ct": int((scene["color_temp"] - 2000) * 347/4500)
                    })
                return False
            else:
                logger.warning(f"Unsupported light action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing light command: {e}")
            return False
    
    async def _set_state(self, state: Dict[str, Any]) -> bool:
        """Set light state."""
        if not self.bridge_ip or not self.api_key:
            return False
            
        try:
            state["transitiontime"] = self.transition_time // 100  # Convert to deciseconds
            
            if self.protocol == "hue":
                url = f"http://{self.bridge_ip}/api/{self.api_key}/lights/{self.light_id}/state"
                async with self.session.put(url, json=state) as response:
                    return response.status == 200
            elif self.protocol == "lifx":
                # Adjust for LIFX API
                url = f"https://api.lifx.com/v1/lights/id:{self.light_id}/state"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                async with self.session.put(url, headers=headers, json=state) as response:
                    return response.status == 200
                    
            return False
            
        except Exception as e:
            logger.error(f"Error setting light state: {e}")
            return False
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current light state."""
        if not self.bridge_ip or not self.api_key:
            return {"on": False, "brightness": 0, "color_temp": 2700}
            
        try:
            if self.protocol == "hue":
                url = f"http://{self.bridge_ip}/api/{self.api_key}/lights/{self.light_id}"
            else:  # lifx
                url = f"https://api.lifx.com/v1/lights/id:{self.light_id}"
                headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with self.session.get(url, headers=headers if self.protocol == "lifx" else {}) as response:
                if response.status == 200:
                    data = await response.json()
                    if self.protocol == "hue":
                        state = data["state"]
                        return {
                            "on": state["on"],
                            "brightness": int(state["bri"] / 2.54),
                            "color_temp": int(2000 + (state["ct"] * 4500/347))
                        }
                    else:  # lifx
                        return {
                            "on": data["power"] == "on",
                            "brightness": int(data["brightness"] * 100),
                            "color_temp": data["color"]["kelvin"]
                        }
                        
                return {"on": False, "brightness": 0, "color_temp": 2700}
                
        except Exception as e:
            logger.error(f"Error getting light state: {e}")
            return {"on": False, "brightness": 0, "color_temp": 2700}
