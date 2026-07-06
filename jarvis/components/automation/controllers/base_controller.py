# jarvis/components/automation/controllers/base_controller.py

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from ....utils.logging_utils import get_logger

logger = get_logger(__name__)

class BaseController(ABC):
    """Base class for device controllers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device_id = config["id"]
        self.device_type = config["type"]
        self.name = config.get("name", self.device_id)
        self.state = {}
    
    async def start(self):
        """Initialize device connection."""
        pass
    
    async def stop(self):
        """Cleanup device connection."""
        pass
    
    @abstractmethod
    async def execute(self, action: str, value: Any = None) -> bool:
        """Execute device command."""
        pass
    
    @abstractmethod
    async def get_state(self) -> Dict[str, Any]:
        """Get current device state."""
        pass