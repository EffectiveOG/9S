# jarvis/core/base_component.py

import asyncio
from typing import Dict, Any, Optional, Set
from abc import ABC, abstractmethod
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class BaseComponent(ABC):
    """Base class for all Jarvis components."""
    
    def __init__(self, name: str):
        """
        Initialize base component.
        
        Args:
            name: Component identifier
        """
        self.name = name
        self.is_running = False
        self.state: Dict[str, Any] = {}
        self.subscribers: Set[asyncio.Queue] = set()
        self._message_queue: Optional[asyncio.Queue] = None
        
    async def initialize(self) -> bool:
        """
        Initialize component. Called before start.
        
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Initializing component: {self.name}")
            return True
        except Exception as e:
            logger.error(f"Error initializing {self.name}: {e}")
            return False
    
    @abstractmethod
    async def start(self):
        """Start component operation."""
        self.is_running = True
        self._message_queue = asyncio.Queue()
        logger.info(f"Started component: {self.name}")
    
    @abstractmethod
    async def stop(self):
        """Stop component operation."""
        self.is_running = False
        self._message_queue = None
        logger.info(f"Stopped component: {self.name}")
    
    async def subscribe(self, queue: asyncio.Queue):
        """
        Subscribe to component messages.
        
        Args:
            queue: Subscriber's message queue
        """
        self.subscribers.add(queue)
        logger.debug(f"New subscriber to {self.name}")
    
    async def unsubscribe(self, queue: asyncio.Queue):
        """
        Unsubscribe from component messages.
        
        Args:
            queue: Subscriber's message queue
        """
        self.subscribers.remove(queue)
        logger.debug(f"Removed subscriber from {self.name}")
    
    async def publish(self, message: Any):
        """
        Publish message to all subscribers.
        
        Args:
            message: Message to publish
        """
        if not self.is_running:
            return
            
        for subscriber in self.subscribers:
            try:
                await subscriber.put(message)
            except Exception as e:
                logger.error(f"Error publishing message from {self.name}: {e}")
    
    async def receive(self) -> Optional[Any]:
        """
        Receive next message.
        
        Returns:
            Optional[Any]: Next message or None if component not running
        """
        if not self.is_running or not self._message_queue:
            return None
        return await self._message_queue.get()
    
    def update_state(self, updates: Dict[str, Any]):
        """
        Update component state.
        
        Args:
            updates: State updates to apply
        """
        self.state.update(updates)
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            "name": self.name,
            "running": self.is_running,
            "state": self.state,
            "subscribers": len(self.subscribers)
        }
    
    async def process_command(self, command: Dict[str, Any]) -> bool:
        """
        Process component command.
        
        Args:
            command: Command to process
        
        Returns:
            bool: Success status
        """
        try:
            command_type = command.get("type")
            if command_type == "start":
                await self.start()
                return True
            elif command_type == "stop":
                await self.stop()
                return True
            elif command_type == "status":
                await self.publish(self.status)
                return True
            else:
                logger.warning(f"Unknown command type for {self.name}: {command_type}")
                return False
        except Exception as e:
            logger.error(f"Error processing command in {self.name}: {e}")
            return False