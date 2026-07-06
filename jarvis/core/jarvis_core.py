# jarvis/core/jarvis_core.py

import asyncio
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import json
import logging
from datetime import datetime
from .base_component import BaseComponent
from .message import Message
from ..components.vision.vision_component import VisionComponent
from ..components.audio.audio_component import AudioComponent
from ..components.memory.memory_component import MemoryComponent
from ..components.automation.automation_component import AutomationComponent
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class JarvisCore:
    """Core system coordinating all Jarvis components."""
    
    def __init__(self, config_path: str = "config/jarvis_config.json"):
        """
        Initialize Jarvis core system.
        
        Args:
            config_path: Path to main configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.components: Dict[str, BaseComponent] = {}
        self.event_queue = asyncio.Queue()
        self.state = {}
        
        # Command processing
        self.command_history = []
        self.command_queue = asyncio.Queue()
        self.max_command_history = self.config.get("max_command_history", 100)
        
        # Component initialization flags
        self.initialized_components: Set[str] = set()
        
        # System state
        self.is_running = False
        self.startup_time = None

        # Background tasks (processing loops + message-bus pumps)
        self._loop_tasks: List[asyncio.Task] = []
        self._pump_tasks: List[asyncio.Task] = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load main configuration file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
        
    def register_component(self, name: str, component: BaseComponent):
        """
        Register a new component with the system.
        
        Args:
            name: Unique name for the component.
            component: Instance of the component to register.
        """
        if name in self.components:
            logger.warning(f"Component '{name}' is already registered. Skipping.")
            return
        
        self.components[name] = component
        logger.info(f"Component '{name}' registered successfully.")
    
    async def start(self):
        """Start Jarvis system and all components."""
        try:
            logger.info("Starting Jarvis...")
            self.is_running = True
            self.startup_time = datetime.now()
            
            # Initialize components
            await self._init_components()

            # Start main processing loops as background tasks.
            # NOTE: this must NOT block, because callers (the web-server
            # lifespan and BackupManager.restart) `await jarvis.start()` and
            # expect it to return once the system is up.
            self._loop_tasks = [
                asyncio.create_task(self._process_events()),
                asyncio.create_task(self._process_commands()),
                asyncio.create_task(self._monitor_system()),
            ]
            logger.info("Jarvis started; processing loops running")

        except Exception as e:
            logger.error(f"Error starting Jarvis: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop Jarvis system and cleanup."""
        logger.info("Stopping Jarvis...")
        self.is_running = False

        # Cancel background tasks (processing loops + bus pumps)
        for task in self._loop_tasks + self._pump_tasks:
            task.cancel()
        self._loop_tasks = []
        self._pump_tasks = []

        # Stop all components
        for component in self.components.values():
            try:
                await component.stop()
            except Exception as e:
                logger.error(f"Error stopping component {component.name}: {e}")
        
        logger.info("Jarvis stopped")
    
    async def _init_components(self):
        """Initialize and start all components."""
        # Initialize vision component
        self.components["vision"] = VisionComponent(
            self.config.get("vision", {})
        )
        
        # Initialize audio component
        self.components["audio"] = AudioComponent(
            self.config.get("audio", {})
        )
        
        # Initialize memory component
        self.components["memory"] = MemoryComponent(
            self.config.get("memory", {})
        )
        
        # Initialize automation component
        self.components["automation"] = AutomationComponent(
            self.config.get("automation", {})
        )
        
        # Start all components
        for name, component in self.components.items():
            try:
                await component.start()
                self.initialized_components.add(name)
                logger.info(f"Component started: {name}")
            except Exception as e:
                logger.error(f"Error starting component {name}: {e}")

        # Connect producers (vision/audio) to consumers (memory/automation/core)
        self._wire_message_bus()

    def _wire_message_bus(self):
        """Route messages published by producers to interested consumers.

        Producers publish Message objects to any asyncio.Queue registered in
        their `subscribers` set. We register one queue per producer and pump
        its messages to the memory + automation handlers and the core event
        queue, so detections/speech actually flow through the system.
        """
        memory = self.components.get("memory")
        automation = self.components.get("automation")

        for producer_name in ("vision", "audio"):
            producer = self.components.get(producer_name)
            if producer is None:
                continue

            handlers = []
            if memory is not None and hasattr(memory, "process_message"):
                handlers.append(memory.process_message)
            if automation is not None and hasattr(automation, "handle_message"):
                handlers.append(automation.handle_message)
            handlers.append(self.event_queue.put)  # let the core react too

            queue: asyncio.Queue = asyncio.Queue()
            producer.subscribers.add(queue)  # subscribe() is async; register directly
            self._pump_tasks.append(
                asyncio.create_task(
                    self._pump_messages(producer_name, queue, handlers)
                )
            )
            logger.info(f"Wired message bus for producer: {producer_name}")

    async def _pump_messages(self, source: str, queue: "asyncio.Queue", handlers):
        """Deliver each message from a producer queue to all handlers."""
        while self.is_running:
            try:
                message = await queue.get()
            except asyncio.CancelledError:
                break
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Bus handler error ({source}): {e}")

    async def _process_events(self):
        """Process system events from components."""
        while self.is_running:
            try:
                message = await self.event_queue.get()
                await self._handle_event(message)
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _process_commands(self):
        """Process user commands."""
        while self.is_running:
            try:
                command = await self.command_queue.get()
                await self._handle_command(command)
            except Exception as e:
                logger.error(f"Error processing command: {e}")
    
    async def _monitor_system(self):
        """Monitor system health and performance."""
        while self.is_running:
            try:
                # Check component health
                for name, component in self.components.items():
                    if name not in self.initialized_components:
                        logger.warning(f"Component not initialized: {name}")
                        # Attempt recovery
                        await self._recover_component(name)
                
                # Update system state
                self.state.update({
                    "uptime": (datetime.now() - self.startup_time).total_seconds(),
                    "components": {name: "healthy" if name in self.initialized_components else "failed"
                                 for name in self.components},
                    "event_queue_size": self.event_queue.qsize(),
                    "command_queue_size": self.command_queue.qsize()
                })
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in system monitor: {e}")
    
    async def _handle_event(self, message: Message):
        """Handle system events."""
        try:
            event_type = message.message_type
            sender = message.sender
            data = message.data
            
            # Log event
            logger.debug(f"Event received: {event_type} from {sender}")
            
            # Handle different event types
            if event_type == "vision_update":
                await self._handle_vision_event(data)
            elif event_type == "speech_recognized":
                await self._handle_speech_event(data)
            elif event_type == "device_update":
                await self._handle_device_event(data)
            elif event_type == "pattern_update":
                await self._handle_pattern_event(data)
            
            # Update system state
            self.state["last_event"] = {
                "type": event_type,
                "sender": sender,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling event: {e}")
    
    async def _handle_command(self, command: Dict[str, Any]):
        """Handle user commands."""
        try:
            command_type = command["type"]
            data = command["data"]
            
            # Log command
            logger.debug(f"Command received: {command_type}")
            
            # Add to history
            self.command_history.append({
                "type": command_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
            
            # Trim history if needed
            if len(self.command_history) > self.max_command_history:
                self.command_history = self.command_history[-self.max_command_history:]
            
            # Process command
            if command_type == "scene_control":
                await self._handle_scene_command(data)
            elif command_type == "device_control":
                await self._handle_device_command(data)
            elif command_type == "system_control":
                await self._handle_system_command(data)
            
        except Exception as e:
            logger.error(f"Error handling command: {e}")
    
    async def _handle_vision_event(self, data: Dict[str, Any]):
        """Handle vision processing events."""
        # Process vision data and update state
        pass
    
    async def _handle_speech_event(self, data: Dict[str, Any]):
        """Handle speech recognition events."""
        text = data["text"].lower()
        
        # Check for system commands
        if "jarvis" in text:
            if "stop" in text or "shutdown" in text:
                await self.stop()
            elif "status" in text:
                await self._report_status()
            else:
                # Convert speech to command
                command = self._parse_speech_command(text)
                if command:
                    await self.command_queue.put(command)
    
    async def _handle_device_event(self, data: Dict[str, Any]):
        """Handle device state update events."""
        device_id = data["device_id"]
        state = data["state"]
        
        # Update device state
        self.state.setdefault("devices", {})[device_id] = state
        
        # Check for automation triggers
        automation = self.components.get("automation")
        if automation:
            await automation.check_triggers(device_id, state)
    
    async def _handle_pattern_event(self, data: Dict[str, Any]):
        """Handle pattern detection events."""
        # Process detected patterns
        pass
    
    async def _handle_scene_command(self, data: Dict[str, Any]):
        """Handle scene control commands."""
        automation = self.components.get("automation")
        if automation:
            if data["action"] == "activate":
                await automation.scene_manager.activate_scene(data["scene"])
            elif data["action"] == "deactivate":
                await automation.scene_manager.deactivate_scene()
    
    async def _handle_device_command(self, data: Dict[str, Any]):
        """Handle device control commands."""
        automation = self.components.get("automation")
        if automation:
            await automation.execute_command(data)
    
    async def _handle_system_command(self, data: Dict[str, Any]):
        """Handle system control commands."""
        action = data["action"]
        
        if action == "restart":
            await self.stop()
            await self.start()
        elif action == "status":
            await self._report_status()
    
    def _parse_speech_command(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse speech text into system command."""
        # Implement command parsing logic
        pass
    
    async def _report_status(self):
        """Report system status through speech."""
        status_text = f"Jarvis has been running for {int(self.state.get('uptime', 0))} seconds. "
        status_text += f"All components are {self._get_health_status()}. "
        
        audio = self.components.get("audio")
        if audio:
            await audio.speak(status_text)
    
    def _get_health_status(self) -> str:
        """Get system health status."""
        all_healthy = all(component in self.initialized_components 
                         for component in self.components)
        return "healthy" if all_healthy else "experiencing issues"
    
    async def _recover_component(self, component_name: str):
        """Attempt to recover failed component."""
        try:
            component = self.components[component_name]
            await component.stop()
            await component.start()
            self.initialized_components.add(component_name)
            logger.info(f"Recovered component: {component_name}")
        except Exception as e:
            logger.error(f"Failed to recover component {component_name}: {e}")

# Example usage:

async def main():
    jarvis = JarvisCore("config/jarvis_config.json")
    
    try:
        await jarvis.start()
        
        # Keep running until stopped
        while jarvis.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        await jarvis.stop()

if __name__ == "__main__":
    asyncio.run(main())