# jarvis/components/automation/automation_component.py

from typing import Dict, Any, List, Optional
import asyncio
from collections import defaultdict
from pathlib import Path
import json
from ...core.base_component import BaseComponent
from ...core.message import Message
from .controllers.base_controller import BaseController
from .controllers.home_assistant_controller import HomeAssistantController
from .controllers.scene_manager import SceneManager
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)

class AutomationComponent(BaseComponent):
    """Main automation component handling device control and automation."""
    
    def __init__(self, config: Dict[str, Any], ha_client=None):
        """
        Initialize automation component.
        
        Args:
            config: Configuration dictionary containing:
                - devices: List of device configurations
                - rules_file: Path to automation rules file
                - command_timeout: Timeout for device commands (seconds)
                - retry_attempts: Number of retry attempts for failed commands
            ha_client: HomeAssistantClient that drives every device (None
                if Home Assistant isn't configured: no device is available)
        """
        super().__init__("automation")
        self.config = config
        self.ha_client = ha_client
        
        # Command configuration
        self.command_timeout = config.get("command_timeout", 5.0)
        self.retry_attempts = config.get("retry_attempts", 3)
        
        # Load automation rules
        self.rules_file = Path(config.get("rules_file", "config/automation_rules.json"))
        self.rules = self._load_rules()
        
        # Initialize device controllers
        self.controllers: Dict[str, BaseController] = {}
        self._init_controllers()

        # Scene orchestration (referenced by the web API and core).
        self.scene_manager = SceneManager({
            "scenes_file": config.get("scenes_file", "config/scenes.json"),
            "controllers": self.controllers,
        })

        # Command processing queues
        self.command_queue = asyncio.Queue()
        
        # State tracking
        self.device_states = defaultdict(dict)
        self.pending_commands = {}
        
    def _load_rules(self) -> Dict[str, Any]:
        """Load automation rules from file."""
        try:
            if self.rules_file.exists():
                with open(self.rules_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Rules file not found: {self.rules_file}")
                return {}
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            return {}
    
    def _init_controllers(self):
        """Create a Home Assistant controller for each configured device."""
        devices = self.config.get("devices", [])
        if devices and self.ha_client is None:
            logger.warning("Home Assistant not configured (set HA_URL and HA_TOKEN); "
                           f"{len(devices)} device(s) unavailable")
            return

        for device_config in devices:
            device_id = device_config.get("id")
            device_type = device_config.get("type", "home_assistant")
            if device_type != "home_assistant":
                logger.warning(f"Unsupported device type '{device_type}' for {device_id}; "
                               "declare it in Home Assistant and use type 'home_assistant'")
                continue
            try:
                self.controllers[device_id] = HomeAssistantController(device_config, self.ha_client)
                logger.info(f"Initialized Home Assistant controller for {device_id} "
                            f"({device_config.get('entity_id')})")
            except Exception as e:
                logger.error(f"Error initializing Home Assistant device {device_id}: {e}")
    
    async def start(self):
        """Start automation component and controllers."""
        try:
            await super().start()  # marks component running so publish() works

            # Start device controllers
            for controller in self.controllers.values():
                await controller.start()

            # Start command processor
            self.command_processor = asyncio.create_task(self._process_commands())

            # NOTE: message delivery is wired by JarvisCore._wire_message_bus,
            # which routes vision/audio messages to self.handle_message().

            logger.info("Automation component started successfully")
            
        except Exception as e:
            logger.error(f"Error starting automation component: {e}")
            raise
    
    async def stop(self):
        """Stop automation component and cleanup."""
        # Stop command processor
        if hasattr(self, 'command_processor'):
            self.command_processor.cancel()
        
        # Stop controllers
        for controller in self.controllers.values():
            await controller.stop()

        await super().stop()
        logger.info("Automation component stopped")
    
    async def handle_message(self, message: Message):
        """Handle incoming messages from other components."""
        if message.message_type == "vision_update":
            await self._handle_vision_update(message.data)
        elif message.message_type == "speech_recognized":
            await self._handle_speech_command(message.data)
        elif message.message_type == "presence_update":
            await self._check_area_rules(message.data)
    
    async def _handle_vision_update(self, vision_data: Dict[str, Any]):
        """Process vision updates for automation triggers."""
        # Check for gesture commands
        if "gestures" in vision_data:
            for gesture in vision_data["gestures"]:
                await self._process_gesture_command(gesture)
        
        # Check for occupancy-based rules
        if "objects" in vision_data:
            await self._check_occupancy_rules(vision_data["objects"])
    
    async def _handle_speech_command(self, speech_data: Dict[str, str]):
        """Process speech commands for device control."""
        command = self._parse_speech_command(speech_data["text"])
        if command:
            await self.execute_command(command)
    
    def _parse_speech_command(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse speech text into device command."""
        # Simple command parsing logic
        text = text.lower()
        command = {}
        
        # Check for device mentions
        for device_id in self.controllers:
            if device_id.lower() in text:
                command["device_id"] = device_id
                break
        
        # Check for common actions
        actions = {
            "turn on": "power_on",
            "turn off": "power_off",
            "increase": "increase",
            "decrease": "decrease",
            "set": "set",
            "switch to": "set_input"
        }
        
        for phrase, action in actions.items():
            if phrase in text:
                command["action"] = action
                break
        
        # Extract parameters
        if "set" in text or "switch to" in text:
            # Try to extract value/input
            words = text.split()
            try:
                value_index = words.index("to") + 1
                if value_index < len(words):
                    command["value"] = words[value_index]
            except ValueError:
                pass
        
        return command if "device_id" in command and "action" in command else None
    
    async def execute_command(self, command: Dict[str, Any]) -> bool:
        """
        Execute device command.
        
        Args:
            command: Dictionary containing:
                - device_id: Target device identifier
                - action: Command action
                - value: Optional command parameter
        
        Returns:
            bool: Success status
        """
        try:
            device_id = command["device_id"]
            if device_id not in self.controllers:
                logger.error(f"Unknown device: {device_id}")
                return False
            
            # Add to command queue
            await self.command_queue.put(command)
            return True
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return False
    
    async def _process_commands(self):
        """Process queued device commands."""
        while True:
            try:
                command = await self.command_queue.get()
                device_id = command["device_id"]
                action = command["action"]
                value = command.get("value")
                
                controller = self.controllers[device_id]
                
                # Execute command with retry logic
                for attempt in range(self.retry_attempts):
                    try:
                        success = await asyncio.wait_for(
                            controller.execute(action, value),
                            timeout=self.command_timeout
                        )
                        
                        if success:
                            # Update device state
                            self.device_states[device_id].update({
                                "last_action": action,
                                "last_value": value,
                                "status": "success"
                            })
                            break
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"Command timeout: {device_id} {action}")
                        continue
                    except Exception as e:
                        logger.error(f"Command error: {e}")
                        if attempt == self.retry_attempts - 1:
                            self.device_states[device_id]["status"] = "error"
                
                # Publish state update
                await self.publish(Message(
                    sender=self.name,
                    message_type="device_update",
                    data={
                        "device_id": device_id,
                        "state": self.device_states[device_id]
                    }
                ))
                
            except Exception as e:
                logger.error(f"Error in command processor: {e}")
    
    async def _process_gesture_command(self, gesture: Dict[str, Any]):
        """Process gesture for device control."""
        # Example gesture mappings
        gesture_commands = {
            "pointing": {"action": "power_toggle"},
            "thumbs_up": {"action": "increase"},
            "thumbs_down": {"action": "decrease"},
            "open_palm": {"action": "power_off"},
            "closed_fist": {"action": "power_on"}
        }
        
        gesture_type = gesture["gesture"]
        if gesture_type in gesture_commands:
            # Find nearest device based on context
            device_id = await self._get_gesture_target()
            if device_id:
                command = gesture_commands[gesture_type]
                command["device_id"] = device_id
                await self.execute_command(command)
    
    async def _get_gesture_target(self) -> Optional[str]:
        """Determine target device for gesture command based on context."""
        # Implementation to determine which device the user is gesturing at
        # Could use vision data, user position, etc.
        return list(self.controllers.keys())[0] if self.controllers else None
    
    async def _check_occupancy_rules(self, objects: List[Dict[str, Any]]):
        """Check and execute occupancy-based automation rules."""
        # Count people in scene
        person_count = sum(1 for obj in objects if obj["class"] == "person")
        
        # Check occupancy rules (rules bound to a Frigate area are handled by
        # _check_area_rules instead)
        for rule in self.rules.get("occupancy_rules", []):
            if "area" in rule:
                continue
            conditions = rule.get("conditions", {})
            if "min_occupancy" in conditions and person_count >= conditions["min_occupancy"]:
                await self._execute_rule_actions(rule["actions"])
            elif "max_occupancy" in conditions and person_count <= conditions["max_occupancy"]:
                await self._execute_rule_actions(rule["actions"])
    
    async def _check_area_rules(self, presence: Dict[str, Any]):
        """Run occupancy rules bound to a Frigate camera/zone ("area").

        Rules fire when a count *crosses* their threshold (e.g. 1 -> 0 for
        max_occupancy 0), not on every update while the condition holds.
        """
        count = presence.get("count", 0)
        previous = presence.get("previous")
        for rule in self.rules.get("occupancy_rules", []):
            if rule.get("area") != presence.get("area"):
                continue
            if rule.get("label", "person") != presence.get("label"):
                continue
            conditions = rule.get("conditions", {})
            low, high = conditions.get("min_occupancy"), conditions.get("max_occupancy")
            crossed_up = low is not None and count >= low and (previous is None or previous < low)
            crossed_down = high is not None and count <= high and (previous is None or previous > high)
            if crossed_up or crossed_down:
                logger.info(f"Rule triggered: {rule.get('name')} ({presence['area']}={count})")
                await self._execute_rule_actions(rule["actions"])

    async def _execute_rule_actions(self, actions: List[Dict[str, Any]]):
        """Execute automation rule actions."""
        for action in actions:
            device_id = action.get("device_id")
            if device_id in self.controllers:
                await self.execute_command({
                    "device_id": device_id,
                    "action": action["type"],
                    "value": action.get("value")
                })