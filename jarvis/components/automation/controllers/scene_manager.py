# jarvis/components/automation/scene_manager.py

from typing import Dict, Any, List, Optional
import asyncio
from pathlib import Path
import json
from ....utils.logging_utils import get_logger

logger = get_logger(__name__)

class SceneManager:
    """Manages automation scenes and device orchestration."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scene manager.
        
        Args:
            config: Configuration dictionary containing:
                - scenes_file: Path to scenes configuration file
                - controllers: Dictionary of device controllers
        """
        self.config = config
        self.scenes_file = Path(config.get("scenes_file", "config/scenes.json"))
        self.controllers = config["controllers"]
        self.scenes = self._load_scenes()
        
        # Track active scene
        self.active_scene = None
    
    def _load_scenes(self) -> Dict[str, Any]:
        """Load scene configurations."""
        try:
            if self.scenes_file.exists():
                with open(self.scenes_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading scenes: {e}")
            return {}
    
    async def activate_scene(self, scene_name: str) -> bool:
        """
        Activate a scene.
        
        Args:
            scene_name: Name of scene to activate
        
        Returns:
            bool: Success status
        """
        if scene_name not in self.scenes:
            logger.error(f"Unknown scene: {scene_name}")
            return False
            
        scene = self.scenes[scene_name]
        success = True
        
        # Execute scene actions in parallel
        tasks = []
        for device_id, actions in scene["devices"].items():
            if device_id in self.controllers:
                for action in actions:
                    tasks.append(
                        self.controllers[device_id].execute(
                            action["type"],
                            action.get("value")
                        )
                    )
        
        # Wait for all actions to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in scene activation: {result}")
                success = False
            elif not result:
                success = False
        
        if success:
            self.active_scene = scene_name
            logger.info(f"Activated scene: {scene_name}")
        
        return success
    
    async def deactivate_scene(self) -> bool:
        """Deactivate current scene."""
        if not self.active_scene:
            return True
            
        scene = self.scenes[self.active_scene]
        success = True
        
        # Execute scene exit actions
        tasks = []
        for device_id, actions in scene.get("exit_actions", {}).items():
            if device_id in self.controllers:
                for action in actions:
                    tasks.append(
                        self.controllers[device_id].execute(
                            action["type"],
                            action.get("value")
                        )
                    )
        
        # Wait for all actions to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        for result in results:
            if isinstance(result, Exception) or not result:
                success = False
        
        self.active_scene = None
        return success
    
    def get_active_scene(self) -> Optional[str]:
        """Get currently active scene name."""
        return self.active_scene
    
    async def toggle_scene(self, scene_name: str) -> bool:
        """Toggle scene on/off."""
        if self.active_scene == scene_name:
            return await self.deactivate_scene()
        return await self.activate_scene(scene_name)
    
    async def create_scene(self, name: str, device_states: Dict[str, List[Dict]]) -> bool:
        """
        Create a new scene from current device states.
        
        Args:
            name: Scene name
            device_states: Dictionary of device states and actions
        """
        try:
            self.scenes[name] = {
                "name": name,
                "devices": device_states,
                "exit_actions": self._generate_exit_actions(device_states)
            }
            
            # Save to file
            await self._save_scenes()
            logger.info(f"Created new scene: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating scene: {e}")
            return False
    
    def _generate_exit_actions(self, device_states: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Generate scene exit actions based on device states."""
        exit_actions = {}
        
        # Generate opposite actions for each device
        for device_id, actions in device_states.items():
            device_exit_actions = []
            for action in actions:
                if action["type"] == "power_on":
                    device_exit_actions.append({"type": "power_off"})
                elif action["type"] == "set_scene":
                    device_exit_actions.append({"type": "set_scene", "value": "default"})
                # Add more opposite actions as needed
            
            if device_exit_actions:
                exit_actions[device_id] = device_exit_actions
                
        return exit_actions
    
    async def _save_scenes(self):
        """Save scenes to configuration file."""
        try:
            with open(self.scenes_file, 'w') as f:
                json.dump(self.scenes, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving scenes: {e}")