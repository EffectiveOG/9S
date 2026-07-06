# jarvis/web/metrics.py

try:
    import psutil
except ImportError:  # optional dependency
    psutil = None
try:
    import GPUtil
except ImportError:  # optional dependency
    GPUtil = None
from datetime import datetime
from typing import Dict, Any, List

class MetricsCollector:
    """Collect system and Jarvis metrics."""
    
    def __init__(self, jarvis_core):
        self.jarvis = jarvis_core
        self.metrics_history = {
            "cpu": [],
            "memory": [],
            "gpu": [],
            "events": [],
            "commands": []
        }
        self.max_history_points = 100
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": self._get_system_metrics(),
            "jarvis": self._get_jarvis_metrics(),
            "components": self._get_component_metrics()
        }
        
        # Update history
        self._update_metrics_history(metrics)
        
        return metrics
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics."""
        if psutil is None:
            return {
                "cpu": {"usage_percent": 0, "frequency": 0, "temperature": 0},
                "memory": {"total": 0, "available": 0, "percent": 0, "used": 0},
                "gpu": [],
            }
        cpu_metrics = {
            "usage_percent": psutil.cpu_percent(interval=1),
            "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "temperature": psutil.sensors_temperatures().get('coretemp', [{}])[0].current
            if hasattr(psutil, 'sensors_temperatures') and psutil.sensors_temperatures().get('coretemp')
            else 0
        }
        
        memory = psutil.virtual_memory()
        memory_metrics = {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        }
        
        gpu_metrics = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_metrics.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature
                })
        except Exception:
            pass
        
        return {
            "cpu": cpu_metrics,
            "memory": memory_metrics,
            "gpu": gpu_metrics
        }
    
    def _get_jarvis_metrics(self) -> Dict[str, Any]:
        """Collect Jarvis-specific metrics."""
        return {
            "uptime": self.jarvis.state.get("uptime", 0),
            "event_queue_size": self.jarvis.event_queue.qsize(),
            "command_queue_size": self.jarvis.command_queue.qsize(),
            "components_healthy": len(self.jarvis.initialized_components),
            "components_total": len(self.jarvis.components)
        }
    
    def _get_component_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Collect metrics from each component."""
        metrics = {}
        
        # Vision metrics
        if "vision" in self.jarvis.components:
            vision = self.jarvis.components["vision"]
            metrics["vision"] = {
                "fps": vision.frame_count / max(1, self.jarvis.state.get("uptime", 1)),
                "detection_count": len(vision.state.get("objects", [])),
                "face_count": len(vision.state.get("faces", []))
            }
        
        # Audio metrics
        if "audio" in self.jarvis.components:
            audio = self.jarvis.components["audio"]
            metrics["audio"] = {
                "speech_detected": audio.is_speech_detected,
                "processing_latency": audio.state.get("processing_latency", 0)
            }
        
        # Memory metrics
        if "memory" in self.jarvis.components:
            memory = self.jarvis.components["memory"]
            metrics["memory"] = {
                "stored_events": memory.state.get("stored_events", 0),
                "database_size": memory.state.get("database_size", 0)
            }
        
        # Automation metrics
        if "automation" in self.jarvis.components:
            automation = self.jarvis.components["automation"]
            metrics["automation"] = {
                "active_devices": len(automation.controllers),
                "active_scene": automation.scene_manager.get_active_scene()
            }
        
        return metrics
    
    def _update_metrics_history(self, metrics: Dict[str, Any]):
        """Update metrics history."""
        timestamp = datetime.now().timestamp()
        
        # Update CPU history
        self.metrics_history["cpu"].append({
            "timestamp": timestamp,
            "value": metrics["system"]["cpu"]["usage_percent"]
        })
        
        # Update memory history
        self.metrics_history["memory"].append({
            "timestamp": timestamp,
            "value": metrics["system"]["memory"]["percent"]
        })
        
        # Update GPU history if available
        if metrics["system"]["gpu"]:
            self.metrics_history["gpu"].append({
                "timestamp": timestamp,
                "value": metrics["system"]["gpu"][0]["load"]
            })
        
        # Trim history if needed
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > self.max_history_points:
                self.metrics_history[key] = self.metrics_history[key][-self.max_history_points:]
