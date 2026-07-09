# jarvis/components/vision/processors/object_detector.py

from typing import Dict, List, Optional
import asyncio
import numpy as np
import torch
from ultralytics import YOLO
from ....utils.logging_utils import get_logger

logger = get_logger(__name__)

class ObjectDetector:
    """YOLOv8 based object detector optimized for M1/M2 Mac."""
    
    def __init__(self, 
                 model_type: str = "yolov8n.pt",  # nano model by default
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: Optional[str] = None):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_type: YOLOv8 model type ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt')
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run the model on ('cpu', 'mps', 'cuda', None for auto)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Automatically select the best available device if none specified
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'  # Use M1/M2 GPU
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        self.device = device
        logger.info(f"Using device: {device}")
        
        # Load model
        try:
            self.model = YOLO(model_type)
            logger.info(f"Loaded YOLOv8 model: {model_type}")
            
            # Model configuration for inference
            self.model.conf = confidence_threshold
            self.model.iou = iou_threshold
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

    async def detect(self, frame: np.ndarray) -> Dict[str, List]:
        """
        Detect objects in frame.
        
        Args:
            frame: numpy array of shape (H, W, C) in BGR format
            
        Returns:
            Dictionary containing detection results with format:
            {
                "objects": [
                    {
                        "class": str,
                        "confidence": float,
                        "bbox": [x1, y1, x2, y2],
                        "segmentation": np.array (if available),
                        "track_id": int (if tracking enabled)
                    },
                    ...
                ]
            }
        """
        try:
            # Run the blocking inference off the event loop.
            detections = await asyncio.to_thread(self._detect_sync, frame)
            return {"objects": detections}

        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return {"objects": []}

    def _detect_sync(self, frame: np.ndarray) -> List[Dict]:
        """Blocking YOLO inference (runs in a worker thread)."""
        results = self.model(
            frame,
            verbose=False,
            device=self.device,
            stream=True  # Enable streaming mode for better memory efficiency
        )

        detections = []
        for result in results:
            for box in result.boxes:
                detection = {
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                }
                if result.masks is not None:
                    detection["segmentation"] = result.masks[0].data
                if hasattr(box, 'id') and box.id is not None:
                    detection["track_id"] = int(box.id[0])
                detections.append(detection)
        return detections

    async def detect_with_tracking(self, frame: np.ndarray) -> Dict[str, List]:
        """
        Detect and track objects across frames.
        
        Similar to detect() but with tracking enabled.
        """
        try:
            results = self.model.track(
                frame,
                verbose=False,
                device=self.device,
                persist=True,  # Maintain tracking state
                conf=self.confidence_threshold,
                iou=self.iou_threshold
            )
            
            # Process results (similar to detect() but with tracking IDs)
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.id is None:
                        continue
                        
                    detection = {
                        "class": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),
                        "track_id": int(box.id[0])
                    }
                    detections.append(detection)
            
            return {"objects": detections}
            
        except Exception as e:
            logger.error(f"Error during object detection and tracking: {e}")
            return {"objects": []}

    def enable_tracking(self):
        """Enable object tracking for subsequent detections."""
        self.model.tracker = "bytetrack.yaml"  # You can also use 'botsort.yaml'
        logger.info("Object tracking enabled")

    def disable_tracking(self):
        """Disable object tracking."""
        self.model.tracker = None
        logger.info("Object tracking disabled")