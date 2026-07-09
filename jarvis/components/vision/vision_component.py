# jarvis/components/vision/vision_component.py

from typing import List, Dict, Any, Optional
import asyncio
import cv2
import numpy as np
from pathlib import Path
from ...core.base_component import BaseComponent
from ...core.message import Message
from .processors.object_detector import ObjectDetector
from .processors.face_recognizer import FaceRecognizer
from .processors.gesture_detector import GestureDetector
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)

class VisionComponent(BaseComponent):
    """Main vision component handling all visual processing tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vision component.
        
        Args:
            config: Configuration dictionary containing:
                - camera_index: Index of camera to use (default: 0)
                - frame_skip: Process every nth frame (default: 2)
                - frame_size: Tuple of (width, height) for camera resolution
                - yolo_model: YOLOv8 model type (default: "yolov8n.pt")
                - object_confidence: Confidence threshold for object detection
                - enable_tracking: Whether to enable object tracking
                - detect_objects: Enable/disable object detection
                - detect_faces: Enable/disable face recognition
                - detect_gestures: Enable/disable gesture detection
                - known_faces_dir: Directory containing known face images
                - device: Device to run models on ('cpu', 'mps', 'cuda', None for auto)
        """
        super().__init__("vision")
        self.config = config
        
        # Camera settings
        self.camera_index = config.get("camera_index", 0)
        self.frame_skip = config.get("frame_skip", 2)
        self.frame_size = config.get("frame_size", (1280, 720))
        self.frame_count = 0
        self.camera = None
        self.latest_frame = None  # most recent captured frame (for the GUI)

        # Camera orientation
        self.flip_horizontal = config.get("flip_horizontal", True)  # Mirror fix
        self.flip_vertical = config.get("flip_vertical", False)
        self.rotation = config.get("rotation", 0)  # 0, 90, 180, or 270 degrees
        
        # Initialize processors based on configuration
        if config.get("detect_objects", True):
            self.object_detector = ObjectDetector(
                model_type=config.get("yolo_model", "yolov8n.pt"),
                confidence_threshold=config.get("object_confidence", 0.25),
                iou_threshold=config.get("iou_threshold", 0.45),
                device=config.get("device", None)
            )
            
            # Enable tracking if configured
            if config.get("enable_tracking", False):
                self.object_detector.enable_tracking()
        else:
            self.object_detector = None

        if config.get("detect_faces", True):
            self.face_recognizer = FaceRecognizer(
                known_faces_dir=config.get("known_faces_dir", "data/known_faces"),
                recognition_threshold=config.get("face_recognition_threshold", 0.6)
            )
        else:
            self.face_recognizer = None

        if config.get("detect_gestures", True):
            self.gesture_detector = GestureDetector()
        else:
            self.gesture_detector = None

        # Store processing flags
        self.detect_objects = config.get("detect_objects", True)
        self.detect_faces = config.get("detect_faces", True)
        self.detect_gestures = config.get("detect_gestures", True)
        
        # Initialize visualization settings
        self.show_preview = config.get("show_preview", False)
        self.preview_window_name = "Jarvis Vision"
        self.preview_scale = config.get("preview_scale", 1.0)

    def get_frame(self):
        """Return the most recently captured frame (or None). Used by the GUI."""
        return self.latest_frame

    async def process_frames(self):
        """Main processing loop for camera frames."""
        try:
            while self.camera and self.camera.isOpened():
                # Offload the blocking camera read so we don't starve the loop
                ret, frame = await asyncio.get_running_loop().run_in_executor(
                    None, self.camera.read
                )
                if not ret:
                    logger.error("Failed to read frame")
                    continue

                # Apply orientation transformations
                frame = self._process_frame_orientation(frame)
                self.latest_frame = frame  # expose for the GUI preview

                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    continue

                try:
                    # Process frame asynchronously
                    results = await self.process_single_frame(frame)
                    
                    # Visualize results if preview is enabled
                    if self.show_preview:
                        visualization = self.visualize_results(frame.copy(), results)
                        
                        # Apply preview scaling if needed
                        if self.preview_scale != 1.0:
                            height, width = visualization.shape[:2]
                            new_width = int(width * self.preview_scale)
                            new_height = int(height * self.preview_scale)
                            visualization = cv2.resize(
                                visualization, 
                                (new_width, new_height),
                                interpolation=cv2.INTER_AREA
                            )
                        
                        # Show frame and handle key events
                        cv2.imshow(self.preview_window_name, visualization)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("Quit command received")
                            break
                        elif key == ord('f'):
                            self.flip_horizontal = not self.flip_horizontal
                            logger.info(f"Horizontal flip: {self.flip_horizontal}")
                        elif key == ord('v'):
                            self.flip_vertical = not self.flip_vertical
                            logger.info(f"Vertical flip: {self.flip_vertical}")
                        elif key == ord('r'):
                            self.rotation = (self.rotation + 90) % 360
                            logger.info(f"Rotation: {self.rotation}")
                    
                    # Publish results
                    if results:
                        await self.publish_results(results)
                        
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
        except Exception as e:
            logger.error(f"Fatal error in process_frames: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Ensure proper cleanup
            if self.camera:
                self.camera.release()
            if self.show_preview:
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Give time for windows to close
            
    def _process_frame_orientation(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply orientation transformations to frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Processed frame with correct orientation
        """
        try:
            # Apply horizontal flip (mirror)
            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)
                
            # Apply vertical flip
            if self.flip_vertical:
                frame = cv2.flip(frame, 0)
                
            # Apply rotation
            if self.rotation != 0:
                if self.rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
            return frame
        except Exception as e:
            logger.error(f"Error processing frame orientation: {e}")
            return frame
    async def start(self):
        """Initialize camera and start vision processing."""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")
            
            # Configure camera
            width, height = self.frame_size
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Additional camera settings for better quality
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
            
            if self.show_preview:
                # Create window before the loop starts
                cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.preview_window_name, width, height)
                logger.info(f"Created preview window: {self.preview_window_name}")
            
            self.is_running = True
            logger.info("Vision component started successfully")
            # Run the capture loop as a background task so start() returns
            # instead of blocking the whole system in process_frames().
            self._frame_task = asyncio.create_task(self.process_frames())

        except Exception as e:
            logger.error(f"Error starting vision component: {e}")
            raise

    async def stop(self):
        """Clean up resources."""
        try:
            self.is_running = False
            if getattr(self, "_frame_task", None):
                self._frame_task.cancel()
                self._frame_task = None
            if self.camera:
                self.camera.release()
            if self.show_preview:
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Give time for windows to close
                logger.info("Closed preview window")
            logger.info("Vision component stopped")
        except Exception as e:
            logger.error(f"Error stopping vision component: {e}")


    async def process_single_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame with all enabled detectors."""
        results = {}
        
        try:
            # Create tasks for enabled processors
            tasks = []
            if self.detect_objects and self.object_detector:
                tasks.append(self.object_detector.detect(frame))
            if self.detect_faces and self.face_recognizer:
                tasks.append(self.face_recognizer.identify(frame))
            if self.detect_gestures and self.gesture_detector:
                tasks.append(self.gesture_detector.detect(frame))
                
            # Gather results from all processors
            if tasks:
                processed_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combine results, handling any exceptions
                for idx, result in enumerate(processed_results):
                    if isinstance(result, Exception):
                        logger.error(f"Detection error in processor {idx}: {result}")
                        continue
                    if isinstance(result, dict):
                        results.update(result)
                    else:
                        logger.warning(f"Unexpected result type from processor {idx}: {type(result)}")
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return results

    async def publish_results(self, results: Dict[str, Any]):
        """Publish detection results to other components."""
        message = Message(
            sender=self.name,
            message_type="vision_update",
            data=results
        )
        await self.publish(message)

    def visualize_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Visualize detection results on frame."""
        try:
            # Draw object detections
            if "objects" in results:
                for obj in results["objects"]:
                    bbox = obj["bbox"]
                    label = f"{obj['class']} {obj['confidence']:.2f}"
                    
                    # Ensure bbox coordinates are valid
                    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
                    x2 = min(frame.shape[1], int(bbox[2]))
                    y2 = min(frame.shape[0], int(bbox[3]))
                    
                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )
                    
                    # Draw label with background
                    label_size = cv2.getTextSize(
                        label, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        2
                    )[0]
                    
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        (0, 255, 0),
                        -1
                    )
                    
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )

            # Draw face detections
            if "faces" in results:
                for face in results["faces"]:
                    bbox = face["location"]
                    label = f"{face['identity']} {face['confidence']:.2f}"
                    
                    # Convert from face_recognition format and ensure valid coordinates
                    x1 = max(0, bbox[3])
                    y1 = max(0, bbox[0])
                    x2 = min(frame.shape[1], bbox[1])
                    y2 = min(frame.shape[0], bbox[2])
                    
                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )
                    
                    # Draw label with background
                    label_size = cv2.getTextSize(
                        label, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        2
                    )[0]
                    
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        (255, 0, 0),
                        -1
                    )
                    
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

            # Draw gesture detections
            if "gestures" in results:
                for gesture in results["gestures"]:
                    if "bounding_box" in gesture:
                        bbox = gesture["bounding_box"]
                        label = f"{gesture['gesture']} {gesture['confidence']:.2f}"
                        
                        # Ensure valid coordinates
                        x1 = max(0, int(bbox[0]))
                        y1 = max(0, int(bbox[1]))
                        x2 = min(frame.shape[1], int(bbox[2]))
                        y2 = min(frame.shape[0], int(bbox[3]))
                        
                        # Draw bounding box
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2
                        )
                        
                        # Draw label with background
                        label_size = cv2.getTextSize(
                            label, 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            2
                        )[0]
                        
                        cv2.rectangle(
                            frame,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            (0, 0, 255),
                            -1
                        )
                        
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2
                        )

            return frame
            
        except Exception as e:
            logger.error(f"Error visualizing results: {e}")
            return frame