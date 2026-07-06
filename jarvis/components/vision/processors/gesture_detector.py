# jarvis/components/vision/processors/gesture_detector.py

import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2
from dataclasses import dataclass
from ....utils.logging_utils import get_logger
import asyncio


logger = get_logger(__name__)

@dataclass
class GestureThresholds:
    """Thresholds for gesture detection."""
    FINGER_EXTENSION: float = 0.1  # Minimum distance for finger extension
    FINGER_CURL: float = 0.02      # Maximum distance for finger curl
    ANGLE_TOLERANCE: float = 15.0  # Degrees
    MIN_CONFIDENCE: float = 0.75   # Minimum confidence to report gesture

class GestureDetector:
    """Gesture detection using MediaPipe Hands."""
    
    def __init__(self,
                 static_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1):
        """
        Initialize gesture detector.
        
        Args:
            static_mode: Whether to treat each frame independently
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            model_complexity: Model complexity (0, 1, or 2)
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands with explicit image mode
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        
        # Define gesture patterns with confidence thresholds
        self.gestures = {
            "open_palm": (self._check_open_palm, 0.8),
            "closed_fist": (self._check_closed_fist, 0.8),
            "pointing": (self._check_pointing, 0.75),
            "victory": (self._check_victory, 0.8),
            "thumbs_up": (self._check_thumbs_up, 0.85),
            "pinch": (self._check_pinch, 0.8),  # New gesture
            "swipe": (self._check_swipe, 0.75)  # New gesture
        }
        self.thresholds = GestureThresholds()
        self.last_hand_positions = {} 

        # Store frame dimensions
        self.frame_width = None
        self.frame_height = None
    
    def _initialize_frame_dimensions(self, frame: np.ndarray):
        """Initialize or update frame dimensions."""
        if frame is not None:
            height, width = frame.shape[:2]
            self.frame_width = int(width)
            self.frame_height = int(height)
            self.frame_dims = (height, width)
        else:
            raise ValueError("Invalid frame provided")

    def _ensure_frame_dimensions(self):
        """Ensure frame dimensions are initialized."""
        if self.frame_width is None or self.frame_height is None:
            raise RuntimeError("Frame dimensions not initialized")

    async def detect(self, frame: np.ndarray, visualize: bool = False) -> Dict[str, List]:
        """
        Detect and optionally visualize hand gestures.
        
        Args:
            frame: Input frame (BGR format)
            visualize: Whether to draw gesture annotations
            
        Returns:
            Dictionary containing detection results and optionally annotated frame
        """
        try:
            # Initialize frame dimensions
            self._initialize_frame_dimensions(frame)
            
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(frame_rgb)
            
            # Process detections
            gestures = []
            if results.multi_hand_landmarks:
                for hand_idx, (landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    try:
                        # Get gesture and tracking info
                        gesture_info = self._classify_gesture(landmarks)
                        track_info = self._track_hand_movement(hand_idx, landmarks)
                        
                        if gesture_info:
                            gesture_name, confidence = gesture_info
                            
                            gesture_data = {
                                "hand_index": hand_idx,
                                "gesture": gesture_name,
                                "confidence": confidence,
                                "landmarks": self._normalize_landmarks(landmarks),
                                "bounding_box": self._get_hand_bbox(landmarks),
                                "hand_side": handedness.classification[0].label,
                                "movement": track_info
                            }
                            
                            gestures.append(gesture_data)
                    except Exception as e:
                        logger.error(f"Error processing hand {hand_idx}: {e}")
                        continue
            
            result = {"gestures": gestures}
            
            # Add visualization if requested
            if visualize:
                try:
                    annotated_frame = self._visualize_detections(frame.copy(), gestures)
                    result["visualization"] = annotated_frame
                except Exception as e:
                    logger.error(f"Visualization error: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in gesture detection: {e}")
            return {"gestures": []}
    

    def _normalize_landmarks(self, landmarks) -> List[Dict[str, float]]:
        """Convert landmarks to normalized coordinate space."""
        self._ensure_frame_dimensions()
        height, width = self.frame_dims
        base_depth = float(width) * 0.1  # Approximate depth scale
        
        normalized = []
        for landmark in landmarks.landmark:
            try:
                normalized.append({
                    "x": float(landmark.x * width),
                    "y": float(landmark.y * height),
                    "z": float(landmark.z * base_depth),
                    "normalized_x": float(landmark.x),
                    "normalized_y": float(landmark.y),
                    "normalized_z": float(landmark.z)
                })
            except Exception as e:
                logger.error(f"Error normalizing landmark: {e}")
                continue
        
        return normalized
    
    def _track_hand_movement(self, hand_idx: int, landmarks) -> Dict[str, float]:
        """Track hand movement between frames."""
        current_pos = np.mean([[l.x, l.y] for l in landmarks.landmark], axis=0)
        
        if hand_idx in self.last_hand_positions:
            last_pos = self.last_hand_positions[hand_idx]
            movement = {
                "delta_x": float(current_pos[0] - last_pos[0]),
                "delta_y": float(current_pos[1] - last_pos[1]),
                "speed": float(np.linalg.norm(current_pos - last_pos))
            }
        else:
            movement = {"delta_x": 0.0, "delta_y": 0.0, "speed": 0.0}
        
        self.last_hand_positions[hand_idx] = current_pos
        return movement
    def _check_pinch(self, landmarks) -> float:
        """Detect pinch gesture (thumb and index finger close together)."""
        try:
            thumb_tip = landmarks.landmark[4]
            index_tip = landmarks.landmark[8]
            
            # Calculate distance between thumb and index tips
            distance = np.sqrt(
                (thumb_tip.x - index_tip.x)**2 +
                (thumb_tip.y - index_tip.y)**2
            )
            
            # Check if other fingers are extended
            other_tips = [landmarks.landmark[i] for i in [12, 16, 20]]
            other_pips = [landmarks.landmark[i] for i in [10, 14, 18]]
            others_curled = all(
                tip.y > pip.y for tip, pip in zip(other_tips, other_pips)
            )
            
            if distance < self.thresholds.FINGER_CURL and others_curled:
                confidence = 1.0 - (distance / self.thresholds.FINGER_CURL)
                return float(confidence)
            
            return 0.0
        except Exception:
            return 0.0
        
    def _check_swipe(self, landmarks) -> float:
        """Detect swipe gesture (flat hand moving sideways)."""
        try:
            # Check if hand is flat
            fingertips = [landmarks.landmark[i] for i in [8, 12, 16, 20]]
            mcp_joints = [landmarks.landmark[i] for i in [5, 9, 13, 17]]
            
            # All fingers should be extended and aligned
            y_vals = [tip.y for tip in fingertips]
            y_std = np.std(y_vals)
            
            fingers_aligned = y_std < 0.05
            fingers_extended = all(
                tip.y < joint.y for tip, joint in zip(fingertips, mcp_joints)
            )
            
            if fingers_aligned and fingers_extended:
                # Calculate confidence based on alignment
                confidence = 1.0 - min(1.0, y_std * 10)
                return float(confidence)
            
            return 0.0
        except Exception:
            return 0.0
    def _visualize_detections(self, frame: np.ndarray, gestures: List[Dict]) -> np.ndarray:
        """Draw detailed gesture annotations on frame."""
        for gesture in gestures:
            # Get hand information
            bbox = gesture["bounding_box"]
            landmarks = gesture["landmarks"]
            movement = gesture["movement"]
            hand_side = gesture["hand_side"]
            
            # Draw hand bounding box
            color = (0, 255, 0) if hand_side == "Right" else (255, 0, 0)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw gesture label with confidence
            label = f"{hand_side} {gesture['gesture']} ({gesture['confidence']:.2f})"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw hand landmarks
            for lm in landmarks:
                # Draw each landmark point
                pos = (int(lm["x"]), int(lm["y"]))
                cv2.circle(frame, pos, 3, (255, 0, 0), -1)
                
                # Draw depth information (size based on Z coordinate)
                radius = int(max(2, min(5, abs(lm["z"]) * 10)))
                cv2.circle(frame, pos, radius, (0, 0, 255), 1)
            
            # Draw movement vector if hand is moving
            if movement["speed"] > 0.01:
                # Calculate center of hand
                center = np.mean([[lm["x"], lm["y"]] for lm in landmarks], axis=0)
                center = tuple(map(int, center))
                
                # Calculate movement endpoint
                end_x = int(center[0] + movement["delta_x"] * 100)
                end_y = int(center[1] + movement["delta_y"] * 100)
                
                # Draw arrow showing movement direction and speed
                cv2.arrowedLine(frame, center, (end_x, end_y), (0, 255, 255), 2)
                
                # Draw speed indicator
                speed_label = f"Speed: {movement['speed']:.2f}"
                cv2.putText(frame, speed_label, (center[0], center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame

    
    def _get_hand_bbox(self, landmarks) -> List[int]:
        """Calculate hand bounding box from landmarks."""
        self._ensure_frame_dimensions()
        
        try:
            x_coords = [float(lm.x * self.frame_width) for lm in landmarks.landmark]
            y_coords = [float(lm.y * self.frame_height) for lm in landmarks.landmark]
            
            x1 = int(max(0, min(x_coords)))
            y1 = int(max(0, min(y_coords)))
            x2 = int(min(self.frame_width, max(x_coords)))
            y2 = int(min(self.frame_height, max(y_coords)))
            
            # Add padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(self.frame_width, x2 + padding)
            y2 = min(self.frame_height, y2 + padding)
            
            return [x1, y1, x2, y2]
            
        except Exception as e:
            logger.error(f"Error calculating bounding box: {e}")
            # Return safe default bbox
            return [0, 0, min(100, self.frame_width), min(100, self.frame_height)]
    
    def _convert_landmarks(self, landmarks) -> List[Dict[str, float]]:
        """Convert landmarks to pixel coordinates."""
        return [{
            "x": min(float(landmark.x * self.frame_width), self.frame_width - 1),
            "y": min(float(landmark.y * self.frame_height), self.frame_height - 1),
            "z": float(landmark.z)
        } for landmark in landmarks.landmark]
    
    def _calculate_bbox(self, landmarks: List[Dict[str, float]]) -> List[int]:
        """Calculate bounding box from landmarks."""
        x_coords = [lm["x"] for lm in landmarks]
        y_coords = [lm["y"] for lm in landmarks]
        
        # Add padding
        padding = 20
        x1 = max(0, int(min(x_coords)) - padding)
        y1 = max(0, int(min(y_coords)) - padding)
        x2 = min(self.frame_width, int(max(x_coords)) + padding)
        y2 = min(self.frame_height, int(max(y_coords)) + padding)
        
        return [x1, y1, x2, y2]
    
    def _classify_gesture(self, landmarks) -> Optional[Tuple[str, float]]:
        """Classify hand gesture based on landmarks."""
        try:
            # Get fingertip landmarks (indices: 4, 8, 12, 16, 20)
            fingertips = [landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]
            # Get finger bases (indices: 2, 5, 9, 13, 17)
            bases = [landmarks.landmark[i] for i in [2, 5, 9, 13, 17]]
            
            # Check fingers extended
            fingers_extended = [
                tip.y < base.y  # Compare y-coordinates
                for tip, base in zip(fingertips[1:], bases[1:])  # Skip thumb
            ]
            
            # Check thumb separately (compare x-coordinate)
            thumb_extended = (
                fingertips[0].x < bases[0].x  # For right hand
                if landmarks.landmark[0].x < landmarks.landmark[9].x  # Check if right hand
                else fingertips[0].x > bases[0].x  # For left hand
            )
            
            # Detect gestures
            if all(fingers_extended) and not thumb_extended:
                return "open_palm", 0.9
            elif not any(fingers_extended) and not thumb_extended:
                return "closed_fist", 0.9
            elif fingers_extended[0] and not any(fingers_extended[1:]) and not thumb_extended:
                return "pointing", 0.9
            elif fingers_extended[0] and fingers_extended[1] and not any(fingers_extended[2:]):
                return "victory", 0.9
            elif not any(fingers_extended) and thumb_extended:
                return "thumbs_up", 0.9
            
            return None
            
        except Exception as e:
            logger.error(f"Error classifying gesture: {e}")
            return None
    
    def _check_open_palm(self, landmarks) -> float:
        """
        Check for open palm gesture.
        All fingers extended and separated.
        """
        try:
            # Get fingertip landmarks
            fingertips = [landmarks.landmark[tip] for tip in [4, 8, 12, 16, 20]]
            
            # Get palm center (landmark 0)
            palm = landmarks.landmark[0]
            
            # Calculate distances from palm to fingertips
            distances = [
                ((tip.x - palm.x) ** 2 + (tip.y - palm.y) ** 2) ** 0.5
                for tip in fingertips
            ]
            
            # Check if all fingers are extended (distance > threshold)
            threshold = 0.1
            fingers_extended = all(d > threshold for d in distances)
            
            # Calculate confidence based on finger separation
            separation = min(
                abs(fingertips[i].x - fingertips[i+1].x)
                for i in range(len(fingertips)-1)
            )
            
            confidence = min(1.0, separation * 5.0) if fingers_extended else 0.0
            return confidence
            
        except Exception:
            return 0.0
    
    def _check_closed_fist(self, landmarks) -> float:
        """
        Check for closed fist gesture.
        All fingers curled inward.
        """
        try:
            # Get fingertip and middle joint landmarks
            fingertips = [landmarks.landmark[tip] for tip in [4, 8, 12, 16, 20]]
            mcp_joints = [landmarks.landmark[joint] for joint in [5, 9, 13, 17]]
            
            # Calculate how curled each finger is
            curl_amounts = []
            for tip, mcp in zip(fingertips[1:], mcp_joints):  # Skip thumb
                curl = (tip.y - mcp.y) > 0  # Fingertip below joint
                curl_amounts.append(curl)
            
            # Check thumb separately
            thumb_curl = (fingertips[0].x - landmarks.landmark[5].x) < 0
            curl_amounts.append(thumb_curl)
            
            # Calculate confidence based on number of curled fingers
            confidence = sum(curl_amounts) / len(curl_amounts)
            return confidence
            
        except Exception:
            return 0.0
    
    def _check_pointing(self, landmarks) -> float:
        """
        Check for pointing gesture.
        Index finger extended, others curled.
        """
        try:
            # Get index fingertip and joints
            index_tip = landmarks.landmark[8]
            index_pip = landmarks.landmark[6]
            
            # Get other fingertips
            other_tips = [landmarks.landmark[i] for i in [4, 12, 16, 20]]
            other_pips = [landmarks.landmark[i] for i in [3, 10, 14, 18]]
            
            # Check if index is extended
            index_extended = (index_tip.y < index_pip.y)
            
            # Check if other fingers are curled
            others_curled = all(
                tip.y > pip.y for tip, pip in zip(other_tips, other_pips)
            )
            
            if index_extended and others_curled:
                # Calculate confidence based on how straight the index finger is
                straightness = abs(index_tip.x - index_pip.x)
                confidence = 1.0 - min(1.0, straightness * 10)
                return confidence
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _check_victory(self, landmarks) -> float:
        """
        Check for victory sign gesture.
        Index and middle fingers extended in V shape.
        """
        try:
            # Get index and middle fingertips
            index_tip = landmarks.landmark[8]
            middle_tip = landmarks.landmark[12]
            
            # Get their base positions
            index_base = landmarks.landmark[5]
            middle_base = landmarks.landmark[9]
            
            # Calculate angle between fingers
            angle = np.arctan2(
                middle_tip.y - middle_base.y,
                middle_tip.x - middle_base.x
            ) - np.arctan2(
                index_tip.y - index_base.y,
                index_tip.x - index_base.x
            )
            angle = abs(np.degrees(angle))
            
            # Ideal V shape is about 30-45 degrees
            if 20 <= angle <= 60:
                confidence = 1.0 - abs(45 - angle) / 45
                return confidence
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _check_thumbs_up(self, landmarks) -> float:
        """
        Check for thumbs up gesture.
        Thumb extended upward, others curled.
        """
        try:
            # Get thumb tip and joint
            thumb_tip = landmarks.landmark[4]
            thumb_ip = landmarks.landmark[3]
            
            # Check if thumb is pointing upward
            thumb_up = (thumb_tip.y < thumb_ip.y)
            
            # Check if other fingers are curled
            other_tips = [landmarks.landmark[i] for i in [8, 12, 16, 20]]
            other_pips = [landmarks.landmark[i] for i in [6, 10, 14, 18]]
            others_curled = all(
                tip.y > pip.y for tip, pip in zip(other_tips, other_pips)
            )
            
            if thumb_up and others_curled:
                # Calculate confidence based on thumb angle
                angle = np.arctan2(
                    thumb_tip.y - thumb_ip.y,
                    thumb_tip.x - thumb_ip.x
                )
                angle = abs(np.degrees(angle))
                
                # Ideal angle is 90 degrees (vertical)
                confidence = 1.0 - abs(90 - angle) / 90
                return confidence
            
            return 0.0
            
        except Exception:
            return 0.0