# tests/test_vision/test_vision_component.py

import pytest
import asyncio
import cv2
import numpy as np
from pathlib import Path
from jarvis.components.vision.vision_component import VisionComponent

@pytest.fixture
def test_config():
    """Provide test configuration for vision component."""
    return {
        "camera_index": 0,
        "frame_skip": 1,
        "frame_size": (640, 480),  # Smaller size for testing
        "show_preview": True,
        
        # YOLOv8 settings
        "yolo_model": "yolov8n.pt",  # Use nano model for faster testing
        "object_confidence": 0.25,
        "iou_threshold": 0.45,
        "enable_tracking": True,
        "device": "mps",  # Adjust based on available hardware
        
        # Component enables/disables
        "detect_objects": True,
        "detect_faces": True,
        "detect_gestures": True,
        
        # Face recognition settings
        "known_faces_dir": "tests/test_data/known_faces",
        "face_recognition_threshold": 0.6
    }

@pytest.fixture
def test_frame():
    """Provide a test frame for vision processing."""
    # Create a blank frame for testing
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.mark.asyncio
async def test_vision_component_initialization(test_config):
    """Test vision component initialization."""
    vision = VisionComponent(test_config)
    assert vision.name == "vision"
    assert vision.camera_index == test_config["camera_index"]
    assert vision.frame_skip == test_config["frame_skip"]

@pytest.mark.asyncio
async def test_vision_component_processing(test_config, test_frame):
    """Test vision component frame processing."""
    vision = VisionComponent(test_config)
    
    # Process a single frame
    results = await vision.process_single_frame(test_frame)
    
    # Check results structure
    assert isinstance(results, dict)
    if "objects" in results:
        assert isinstance(results["objects"], list)
    if "faces" in results:
        assert isinstance(results["faces"], list)
    if "gestures" in results:
        assert isinstance(results["gestures"], list)

@pytest.mark.asyncio
async def test_message_publishing(test_config, test_frame):
    """Test message publishing from vision component."""
    vision = VisionComponent(test_config)
    
    # Create a mock message receiver
    received_messages = []
    
    async def mock_receiver(message):
        received_messages.append(message)
    
    # Add mock receiver to subscribers
    vision.subscribe(mock_receiver)
    
    # Process frame and check if message was published
    await vision.process_single_frame(test_frame)
    
    assert len(received_messages) > 0
    assert received_messages[0].sender == "vision"
    assert received_messages[0].message_type == "vision_update"

def test_visualization(test_config, test_frame):
    """Test visualization of detection results."""
    vision = VisionComponent(test_config)
    
    # Create mock results
    mock_results = {
        "objects": [
            {
                "class": "person",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 200]
            }
        ],
        "faces": [
            {
                "location": (50, 150, 100, 100),
                "identity": "test_person",
                "confidence": 0.85
            }
        ]
    }
    
    # Test visualization
    visualized_frame = vision.visualize_results(test_frame.copy(), mock_results)
    assert isinstance(visualized_frame, np.ndarray)
    assert visualized_frame.shape == test_frame.shape