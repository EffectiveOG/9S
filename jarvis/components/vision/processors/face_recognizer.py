# jarvis/components/vision/processors/face_recognizer.py

import face_recognition
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor
import asyncio
from jarvis.utils.logging_utils import get_logger

logger = get_logger(__name__)

class FaceRecognizer:
    """Face recognition processor using face_recognition library."""
    
    def __init__(self, known_faces_dir: str, recognition_threshold: float = 0.6):
        """Initialize face recognizer."""
        self.recognition_threshold = recognition_threshold
        self.known_faces_dir = Path(known_faces_dir)
        self.known_face_encodings = {}
        self.known_face_names = []
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.cache_encodings = True
        self.known_faces = self._load_known_faces(str(self.known_faces_dir))
        
    def _load_known_faces(self, faces_dir: str) -> Dict[str, List[np.ndarray]]:
        """Load and encode known faces from directory."""
        known_faces = {}
        logger.info(f"Attempting to load faces from: {faces_dir}")
        
        if not os.path.exists(faces_dir):
            logger.warning(f"Known faces directory not found: {faces_dir}")
            os.makedirs(faces_dir, exist_ok=True)
            return known_faces
            
        try:
            # Log directory contents
            logger.info(f"Directory contents: {os.listdir(faces_dir)}")
            
            for person_name in os.listdir(faces_dir):
                person_dir = os.path.join(faces_dir, person_name)
                if os.path.isdir(person_dir):
                    logger.info(f"Processing directory for person: {person_name}")
                    encodings = []
                    for image_file in os.listdir(person_dir):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(person_dir, image_file)
                            logger.info(f"Processing image: {image_path}")
                            encoding = self._encode_face(image_path)
                            if encoding is not None:
                                encodings.append(encoding)
                    if encodings:
                        known_faces[person_name] = encodings
                        logger.info(f"Loaded {len(encodings)} encodings for {person_name}")
                    else:
                        logger.warning(f"No valid encodings found for {person_name}")
            
            logger.info(f"Loaded {len(known_faces)} known faces")
            return known_faces
            
        except Exception as e:
            logger.error(f"Error loading known faces: {e}")
            return {}
        
    def _initialize_face_encodings(self):
        """Load known face encodings from cache or create new ones."""
        cache_file = self.known_faces_dir / "encodings_cache.pkl"
        
        if self.cache_encodings and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.known_face_encodings = cached_data['encodings']
                    self.known_face_names = cached_data['names']
                logger.info(f"Loaded {len(self.known_face_names)} face encodings from cache")
                return
            except Exception as e:
                logger.error(f"Failed to load face encodings cache: {e}")
        
        # Create new encodings
        self._load_face_encodings()
        
        # Save to cache if enabled
        if self.cache_encodings:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'encodings': self.known_face_encodings,
                        'names': self.known_face_names
                    }, f)
                logger.info("Saved face encodings to cache")
            except Exception as e:
                logger.error(f"Failed to save face encodings cache: {e}")
    
    def _load_face_encodings(self):
        """Load and encode faces from the known_faces directory."""
        for person_dir in self.known_faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
                
            person_name = person_dir.name
            encodings = []
            
            # Process each image in person's directory
            for image_path in person_dir.glob("*.jpg"):
                try:
                    encoding = self._encode_face(str(image_path))
                    if encoding is not None:
                        encodings.append(encoding)
                except Exception as e:
                    logger.error(f"Error encoding face {image_path}: {e}")
            
            if encodings:
                self.known_face_encodings[person_name] = encodings
                self.known_face_names.append(person_name)
                
        logger.info(f"Loaded {len(self.known_face_names)} persons' face encodings")
    
    def _encode_face(self, image_path: str) -> Optional[np.ndarray]:
        """Encode a single face image."""
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            return encodings[0] if encodings else None
        except Exception as e:
            logger.error(f"Error encoding face from {image_path}: {e}")
            return None
    
    async def identify(self, frame: np.ndarray) -> Dict[str, List]:
        """Identify faces in frame."""
        try:
            # Get face locations and encodings
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            # Match faces against known faces
            results = []
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = await self._match_face(face_encoding)
                results.append({
                    "location": face_location,
                    "identity": matches[0] if matches else "unknown",
                    "confidence": float(matches[1]) if matches else 0.0
                })
                
            return {"faces": results}
            
        except Exception as e:
            logger.error(f"Error identifying faces: {e}")
            return {"faces": []}
    
    async def _async_face_locations(self, frame: np.ndarray) -> List:
        """Get face locations asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            face_recognition.face_locations,
            frame
        )
    
    async def _async_face_encodings(self, frame: np.ndarray, locations: List) -> List:
        """Get face encodings asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            face_recognition.face_encodings,
            frame,
            locations
        )
    
    async def _match_face(self, face_encoding: np.ndarray) -> Optional[Tuple[str, float]]:
        """Match a face encoding against known faces."""
        try:
            best_match = None
            best_confidence = 0.0
            
            for name, known_encodings in self.known_faces.items():
                # Calculate distances to all known encodings for this person
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                # Convert distance to confidence (0-1 scale)
                confidence = 1.0 - min(distances)
                
                if confidence > self.recognition_threshold and confidence > best_confidence:
                    best_match = name
                    best_confidence = confidence
            
            return (best_match, best_confidence) if best_match else None
            
        except Exception as e:
            logger.error(f"Error matching face: {e}")
            return None
