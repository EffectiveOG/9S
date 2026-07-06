# jarvis/components/memory/memory_component.py

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio
from ...core.base_component import BaseComponent
from ...core.message import Message
from ..vision.vision_component import VisionComponent
from .database.schemas import create_tables
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)

class MemoryComponent(BaseComponent):
    """Component for managing Jarvis's memory and data persistence."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize memory component.
        
        Args:
            config: Configuration dictionary containing:
                - db_path: Path to SQLite database
                - retention_days: Number of days to retain data
        """
        super().__init__("memory")
        self.config = config
        self.db_path = Path(config.get("db_path", "data/jarvis.db"))
        self.retention_days = config.get("retention_days", 30)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cleanup_task = None
    
    async def start(self):
        """Start memory component and initialize database."""
        await super().start()
        
        try:
            # Initialize database connection
            self.conn = sqlite3.connect(self.db_path)
            create_tables(self.conn)
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            logger.info("Memory component started successfully")
            
        except Exception as e:
            logger.error(f"Error starting memory component: {e}")
            raise
    
    async def stop(self):
        """Stop memory component and cleanup."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            
        if self.conn:
            self.conn.close()
            
        await super().stop()
        logger.info("Memory component stopped")
    
    async def process_message(self, message: Message):
        """Process incoming messages and store relevant data."""
        try:
            if message.message_type == "vision_update":
                await self._store_vision_data(message.data)
            elif message.message_type == "speech_recognized":
                await self._store_speech_data(message.data)
            elif message.message_type == "device_update":
                await self._store_device_data(message.data)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _store_vision_data(self, data: Dict[str, Any]):
        """Store vision detection results."""
        try:
            timestamp = datetime.now().isoformat()
            
            # Store object detections
            if "objects" in data:
                for obj in data["objects"]:
                    self.conn.execute(
                        """
                        INSERT INTO object_detections 
                        (timestamp, class_name, confidence, bbox, tracking_id)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            timestamp,
                            obj["class"],
                            obj["confidence"],
                            json.dumps(obj["bbox"]),
                            obj.get("track_id")
                        )
                    )
            
            # Store face detections
            if "faces" in data:
                for face in data["faces"]:
                    self.conn.execute(
                        """
                        INSERT INTO face_detections 
                        (timestamp, identity, confidence, location)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            timestamp,
                            face["identity"],
                            face["confidence"],
                            json.dumps(face["location"])
                        )
                    )
            
            # Store gesture detections
            if "gestures" in data:
                for gesture in data["gestures"]:
                    self.conn.execute(
                        """
                        INSERT INTO gesture_detections 
                        (timestamp, gesture_type, confidence, hand_index)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            timestamp,
                            gesture["gesture"],
                            gesture["confidence"],
                            gesture["hand_index"]
                        )
                    )
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing vision data: {e}")
            self.conn.rollback()
    
    async def get_recent_detections(self, 
                                  detection_type: str,
                                  minutes: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent detection results.
        
        Args:
            detection_type: Type of detection ('object', 'face', 'gesture')
            minutes: Time window in minutes
        
        Returns:
            List of detection results
        """
        try:
            table_name = f"{detection_type}_detections"
            timestamp = (datetime.now() - timedelta(minutes=minutes)).isoformat()
            
            cursor = self.conn.execute(
                f"SELECT * FROM {table_name} WHERE timestamp > ? ORDER BY timestamp DESC",
                (timestamp,)
            )
            
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                # Parse JSON fields
                if "bbox" in result:
                    result["bbox"] = json.loads(result["bbox"])
                if "location" in result:
                    result["location"] = json.loads(result["location"])
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving recent detections: {e}")
            return []
    
    async def _periodic_cleanup(self):
        """Periodically clean up old data."""
        while self.is_running:
            try:
                # Calculate cutoff date
                cutoff = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
                
                # Delete old records from all tables
                tables = ["object_detections", "face_detections", "gesture_detections"]
                for table in tables:
                    self.conn.execute(
                        f"DELETE FROM {table} WHERE timestamp < ?",
                        (cutoff,)
                    )
                
                self.conn.commit()
                logger.debug("Completed periodic data cleanup")
                
                # Wait for next cleanup cycle (daily)
                await asyncio.sleep(24 * 60 * 60)
                
            except Exception as e:
                logger.error(f"Error during periodic cleanup: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            stats = {}
            
            # Get record counts
            for table in ["object_detections", "face_detections", "gesture_detections"]:
                cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Get database size
            stats["database_size"] = self.db_path.stat().st_size
            
            # Get retention period
            stats["retention_days"] = self.retention_days
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {}