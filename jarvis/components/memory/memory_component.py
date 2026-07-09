# jarvis/components/memory/memory_component.py

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
import asyncio
from ...core.base_component import BaseComponent
from ...core.message import Message
from .database.schemas import create_tables
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class MemoryComponent(BaseComponent):
    """Component for managing Jarvis's memory and data persistence.

    Every database access is serialized through an asyncio.Lock and executed
    via asyncio.to_thread, so the blocking sqlite3 calls never stall the event
    loop. The connection is opened with check_same_thread=False because it is
    used from worker threads.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("memory")
        self.config = config
        self.db_path = Path(config.get("db_path", "data/jarvis.db"))
        self.retention_days = config.get("retention_days", 30)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cleanup_task = None
        self._db_lock = asyncio.Lock()

    async def start(self):
        """Start memory component and initialize database."""
        await super().start()
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            await asyncio.to_thread(create_tables, self.conn)
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
            conn, self.conn = self.conn, None
            await asyncio.to_thread(conn.close)
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

    async def _execute(self, fn, *args):
        """Run a blocking DB function under the lock, off the event loop."""
        if self.conn is None:
            return None
        async with self._db_lock:
            return await asyncio.to_thread(fn, *args)

    # ---- writes -----------------------------------------------------------
    async def _store_vision_data(self, data: Dict[str, Any]):
        await self._execute(self._store_vision_sync, data)

    def _store_vision_sync(self, data: Dict[str, Any]):
        try:
            timestamp = datetime.now().isoformat()
            for obj in data.get("objects", []):
                self.conn.execute(
                    "INSERT INTO object_detections "
                    "(timestamp, class_name, confidence, bbox, tracking_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (timestamp, obj.get("class"), obj.get("confidence"),
                     json.dumps(obj.get("bbox")), obj.get("track_id")),
                )
            for face in data.get("faces", []):
                self.conn.execute(
                    "INSERT INTO face_detections "
                    "(timestamp, identity, confidence, location) VALUES (?, ?, ?, ?)",
                    (timestamp, face.get("identity"), face.get("confidence"),
                     json.dumps(face.get("location"))),
                )
            for gesture in data.get("gestures", []):
                self.conn.execute(
                    "INSERT INTO gesture_detections "
                    "(timestamp, gesture_type, confidence, hand_index) VALUES (?, ?, ?, ?)",
                    (timestamp, gesture.get("gesture"), gesture.get("confidence"),
                     gesture.get("hand_index")),
                )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error storing vision data: {e}")
            self.conn.rollback()

    async def _store_speech_data(self, data: Dict[str, Any]):
        await self._execute(self._store_speech_sync, data)

    def _store_speech_sync(self, data: Dict[str, Any]):
        try:
            self.conn.execute(
                "INSERT INTO speech_events (timestamp, text) VALUES (?, ?)",
                (datetime.now().isoformat(), data.get("text", "")),
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error storing speech data: {e}")
            self.conn.rollback()

    async def _store_device_data(self, data: Dict[str, Any]):
        await self._execute(self._store_device_sync, data)

    def _store_device_sync(self, data: Dict[str, Any]):
        try:
            self.conn.execute(
                "INSERT INTO device_events (timestamp, device_id, state) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), data.get("device_id"),
                 json.dumps(data.get("state"))),
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error storing device data: {e}")
            self.conn.rollback()

    # ---- reads ------------------------------------------------------------
    async def get_recent_detections(self, detection_type: str,
                                    minutes: int = 5) -> List[Dict[str, Any]]:
        """Get recent detection results for 'object', 'face', or 'gesture'."""
        result = await self._execute(self._get_recent_sync, detection_type, minutes)
        return result or []

    def _get_recent_sync(self, detection_type: str, minutes: int) -> List[Dict[str, Any]]:
        try:
            # Whitelist the table name (never interpolate untrusted input).
            allowed = {
                "object": "object_detections",
                "face": "face_detections",
                "gesture": "gesture_detections",
            }
            table_name = allowed.get(detection_type)
            if table_name is None:
                logger.warning(f"Unknown detection type: {detection_type}")
                return []
            timestamp = (datetime.now() - timedelta(minutes=minutes)).isoformat()
            cursor = self.conn.execute(
                f"SELECT * FROM {table_name} WHERE timestamp > ? ORDER BY timestamp DESC",
                (timestamp,),
            )
            columns = [d[0] for d in cursor.description]
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                if result.get("bbox"):
                    result["bbox"] = json.loads(result["bbox"])
                if result.get("location"):
                    result["location"] = json.loads(result["location"])
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Error retrieving recent detections: {e}")
            return []

    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        result = await self._execute(self._get_statistics_sync)
        return result or {}

    def _get_statistics_sync(self) -> Dict[str, Any]:
        try:
            stats = {}
            for table in ["object_detections", "face_detections", "gesture_detections"]:
                cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            stats["database_size"] = self.db_path.stat().st_size
            stats["retention_days"] = self.retention_days
            return stats
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {}

    # ---- maintenance ------------------------------------------------------
    async def _periodic_cleanup(self):
        """Periodically clean up old data (daily)."""
        while self.is_running:
            try:
                await self._execute(self._cleanup_sync)
                logger.debug("Completed periodic data cleanup")
                await asyncio.sleep(24 * 60 * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during periodic cleanup: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error

    def _cleanup_sync(self):
        cutoff = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
        tables = ["object_detections", "face_detections", "gesture_detections",
                  "speech_events", "device_events"]
        for table in tables:
            try:
                self.conn.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,))
            except sqlite3.OperationalError:
                pass  # table may not exist yet
        self.conn.commit()
