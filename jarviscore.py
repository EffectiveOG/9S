from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Dict, List, Optional, Any
import queue
import threading
import sqlite3
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class Message:
    """Base class for all internal messages between components"""
    timestamp: float
    source: str
    message_type: str
    data: Any

class ComponentBase(ABC):
    """Base class for all Jarvis components"""
    
    def __init__(self, name: str, message_queue: asyncio.Queue):
        self.name = name
        self.message_queue = message_queue
        self.running = False
        self.logger = logging.getLogger(name)
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the component"""
        pass
        
    @abstractmethod
    async def process_message(self, message: Message) -> None:
        """Process incoming messages"""
        pass
    
    async def send_message(self, message_type: str, data: Any) -> None:
        """Send a message to the central message queue"""
        message = Message(
            timestamp=datetime.now().timestamp(),
            source=self.name,
            message_type=message_type,
            data=data
        )
        await self.message_queue.put(message)
        
    async def run(self):
        """Main component loop"""
        self.running = True
        try:
            await self.initialize()
            while self.running:
                message = await self.message_queue.get()
                if message.message_type == "SHUTDOWN" and message.data.get("target") in [self.name, "all"]:
                    break
                await self.process_message(message)
        except Exception as e:
            self.logger.error(f"Error in component {self.name}: {str(e)}")
            raise
        finally:
            self.running = False

class MemoryManager:
    """Manages persistent storage and retrieval of Jarvis's memory"""
    
    def __init__(self, db_path: str = "jarvis_memory.db"):
        self.db_path = db_path
        self.conn = None
        self.setup_database()
    
    def setup_database(self):
        """Initialize the SQLite database with required tables"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create tables for different types of memories
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                interaction_type TEXT,
                data TEXT,
                context TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                preference_type TEXT,
                preference_value TEXT,
                last_updated REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                pattern_data TEXT,
                confidence REAL,
                last_used REAL
            )
        """)
        
        self.conn.commit()
    
    def store_interaction(self, interaction_type: str, data: dict, context: Optional[dict] = None):
        """Store an interaction in the database"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO interactions (timestamp, interaction_type, data, context) VALUES (?, ?, ?, ?)",
            (datetime.now().timestamp(), interaction_type, json.dumps(data), json.dumps(context) if context else None)
        )
        self.conn.commit()
    
    def get_recent_interactions(self, limit: int = 10) -> List[dict]:
        """Retrieve recent interactions"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        return [
            {
                "timestamp": row[1],
                "type": row[2],
                "data": json.loads(row[3]),
                "context": json.loads(row[4]) if row[4] else None
            }
            for row in cursor.fetchall()
        ]

class JarvisCore:
    """Main controller class for Jarvis"""
    
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.components: Dict[str, ComponentBase] = {}
        self.memory = MemoryManager()
        self.logger = logging.getLogger("JarvisCore")
        
    async def register_component(self, component: ComponentBase):
        """Register a new component with Jarvis"""
        self.components[component.name] = component
        self.logger.info(f"Registered component: {component.name}")
    
    async def start(self):
        """Start all components"""
        component_tasks = []
        for component in self.components.values():
            task = asyncio.create_task(component.run())
            component_tasks.append(task)
        
        await asyncio.gather(*component_tasks)
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        for component in self.components.values():
            await self.message_queue.put(Message(
                timestamp=datetime.now().timestamp(),
                source="JarvisCore",
                message_type="SHUTDOWN",
                data={"target": component.name}
            ))

# Example usage and setup
async def main():
    # Initialize core
    jarvis = JarvisCore()
    
    # Start Jarvis
    try:
        await jarvis.start()
    except KeyboardInterrupt:
        await jarvis.shutdown()

if __name__ == "__main__":
    asyncio.run(main())