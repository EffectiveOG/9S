# jarvis/core/message.py

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class Message:
    """Message class for inter-component communication."""
    
    sender: str
    """Component that generated the message"""
    
    message_type: str
    """Type of message (e.g., 'vision_update', 'speech_recognized', 'device_update')"""
    
    data: Dict[str, Any]
    """Message payload"""
    
    timestamp: datetime = None
    """Message creation timestamp"""
    
    priority: int = 0
    """Message priority (0-10, higher is more important)"""
    
    id: Optional[str] = None
    """Unique message identifier"""
    
    correlation_id: Optional[str] = None
    """ID for correlating related messages"""
    
    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
            
        if self.id is None:
            self.id = f"{self.sender}-{self.timestamp.timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "id": self.id,
            "sender": self.sender,
            "type": self.message_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "correlation_id": self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary format."""
        return cls(
            sender=data["sender"],
            message_type=data["type"],
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=data.get("priority", 0),
            id=data.get("id"),
            correlation_id=data.get("correlation_id")
        )
    
    def __str__(self) -> str:
        """String representation of message."""
        return (f"Message(id={self.id}, type={self.message_type}, "
                f"sender={self.sender}, priority={self.priority})")
    
    def is_command(self) -> bool:
        """Check if message is a command."""
        return self.message_type.startswith("command_")
    
    def is_event(self) -> bool:
        """Check if message is an event."""
        return self.message_type.startswith("event_")
    
    def is_response(self) -> bool:
        """Check if message is a response."""
        return self.message_type.startswith("response_")
    
    def create_response(self, data: Dict[str, Any]) -> 'Message':
        """Create a response message to this message."""
        return Message(
            sender=self.sender,
            message_type=f"response_{self.message_type}",
            data=data,
            correlation_id=self.id,
            priority=self.priority
        )