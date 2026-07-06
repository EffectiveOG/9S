# tests/test_component.py

import asyncio
from jarvis.core import BaseComponent, Message


class TestComponent(BaseComponent):
    """A simple test component that echoes messages"""

    def __init__(self, name: str):
        super().__init__(name)
        self.message_count = 0

    async def initialize(self) -> bool:
        """Initialize the component."""
        self.logger.info(f"Initializing {self.name}")
        # Subscribe to messages during initialization
        await self.subscribe("TEST_MESSAGE")
        await self.subscribe("PING")
        return True

    async def start(self):
        """Start the component."""
        self.logger.info(f"Starting {self.name}")
        await asyncio.sleep(1)

    async def stop(self):
        """Stop the component."""
        self.logger.info(f"Stopping {self.name}")
        await asyncio.sleep(1)

    async def process_message(self, message: Message) -> None:
        """Process incoming messages."""
        self.message_count += 1
        self.logger.info(f"Received message #{self.message_count}: {message.message_type} from {message.source}")

        if message.message_type == "PING":
            # Respond to ping with pong
            await self.send_message("PONG", {
                "original_sender": message.source,
                "ping_count": self.message_count
            })
        elif message.message_type == "TEST_MESSAGE":
            # Echo the message
            await self.send_message("TEST_RESPONSE", {
                "original_message": message.data,
                "processed_by": self.name
            })