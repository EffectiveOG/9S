# tests/run_test.py

import asyncio
import logging
from jarvis.core import JarvisCore
from test_component import TestComponent


async def run_test():
    """Run the test for TestComponent."""
    # Create Jarvis core
    jarvis = JarvisCore()

    # Create test components
    sender = TestComponent("Sender")
    receiver = TestComponent("Receiver")

    # Register components using the new method
    jarvis.register_component("Sender", sender)
    jarvis.register_component("Receiver", receiver)

    # Start the system
    try:
        await jarvis.start()

        # Send test messages
        await sender.send_message("PING", {"test": "data"})
        await asyncio.sleep(1)

        await sender.send_message("TEST_MESSAGE", {
            "content": "Hello, Jarvis!",
            "timestamp": "2024-01-01 12:00:00"
        })
        await asyncio.sleep(1)

        # Send multiple rapid messages
        for i in range(5):
            await sender.send_message("TEST_MESSAGE", {
                "content": f"Rapid message {i}",
                "timestamp": "2024-01-01 12:00:00"
            })
            await asyncio.sleep(0.5)

        # Wait for messages to be processed
        await asyncio.sleep(2)

    finally:
        # Shutdown the system
        await jarvis.stop()

def main():
    """Entry point for running the test."""
    logging.basicConfig(level=logging.INFO)  # Set logging level for debugging
    asyncio.run(run_test())

if __name__ == "__main__":
    main()