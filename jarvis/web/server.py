# jarvis/web/server.py

from fastapi import FastAPI, WebSocket, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
import uvicorn
from typing import Dict, Any
import asyncio
from pathlib import Path

# Import internal modules using relative imports
from jarvis.core.jarvis_core import JarvisCore
from jarvis.utils.logging_utils import get_logger
from jarvis.web.security import SecurityManager
from jarvis.web.metrics import MetricsCollector
from jarvis.web.backup import BackupManager

# Setup logging
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    global jarvis, metrics_collector, backup_manager
    try:
        # Initialize core components
        logger.info("Initializing Jarvis components...")
        jarvis = JarvisCore("config/jarvis_config.json")
        
        # Initialize optional components with error handling
        try:
            metrics_collector = MetricsCollector(jarvis)
        except Exception as e:
            logger.warning(f"Failed to initialize metrics collector: {e}")
            metrics_collector = None

        try:
            backup_manager = BackupManager(jarvis)
        except Exception as e:
            logger.warning(f"Failed to initialize backup manager: {e}")
            backup_manager = None
        
        # Start Jarvis
        await jarvis.start()
        
        # Log initialization status for each component
        for component_name, component in jarvis.components.items():
            if component_name in jarvis.initialized_components:
                logger.info(f"Component initialized successfully: {component_name}")
            else:
                logger.warning(f"Component failed to initialize: {component_name}")
        
        logger.info("Jarvis web interface started successfully")
        
    except Exception as e:
        logger.error(f"Critical error starting Jarvis web interface: {str(e)}", exc_info=True)
        raise
    
    yield  # Server is running
    
    # Shutdown
    try:
        if jarvis:
            logger.info("Shutting down Jarvis...")
            await jarvis.stop()
            logger.info("Jarvis web interface stopped successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}", exc_info=True)

# Initialize FastAPI app
app = FastAPI(
    title="Jarvis Control Interface",
    description="Web interface for Jarvis AI Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS (Adjust allow_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - restrict in production
    # allow_credentials MUST be False when allow_origins is "*": the browser
    # rejects that combination, and auth here uses a Bearer header (not cookies).
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
jarvis: JarvisCore = None
security = SecurityManager()
metrics_collector = None
backup_manager = None
websocket_clients = set()

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve main dashboard page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return "<h1>Welcome to Jarvis</h1>"

@app.get("/api/status")
async def get_status():
    """Get current system status."""
    if not jarvis:
        raise HTTPException(status_code=503, detail="Jarvis not initialized")
    return {
        "status": "running" if jarvis.is_running else "stopped",
        "uptime": jarvis.state.get("uptime", 0),
        "components": jarvis.state.get("components", {}),
        "last_event": jarvis.state.get("last_event", {})
    }

@app.get("/api/components")
async def get_components():
    """Get status of all components."""
    if not jarvis:
        raise HTTPException(status_code=503, detail="Jarvis not initialized")
    return {
        name: {
            "status": "healthy" if name in jarvis.initialized_components else "failed",
            "type": component.__class__.__name__
        }
        for name, component in jarvis.components.items()
    }

@app.post("/api/command")
async def send_command(command: Dict[str, Any]):
    """Send command to Jarvis."""
    if not jarvis:
        raise HTTPException(status_code=503, detail="Jarvis not initialized")
    await jarvis.command_queue.put(command)
    return {"status": "success", "message": "Command sent"}

@app.get("/api/scenes")
async def get_scenes():
    """Get available scenes."""
    if not jarvis or "automation" not in jarvis.components:
        raise HTTPException(status_code=503, detail="Automation not available")
    automation = jarvis.components["automation"]
    return {
        "scenes": list(automation.scene_manager.scenes.keys()),
        "active_scene": automation.scene_manager.get_active_scene()
    }

@app.post("/api/scenes/{scene_name}/activate")
async def activate_scene(scene_name: str):
    """Activate a scene."""
    if not jarvis or "automation" not in jarvis.components:
        raise HTTPException(status_code=503, detail="Automation not available")
    automation = jarvis.components["automation"]
    success = await automation.scene_manager.activate_scene(scene_name)
    return {"status": "success" if success else "failed"}

@app.get("/api/devices")
async def get_devices():
    """Get status of all devices."""
    if not jarvis or "automation" not in jarvis.components:
        raise HTTPException(status_code=503, detail="Automation not available")
    automation = jarvis.components["automation"]
    return {
        device_id: controller.get_state()
        for device_id, controller in automation.controllers.items()
    }

async def broadcast_state():
    """Broadcast system state to all connected clients."""
    while True:
        if jarvis and websocket_clients:
            state = {
                "system": {
                    "status": "running" if jarvis.is_running else "stopped",
                    "uptime": jarvis.state.get("uptime", 0),
                    "components": jarvis.state.get("components", {})
                },
                "devices": jarvis.state.get("devices", {}),
                "last_event": jarvis.state.get("last_event", {})
            }
            for client in websocket_clients.copy():
                try:
                    await client.send_json(state)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")
                    websocket_clients.remove(client)
        await asyncio.sleep(1)  # Update every second

# Authentication endpoint
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Handle user login."""
    user = security.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = security.create_access_token(
        data={"sub": user.username}
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Metrics endpoint
@app.get("/api/metrics")
async def get_metrics(current_user: Dict = Depends(security.get_current_user)):
    """Get system metrics."""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics not available")
    return await metrics_collector.collect_metrics()

# Backup endpoints
@app.post("/api/backup")
async def create_backup(current_user: Dict = Depends(security.get_current_user)):
    """Create system backup."""
    if not backup_manager:
        raise HTTPException(status_code=503, detail="Backup not available")
    backup_path = await backup_manager.create_backup()
    if not backup_path:
        raise HTTPException(status_code=500, detail="Backup failed")
    return {"backup_file": str(backup_path)}

@app.post("/api/restore/{backup_file}")
async def restore_backup(
    backup_file: str,
    current_user: Dict = Depends(security.get_current_user)
):
    """Restore system from backup."""
    if not backup_manager:
        raise HTTPException(status_code=503, detail="Backup not available")
    # Prevent path traversal: only allow a bare filename inside backups/.
    backups_root = Path("backups").resolve()
    backup_path = (backups_root / backup_file).resolve()
    if backup_path.parent != backups_root or not backup_path.exists():
        raise HTTPException(status_code=400, detail="Invalid backup file")
    success = await backup_manager.restore_backup(backup_path)
    if not success:
        raise HTTPException(status_code=500, detail="Restore failed")
    return {"status": "success"}

@app.get("/api/backups")
async def list_backups(current_user: Dict = Depends(security.get_current_user)):
    """List available backups."""
    if not backup_manager:
        raise HTTPException(status_code=503, detail="Backup not available")
    backups = []
    for backup_file in backup_manager.backup_dir.glob("*.tar.gz"):
        backups.append({
            "filename": backup_file.name,
            "size": backup_file.stat().st_size,
            "created": backup_file.stat().st_mtime,
            "path": str(backup_file)
        })
    return {"backups": sorted(backups, key=lambda x: x["created"], reverse=True)}

# Enhanced WebSocket handling
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections with authentication."""
    try:
        await websocket.accept()
        # Add timeout for authentication
        try:
            auth_data = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=10.0  # 10-second timeout
            )
        except asyncio.TimeoutError:
            logger.warning("WebSocket authentication timeout")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        if not security.verify_token(auth_data.get("token")):
            logger.warning("Invalid WebSocket authentication token")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        websocket_clients.add(websocket)
        metrics_task = None
        try:
            # Start metrics broadcast for this client
            if metrics_collector:
                metrics_task = asyncio.create_task(broadcast_metrics(websocket))
            # Handle incoming messages
            while True:
                data = await websocket.receive_json()
                await handle_websocket_message(websocket, data)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if metrics_task:
                metrics_task.cancel()
            websocket_clients.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass

async def broadcast_metrics(websocket: WebSocket):
    """Broadcast metrics to specific client."""
    try:
        while True:
            if metrics_collector:
                metrics = await metrics_collector.collect_metrics()
                await websocket.send_json({
                    "type": "metrics",
                    "data": metrics
                })
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error broadcasting metrics: {e}")

async def handle_websocket_message(websocket: WebSocket, message: Dict):
    """Handle incoming WebSocket messages."""
    try:
        message_type = message.get("type")
        data = message.get("data", {})
        if message_type == "command":
            await jarvis.command_queue.put(data)
            await websocket.send_json({
                "type": "command_response",
                "data": {"status": "accepted"}
            })
        elif message_type == "get_metrics":
            if metrics_collector:
                metrics = await metrics_collector.collect_metrics()
                await websocket.send_json({
                    "type": "metrics",
                    "data": metrics
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Metrics collector not available"}
                })
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")
        await websocket.send_json({
            "type": "error",
            "data": {"message": str(e)}
        })

# Start background tasks
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(broadcast_state())

# Start server if run directly
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        ssl_keyfile="config/ssl/key.pem",
        ssl_certfile="config/ssl/cert.pem"
    )