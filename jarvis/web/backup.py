# jarvis/web/backup.py

import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _safe_extractall(tar: tarfile.TarFile, dest: Path):
    """Extract a tar archive, refusing any member that would write outside
    `dest` (protects against path-traversal / CVE-2007-4559)."""
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not (target == dest or str(target).startswith(str(dest) + os.sep)):
            raise ValueError(f"Unsafe path in archive: {member.name}")
    tar.extractall(dest)

class BackupManager:
    """Manage system backup and restore."""
    
    def __init__(self, jarvis_core):
        self.jarvis = jarvis_core
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
    
    async def create_backup(self) -> Optional[Path]:
        """Create system backup."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"jarvis_backup_{timestamp}.tar.gz"
            
            # Stop Jarvis temporarily
            was_running = self.jarvis.is_running
            if was_running:
                await self.jarvis.stop()
            
            try:
                with tarfile.open(backup_path, "w:gz") as tar:
                    # Backup configuration
                    tar.add("config", arcname="config")
                    
                    # Backup database
                    tar.add("data", arcname="data")
                    
                    # Backup known faces
                    if Path("known_faces").exists():
                        tar.add("known_faces", arcname="known_faces")
                    
                    # Save current state
                    state_file = self.backup_dir / "state.json"
                    with open(state_file, 'w') as f:
                        json.dump(self.jarvis.state, f)
                    tar.add(state_file, arcname="state.json")
                    state_file.unlink()
                
                logger.info(f"Backup created: {backup_path}")
                return backup_path
                
            finally:
                # Restart Jarvis if it was running
                if was_running:
                    await self.jarvis.start()
                    
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None
    
    async def restore_backup(self, backup_path: Path) -> bool:
        """Restore system from backup."""
        try:
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup not found: {backup_path}")
            
            # Stop Jarvis
            was_running = self.jarvis.is_running
            if was_running:
                await self.jarvis.stop()
            
            try:
                # Create temporary directory
                restore_dir = Path("restore_temp")
                restore_dir.mkdir(exist_ok=True)
                
                # Extract backup (guarded against path traversal)
                with tarfile.open(backup_path, "r:gz") as tar:
                    _safe_extractall(tar, restore_dir)
                
                # Restore configuration
                if (restore_dir / "config").exists():
                    shutil.rmtree("config", ignore_errors=True)
                    shutil.copytree(restore_dir / "config", "config")
                
                # Restore database
                if (restore_dir / "data").exists():
                    shutil.rmtree("data", ignore_errors=True)
                    shutil.copytree(restore_dir / "data", "data")
                
                # Restore known faces
                if (restore_dir / "known_faces").exists():
                    shutil.rmtree("known_faces", ignore_errors=True)
                    shutil.copytree(restore_dir / "known_faces", "known_faces")
                
                # Restore state
                if (restore_dir / "state.json").exists():
                    with open(restore_dir / "state.json", 'r') as f:
                        self.jarvis.state = json.load(f)
                
                logger.info(f"Restored backup: {backup_path}")
                return True
                
            finally:
                # Cleanup
                shutil.rmtree(restore_dir, ignore_errors=True)
                
                # Restart Jarvis if it was running
                if was_running:
                    await self.jarvis.start()
                    
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False