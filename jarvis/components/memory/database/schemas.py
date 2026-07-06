# jarvis/components/memory/database/schemas.py

def create_tables(conn):
    """Create database tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS object_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox TEXT NOT NULL,
            tracking_id INTEGER
        );
        
        CREATE TABLE IF NOT EXISTS face_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            identity TEXT NOT NULL,
            confidence REAL NOT NULL,
            location TEXT NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS gesture_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            gesture_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            hand_index INTEGER NOT NULL
        );
        
        CREATE INDEX IF NOT EXISTS idx_object_timestamp 
        ON object_detections(timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_face_timestamp 
        ON face_detections(timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_gesture_timestamp 
        ON gesture_detections(timestamp);
    """)