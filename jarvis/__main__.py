# jarvis/__main__.py

import uvicorn
import os
from pathlib import Path

def main():
    # Ensure we're in the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Start the server
    uvicorn.run(
        "jarvis.web.server:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()