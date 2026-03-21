"""Run FastAPI server."""
import uvicorn
from config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("main:app", host=s.api_host, port=s.api_port, reload=True)
