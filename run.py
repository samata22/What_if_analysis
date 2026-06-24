"""
run.py
Start the FastAPI server via uvicorn.
Usage: python run.py
"""
import uvicorn
from config.settings import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
