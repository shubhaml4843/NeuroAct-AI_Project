"""Main entrypoint for NeuroAct AI."""
import uvicorn
from app.interface.fastapi_main import app

if __name__ == "__main__":
    print("🚀 Starting NeuroAct AI Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)