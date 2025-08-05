"""Main entrypoint for NeuroAct AI."""
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting NeuroAct AI Server...")
    uvicorn.run("app.interface.fastapi_main:app", host="0.0.0.0", port=8000, reload=True)