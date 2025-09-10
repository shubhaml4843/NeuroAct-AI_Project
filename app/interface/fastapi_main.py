"""FastAPI endpoint server."""
from fastapi import FastAPI
from pydantic import BaseModel
from app.core.orchestrator import NeuroActOrchestrator
from app.utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="NeuroAct AI", version="0.1.0")
orchestrator = NeuroActOrchestrator()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "NeuroAct AI is running", "status": "ready"}

@app.post("/process")
async def process_query(request: QueryRequest):
    """Process user query through NeuroAct system."""
    try:
        result = orchestrator.process_query(request.query)
        return result
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        return {"status": "failed", "error": str(e)}

@app.get("/status")
async def system_status():
    """Get system status."""
    return orchestrator.get_system_status()

@app.get("/agents")
async def list_agents():
    """List available agents."""
    status = orchestrator.get_system_status()
    return {"agents": list(status["agents"].keys())}