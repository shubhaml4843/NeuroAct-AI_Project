"""FastAPI endpoint server."""
from fastapi import FastAPI

app = FastAPI(title="NeuroAct AI", version="0.1.0")

@app.get("/")
async def root():
    return {"message": "NeuroAct AI is running"}

@app.post("/task")
async def execute_task(task: dict):
    """Execute AI task."""
    return {"status": "completed", "result": "Task executed successfully"}

@app.get("/agents")
async def list_agents():
    """List available agents."""
    return {"agents": ["CodeAgent", "MLAgent", "DataAgent", "EvalAgent", "PlannerAgent"]}