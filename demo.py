#!/usr/bin/env python3
"""
NeuroAct-AI Demo Server
Public showcase version with limited functionality
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any

app = FastAPI(
    title="NeuroAct-AI Demo",
    description="Multi-Agent AI Platform - Public Showcase",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str
    agent: str = "auto"

class DemoResponse(BaseModel):
    status: str
    message: str
    agent_used: str
    demo_output: Dict[str, Any]

# Demo responses for showcase
DEMO_RESPONSES = {
    "data": {
        "agent_used": "DataAgent",
        "demo_output": {
            "action": "Data analysis simulation",
            "result": "Dataset loaded: 1000 rows, 5 columns. Quality score: 95%",
            "insights": ["No missing values", "2 outliers detected", "Normal distribution"]
        }
    },
    "ml": {
        "agent_used": "MLAgent", 
        "demo_output": {
            "action": "ML model training simulation",
            "result": "Random Forest model trained. Accuracy: 94.2%",
            "metrics": {"precision": 0.94, "recall": 0.92, "f1": 0.93}
        }
    },
    "code": {
        "agent_used": "CodeAgent",
        "demo_output": {
            "action": "Code generation simulation",
            "result": "REST API generated with 5 endpoints",
            "features": ["CRUD operations", "Authentication", "Error handling"]
        }
    }
}

@app.get("/")
async def root():
    return {
        "message": "NeuroAct-AI Demo Server",
        "status": "running",
        "note": "This is a showcase version. Full functionality in private repository."
    }

@app.post("/chat", response_model=DemoResponse)
async def chat_demo(request: QueryRequest):
    """Demo chat endpoint with simulated responses"""
    
    query_lower = request.query.lower()
    
    # Simple keyword matching for demo
    if any(word in query_lower for word in ["data", "analyze", "csv", "dataset"]):
        demo_type = "data"
    elif any(word in query_lower for word in ["model", "predict", "ml", "train"]):
        demo_type = "ml"
    elif any(word in query_lower for word in ["code", "api", "function", "generate"]):
        demo_type = "code"
    else:
        demo_type = "data"  # default
    
    response_data = DEMO_RESPONSES[demo_type]
    
    return DemoResponse(
        status="success",
        message=f"Demo response for: {request.query}",
        agent_used=response_data["agent_used"],
        demo_output=response_data["demo_output"]
    )

@app.get("/agents")
async def list_agents():
    """List available agents in demo"""
    return {
        "available_agents": [
            "DataAgent - Data processing and analysis",
            "MLAgent - Machine learning and predictions", 
            "CodeAgent - Code generation and review",
            "VisualizationAgent - Charts and dashboards",
            "NLPAgent - Text processing and analysis"
        ],
        "note": "Demo versions with limited functionality"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "demo-1.0.0"}

if __name__ == "__main__":
    print("🚀 Starting NeuroAct-AI Demo Server...")
    print("📝 This is a showcase version")
    print("🔒 Full version available in private repository")
    uvicorn.run(app, host="0.0.0.0", port=8700)