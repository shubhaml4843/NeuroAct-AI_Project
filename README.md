# NeuroAct AI 🧠

A multi-agent AI system with reinforcement learning capabilities, featuring specialized agents for code generation, machine learning, data processing, and task evaluation.

## 🏗️ Architecture

- **Agents**: Specialized AI agents for different domains
- **Core**: LangGraph workflow orchestration with ReAct + ToT planning
- **MCP**: Multi-agent Communication Protocol for inter-agent messaging
- **Memory**: RAG pipeline with vector store for knowledge retention
- **Tools**: Integrated tools for code execution, ML training, and data processing
- **Interface**: FastAPI server with REST endpoints

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export HUGGINGFACE_TOKEN="your-token-here"
   ```

3. **Run the server**:
   ```bash
   python run.py
   ```

4. **Access the API**:
   - Server: http://localhost:8000
   - Docs: http://localhost:8000/docs

## 🧪 Testing

Run tests with:
```bash
python -m pytest tests/
```

## 📁 Project Structure

```
neuroact-ai/
├── app/                    # Main application
│   ├── agents/            # Specialized AI agents
│   ├── core/              # Core workflow logic
│   ├── mcp/               # Multi-agent communication
│   ├── memory/            # RAG and vector store
│   ├── tools/             # Agent tools and utilities
│   └── interface/         # FastAPI server
├── tests/                 # Unit and integration tests
├── scripts/               # Development utilities
└── run.py                 # Main entry point
```

## 🤖 Available Agents

- **CodeAgent**: Code generation and review
- **MLAgent**: Machine learning model training
- **DataAgent**: Data processing and analysis
- **EvalAgent**: Quality assessment and feedback
- **PlannerAgent**: Task orchestration and planning

## 🔧 Development

- **Seed vector store**: `python scripts/seed_vector_store.py`
- **Run pipeline**: `python scripts/run_pipeline.py`
- **Format code**: `black .`
- **Lint code**: `flake8 .`