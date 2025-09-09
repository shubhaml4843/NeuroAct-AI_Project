# NeuroAct AI 🧠

A multi-agent AI system with reinforcement learning capabilities, featuring specialized agents for code generation, machine learning, data processing, and task evaluation.

## 🏗️ Architecture

![NeuroAct AI Architecture](./docs/Architecture.png)

*System architecture showing the multi-agent workflow with Ollama integration*

### Core Components:
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
pip install pydantic networkx langgraph asyncio
pip install pandas numpy torch tensorflow scikit-learn
pip install matplotlib seaborn plotly streamlit
```

## 📁 Project Structure

```
NeuroAct-AI_Project/
├── app/
│   ├── agents/                    # Agent implementations
│   │   ├── __init__.py
│   │   ├── code_agent.py          # Code execution agent
│   │   ├── CriticAgent.py         # Quality review agent
│   │   ├── data_agent.py          # Data processing agent
│   │   ├── Deep_learning_Agent.py # Neural network training
│   │   ├── eval_agent.py          # Model evaluation
│   │   ├── ml_agent.py            # Traditional ML
│   │   ├── OptimizerAgent.py      # Hyperparameter tuning
│   │   ├── planner_agent.py       # Workflow orchestration
│   │   ├── RetrievalAgent.py      # External data fetching
│   │   └── VisualizationAgent.py  # Data visualization
│   ├── core/                      # Core logic
│   │   ├── __init__.py
│   │   ├── langgraph_flow.py      # LangGraph workflows
│   │   ├── reward_loop.py         # RL feedback system
│   │   └── task_parser.py         # Query parsing
│   ├── interface/                 # API and UI
│   │   ├── __init__.py
│   │   ├── fastapi_main.py        # FastAPI server
│   │   └── ui_helpers.py          # UI utilities
│   ├── mcp/                       # Task management
│   │   ├── mcp_dispatcher.py      # Task routing
│   │   └── mcp_schema.py          # Task definitions
│   ├── memory/                    # Memory systems
│   │   ├── embedder.py            # Text embeddings
│   │   ├── rag.py                 # RAG implementation
│   │   └── vector_store.py        # Vector database
│   ├── tools/                     # Specialized tools
│   │   ├── __init__.py
│   │   ├── data_utils.py          # Data utilities
│   │   ├── github_api.py          # GitHub integration
│   │   ├── json_editor.py         # JSON utilities
│   │   └── data_utils.py          # Data processing tools
│   └── interface/                 # FastAPI server
│       ├── __init__.py
│       ├── fastapi_main.py        # Main API server
│       └── ui_helpers.py          # UI response helpers
├── scripts/                       # Development utilities
│   ├── run_pipeline.py            # CLI pipeline runner
│   └── seed_vector_store.py       # Vector store seeding
├── docs/                          # Documentation
│   └── Architecture.png           # System architecture diagram
├── data/                          # Data storage (auto-created)
│   ├── vectors/                   # Vector embeddings
│   └── logs/                      # Application logs
├── .env                           # Environment variables
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── run.py                         # Main entry point
```

## 🤖 Available Agents

- **CodeAgent**: Secure code generation, execution, and review with subprocess isolation, pattern matching, and comprehensive templates for APIs, databases, algorithms, and data processing
- **MLAgent**: Traditional machine learning with scikit-learn, feature engineering, model selection, hyperparameter tuning, and performance evaluation
- **DataAgent**: Complete data pipeline with loading (CSV/JSON/Excel), cleaning, statistical analysis, transformation, and quality assessment
- **DeepLearningAgent**: Neural network training with PyTorch/TensorFlow, custom architectures, transfer learning, and distributed training support
- **EvalAgent**: Model evaluation and quality assessment with metrics calculation, performance analysis, and validation reporting
- **PlannerAgent**: Advanced workflow orchestration using NetworkX + LangGraph with DAG validation, parallel execution, and retry strategies
- **CriticAgent**: Code and model quality review with best practices validation, security analysis, and improvement recommendations
- **OptimizerAgent**: Hyperparameter optimization using Optuna, Bayesian optimization, and automated model tuning
- **RetrievalAgent**: External data fetching from APIs, databases, and web sources with caching and rate limiting
- **VisualizationAgent**: Data visualization and reporting with matplotlib, seaborn, plotly, and interactive dashboards


## 🔧 Development

- **Seed vector store**: `python scripts/seed_vector_store.py`
- **Run pipeline**: `python scripts/run_pipeline.py`
- **Format code**: `black .`
- **Lint code**: `flake8 .`
