# NeuroAct AI 🧠

A multi-agent AI system with reinforcement learning capabilities, featuring specialized agents for code generation, machine learning, data processing, and task evaluation.

## 🏗️ Architecture

![NeuroAct AI Architecture](./docs/NeuroAct.png)

*System architecture showing the multi-agent workflow with RLHF integration*

📊 **[Complete System Flowchart](./NeuroAct_Project_Flowchart.md)** - Detailed workflow diagrams and architecture documentation

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
   - Server: http://localhost:8700
   - Docs: http://localhost:8700/docs

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
│   │   ├── ml_agent.py            # Traditional ML
│   │   ├── model_evaluation_agent.py # Model evaluation
│   │   ├── nlp_agent.py           # Natural language processing
│   │   ├── planner_agent.py       # Workflow orchestration
│   │   ├── RetrievalAgent.py      # External data fetching
│   │   └── VisualizationAgent.py  # Data visualization
│   ├── RLHF_Implementation/       # Reinforcement Learning from Human Feedback
│   │   ├── __init__.py
│   │   ├── feedback_collector.py  # User feedback collection
│   │   ├── feedback_models.py     # Data structures
│   │   ├── reward_model.py        # Reward model training
│   │   ├── policy_optimizer.py    # Policy optimization
│   │   └── integration_manager.py # Agent integration
│   ├── core/                      # Core logic
│   │   ├── __init__.py
│   │   ├── langgraph_flow.py      # LangGraph workflows
│   │   ├── orchestrator.py        # Main system coordinator
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

## 🤖 Available Agents (10 Specialized Agents)

- **DataAgent**: Complete data pipeline with loading, cleaning, statistical analysis, EDA, outlier detection, and normality testing
- **VisualizationAgent**: LLM-powered dynamic plot generation with matplotlib, seaborn, plotly, and intelligent insights
- **MLAgent**: Traditional machine learning with auto-model selection, feature engineering, and hyperparameter tuning
- **CodeAgent**: Secure code generation, execution, and review with comprehensive templates and best practices
- **NLPAgent**: Natural language processing with sentiment analysis, NER, summarization, and text embeddings
- **DeepLearningAgent**: Neural network training with CNN, RNN, LSTM, image annotation, and optimization features
- **ModelEvaluationAgent**: Model evaluation, optimization, benchmarking, and performance analysis
- **CriticAgent**: Smart query routing, multi-agent coordination, and quality review with improvement recommendations
- **PlannerAgent**: Advanced workflow orchestration with multi-dimensional analysis and intelligent agent selection
- **RetrievalAgent**: Multi-source search, RAG implementation, and knowledge retrieval with caching

## 🧠 RLHF Learning System

- **FeedbackCollector**: Advanced feedback collection with multi-agent performance tracking and analytics
- **RewardModel**: Neural reward model training from human preferences and feedback data
- **PolicyOptimizer**: Continuous system improvement through reinforcement learning from human feedback
- **IntegrationManager**: Seamless integration with existing agents for non-disruptive learning

## 🔧 Development

- **Seed vector store**: `python scripts/seed_vector_store.py`
- **Run pipeline**: `python scripts/run_pipeline.py`
- **Format code**: `black .`
- **Lint code**: `flake8 .`
