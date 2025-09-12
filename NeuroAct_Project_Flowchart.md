# NeuroAct AI Project - Complete System Flowchart

## 🎯 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           NEUROACT AI SYSTEM                                   │
│                     Multi-Agent AI Platform with RLHF                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   FastAPI Web   │  │   Streamlit     │  │   Jupyter                        │
│  │   Interface     │  │   Dashboard     │  │   Notebooks     │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CORE ORCHESTRATION                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │  Orchestrator   │  │  Task Parser    │  │  LangGraph      │                │
│  │   (main.py)     │  │                 │  │   Flow          │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         INTELLIGENT ROUTING                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │  CriticAgent    │  │  PlannerAgent   │  │  MCP Dispatcher │                │
│  │  (Coordinator)  │  │  (Smart Route)  │  │  (Messages)     │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          10 SPECIALIZED AGENTS                                 │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   DataAgent     │  │ VisualizationA. │  │    MLAgent      │                │
│  │ • Data Clean    │  │ • Plot Generate │  │ • Model Train   │                │
│  │ • EDA Analysis  │  │ • Chart Create  │  │ • Prediction    │                │
│  │ • Statistics    │  │ • Insights      │  │ • AutoML        │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   CodeAgent     │  │    NLPAgent     │  │ DeepLearningA.  │                │
│  │ • Code Generate │  │ • Text Process  │  │ • Neural Nets   │                │
│  │ • Execute Code  │  │ • Sentiment     │  │ • CNN/RNN       │                │
│  │ • Debug/Review  │  │ • Summarize     │  │ • Image Annot.  │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │ ModelEvalAgent  │  │ RetrievalAgent  │  │   (Future)      │                │
│  │ • Evaluate      │  │ • RAG Search    │  │   Expansion     │                │
│  │ • Optimize      │  │ • Knowledge     │  │   Agents        │                │
│  │ • Benchmark     │  │ • Context       │  │                 │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MEMORY & STORAGE                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │  Vector Store   │  │   RAG Memory    │  │   Embeddings    │                │
│  │  (ChromaDB)     │  │   (Context)     │  │   (Sentence-T)  │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      RLHF LEARNING SYSTEM                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │ FeedbackCollect │  │  Reward Model   │  │ Policy Optimize │                │
│  │ • User Feedback │  │ • Learn Prefer. │  │ • Improve Agent │                │
│  │ • Performance   │  │ • Quality Score │  │ • Better Route  │                │
│  │ • Analytics     │  │ • Training Data │  │ • Continuous    │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TOOLS & UTILITIES                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │  Data Loader    │  │   Web Scraper   │  │   API Client    │                │
│  │  Logger System  │  │   Cache Manager │  │   Validators    │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Detailed Workflow Process

### 1. User Query Processing Flow

```
User Input
    │
    ▼
┌─────────────────┐
│   FastAPI       │ ──► Receive user query
│   Interface     │     Parse request
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Orchestrator   │ ──► Initialize session
│   (Core)        │     Log request
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Task Parser    │ ──► Analyze query intent
│                 │     Extract parameters
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  CriticAgent    │ ──► Route to best agent
│  PlannerAgent   │     Multi-agent coordination
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Specialized     │ ──► Execute specific task
│ Agent           │     Generate response
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Response        │ ──► Format output
│ Processing      │     Add metadata
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ User Feedback   │ ──► Collect satisfaction
│ Collection      │     Store for RLHF
└─────────────────┘
```

### 2. Agent Specialization Matrix

```
┌─────────────────┐    ┌─────────────────────────────────────────────────────┐
│   Query Type    │    │                Agent Routing                        │
├─────────────────┼────┼─────────────────────────────────────────────────────┤
│ "Clean data"    │ ──►│ DataAgent → Statistical analysis → Visualization    │
│ "Plot chart"    │ ──►│ VisualizationAgent → Generate plots → Insights     │
│ "Train model"   │ ──►│ MLAgent → Model training → Evaluation               │
│ "Generate code" │ ──►│ CodeAgent → Code creation → Execution               │
│ "Analyze text"  │ ──►│ NLPAgent → Text processing → Sentiment              │
│ "Deep learning" │ ──►│ DeepLearningAgent → Neural nets → Training          │
│ "Evaluate"      │ ──►│ ModelEvaluationAgent → Metrics → Optimization       │
│ "Search info"   │ ──►│ RetrievalAgent → RAG search → Context               │
│ "Complex task"  │ ──►│ Multi-agent → Coordination → Combined result        │
└─────────────────┘    └─────────────────────────────────────────────────────┘
```

### 3. RLHF Learning Cycle

```
┌─────────────────┐
│  User Query     │
└─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐
│ Agent Response  │───►│ User Feedback   │
│ (Quality?)      │    │ (👍/👎/Rating)  │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Store Context   │    │ Feedback DB     │
│ (Query+Response)│    │ (Training Data) │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────┬───────────────┘
                 ▼
┌─────────────────┐    ┌─────────────────┐
│ Reward Model    │───►│ Policy Update   │
│ Training        │    │ (Better Routing)│
└─────────────────┘    └─────────────────┘
         │                       │
         └───────┬───────────────┘
                 ▼
┌─────────────────┐
│ Improved System │ ──► Better responses for future queries
│ Performance     │
└─────────────────┘
```

## 📊 Data Flow Architecture

### Input → Processing → Output

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     INPUT       │    │   PROCESSING    │    │     OUTPUT      │
│                 │    │                 │    │                 │
│ • User Query    │───►│ • Agent Routing │───►│ • Formatted     │
│ • Data Files    │    │ • Task Execution│    │   Response      │
│ • Parameters    │    │ • Multi-Agent   │    │ • Visualizations│
│ • Context       │    │   Coordination  │    │ • Generated Code│
│                 │    │ • Memory Access │    │ • Analysis      │
│                 │    │ • Tool Usage    │    │ • Insights      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ • Validation    │    │ • Error Handle  │    │ • Success Metrics│
│ • Preprocessing │    │ • Logging       │    │ • Performance   │
│ • Security      │    │ • Monitoring    │    │ • Feedback Hook │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🏗️ Project Structure Flow

```
NeuroAct-AI_Project/
│
├── main.py ──────────────────► Application Entry Point
│
├── app/
│   ├── core/ ─────────────────► System Orchestration
│   │   ├── orchestrator.py ───► Main Coordinator
│   │   ├── task_parser.py ────► Query Analysis
│   │   └── langgraph_flow.py ─► Workflow Management
│   │
│   ├── agents/ ───────────────► 10 Specialized Agents
│   │   ├── CriticAgent.py ────► Smart Routing
│   │   ├── PlannerAgent.py ───► Task Planning
│   │   ├── DataAgent.py ──────► Data Processing
│   │   ├── VisualizationAgent.py ► Plot Generation
│   │   ├── MLAgent.py ────────► Machine Learning
│   │   ├── CodeAgent.py ──────► Code Generation
│   │   ├── NLPAgent.py ───────► Text Processing
│   │   ├── DeepLearningAgent.py ► Neural Networks
│   │   ├── ModelEvaluationAgent.py ► Model Assessment
│   │   └── RetrievalAgent.py ─► Knowledge Search
│   │
│   ├── RLHF_Implementation/ ──► Learning System
│   │   ├── feedback_collector.py ► User Feedback
│   │   ├── reward_model.py ───► Preference Learning
│   │   └── policy_optimizer.py ► System Improvement
│   │
│   ├── memory/ ───────────────► Knowledge Storage
│   │   ├── vector_store.py ───► Embeddings
│   │   ├── rag.py ────────────► Retrieval
│   │   └── embedder.py ───────► Text Encoding
│   │
│   ├── tools/ ────────────────► Utility Functions
│   │   ├── data_loader.py ────► File Processing
│   │   ├── web_scraper.py ────► Web Data
│   │   └── api_client.py ─────► External APIs
│   │
│   ├── interface/ ────────────► User Interfaces
│   │   └── fastapi_main.py ───► Web API
│   │
│   └── utils/ ────────────────► Common Utilities
│       ├── logger.py ─────────► Logging System
│       └── config.py ─────────► Configuration
│
├── data/ ────────────────────► Data Storage
├── logs/ ────────────────────► System Logs
└── requirements.txt ─────────► Dependencies
```

## 🎯 Execution Flow Summary

1. **User Input** → FastAPI Interface
2. **Query Analysis** → Task Parser
3. **Smart Routing** → CriticAgent/PlannerAgent
4. **Task Execution** → Specialized Agent
5. **Response Generation** → Formatted Output
6. **Feedback Collection** → RLHF System
7. **Continuous Learning** → System Improvement

This comprehensive flowchart shows how your NeuroAct AI system processes queries through intelligent routing to specialized agents, with continuous learning through RLHF feedback collection.