# NeuroAct AI System - Complete Architecture

## 🎯 System Overview

```
                    ┌─────────────────────────────────────┐
                    │         NEUROACT AI SYSTEM         │
                    │    Multi-Agent Platform + RLHF     │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
```

## 🌐 User Interface Layer

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI Web   │    │   REST API      │    │   CLI Runner    │
│   Interface     │    │   Endpoints     │    │   (run.py)      │
│   Port: 8700    │    │   /chat /docs   │    │   Direct Exec   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
```

## 🧠 Core System Architecture

```
                    ┌─────────────────────────────────────┐
                    │        CORE ORCHESTRATION           │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
         ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
         │  Orchestrator   │ │  Task Parser    │ │  LangGraph      │
         │  (main.py)      │ │  Query Analysis │ │  Multi-Agent    │
         │  Main Control   │ │  Intent Extract │ │  Coordination   │
         └─────────────────┘ └─────────────────┘ └─────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       INTELLIGENT ROUTING           │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
         ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
         │  CriticAgent    │ │  PlannerAgent   │ │  MCP Dispatcher │
         │  Coordinator    │ │  Smart Router   │ │  Messages       │
         └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## 🔗 LangGraph Multi-Agent Coordination

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY LANGGRAPH FOR MULTI-AGENT SYSTEMS?                    │
└─────────────────────────────────────────────────────────────────────────────┘

✅ **Agent Workflow Management**
   ┌─────────────────┐
   │ • State Graphs    │ ──► Manages agent execution flow
   │ • Node Routing    │ ──► Routes between 10 specialized agents
   │ • Parallel Exec   │ ──► Multiple agents can work simultaneously
   │ • Conditional     │ ──► Smart routing based on query type
   └─────────────────┘

✅ **Multi-Agent Coordination**
   ┌─────────────────┐
   │ • Agent Handoffs  │ ──► DataAgent → VisualizationAgent
   │ • State Sharing   │ ──► Agents share context and results
   │ • Error Recovery  │ ──► Fallback to alternative agents
   │ • Result Merging  │ ──► Combine outputs from multiple agents
   └─────────────────┘

✅ **Complex Task Handling**
   ┌─────────────────┐
   │ • Multi-Step      │ ──► "Clean data, train model, plot results"
   │ • Dependencies    │ ──► Agent B waits for Agent A completion
   │ • Branching       │ ──► Different paths based on data type
   │ • Loops & Retry   │ ──► Retry failed operations
   └─────────────────┘

Example Multi-Agent Flow:
User: "Clean my data, check normality, and create appropriate plots"

┌─────────────┐    ┌─────────────────┐    ┌───────────────────┐
│  DataAgent   │──►│ Statistical Test │──►│ VisualizationAgent │
│ Clean Data  │    │ Normality Check  │    │ Plot Distribution  │
└─────────────┘    └─────────────────┘    └───────────────────┘

🔗 **LangGraph manages this entire workflow automatically!**
```

## 🤖 10 Specialized Agents

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENT ECOSYSTEM                                   │
└─────────────────────────────────────────────────────────────────────────────┘

Row 1: Data Processing & Analysis
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   DataAgent     │ │ VisualizationA. │ │    MLAgent      │ │ ModelEvalAgent  │
│                 │ │                 │ │                 │ │                 │
│ • Data Clean    │ │ • Plot Generate │ │ • Model Train   │ │ • Evaluate      │
│ • EDA Analysis  │ │ • Chart Create  │ │ • Prediction    │ │ • Optimize      │
│ • Statistics    │ │ • Insights      │ │ • AutoML        │ │ • Benchmark     │
│ • Outliers      │ │ • LLM-Powered   │ │ • Sklearn       │ │ • Metrics       │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘

Row 2: Code & Intelligence
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   CodeAgent     │ │    NLPAgent     │ │ DeepLearningA.  │ │ RetrievalAgent  │
│                 │ │                 │ │                 │ │                 │
│ • Code Generate │ │ • Text Process  │ │ • Neural Nets   │ │ • RAG Search    │
│ • Execute Code  │ │ • Sentiment     │ │ • CNN/RNN/LSTM  │ │ • Knowledge     │
│ • Debug/Review  │ │ • Summarize     │ │ • Image Annot.  │ │ • Multi-Source  │
│ • Security      │ │ • Embeddings    │ │ • TensorFlow    │ │ • Context       │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘

Row 3: Coordination
┌─────────────────┐ ┌─────────────────┐
│  CriticAgent    │ │  PlannerAgent   │
│                 │ │                 │
│ • Smart Route   │ │ • Multi-Dim     │
│ • Coordinate    │ │ • Analysis      │
│ • Quality Check │ │ • Agent Select  │
│ • Multi-Agent   │ │ • Workflow      │
└─────────────────┘ └─────────────────┘
```

## 💾 Memory & Storage System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY ARCHITECTURE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Vector Store   │ │   RAG Memory    │ │   Embeddings    │ │   Cache Layer   │
│                 │ │                 │ │                 │ │                 │
│ • ChromaDB      │ │ • Context       │ │ • Sentence-T    │ │ • Redis         │
│ • Embeddings    │ │ • History       │ │ • Text Vectors  │ │ • Fast Access   │
│ • Similarity    │ │ • Knowledge     │ │ • Semantic      │ │ • Session Data  │
│ • Search        │ │ • Retrieval     │ │ • Matching      │ │ • Performance   │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## 🧠 RLHF Learning System (7 Components)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REINFORCEMENT LEARNING FROM HUMAN FEEDBACK               │
└─────────────────────────────────────────────────────────────────────────────┘

Core Components:
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│FeedbackCollector│ │ Feedback Models │ │ Feedback Storage│
│                 │ │                 │ │                 │
│ • User Feedback │ │ • Data Schemas  │ │ • SQLite DB     │
│ • Performance   │ │ • Validation    │ │ • Persistence   │
│ • Analytics     │ │ • Structures    │ │ • Retrieval     │
│ • Multi-Agent   │ │ • Type Safety   │ │ • Query System  │
└─────────────────┘ └─────────────────┘ └─────────────────┘

Learning Components:
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Reward Model   │ │ Policy Optimizer│ │Integration Mgr  │
│                 │ │                 │ │                 │
│ • Learn Prefer. │ │ • Improve Agent │ │ • Agent Hooks   │
│ • Quality Score │ │ • Better Route  │ │ • Coordination  │
│ • Training Data │ │ • Continuous    │ │ • Non-disrupt   │
│ • Neural Net    │ │ • PPO/TRPO      │ │ • Seamless      │
└─────────────────┘ └─────────────────┘ └─────────────────┘

Module Coordination:
┌─────────────────┐
│   __init__.py   │
│                 │
│ • Module Setup  │
│ • Imports       │
│ • Coordination  │
│ • Entry Point   │
└─────────────────┘
```

## 🔄 Complete Workflow Process

```
1. USER INPUT
   │
   ├─► FastAPI Interface ──► Parse Request ──► Validate Input
   │
   ▼
2. CORE PROCESSING
   │
   ├─► Orchestrator ──► Initialize Session ──► Log Request
   │
   ├─► Task Parser ──► Analyze Intent ──► Extract Parameters
   │
   ▼
3. INTELLIGENT ROUTING
   │
   ├─► CriticAgent ──► Evaluate Query ──► Route Decision
   │
   ├─► PlannerAgent ──► Multi-Dimensional Analysis ──► Agent Selection
   │
   ▼
4. AGENT EXECUTION
   │
   ├─► Specialized Agent ──► Execute Task ──► Generate Response
   │
   ├─► Memory Access ──► Context Retrieval ──► Enhanced Output
   │
   ▼
5. RESPONSE & LEARNING
   │
   ├─► Format Output ──► Add Metadata ──► Return Response
   │
   ├─► Feedback Collection ──► Store Context ──► RLHF Training
   │
   ▼
6. CONTINUOUS IMPROVEMENT
   │
   └─► Reward Model ──► Policy Update ──► Better Performance
```

## 🎯 Agent Routing Matrix

```
┌─────────────────┬─────────────────────────────────────────────────────────┐
│   Query Type    │                 Routing Decision                        │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ "Clean data"    │ DataAgent → Statistical Analysis → VisualizationAgent   │
│ "Plot chart"    │ VisualizationAgent → LLM Plot Generation → Insights    │
│ "Train model"   │ MLAgent → Model Training → ModelEvaluationAgent        │
│ "Generate code" │ CodeAgent → Code Creation → Execution → Review         │
│ "Analyze text"  │ NLPAgent → Text Processing → Sentiment → Embeddings    │
│ "Deep learning" │ DeepLearningAgent → Neural Networks → Training         │
│ "Evaluate"      │ ModelEvaluationAgent → Metrics → Optimization          │
│ "Search info"   │ RetrievalAgent → RAG Search → Context Enhancement      │
│ "Complex task"  │ Multi-Agent → CriticAgent Coordination → Combined      │
└─────────────────┴─────────────────────────────────────────────────────────┘
```

## 📊 RLHF Learning Cycle

```
┌─────────────────┐
│  User Query     │ ──► "Plot my data distribution"
└─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Agent Response  │────►│ User Feedback   │
│ Generated Plot  │     │ 👍 Great plot!  │
└─────────────────┘     └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Store Context   │     │ Feedback DB     │
│ Query+Response  │     │ Training Data   │
│ Performance     │     │ Reward Signal   │
└─────────────────┘     └─────────────────┘
         │                       │
         └───────┬───────────────┘
                 ▼
┌─────────────────┐     ┌─────────────────┐
│ Reward Model    │────►│ Policy Update   │
│ Learn Preference│     │ Better Routing  │
│ Quality Scoring │     │ Improved Agents │
└─────────────────┘     └─────────────────┘
         │                       │
         └───────┬───────────────┘
                 ▼
┌─────────────────┐
│ Enhanced System │ ──► Better responses for future queries
│ Performance     │     Higher user satisfaction
└─────────────────┘
```

## 🎆 Key Benefits of LangGraph Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH ADVANTAGES FOR NEUROACT                        │
└─────────────────────────────────────────────────────────────────────────────┘

🚀 **Scalable Multi-Agent Orchestration**
   • Handles 10+ specialized agents seamlessly
   • Dynamic routing based on query complexity
   • Parallel execution for performance
   • State management across agent interactions

🧠 **Intelligent Workflow Management**
   • Conditional branching (if data is text → NLPAgent, if numerical → DataAgent)
   • Sequential dependencies (DataAgent → MLAgent → ModelEvaluationAgent)
   • Error handling and fallback strategies
   • Loop support for iterative processes

🔄 **Complex Task Decomposition**
   • "Train a model and visualize results" → Multiple agents coordinated
   • "Analyze this dataset" → DataAgent + VisualizationAgent + insights
   • "Generate and test code" → CodeAgent + execution + review cycle
   • "Research and summarize" → RetrievalAgent + NLPAgent + formatting

📊 **Performance & Reliability**
   • Asynchronous agent execution
   • Built-in retry mechanisms
   • State persistence across failures
   • Memory-efficient agent switching

🎆 **Why LangGraph vs Traditional Approaches?**

   Traditional Multi-Agent:
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ ❌ Manual routing logic                                              │
   │ ❌ Hard-coded agent sequences                                       │
   │ ❌ Difficult error handling                                         │
   │ ❌ No built-in state management                                     │
   │ ❌ Complex coordination code                                        │
   └─────────────────────────────────────────────────────────────────────────────┘

   LangGraph Multi-Agent:
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ ✅ Declarative workflow definition                                   │
   │ ✅ Dynamic routing with conditions                                  │
   │ ✅ Built-in error handling and recovery                            │
   │ ✅ Automatic state management                                       │
   │ ✅ Visual workflow representation                                   │
   │ ✅ Easy to modify and extend                                        │
   └─────────────────────────────────────────────────────────────────────────────┘

🎯 **Perfect for NeuroAct's 10-Agent Architecture**
   • Each agent becomes a "node" in the LangGraph workflow
   • CriticAgent and PlannerAgent act as "routing nodes"
   • Complex queries automatically decomposed into multi-agent workflows
   • RLHF feedback can influence routing decisions over time
```

## 🏗️ Complete Project File Structure

```
NeuroAct-AI_Project/
│
├── 🚀 main.py ──────────────────────► Application Entry Point
├── 🏃 run.py ───────────────────────► CLI Runner
│
├── 📁 app/
│   │
│   ├── 🧠 core/ ────────────────────► System Orchestration (4 files)
│   │   ├── __init__.py ─────────────► Module Initialization
│   │   ├── orchestrator.py ─────────► Main System Coordinator
│   │   ├── task_parser.py ──────────► Query Analysis & Intent Extraction
│   │   └── langgraph_flow.py ───────► Multi-Agent Workflow Management
│   │
│   ├── 🤖 agents/ ──────────────────► 10 Specialized Agents
│   │   ├── __init__.py ─────────────► Agent Module Initialization
│   │   ├── CriticAgent.py ──────────► Smart Routing & Multi-Agent Coordination
│   │   ├── PlannerAgent.py ─────────► Multi-Dimensional Analysis & Agent Selection
│   │   ├── data_agent.py ───────────► Data Processing, EDA & Statistical Analysis
│   │   ├── VisualizationAgent.py ───► LLM-Powered Dynamic Plot Generation
│   │   ├── ml_agent.py ─────────────► Machine Learning & AutoML Pipeline
│   │   ├── code_agent.py ───────────► Code Generation, Execution & Security Review
│   │   ├── nlp_agent.py ────────────► Text Processing, Sentiment & Embeddings
│   │   ├── Deep_learning_Agent.py ──► Neural Networks, CNN/RNN & Training
│   │   ├── model_evaluation_agent.py ► Model Assessment & Optimization
│   │   └── RetrievalAgent.py ───────► RAG, Knowledge Search & Context Enhancement
│   │
│   ├── 🧠 RLHF_Implementation/ ─────► Reinforcement Learning from Human Feedback (7 Components)
│   │   ├── __init__.py ─────────────► RLHF Module Initialization & Coordination
│   │   ├── feedback_collector.py ───► Advanced User Feedback Collection & Analytics
│   │   ├── feedback_models.py ──────► Data Structures, Schemas & Validation
│   │   ├── feedback_storage.py ─────► SQLite Database & Persistence Layer
│   │   ├── reward_model.py ─────────► Neural Reward Model & Preference Learning
│   │   ├── policy_optimizer.py ─────► PPO/TRPO Policy Optimization & System Improvement
│   │   └── integration_manager.py ──► Non-disruptive Agent Integration Hooks
│   │
│   ├── 💾 memory/ ──────────────────► Knowledge Storage & Context Management (3 files)
│   │   ├── vector_store.py ─────────► ChromaDB Vector Database & Similarity Search
│   │   ├── rag.py ──────────────────► Retrieval Augmented Generation & Context
│   │   └── embedder.py ─────────────► Sentence Transformers & Text Encoding
│   │
│   ├── 🛠️ tools/ ───────────────────► Comprehensive Tool Suite (15 files)
│   │   ├── __init__.py ─────────────► Tools Module Initialization
│   │   ├── api_client.py ───────────► External API Integration & HTTP Client
│   │   ├── cache_manager.py ────────► Redis Caching & Performance Optimization
│   │   ├── data_utils.py ───────────► Data Processing Utilities & Helpers
│   │   ├── data_validator.py ───────► Data Validation & Quality Checks
│   │   ├── document_processor.py ───► PDF, Word, Text Document Processing
│   │   ├── feed_reader.py ──────────► RSS/Atom Feed Reading & Parsing
│   │   ├── github_api.py ───────────► GitHub API Integration & Repository Access
│   │   ├── huggingface_loader.py ───► HuggingFace Model Loading & Management
│   │   ├── json_editor.py ──────────► JSON Manipulation & Editing Tools
│   │   ├── lora_adapter.py ─────────► LoRA Fine-tuning & Model Adaptation
│   │   ├── python_repl.py ─────────► Python Code Execution & REPL Environment
│   │   ├── qlora_trainer.py ────────► QLoRA Training & Quantized Fine-tuning
│   │   ├── web_scraper.py ──────────► Web Data Extraction & Scraping
│   │   └── web_search.py ───────────► Web Search & Information Retrieval
│   │
│   ├── 📡 mcp/ ─────────────────────► Multi-Agent Communication Protocol (2 files)
│   │   ├── mcp_dispatcher.py ───────► Message Routing & Task Distribution
│   │   └── mcp_schema.py ───────────► Message Schemas & Protocol Definitions
│   │
│   ├── 🌐 interface/ ───────────────► User Interface Layer (2 files)
│   │   ├── __init__.py ─────────────► Interface Module Initialization
│   │   └── fastapi_main.py ─────────► FastAPI Web Server & REST API Endpoints
│   │
│   ├── 🔧 utils/ ───────────────────► Common Utilities (3 files)
│   │   ├── __init__.py ─────────────► Utils Module Initialization
│   │   ├── data_loader.py ──────────► File Loading & Data Import Utilities
│   │   └── logger.py ───────────────► Comprehensive Logging System
│   │
│   └── ⚙️ config.py ────────────────► Global Configuration Management
│
├── 📊 data/ ────────────────────────► Data Storage, Vectors & Embeddings
│   └── vectors/ ────────────────────► Vector Database Storage
├── 📝 logs/ ────────────────────────► System Logs, Monitoring & Agent Performance
├── 📚 docs/ ────────────────────────► Documentation & Architecture Diagrams
│   └── NeuroAct.png ────────────────► System Architecture Diagram
├── 🧪 scripts/ ─────────────────────► Development & Deployment Scripts
│   ├── run_pipeline.py ─────────────► Pipeline Execution Script
│   └── seed_vector_store.py ────────► Vector Store Initialization
├── 📋 requirements.txt ─────────────► Python Dependencies & Libraries (130+ packages)
├── 🔒 .env ─────────────────────────► Environment Variables & API Keys
├── 📄 README.md ────────────────────► Comprehensive Project Documentation
├── 📊 NeuroAct_System_Architecture.md ► Complete System Architecture & Flowcharts
└── 🚫 .gitignore ───────────────────► Git Ignore Rules & Patterns
```

## 🎯 Key Features Summary

### ✅ Multi-Agent Intelligence
- **10 Specialized Agents** for different domains
- **Smart Routing** with CriticAgent & PlannerAgent
- **Multi-Agent Coordination** for complex tasks

### ✅ RLHF Learning System
- **7 Component Architecture** for comprehensive learning
- **User Feedback Collection** with advanced analytics
- **Continuous Improvement** through reward modeling

### ✅ Production Ready
- **FastAPI Web Interface** with REST endpoints
- **Comprehensive Logging** and monitoring
- **Scalable Architecture** with memory management

### ✅ Advanced Capabilities
- **LLM-Powered Visualization** with dynamic plot generation
- **RAG Knowledge System** with vector storage
- **Auto-ML Pipeline** with model evaluation
- **Code Generation & Execution** with security

This architecture provides a complete, production-ready multi-agent AI system with continuous learning capabilities through RLHF.