# AGENTS.md - AI Coding Agent Guidelines

> Multi-agent based automatic data analysis and intelligent report generation system
> 멀티 에이전트 기반 자동 데이터 분석 및 지능형 보고서 생성시스템

## Project Overview

This project implements a LangGraph-based multi-agent system that:
1. Loads and preprocesses data (CSV files)
2. Analyzes data using LLM-generated Python code executed in a sandbox
3. Evaluates analysis results with a critic agent
4. Generates reports (Markdown, planned: PPT/HTML)
5. Supports Human-in-the-Loop (HITL) review and feedback

## Tech Stack

- **Python 3.11+** (primary language)
- **LangGraph**: Agent orchestration, state management, HITL control
- **LangChain**: LLM integration (Google Gemini via `langchain-google-genai`)
- **LangChain Experimental**: PythonREPL sandboxed code execution
- **FastAPI + Uvicorn**: Web API (planned)
- **python-pptx**: PowerPoint report generation (planned)
- **pandas/numpy/matplotlib/seaborn**: Data analysis stack

---

## Build / Run / Test Commands

### Environment Setup
```bash
# Create and activate virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Copy .env.example to .env and add GOOGLE_API_KEY
```

### Run Application
```bash
# Main entry point - runs the complete agent workflow
python -m src.main

# Check available Google Gemini models
python check_models.py
```

### Testing (not yet implemented)
```bash
# When tests are added:
pytest                           # Run all tests
pytest tests/test_file.py       # Run single test file
pytest -k "test_name"           # Run specific test by name
pytest -x                       # Stop on first failure
```

---

## Project Structure

```
src/
├── main.py                 # Entry point, graph execution with HITL demo
├── agent/
│   ├── __init__.py         # Package marker (empty)
│   ├── graph.py            # LangGraph workflow definition (StateGraph)
│   ├── nodes.py            # Node implementations (load, preprocess, analyze, etc.)
│   ├── state.py            # AgentState TypedDict definition
│   └── tools.py            # Python REPL tool for sandboxed execution
└── data/
    └── sample.csv          # Sample data for testing
```

---

## Code Style Guidelines

### Imports
```python
# Order: stdlib → third-party → local
import os
import re
from typing import TypedDict, Annotated, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd

from .state import AgentState
from .tools import python_repl_tool
```

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Functions | snake_case | `load_data`, `analyze_data` |
| Classes | PascalCase | `AgentState`, `ChatGoogleGenerativeAI` |
| Constants | UPPER_SNAKE | `GOOGLE_API_KEY` |
| Variables | snake_case | `file_path`, `raw_data` |
| Private | _prefix | `_internal_helper` |

### Type Hints
```python
# Always use type hints for function signatures
def load_data(state: AgentState) -> AgentState:
    ...

# Use TypedDict for complex state
class AgentState(TypedDict):
    file_path: str
    raw_data: Optional[dict]
    analysis_results: Annotated[List[str], merge_logs]

# Use Optional for nullable fields
evaluation_feedback: Optional[str]
```

### Error Handling
```python
# Use try/except with informative error messages
try:
    df = pd.read_csv(file_path)
    return {"raw_data": df.to_dict(), "steps_log": [f"Loaded: {file_path}"]}
except Exception as e:
    return {"steps_log": [f"Error loading data: {e}"]}
```

### LLM Response Handling
```python
# ALWAYS check if LLM response content is a list (multimodal responses)
response = llm.invoke(prompt)
content = response.content
if isinstance(content, list):
    content = "".join([str(c) for c in content])
```

### LangGraph Node Pattern
```python
# Nodes receive state and return partial state updates
def node_function(state: AgentState) -> AgentState:
    # 1. Extract needed state fields
    file_path = state['file_path']
    
    # 2. Perform operation
    result = some_operation(file_path)
    
    # 3. Return partial state update (merged automatically)
    return {
        "analysis_results": [result],
        "steps_log": ["Operation completed"]
    }
```

### State Merging (Annotated Lists)
```python
# Use custom merge functions for accumulating lists
def merge_logs(left: List[str], right: List[str]) -> List[str]:
    if right is None: return left
    if left is None: return right
    return left + right

# Apply with Annotated
steps_log: Annotated[List[str], merge_logs]
```

---

## Key Patterns

### Conditional Routing in LangGraph
```python
def should_continue(state: AgentState):
    if state.get('evaluation_feedback') == "APPROVE":
        return "generate_report"
    return "analyze_data"

workflow.add_conditional_edges(
    "evaluate_results",
    should_continue,
    {"analyze_data": "analyze_data", "generate_report": "generate_report"}
)
```

### HITL (Human-in-the-Loop) Pattern
```python
# Compile with interrupt points
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_review"]  # Pause before this node
)

# Resume with human feedback
app.update_state(thread, {"human_feedback": "APPROVE"})
for event in app.stream(None, thread):  # None resumes from checkpoint
    ...
```

### Python REPL Tool Usage
```python
# The REPL maintains state across invocations
repl = PythonREPL()  # Global instance

# Load data into REPL environment
repl.run("import pandas as pd\ndf = pd.read_csv('data.csv')")

# Execute analysis (df is available from previous run)
result = repl.run("print(df.describe())")
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google AI API key for Gemini models |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key (for LLM tracing) |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key (for LLM tracing) |
| `LANGFUSE_HOST` | No | Langfuse host URL (default: https://cloud.langfuse.com) |

### Langfuse Setup (LLM Observability)

Langfuse provides tracing and observability for LLM calls, replacing LangSmith.

1. Create account at https://cloud.langfuse.com
2. Create a project and get API keys
3. Copy `.env.example` to `.env` and fill in credentials:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Tracing is automatically enabled when these environment variables are set.

---

## Common Tasks for Agents

### Adding a New Node
1. Define function in `src/agent/nodes.py`
2. Register in `src/agent/graph.py` via `workflow.add_node()`
3. Add edges to connect the node in the graph

### Modifying State
1. Add new field to `AgentState` in `src/agent/state.py`
2. Use `Optional[T]` if field may be absent
3. Use `Annotated[List[T], merge_func]` for accumulating lists

### Adding New Tools
1. Create in `src/agent/tools.py` using `@tool` decorator
2. Import and use via `tool.invoke(args)` in nodes

---

## Do NOT

- Suppress type errors with `# type: ignore` or `as any`
- Leave empty except blocks
- Hardcode API keys (use `.env`)
- Commit `.env` or `venv/` to version control
- Modify global REPL state without understanding side effects
