# TaskManager - Recursive Multi-Agent Task Orchestration System

A production-ready LangGraph-based orchestration system that recursively decomposes complex objectives and executes specialized sub-agents (PDF, Excel, OCR, WebSearch, CodeInterpreter) with intelligent coordination through blackboard pattern knowledge sharing.

**Version**: 2.4 | **Status**: Production Ready ✅

## Features

- **Graph-of-Thought Planning**: Non-linear task graphs with cross-branch dependencies (Task A depends on B AND C)
- **Recursive Decomposition**: Multi-level hierarchical task breakdown (up to 5 levels)
- **5 Specialized Agents**: PDF (5 ops) | Excel (6 ops) | OCR (8 ops) | WebSearch (7 ops) | CodeInterpreter (4 ops)
- **Cross-Agent Workflows**: Automatic chaining (PDF→OCR→Excel) via file pointers
- **Research Synthesis**: Auto-detect and flag data contradictions with human escalation
- **Agentic Debate**: Consensus-based validation with Fact-Checker & Lead Researcher personas for high-confidence conflict resolution
- **Multimodal Vision**: Charts, diagrams, heatmaps analyzed via LLM vision models
- **Multi-Provider Support**: Anthropic Claude, OpenAI GPT, Google Gemini, Local Ollama
- **Flexible API Configuration**: Generic LLM endpoint configuration for any provider (no vendor lock-in)
- **Knowledge Sharing**: Blackboard pattern for findings across hierarchy levels
- **State Persistence**: LangGraph checkpointing for fault tolerance

## Quick Start

### 1. Install & Configure

```bash
pip install langchain langchain-core python-dotenv langgraph langchain-anthropic

cp .env.example .env
# Edit .env with your API key
```

**Example .env**:
```
ANTHROPIC_API_KEY=sk-ant-...
AGENT_LLM_PROVIDER=anthropic
AGENT_LLM_MODEL=claude-sonnet-4-20250514
AGENT_MAX_ITERATIONS=100
```

### 2. Basic Usage

```python
from task_manager import TaskManagerAgent, AgentConfig
from task_manager.config import EnvConfig

EnvConfig.load_env_file()
config = AgentConfig.from_env(prefix="AGENT_")

agent = TaskManagerAgent(
    objective="Analyze the quarterly report and create summary",
    config=config
)

result = agent.run(thread_id="task-001")
print(result)
```

## Installation

```bash
# Virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install langchain langchain-core python-dotenv langgraph

# Choose LLM provider
pip install langchain-anthropic      # Recommended
# OR: pip install langchain-openai / langchain-google-genai

# Verify
python -c "from task_manager import TaskManagerAgent; print('✓ Success')"
```

## Configuration

### Option 1: Environment Variables (Recommended)

```bash
# API Key (generic - works with any LLM provider)
LLM_API_KEY=your-api-key-here

# LLM Endpoint Configuration (Optional - for custom or self-hosted endpoints)
LLM_API_BASE_URL=https://api.provider.com          # Default: based on provider
LLM_API_ENDPOINT_PATH=v1beta                       # Default: based on provider
LLM_API_VERSION=v1alpha                            # Default: based on provider

# Use native provider SDK (if available)
USE_NATIVE_SDK=false

# LLM Settings
AGENT_LLM_PROVIDER=anthropic
AGENT_LLM_MODEL=claude-sonnet-4-20250514
AGENT_LLM_TEMPERATURE=0.2
AGENT_LLM_MAX_TOKENS=2000

# Agent Settings
AGENT_MAX_ITERATIONS=100
AGENT_ENABLE_SEARCH=true
AGENT_TIMEOUT=30
AGENT_MAX_RETRIES=3
AGENT_LOG_LEVEL=INFO

# Vision/Multimodal (Optional)
VISION_LLM_MODEL=gemini-2.5-pro-vision
ENABLE_VISION_ANALYSIS=true
```

### Option 2: Explicit Configuration

```python
from task_manager import AgentConfig
from task_manager.config import LLMConfig

config = AgentConfig(
    llm=LLMConfig(
        provider="anthropic",
        model_name="claude-sonnet-4-20250514",
        temperature=0.2,
        api_base_url="https://api.anthropic.com",  # Optional
        api_endpoint_path="v1"                      # Optional
    ),
    max_iterations=50
)

agent = TaskManagerAgent(objective="...", config=config)
```

### Option 3: Dictionary/JSON

```python
import json
from task_manager import AgentConfig

config_dict = {
    "llm": {
        "provider": "openai",
        "model_name": "gpt-4-turbo",
        "api_base_url": "https://api.openai.com"
    },
    "max_iterations": 100
}
config = AgentConfig.from_dict(config_dict)
agent = TaskManagerAgent(objective="...", config=config)
```

## Supported LLM Providers

| Provider | Model | Setup | Cost |
|----------|-------|-------|------|
| **Anthropic** | claude-sonnet-4-20250514 | `pip install langchain-anthropic` | High quality |
| **OpenAI** | gpt-4-turbo | `pip install langchain-openai` | Premium |
| **Google** | gemini-pro | `pip install langchain-google-genai` | Cost-effective |
| **Ollama** | llama2 | Local: `ollama serve` | Free |

**Configuration Examples**:

```python
# Anthropic
config = AgentConfig(
    llm=LLMConfig(provider="anthropic", model_name="claude-opus-4-20250805")
)

# OpenAI
config = AgentConfig(
    llm=LLMConfig(provider="openai", model_name="gpt-4-turbo")
)

# Google
config = AgentConfig(
    llm=LLMConfig(provider="google", model_name="gemini-pro")
)

# Local Ollama
config = AgentConfig(
    llm=LLMConfig(provider="local", model_name="llama2", base_url="http://localhost:11434")
)
```

## Project Structure

```
TaskManager/
├── task_manager/
│   ├── core/              # Orchestration (agent, workflow, planner)
│   ├── models/            # Data structures (state, task, enums)
	│   ├── sub_agents/        # Specialized agents (PDF, Excel, OCR, WebSearch, CodeInterpreter)
│   ├── config/            # Configuration (LLMConfig, AgentConfig)
│   └── utils/             # Logger, PromptBuilder
├── examples/              # Working examples
├── tests/                 # Unit tests
├── .env.example           # Configuration template
└── README.md / ARCHITECTURE_OVERVIEW.md
```

**For detailed architecture, components, and data models see [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)**

## Usage Examples

### Example 1: Document Analysis with Cross-Agent Chaining

```python
from task_manager import TaskManagerAgent, AgentConfig
from task_manager.config import EnvConfig

EnvConfig.load_env_file()
config = AgentConfig.from_env(prefix="AGENT_")

# Objective triggers automatic PDF → OCR → Excel chain
objective = """
Process the marketing report:
1. Extract text and find charts
2. Analyze charts for trends
3. Create Excel summary with findings
"""

agent = TaskManagerAgent(objective=objective, config=config)
result = agent.run(thread_id="report-001")

# Internally: PDF Agent → (auto) OCR Agent → (auto) Excel Agent
# No manual routing needed - automatic chaining via file pointers
```

### Example 2: Research with Synthesis & Conflict Detection

```python
objective = """
Research population statistics for major cities:
1. Search web for government statistics
2. Find relevant census PDFs
3. Extract charts and tables
4. Cross-check for contradictions
"""

agent = TaskManagerAgent(objective=objective, config=config)
result = agent.run(thread_id="research-001")

# Synthesis automatically runs after all research tasks complete
# Flags contradictions with severity levels (CRITICAL/HIGH/MEDIUM/LOW)
if result.get("requires_human_review"):
    print(f"Conflicts detected: {result.get('human_feedback')}")
```

### Example 3: Configuration Switching

```python
from task_manager.config import LLMConfig, AgentConfig

# Try different providers easily
for provider in ["anthropic", "openai", "google"]:
    config = AgentConfig(
        llm=LLMConfig(provider=provider, model_name="..."),
        max_iterations=50
    )
    agent = TaskManagerAgent(objective="...", config=config)
    result = agent.run()
    # Compare results across providers
```

### Example 4: Advanced - Custom Configuration

```python
config = AgentConfig(
    llm=LLMConfig(
        provider="anthropic",
        model_name="claude-opus-4-20250805",
        temperature=0.1,        # More deterministic
        max_tokens=4000,
        timeout=60
    ),
    max_iterations=100,
    max_retries=5,
    timeout=60,
    log_level="DEBUG",
    debug=True
)

agent = TaskManagerAgent(
    objective="Complex analysis task",
    config=config,
    metadata={"priority": "high", "user": "analyst"}
)

result = agent.run(thread_id="advanced-001")
```

## Key Capabilities

## Advanced Features

### Cross-Agent Chaining (v2.1)

Agents automatically handoff results to downstream agents without re-selection:

- **PDF → OCR**: If charts found, routes to OCR automatically
- **OCR → Excel**: If tables extracted, routes to Excel automatically  
- **WebSearch → Excel**: If CSV generated, routes to Excel automatically

File pointers maintain exact file paths; blackboard tracks complete data lineage.

### Research Synthesis & Conflict Detection (v2.2)

Automatic synthesis node triggers when:
- All research tasks complete
- Multiple agents contributed findings (≥2)
- Analyzes entire blackboard for contradictions

**Detects**:
- CRITICAL: Major data conflicts
- HIGH: Significant discrepancies
- MEDIUM: Minor inconsistencies
- LOW: Trivial differences

**Routes** to human review if critical conflicts found.

### Agentic Debate: Consensus-Based Conflict Resolution (v2.3)

When synthesis flags contradictions with score > 0.7, agentic debate automatically activates:

**Two Personas Debate Validity**:
1. **Fact-Checker** (conservative): Questions assumptions, demands evidence, prioritizes data reliability
2. **Lead Researcher** (inferential): Considers context, evaluates methodologies, makes reasoned judgments

**Debate Process**:
- Both personas independently analyze conflicting evidence
- Exchange arguments and positions
- Neutral arbiter synthesizes perspectives into consensus
- Records debate arguments in blackboard for transparency
- Returns high-confidence verdict with reasoning

**Outcomes**:
- ✓ Strong consensus: Conflicting data resolved with confidence
- ✗ Continued disagreement: Escalates to human review with both positions documented
- → Builds institutional knowledge: Debate patterns inform future similar conflicts

**Example Output**:
```
[DEBATE] Contradiction score: 0.85 exceeds threshold - initiating debate
[DEBATE] Fact-Checker Position: "Source A's methodology is flawed..."
[DEBATE] Lead Researcher Position: "Source B likely measured different scope..."
[DEBATE] Consensus: Source A is more reliable (fact-checked approach = 75% confidence)
```

### Graph-of-Thought Planning: Non-Linear Task Execution (v2.4)

Advanced planning system that supports complex, non-linear task workflows with cross-branch dependencies, enabling maximum parallelism and sophisticated task orchestration patterns.

**Key Concepts**:
- **Traditional Tree Planning**: Task depends only on its parent completing
  ```
  Root
  ├── Search Data
  ├── Extract PDF
  └── Analyze Excel
  ```
- **Graph-of-Thought Planning**: Tasks can depend on multiple OTHER tasks (graph edges)
  ```
  Root
  ├── Search Data (ready now)
  ├── Extract PDF (ready now)
  ├── Process Excel (ready now)
  └── Synthesize (depends on Data AND PDF AND Excel)
  ```

**Dependency Features**:
- **Multiple Dependencies**: Task waits for multiple sources (not just parent)
- **Parallel Execution**: Independent tasks run simultaneously
- **Diamond Patterns**: Supported (A→C, B→C, C→D)
- **Complex Workflows**: Non-linear chains, multi-level hierarchies with dependency resolution
- **Maximum Efficiency**: Ready independent tasks execute in parallel, blocking only when dependencies unsatisfied

**Task Status Management**:
```python
from task_manager.models import PlanNode

# Task with no dependencies (ready immediately)
task_1 = PlanNode(
    task_id="search",
    parent_id="root",
    depth=1,
    description="Search for company data",
    status="pending",
    priority=1,
    dependency_task_ids=[],  # No cross-task dependencies
    estimated_effort="low"
)

# Task with multiple dependencies (waits for all to complete)
task_4 = PlanNode(
    task_id="synthesize",
    parent_id="root",
    depth=1,
    description="Synthesize findings from all sources",
    status="pending",
    priority=4,
    dependency_task_ids=["search", "pdf_extract", "excel_process"],  # Graph dependency
    estimated_effort="high"
)
```

**Execution Flow**:
1. **Plan Creation**: LLM identifies tasks and their cross-branch dependencies
2. **Ready Task Selection**: Only tasks with all dependencies completed are available
3. **Parallel Execution**: Ready independent tasks execute simultaneously
4. **Dependency Checking**: Before task starts, all dependency_task_ids verified COMPLETED
5. **Error Handling**: Failed dependency blocks dependent tasks (automatic escalation)

**Example Workflow**:
```
Time 0: Search, PDF, Excel all ready → Start all 3 in parallel
Time 5: Search completes → Check Synthesize deps (need PDF & Excel still)
Time 8: PDF completes → Check Synthesize deps (need Excel still)
Time 12: Excel completes → Synthesize ready! Start
Time 15: Synthesize completes → Report task ready → Final step
```

**Configuration**:
```python
# Enable graph dependency checking (enabled by default)
config = AgentConfig(
    llm=LLMConfig(
        provider="anthropic",
        model_name="claude-sonnet-4-20250514"
    )
)
agent = TaskManagerAgent(objective="...", config=config)
# Graph-of-Thought planning automatically activated during decomposition
```

**Benefits**:
- **Complex Workflows**: Support sophisticated task patterns (diamond, chains, multi-level)
- **Maximum Parallelism**: Independent tasks execute simultaneously, reducing total execution time
- **Clear Dependencies**: Explicit dependency specification prevents ambiguity
- **Scalability**: Works with any number of tasks and dependency relationships
- **Transparency**: Easy to understand why tasks are blocked or ready
- **Backward Compatible**: Existing code continues to work without modification

### Multimodal Vision Analysis (v2.1)

Leverages LLM vision for advanced image understanding:
- Charts & graphs: Trend analysis, axis interpretation
- Diagrams: Component recognition, relationships
- Heatmaps: Color scales, spatial patterns
- Complex tables: Nested structures, merged cells

**Configuration**:
```
ENABLE_VISION_ANALYSIS=true
VISION_LLM_PROVIDER=google|openai|anthropic
AUTO_DETECT_CHARTS=true
```

## Environment Variables

```
# API Keys
ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY

# LLM Config
AGENT_LLM_PROVIDER, AGENT_LLM_MODEL, AGENT_LLM_TEMPERATURE

# Agent Settings
AGENT_MAX_ITERATIONS, AGENT_TIMEOUT, AGENT_MAX_RETRIES, AGENT_LOG_LEVEL

# Vision Analysis (v2.1)
ENABLE_VISION_ANALYSIS, VISION_LLM_PROVIDER, AUTO_DETECT_CHARTS
```

See [Configuration](#configuration) section for detailed setup.

## Production Deployment

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY task_manager/ task_manager/
ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
ENV AGENT_LLM_PROVIDER=anthropic
ENV AGENT_LOG_LEVEL=WARNING
CMD ["python", "examples/karnataka_data_collection.py"]
```

### Kubernetes

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: task-manager-job
spec:
  template:
    spec:
      containers:
      - name: task-manager
        image: task-manager:latest
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: task-manager-secrets
              key: ANTHROPIC_API_KEY
        - name: AGENT_LLM_PROVIDER
          value: "anthropic"
        - name: AGENT_LOG_LEVEL
          value: "WARNING"
```

### Deployment Checklist

- ✓ Create production .env or set environment variables
- ✓ Test with actual API keys
- ✓ Set `AGENT_LOG_LEVEL=WARNING` or `ERROR`
- ✓ Set `AGENT_DEBUG=false`
- ✓ Configure appropriate timeouts and retries
- ✓ Monitor API usage and costs
- ✓ Set up error logging and alerts
- ✓ Never commit .env to version control

## Troubleshooting

### "ModuleNotFoundError: No module named 'langchain_anthropic'"

**Solution**: Install missing provider:
```bash
pip install langchain-anthropic    # For Anthropic
pip install langchain-openai       # For OpenAI
pip install langchain-google-genai # For Google
pip install langchain-community    # For Ollama
```

### "Error: Could not authenticate with the API"

**Causes**: API key not set, incorrect, expired, or insufficient permissions

**Solutions**:
1. Verify API key in `.env`
2. Get fresh key from provider console
3. Check provider documentation for required permissions

### "Connection timeout"

**Solutions**:
```bash
# Increase timeouts
AGENT_TIMEOUT=60
AGENT_LLM_TIMEOUT=60
```
- Check network connectivity
- Verify LLM provider is responding
- For Ollama: ensure `ollama serve` is running

### "Task gives poor results"

**Solutions**:
```bash
# More deterministic output
AGENT_LLM_TEMPERATURE=0.2

# More exploration
AGENT_MAX_ITERATIONS=100

# Debug mode
AGENT_DEBUG=true
AGENT_LOG_LEVEL=DEBUG
```
- Make objective more specific
- Try different LLM provider
- Check task metadata context

### "Out of quota / Rate limited"

**Solutions**:
- Check API usage dashboard  
- Reduce `AGENT_MAX_ITERATIONS`
- Add delays between requests
- Upgrade API plan

## Technical Architecture

For comprehensive technical documentation including:
- System architecture (5-layer design)
- Component details (Agent, MasterPlanner, Workflow)
- Data models & structures
- Complete API reference
- File organization & dependencies

**→ See [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)**

## Testing

```bash
python -m pytest tests/
# or
python -m unittest discover tests/
```

## License

MIT License - See LICENSE file

## Support & Resources

- **Technical Details**: [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) - Comprehensive architecture & API reference
- **Examples**: `examples/configuration_examples.py` - Configuration patterns
- **Working Example**: `examples/karnataka_data_collection.py` - Full example
- **Configuration Template**: `.env.example`

## Contributing

Contributions welcome:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

**TaskManager v2.4** | Production Ready ✅ | January 25, 2026
