"""
AceTaskAgent Web UI Server

FastAPI-based server providing:
- REST API for workflow CRUD, execution management, and alerts
- AI Chat assistant powered by DeepSeek
- WebSocket for real-time execution monitoring
- Static file serving for the frontend SPA
"""

import asyncio
import json
import uuid
import os
import sys
import ast
import re
import hashlib
import time
import httpx
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime
from typing import List, Set, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ui.workflow_store import WorkflowStore
from ui.execution_engine import ExecutionEngine

# Import config properties loader and AI context logger
sys.path.insert(0, str(Path(__file__).parent.parent))
from task_manager.config.config_properties import ConfigProperties
from task_manager.utils.ai_context_logger import AIContextLogger, get_ai_logger, LogCategory

# ============================================================================
# APP SETUP
# ============================================================================

# Load config.properties
config = ConfigProperties.load()

# Initialize AI Context Logger
ai_logger = AIContextLogger.get_instance(
    log_file=config.get("logging.ai_context_file", "./logs/ai_context_log.jsonl"),
    json_log_file=config.get("logging.json_log_file", "./logs/ace_agent_structured.jsonl"),
    max_memory_entries=config.get_int("logging.ai_context_max_entries", 500),
    enabled=config.get_bool("logging.enable_ai_context", True),
)

ai_logger.log_info(
    "AceTaskAgent UI Server initializing",
    category=LogCategory.SYSTEM,
    tags=["startup"],
    metadata={"config_loaded": True, "ai_chat_enabled": config.get_bool("aichat.enabled", True)},
)

app = FastAPI(
    title="AceTaskAgent UI",
    description="Workflow Designer, Execution Monitor & Alert System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
workflow_store = WorkflowStore()
connected_clients: Set[WebSocket] = set()


async def broadcast(message: dict):
    """Broadcast a message to all connected WebSocket clients."""
    disconnected = set()
    for ws in connected_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.add(ws)
    connected_clients.difference_update(disconnected)


execution_engine = ExecutionEngine(broadcast_fn=broadcast)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class WorkflowCreate(BaseModel):
    name: str
    description: str = ""
    nodes: List[dict] = []
    edges: List[dict] = []
    tags: List[str] = []


class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: Optional[List[dict]] = None
    edges: Optional[List[dict]] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None


class ExecutionStart(BaseModel):
    workflow_id: str
    objective: str
    config: dict = {}


class AlertAck(BaseModel):
    alert_id: str


class AgentRunRequest(BaseModel):
    objective: str
    operation: str = "search"
    parameters: dict = {}
    input_data: dict = {}
    timeout_seconds: int = 60


class AIChatMessage(BaseModel):
    role: str = "user"
    content: str


class AIChatRequest(BaseModel):
    message: str
    history: List[AIChatMessage] = []
    include_logs: bool = True


# In-memory conversation store for AI chat (per-session)
_chat_sessions: Dict[str, List[Dict[str, str]]] = {}


# ============================================================================
# STATIC FILES
# ============================================================================

static_dir = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


# Mount static files AFTER explicit routes
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================================
# WEBSOCKET
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    ai_logger.log_info(
        f"WebSocket client connected (total={len(connected_clients)})",
        category=LogCategory.NETWORK,
        tags=["websocket", "connect"],
    )
    try:
        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "data": {
                "stats": execution_engine.get_stats(),
                "alerts": [a.to_dict() for a in execution_engine.get_alerts(limit=20)],
            },
            "timestamp": datetime.now().isoformat(),
        })

        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            # Handle ping/pong
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})

    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        ai_logger.log_info(
            f"WebSocket client disconnected (remaining={len(connected_clients)})",
            category=LogCategory.NETWORK,
            tags=["websocket", "disconnect"],
        )
    except Exception as e:
        connected_clients.discard(websocket)
        ai_logger.log_error(f"WebSocket error: {e}", exc=e, category=LogCategory.NETWORK)


# ============================================================================
# WORKFLOW API
# ============================================================================

@app.get("/api/workflows")
async def list_workflows(status: Optional[str] = None):
    workflows = workflow_store.list_all(status=status)
    return {"workflows": [w.to_dict() for w in workflows]}


@app.get("/api/workflows/templates")
async def get_templates():
    return {"templates": workflow_store.get_templates()}


@app.get("/api/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    wf = workflow_store.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return wf.to_dict()


@app.post("/api/workflows")
async def create_workflow(data: WorkflowCreate):
    wf = workflow_store.create(
        name=data.name,
        description=data.description,
        nodes=data.nodes,
        edges=data.edges,
        tags=data.tags,
    )
    ai_logger.log_workflow_event(
        "created", wf.id, wf.name,
        details={"node_count": len(data.nodes), "tags": data.tags},
    )
    return wf.to_dict()


@app.put("/api/workflows/{workflow_id}")
async def update_workflow(workflow_id: str, data: WorkflowUpdate):
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    wf = workflow_store.update(workflow_id, **updates)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return wf.to_dict()


@app.delete("/api/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    if not workflow_store.delete(workflow_id):
        raise HTTPException(status_code=404, detail="Workflow not found")
    return {"status": "deleted"}


@app.post("/api/workflows/{workflow_id}/duplicate")
async def duplicate_workflow(workflow_id: str, new_name: Optional[str] = None):
    wf = workflow_store.duplicate(workflow_id, new_name)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return wf.to_dict()


@app.post("/api/workflows/{workflow_id}/generate-code")
async def generate_langgraph_code(workflow_id: str):
    """Generate executable LangGraph Python code from a workflow definition."""
    wf = workflow_store.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")

    code = _generate_langgraph_code(wf)
    return {"workflow_id": wf.id, "name": wf.name, "code": code}


def _generate_langgraph_code(wf) -> str:
    """
    Convert a WorkflowDefinition (nodes + edges) into runnable LangGraph code.

    The generated code:
    - Creates a TypedDict state with per-node result slots
    - Registers one node function per workflow node (delegates to sub-agent)
    - Wires edges exactly as drawn in the designer
    - Passes upstream outputs to each node via state
    """
    from ui.workflow_store import WorkflowDefinition  # local import to avoid circles

    nodes = wf.nodes
    edges = wf.edges

    # Build adjacency for conditional routing
    adjacency: Dict[str, list] = {}
    for e in edges:
        src = e.source if hasattr(e, "source") else e["source"]
        tgt = e.target if hasattr(e, "target") else e["target"]
        adjacency.setdefault(src, []).append(tgt)

    # Find entry nodes (no incoming edge)
    targets = {e.target if hasattr(e, "target") else e["target"] for e in edges}
    sources = {e.source if hasattr(e, "source") else e["source"] for e in edges}
    node_ids = [n.id if hasattr(n, "id") else n["id"] for n in nodes]
    entry_nodes = [nid for nid in node_ids if nid not in targets]
    if not entry_nodes:
        entry_nodes = node_ids[:1]  # fallback

    # Helpers
    def _node_var(n):
        """Safe Python identifier from a node."""
        label = n.label if hasattr(n, "label") else n.get("label", "")
        ntype = n.type if hasattr(n, "type") else n.get("type", "task")
        nid = n.id if hasattr(n, "id") else n.get("id", "")
        safe = (label or ntype).lower().replace(" ", "_").replace("-", "_")
        # Remove non-alphanumeric chars
        safe = "".join(c for c in safe if c.isalnum() or c == "_")
        return f"{safe}_{nid}"

    def _node_fn_name(n):
        return f"run_{_node_var(n)}"

    node_map = {}  # id -> node object
    for n in nodes:
        nid = n.id if hasattr(n, "id") else n["id"]
        node_map[nid] = n

    lines = []
    lines.append('"""')
    lines.append(f'Auto-generated LangGraph workflow: {wf.name}')
    lines.append(f'Description: {wf.description or "N/A"}')
    lines.append(f'Generated at: {{datetime.now().isoformat()}}')
    lines.append('"""')
    lines.append('')
    lines.append('import importlib')
    lines.append('from typing import TypedDict, Any, Optional, Dict')
    lines.append('from datetime import datetime')
    lines.append('from langgraph.graph import StateGraph, END')
    lines.append('from langgraph.checkpoint.memory import MemorySaver')
    lines.append('')
    lines.append('')
    lines.append('# ── Sub-agent registry ──────────────────────────────────────────')
    lines.append('_SUB_AGENT_REGISTRY = {')
    lines.append('    "web_search":       ("task_manager.sub_agents.web_search_agent",      "WebSearchAgent",       "search"),')
    lines.append('    "pdf":              ("task_manager.sub_agents.pdf_agent",              "PDFAgent",             "extract_text"),')
    lines.append('    "excel":            ("task_manager.sub_agents.excel_agent",            "ExcelAgent",           "analyze"),')
    lines.append('    "ocr":              ("task_manager.sub_agents.ocr_image_agent",        "OCRImageAgent",        "extract_text"),')
    lines.append('    "code_interpreter": ("task_manager.sub_agents.code_interpreter_agent", "CodeInterpreterAgent", "execute_code"),')
    lines.append('    "data_extraction":  ("task_manager.sub_agents.data_extraction_agent",  "DataExtractionAgent",  "extract"),')
    lines.append('    "problem_solver":   ("task_manager.sub_agents.problem_solver_agent",   "ProblemSolverAgent",   "analysis"),')
    lines.append('    "document":         ("task_manager.sub_agents.document_agent",         "DocumentAgent",        "generate_report"),')
    lines.append('    "mermaid":          ("task_manager.sub_agents.mermaid_agent",          "MermaidAgent",         "generate_diagram"),')
    lines.append('}')
    lines.append('')
    lines.append('')
    lines.append('def _load_agent(agent_type: str):')
    lines.append('    """Instantiate a sub-agent by type. Returns (instance, default_op)."""')
    lines.append('    entry = _SUB_AGENT_REGISTRY.get(agent_type)')
    lines.append('    if not entry:')
    lines.append('        return None, None')
    lines.append('    mod = importlib.import_module(entry[0])')
    lines.append('    cls = getattr(mod, entry[1])')
    lines.append('    return cls(), entry[2]')
    lines.append('')
    lines.append('')

    # ── State definition ──
    lines.append('# ── Workflow state ────────────────────────────────────────────')
    lines.append('class WorkflowState(TypedDict):')
    lines.append('    """Typed state passed through the LangGraph workflow."""')
    lines.append('    objective: str')
    for n in nodes:
        nid = n.id if hasattr(n, "id") else n["id"]
        lines.append(f'    result_{nid}: Optional[Dict[str, Any]]  # output of {_node_var(n)}')
    lines.append('    final_output: Optional[Dict[str, Any]]')
    lines.append('')
    lines.append('')

    # ── Node functions ──
    lines.append('# ── Node functions ────────────────────────────────────────────')
    for n in nodes:
        nid = n.id if hasattr(n, "id") else n["id"]
        ntype = n.type if hasattr(n, "type") else n.get("type", "unknown")
        label = n.label if hasattr(n, "label") else n.get("label", "")
        instructions = n.instructions if hasattr(n, "instructions") else n.get("instructions", "")
        config = n.config if hasattr(n, "config") else n.get("config", {})

        # Determine predecessors for I/O wiring
        preds = []
        for e in edges:
            tgt = e.target if hasattr(e, "target") else e["target"]
            src = e.source if hasattr(e, "source") else e["source"]
            if tgt == nid:
                preds.append(src)

        fn_name = _node_fn_name(n)
        lines.append(f'def {fn_name}(state: WorkflowState) -> dict:')
        lines.append(f'    """')
        lines.append(f'    Node: {label} (type={ntype})')
        if instructions:
            lines.append(f'    Instructions: {instructions}')
        lines.append(f'    """')

        # Collect upstream I/O
        if preds:
            lines.append(f'    # Collect upstream outputs')
            lines.append(f'    upstream = {{}}')
            for pid in preds:
                lines.append(f'    if state.get("result_{pid}"):')
                lines.append(f'        upstream["{pid}"] = state["result_{pid}"]')
            lines.append(f'    merged_text = "\\n---\\n".join(')
            lines.append(f'        str(v.get("output", v))[:2000] for v in upstream.values()')
            lines.append(f'    ) if upstream else ""')
        else:
            lines.append(f'    upstream = {{}}')
            lines.append(f'    merged_text = ""')

        lines.append(f'')
        lines.append(f'    # Build task description')
        if instructions:
            lines.append(f'    task_desc = {repr(instructions)} + "\\nWorkflow objective: " + state["objective"]')
        else:
            lines.append(f'    task_desc = state["objective"]')
        lines.append(f'    if merged_text:')
        lines.append(f'        task_desc += "\\n\\nInput from previous task(s):\\n" + merged_text')
        lines.append(f'')
        lines.append(f'    # Execute sub-agent')
        lines.append(f'    agent, default_op = _load_agent("{ntype}")')
        lines.append(f'    if agent is not None:')
        lines.append(f'        request = {{')
        lines.append(f'            "task_id": "{nid}",')
        lines.append(f'            "task_description": task_desc,')
        lines.append(f'            "task_type": "atomic",')
        lines.append(f'            "operation": {repr(config.get("operation", ""))} or default_op,')
        lines.append(f'            "parameters": {repr(config)},')
        lines.append(f'            "input_data": {{"upstream_results": upstream, "merged_text": merged_text}},')
        lines.append(f'            "temp_folder": "./temp_folder",')
        lines.append(f'            "output_folder": "./output_folder",')
        lines.append(f'            "cache_enabled": True,')
        lines.append(f'            "blackboard": [],')
        lines.append(f'            "relevant_entries": [],')
        lines.append(f'            "max_retries": 1,')
        lines.append(f'            "timeout_seconds": 120,')
        lines.append(f'        }}')
        lines.append(f'        result = agent.execute_task(request)')
        lines.append(f'    else:')
        lines.append(f'        result = {{"output": f"[no agent for {ntype}] {{task_desc[:200]}}", "status": "skipped"}}')
        lines.append(f'')
        lines.append(f'    return {{"result_{nid}": result}}')
        lines.append(f'')
        lines.append(f'')

    # ── Graph construction ──
    lines.append('# ── Build LangGraph workflow ───────────────────────────────────')
    lines.append('def build_workflow() -> StateGraph:')
    lines.append(f'    """Construct the "{wf.name}" workflow graph."""')
    lines.append(f'    graph = StateGraph(WorkflowState)')
    lines.append(f'')

    # Add nodes
    for n in nodes:
        nid = n.id if hasattr(n, "id") else n["id"]
        label = n.label if hasattr(n, "label") else n.get("label", "")
        lines.append(f'    graph.add_node("{nid}", {_node_fn_name(n)})  # {label}')

    lines.append(f'')

    # Set entry point
    lines.append(f'    # Entry point')
    if len(entry_nodes) == 1:
        lines.append(f'    graph.set_entry_point("{entry_nodes[0]}")')
    else:
        lines.append(f'    graph.set_entry_point("{entry_nodes[0]}")  # first entry node')

    lines.append(f'')

    # Add edges
    lines.append(f'    # Edges (as drawn in the designer)')
    # Group edges by source
    for src_id in node_ids:
        successors = adjacency.get(src_id, [])
        if not successors:
            # Terminal node → END
            lines.append(f'    graph.add_edge("{src_id}", END)')
        elif len(successors) == 1:
            # Simple edge
            lines.append(f'    graph.add_edge("{src_id}", "{successors[0]}")')
        else:
            # Fan-out: conditional edges (all branches fire, no condition)
            src_node = node_map.get(src_id)
            src_label = (src_node.label if hasattr(src_node, "label") else src_node.get("label", src_id)) if src_node else src_id
            lines.append(f'    # {src_label} fans out to {len(successors)} nodes')
            for tgt in successors:
                lines.append(f'    graph.add_edge("{src_id}", "{tgt}")')

    lines.append(f'')
    lines.append(f'    return graph')
    lines.append(f'')
    lines.append(f'')

    # ── Runner ──
    lines.append('# ── Execute workflow ─────────────────────────────────────────')
    lines.append('def run_workflow(objective: str) -> dict:')
    lines.append('    """Compile and run the workflow with the given objective."""')
    lines.append('    graph = build_workflow()')
    lines.append('    memory = MemorySaver()')
    lines.append('    app = graph.compile(checkpointer=memory)')
    lines.append('    ')
    lines.append('    initial_state: WorkflowState = {')
    lines.append('        "objective": objective,')
    for n in nodes:
        nid = n.id if hasattr(n, "id") else n["id"]
        lines.append(f'        "result_{nid}": None,')
    lines.append('        "final_output": None,')
    lines.append('    }')
    lines.append('    ')
    lines.append('    config = {"configurable": {"thread_id": "workflow-run-1"}}')
    lines.append('    result = app.invoke(initial_state, config)')
    lines.append('    return result')
    lines.append('')
    lines.append('')
    lines.append('if __name__ == "__main__":')
    lines.append('    import sys')
    lines.append('    objective = sys.argv[1] if len(sys.argv) > 1 else "Run the workflow"')
    lines.append('    output = run_workflow(objective)')
    lines.append('    print("\\n=== Workflow Complete ===")')
    lines.append('    for key, val in output.items():')
    lines.append('        if key.startswith("result_") and val is not None:')
    lines.append('            print(f"  {key}: {str(val)[:200]}")')

    return "\n".join(lines)


# ============================================================================
# EXECUTION API
# ============================================================================

@app.post("/api/executions")
async def start_execution(data: ExecutionStart):
    wf = workflow_store.get(data.workflow_id)
    if not wf:
        ai_logger.log_warning(
            f"Execution start failed: workflow {data.workflow_id} not found",
            category=LogCategory.EXECUTION,
            tags=["execution", "not_found"],
        )
        raise HTTPException(status_code=404, detail="Workflow not found")

    ai_logger.log_execution_event(
        "started", data.workflow_id,
        task_name=wf.name,
        details={"objective": data.objective[:200]},
    )

    instance = await execution_engine.start_execution(
        workflow_id=wf.id,
        workflow_name=wf.name,
        objective=data.objective,
        nodes=[n.to_dict() for n in wf.nodes],
        edges=[e.to_dict() for e in wf.edges],
        config=data.config,
    )
    return instance.to_dict()


@app.get("/api/executions")
async def list_executions(status: Optional[str] = None, limit: int = 20):
    execs = execution_engine.list_executions(status=status, limit=limit)
    return {"executions": [e.to_dict() for e in execs]}


@app.get("/api/executions/{execution_id}")
async def get_execution(execution_id: str):
    ex = execution_engine.get_execution(execution_id)
    if not ex:
        raise HTTPException(status_code=404, detail="Execution not found")
    return ex.to_dict()


@app.post("/api/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str):
    success = await execution_engine.cancel_execution(execution_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel execution")
    return {"status": "cancelled"}


@app.get("/api/executions/stats/summary")
async def execution_stats():
    return execution_engine.get_stats()


# ============================================================================
# ALERTS API
# ============================================================================

@app.get("/api/alerts")
async def get_alerts(severity: Optional[str] = None, limit: int = 50):
    alerts = execution_engine.get_alerts(severity=severity, limit=limit)
    return {"alerts": [a.to_dict() for a in alerts]}


@app.post("/api/alerts/acknowledge")
async def acknowledge_alert(data: AlertAck):
    if not execution_engine.acknowledge_alert(data.alert_id):
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "acknowledged"}


@app.post("/api/alerts/clear")
async def clear_alerts(severity: Optional[str] = None):
    count = execution_engine.clear_alerts(severity=severity)
    return {"cleared": count}


# ============================================================================
# AGENT INFO API
# ============================================================================

@app.get("/api/agents")
async def list_agents():
    """Return available agent types for the workflow designer."""
    return {
        "agents": [
            {
                "type": "web_search",
                "label": "Web Search",
                "icon": "search",
                "color": "#3b82f6",
                "description": "Search the web for information using DuckDuckGo",
                "config_schema": {
                    "search_depth": {"type": "select", "options": ["basic", "deep"], "default": "deep"},
                    "max_results": {"type": "number", "default": 10, "min": 1, "max": 50},
                },
            },
            {
                "type": "pdf",
                "label": "PDF Agent",
                "icon": "file-text",
                "color": "#ef4444",
                "description": "Extract text, tables, and images from PDF documents",
                "config_schema": {
                    "extract_tables": {"type": "boolean", "default": True},
                    "extract_images": {"type": "boolean", "default": False},
                },
            },
            {
                "type": "excel",
                "label": "Excel Agent",
                "icon": "table",
                "color": "#22c55e",
                "description": "Process and analyze Excel/CSV spreadsheet data",
                "config_schema": {
                    "operations": {"type": "multiselect", "options": ["analyze", "chart", "filter", "transform"]},
                },
            },
            {
                "type": "ocr",
                "label": "OCR/Image Agent",
                "icon": "image",
                "color": "#f59e0b",
                "description": "Extract text from images using OCR and analyze visual content",
                "config_schema": {
                    "engine": {"type": "select", "options": ["tesseract", "vision_llm"], "default": "tesseract"},
                },
            },
            {
                "type": "code_interpreter",
                "label": "Code Interpreter",
                "icon": "code",
                "color": "#8b5cf6",
                "description": "Execute Python code for computation and analysis",
                "config_schema": {
                    "language": {"type": "select", "options": ["python"], "default": "python"},
                    "timeout": {"type": "number", "default": 30, "min": 5, "max": 120},
                },
            },
            {
                "type": "data_extraction",
                "label": "Data Extraction",
                "icon": "database",
                "color": "#06b6d4",
                "description": "Extract structured data from various sources",
                "config_schema": {
                    "format": {"type": "select", "options": ["structured", "raw", "json"], "default": "structured"},
                },
            },
            {
                "type": "problem_solver",
                "label": "Problem Solver",
                "icon": "lightbulb",
                "color": "#ec4899",
                "description": "LLM-based analysis and problem solving",
                "config_schema": {
                    "mode": {"type": "select", "options": ["analysis", "synthesis", "debate"], "default": "analysis"},
                },
            },
            {
                "type": "document",
                "label": "Document Generator",
                "icon": "file-output",
                "color": "#14b8a6",
                "description": "Generate reports, summaries, and documents",
                "config_schema": {
                    "output_format": {"type": "select", "options": ["markdown", "html", "pdf"], "default": "markdown"},
                },
            },
            {
                "type": "mermaid",
                "label": "Mermaid Diagrams",
                "icon": "git-branch",
                "color": "#a855f7",
                "description": "Generate Mermaid diagrams and flowcharts",
                "config_schema": {
                    "diagram_type": {"type": "select", "options": ["flowchart", "sequence", "class", "state"], "default": "flowchart"},
                },
            },
        ]
    }


# ============================================================================
# AGENT DETAIL & DIRECT RUN API
# ============================================================================

# Sub-agent registry: type -> (module_path, class_name, default_operation)
_AGENT_REGISTRY = {
    "web_search":       ("task_manager.sub_agents.web_search_agent",     "WebSearchAgent",       "search"),
    "pdf":              ("task_manager.sub_agents.pdf_agent",             "PDFAgent",             "extract_text"),
    "excel":            ("task_manager.sub_agents.excel_agent",           "ExcelAgent",           "analyze"),
    "ocr":              ("task_manager.sub_agents.ocr_image_agent",       "OCRImageAgent",        "extract_text"),
    "code_interpreter": ("task_manager.sub_agents.code_interpreter_agent","CodeInterpreterAgent", "execute_code"),
    "data_extraction":  ("task_manager.sub_agents.data_extraction_agent", "DataExtractionAgent",  "extract"),
    "problem_solver":   ("task_manager.sub_agents.problem_solver_agent",  "ProblemSolverAgent",   "analysis"),
    "document":         ("task_manager.sub_agents.document_agent",        "DocumentAgent",        "generate_report"),
    "mermaid":          ("task_manager.sub_agents.mermaid_agent",         "MermaidAgent",         "generate_diagram"),
}

_CORE_AGENTS = [
    {
        "name": "TaskManagerAgent",
        "module": "task_manager.core.agent",
        "role": "Main orchestrator — breaks objectives into tasks, runs LangGraph workflow",
        "entry_point": "TaskManagerAgent(objective).run()",
    },
    {
        "name": "MasterPlanner",
        "module": "task_manager.core.master_planner",
        "role": "LLM-driven planner — decomposes tasks, assigns sub-agents, tracks blackboard",
        "entry_point": "MasterPlanner.plan(state)",
    },
    {
        "name": "TaskRelayAgent",
        "module": "task_manager.core.task_relay_agent",
        "role": "Routes completed sub-task results back into the main workflow state",
        "entry_point": "TaskRelayAgent.relay(state)",
    },
]


@app.get("/api/agents/detail")
async def list_agents_detail():
    """
    Introspect all real sub-agents and core agents.
    Returns live metadata including supported_operations loaded from code.
    """
    import importlib

    def _collect():
        sub_agents = []
        for agent_type, (module_path, class_name, default_op) in _AGENT_REGISTRY.items():
            entry = {
                "type": agent_type,
                "class": class_name,
                "module": module_path,
                "default_operation": default_op,
                "supported_operations": [],
                "agent_name": agent_type,
                "status": "ok",
                "error": None,
            }
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                instance = cls()
                entry["agent_name"] = getattr(instance, "agent_name", class_name)
                entry["supported_operations"] = getattr(instance, "supported_operations", [])
            except Exception as exc:
                entry["status"] = "unavailable"
                entry["error"] = str(exc)[:200]
            sub_agents.append(entry)
        return {"sub_agents": sub_agents, "core_agents": _CORE_AGENTS}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _collect)


@app.post("/api/agents/{agent_type}/run")
async def run_agent_directly(agent_type: str, data: AgentRunRequest):
    """
    Trigger a specific sub-agent directly without a full workflow.
    Constructs an AgentExecutionRequest and calls execute_task().
    """
    import importlib

    if agent_type not in _AGENT_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent type '{agent_type}'. Available: {list(_AGENT_REGISTRY)}",
        )

    module_path, class_name, default_op = _AGENT_REGISTRY[agent_type]
    operation = data.operation or default_op
    task_id = str(uuid.uuid4())

    ai_logger.log_info(
        f"Direct agent run: type={agent_type} op={operation} objective={data.objective[:120]}",
        category=LogCategory.EXECUTION,
        tags=["direct_run", agent_type],
    )

    def _run():
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        instance = cls()

        request = {
            "task_id": task_id,
            "task_description": data.objective,
            "task_type": "atomic",
            "operation": operation,
            "parameters": data.parameters,
            "input_data": data.input_data,
            "temp_folder": str(Path("./temp_folder").resolve()),
            "output_folder": str(Path("./output_folder").resolve()),
            "cache_enabled": True,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 1,
            "timeout_seconds": data.timeout_seconds,
        }

        return instance.execute_task(request)

    started = datetime.now().isoformat()
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run)
        duration_ms = int((datetime.now().timestamp() - datetime.fromisoformat(started).timestamp()) * 1000)
        ai_logger.log_execution_event(
            "agent_direct_run_ok", task_id,
            task_name=f"{agent_type}/{operation}",
            details={"objective": data.objective[:200]},
        )
        return {
            "task_id": task_id,
            "agent_type": agent_type,
            "operation": operation,
            "started_at": started,
            "duration_ms": duration_ms,
            "result": result,
        }
    except Exception as exc:
        ai_logger.log_error(
            f"Direct agent run failed: {agent_type}/{operation} — {exc}",
            exc=exc,
            category=LogCategory.EXECUTION,
        )
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# AI CHAT ASSISTANT API
# ============================================================================

@app.post("/api/aichat")
async def ai_chat(data: AIChatRequest):
    """
    AI Chat endpoint powered by DeepSeek.
    Sends user message along with system context (recent logs, health summary)
    to the DeepSeek API and returns the assistant response.
    """
    ai_logger.log_info(
        f"AI Chat request received: {data.message[:100]}",
        category=LogCategory.AI_CHAT,
        tags=["chat_request"],
    )

    # Build system prompt with optional log context
    system_prompt = config.get(
        "aichat.system_prompt",
        "You are the AceTaskAgent AI Assistant. Help users with product information, "
        "development guidance, workflow design, troubleshooting, and debugging."
    )

    if data.include_logs:
        # Inject recent logs and health summary for AI context
        health = ai_logger.get_system_health_summary()
        recent_errors = ai_logger.get_error_summary(limit=10)
        recent_logs = ai_logger.get_recent_logs(limit=30)

        context_block = (
            "\n\n--- APPLICATION CONTEXT (for your reference) ---\n"
            f"System Health: {health.get('health_status', 'unknown')} "
            f"(errors={health.get('error_count', 0)}, warnings={health.get('warning_count', 0)})\n"
        )
        if recent_errors:
            context_block += "Recent Errors:\n"
            for err in recent_errors[:5]:
                context_block += f"  - [{err.get('timestamp','')}] {err.get('message','')}\n"
        if recent_logs:
            context_block += f"\nRecent Activity ({len(recent_logs)} entries):\n"
            for log in recent_logs[:10]:
                context_block += (
                    f"  [{log.get('level','')}][{log.get('category','')}] "
                    f"{log.get('message','')}\n"
                )
        context_block += "--- END CONTEXT ---"
        system_prompt = (system_prompt or "") + context_block

    # Build messages payload
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    max_history = config.get_int("aichat.max_history", 20)
    for msg in data.history[-max_history:]:
        messages.append({"role": msg.role, "content": msg.content})

    # Add current user message
    messages.append({"role": "user", "content": data.message})

    # Call DeepSeek API
    api_key = config.get("deepseek.api.key") or os.getenv("LLM_API_KEY", "")
    base_url = config.get("deepseek.api.base_url", "https://api.deepseek.com") or "https://api.deepseek.com"
    endpoint_path = config.get("deepseek.api.endpoint_path", "v1") or "v1"
    model = config.get("aichat.model", config.get("deepseek.api.model", "deepseek-chat")) or "deepseek-chat"
    temperature = config.get_float("aichat.temperature", 0.7)
    max_tokens = config.get_int("aichat.max_tokens", 2048)
    timeout = config.get_int("deepseek.api.timeout", 60)

    url = f"{base_url.rstrip('/')}/{endpoint_path.strip('/')}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)

        duration_ms = (time.time() - start_time) * 1000

        if resp.status_code != 200:
            error_text = resp.text[:300]
            ai_logger.log_api_call(
                endpoint=url, method="POST", status_code=resp.status_code,
                duration_ms=duration_ms, error=error_text,
            )
            raise HTTPException(status_code=502, detail=f"DeepSeek API error: {error_text}")

        result = resp.json()
        assistant_message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens = result.get("usage", {}).get("total_tokens")

        ai_logger.log_llm_interaction(
            provider="deepseek", model=model,
            prompt_summary=data.message[:200],
            response_summary=assistant_message[:200],
            tokens_used=tokens,
            duration_ms=duration_ms,
        )

        return {
            "response": assistant_message,
            "tokens_used": tokens,
            "duration_ms": round(duration_ms, 1),
        }

    except httpx.TimeoutException:
        duration_ms = (time.time() - start_time) * 1000
        ai_logger.log_error(
            "DeepSeek API timeout",
            category=LogCategory.AI_CHAT,
            tags=["timeout"],
            performance={"duration_ms": duration_ms},
        )
        raise HTTPException(status_code=504, detail="DeepSeek API request timed out")
    except HTTPException:
        raise
    except Exception as e:
        ai_logger.log_error(f"AI Chat error: {str(e)}", exc=e, category=LogCategory.AI_CHAT)
        raise HTTPException(status_code=500, detail=f"AI Chat error: {str(e)}")


@app.get("/api/aichat/health")
async def ai_chat_health():
    """Return system health summary for the AI chat context panel."""
    return ai_logger.get_system_health_summary()


@app.get("/api/logs/recent")
async def get_recent_logs(
    limit: int = 50,
    level: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
):
    """Return recent structured log entries."""
    logs = ai_logger.get_recent_logs(limit=limit, level=level, category=category, search_text=search)
    return {"logs": logs, "total": len(logs)}


# ============================================================================
# LANGGRAPH WORKFLOW INTROSPECTION API
# ============================================================================

@app.get("/api/langgraph/workflow")
async def get_langgraph_workflow():
    """
    Return the complete real LangGraph workflow topology as structured JSON.
    This reflects the actual graph wired in task_manager/core/workflow.py.
    """
    nodes = [
        # ── Core orchestration nodes ──────────────────────────────────────
        {
            "id": "initialize",
            "label": "Initialize",
            "category": "core",
            "description": "Creates the root task from the user objective and sets up initial workflow state.",
            "color": "#6366f1",
            "icon": "play",
        },
        {
            "id": "select_task",
            "label": "Select Task",
            "category": "core",
            "description": "Picks the next PENDING task from the queue. Routes to analyze_task, human_review, or END.",
            "color": "#6366f1",
            "icon": "list-checks",
        },
        {
            "id": "analyze_task",
            "label": "Analyze Task",
            "category": "core",
            "description": "LLM determines whether to break the task into subtasks or execute directly via a specific sub-agent.",
            "color": "#6366f1",
            "icon": "brain",
        },
        {
            "id": "breakdown_task",
            "label": "Breakdown Task",
            "category": "core",
            "description": "Master Planner decomposes the task into parallel/sequential subtasks and adds them to the queue.",
            "color": "#6366f1",
            "icon": "git-fork",
        },
        {
            "id": "aggregate_results",
            "label": "Aggregate Results",
            "category": "core",
            "description": "Collects completed sub-task results onto the blackboard. Routes to synthesis, human review, or continues.",
            "color": "#6366f1",
            "icon": "layers",
        },
        {
            "id": "synthesize_research",
            "label": "Synthesize Research",
            "category": "synthesis",
            "description": "Multi-level synthesis node: detects contradictions across all agent findings and produces a unified report.",
            "color": "#0ea5e9",
            "icon": "sparkles",
        },
        {
            "id": "agentic_debate",
            "label": "Agentic Debate",
            "category": "synthesis",
            "description": "Spawns Fact-Checker & Lead Researcher personas to resolve contradictions when score > 0.7. Produces consensus.",
            "color": "#0ea5e9",
            "icon": "message-square",
        },
        {
            "id": "auto_synthesis",
            "label": "Auto Synthesis",
            "category": "synthesis",
            "description": "Observer-pattern node triggered by OCR/WebSearch events (last_updated_key). Runs incremental synthesis.",
            "color": "#0ea5e9",
            "icon": "zap",
        },
        {
            "id": "handle_error",
            "label": "Handle Error",
            "category": "control",
            "description": "Logs errors and resets failed task state. Always routes back to select_task.",
            "color": "#ef4444",
            "icon": "alert-triangle",
        },
        {
            "id": "human_review",
            "label": "Human Review",
            "category": "control",
            "description": "Pauses workflow for human intervention on failed tasks. Routes to analyze_task (retry) or select_task (skip).",
            "color": "#f59e0b",
            "icon": "user-check",
        },
        # ── Sub-agent execution nodes ──────────────────────────────────────
        {
            "id": "execute_task",
            "label": "Execute Task",
            "category": "subagent",
            "description": "Generic task execution fallback. Handles task types not matched by specialised execution nodes.",
            "color": "#64748b",
            "icon": "cpu",
        },
        {
            "id": "execute_pdf_task",
            "label": "PDF Agent",
            "category": "subagent",
            "description": "Extracts text, tables and images from PDF documents. Chains to OCR if images are found.",
            "color": "#ef4444",
            "icon": "file-text",
        },
        {
            "id": "execute_excel_task",
            "label": "Excel Agent",
            "category": "subagent",
            "description": "Processes and analyses Excel/CSV data. Can receive chained output from OCR or Web Search.",
            "color": "#22c55e",
            "icon": "table",
        },
        {
            "id": "execute_ocr_task",
            "label": "OCR / Image Agent",
            "category": "subagent",
            "description": "Extracts text from images via Tesseract or Vision LLM. Chains to Excel if a table is detected.",
            "color": "#f59e0b",
            "icon": "image",
        },
        {
            "id": "execute_web_search_task",
            "label": "Web Search Agent",
            "category": "subagent",
            "description": "Searches the web via DuckDuckGo. Chains to Excel if a CSV is generated; triggers auto-synthesis on web_findings event.",
            "color": "#3b82f6",
            "icon": "search",
        },
        {
            "id": "execute_code_interpreter_task",
            "label": "Code Interpreter",
            "category": "subagent",
            "description": "Executes Python code for computation and data analysis. Chains to OCR if charts/images are produced.",
            "color": "#8b5cf6",
            "icon": "code",
        },
        {
            "id": "execute_data_extraction_task",
            "label": "Data Extraction",
            "category": "subagent",
            "description": "Extracts structured data from documents and APIs. Feeds results into the blackboard.",
            "color": "#06b6d4",
            "icon": "database",
        },
        {
            "id": "execute_problem_solver_task",
            "label": "Problem Solver",
            "category": "subagent",
            "description": "LLM-based analysis, reasoning and synthesis agent.",
            "color": "#ec4899",
            "icon": "lightbulb",
        },
        {
            "id": "execute_document_task",
            "label": "Document Generator",
            "category": "subagent",
            "description": "Generates Markdown / HTML / PDF reports and documents from aggregated findings.",
            "color": "#14b8a6",
            "icon": "file-output",
        },
        {
            "id": "__end__",
            "label": "END",
            "category": "terminal",
            "description": "Workflow complete. Final state contains the synthesised output and all sub-task results.",
            "color": "#374151",
            "icon": "flag",
        },
    ]

    edges = [
        # ── Entry ──────────────────────────────────────────────────────────
        {"id": "e1",  "source": "initialize",      "target": "select_task",                  "type": "direct",      "label": ""},
        # ── select_task conditional ────────────────────────────────────────
        {"id": "e2",  "source": "select_task",      "target": "analyze_task",                 "type": "conditional", "label": "pending task found"},
        {"id": "e3",  "source": "select_task",      "target": "handle_error",                 "type": "conditional", "label": "error state"},
        {"id": "e4",  "source": "select_task",      "target": "human_review",                 "type": "conditional", "label": "requires_human_review"},
        {"id": "e5",  "source": "select_task",      "target": "__end__",                      "type": "conditional", "label": "no pending tasks"},
        # ── analyze_task conditional ───────────────────────────────────────
        {"id": "e6",  "source": "analyze_task",     "target": "breakdown_task",               "type": "conditional", "label": "breakdown"},
        {"id": "e7",  "source": "analyze_task",     "target": "execute_task",                 "type": "conditional", "label": "generic execute"},
        {"id": "e8",  "source": "analyze_task",     "target": "execute_pdf_task",             "type": "conditional", "label": "pdf"},
        {"id": "e9",  "source": "analyze_task",     "target": "execute_excel_task",           "type": "conditional", "label": "excel"},
        {"id": "e10", "source": "analyze_task",     "target": "execute_ocr_task",             "type": "conditional", "label": "ocr"},
        {"id": "e11", "source": "analyze_task",     "target": "execute_web_search_task",      "type": "conditional", "label": "web_search"},
        {"id": "e12", "source": "analyze_task",     "target": "execute_code_interpreter_task","type": "conditional", "label": "code_interpreter"},
        {"id": "e13", "source": "analyze_task",     "target": "execute_data_extraction_task", "type": "conditional", "label": "data_extraction"},
        {"id": "e14", "source": "analyze_task",     "target": "execute_problem_solver_task",  "type": "conditional", "label": "problem_solver"},
        {"id": "e15", "source": "analyze_task",     "target": "execute_document_task",        "type": "conditional", "label": "document"},
        {"id": "e16", "source": "analyze_task",     "target": "handle_error",                 "type": "conditional", "label": "error"},
        {"id": "e17", "source": "analyze_task",     "target": "human_review",                 "type": "conditional", "label": "review"},
        # ── breakdown ─────────────────────────────────────────────────────
        {"id": "e18", "source": "breakdown_task",   "target": "select_task",                  "type": "direct",      "label": "subtasks queued"},
        # ── generic execute ────────────────────────────────────────────────
        {"id": "e19", "source": "execute_task",     "target": "aggregate_results",            "type": "direct",      "label": ""},
        # ── PDF chain ─────────────────────────────────────────────────────
        {"id": "e20", "source": "execute_pdf_task", "target": "execute_ocr_task",             "type": "chain",       "label": "images found → chain"},
        {"id": "e21", "source": "execute_pdf_task", "target": "aggregate_results",            "type": "conditional", "label": "no images"},
        # ── OCR chain ─────────────────────────────────────────────────────
        {"id": "e22", "source": "execute_ocr_task", "target": "auto_synthesis",               "type": "chain",       "label": "ocr_results event"},
        {"id": "e23", "source": "execute_ocr_task", "target": "execute_excel_task",           "type": "chain",       "label": "table extracted → chain"},
        {"id": "e24", "source": "execute_ocr_task", "target": "aggregate_results",            "type": "conditional", "label": "normal"},
        # ── Web Search chain ──────────────────────────────────────────────
        {"id": "e25", "source": "execute_web_search_task", "target": "auto_synthesis",        "type": "chain",       "label": "web_findings event"},
        {"id": "e26", "source": "execute_web_search_task", "target": "execute_excel_task",    "type": "chain",       "label": "CSV generated → chain"},
        {"id": "e27", "source": "execute_web_search_task", "target": "aggregate_results",     "type": "conditional", "label": "normal"},
        # ── Code Interpreter chain ────────────────────────────────────────
        {"id": "e28", "source": "execute_code_interpreter_task", "target": "execute_ocr_task","type": "chain",       "label": "charts/images → chain"},
        {"id": "e29", "source": "execute_code_interpreter_task", "target": "aggregate_results","type": "conditional","label": "no images"},
        # ── Simple aggregation edges ──────────────────────────────────────
        {"id": "e30", "source": "execute_excel_task",            "target": "aggregate_results","type": "direct",      "label": ""},
        {"id": "e31", "source": "execute_data_extraction_task",  "target": "aggregate_results","type": "direct",      "label": ""},
        {"id": "e32", "source": "execute_problem_solver_task",   "target": "aggregate_results","type": "direct",      "label": ""},
        {"id": "e33", "source": "execute_document_task",         "target": "aggregate_results","type": "direct",      "label": ""},
        {"id": "e34", "source": "auto_synthesis",                "target": "aggregate_results","type": "direct",      "label": "observer complete"},
        # ── aggregate_results conditional ─────────────────────────────────
        {"id": "e35", "source": "aggregate_results", "target": "human_review",                "type": "conditional", "label": "failed task"},
        {"id": "e36", "source": "aggregate_results", "target": "synthesize_research",         "type": "conditional", "label": "all done + ≥2 agents"},
        {"id": "e37", "source": "aggregate_results", "target": "select_task",                 "type": "conditional", "label": "continue"},
        {"id": "e38", "source": "aggregate_results", "target": "__end__",                     "type": "conditional", "label": "complete"},
        {"id": "e39", "source": "aggregate_results", "target": "__end__",                     "type": "conditional", "label": "max_iterations"},
        # ── synthesize_research conditional ──────────────────────────────
        {"id": "e40", "source": "synthesize_research", "target": "agentic_debate",            "type": "conditional", "label": "contradiction score > 0.7"},
        {"id": "e41", "source": "synthesize_research", "target": "select_task",               "type": "conditional", "label": "continue"},
        {"id": "e42", "source": "synthesize_research", "target": "__end__",                   "type": "conditional", "label": "complete"},
        {"id": "e43", "source": "synthesize_research", "target": "__end__",                   "type": "conditional", "label": "max_iterations"},
        # ── agentic_debate conditional ────────────────────────────────────
        {"id": "e44", "source": "agentic_debate", "target": "select_task",                    "type": "conditional", "label": "continue"},
        {"id": "e45", "source": "agentic_debate", "target": "__end__",                        "type": "conditional", "label": "complete"},
        {"id": "e46", "source": "agentic_debate", "target": "__end__",                        "type": "conditional", "label": "max_iterations"},
        # ── error / review ─────────────────────────────────────────────────
        {"id": "e47", "source": "handle_error",   "target": "select_task",                    "type": "direct",      "label": ""},
        {"id": "e48", "source": "human_review",   "target": "analyze_task",                   "type": "conditional", "label": "reset to PENDING"},
        {"id": "e49", "source": "human_review",   "target": "select_task",                    "type": "conditional", "label": "completed / failed"},
    ]

    # Layer layout hints (y-band), used by the front-end for DAG placement
    layer_map = {
        "initialize": 0,
        "select_task": 1,
        "analyze_task": 2,
        "breakdown_task": 3,
        "execute_task": 4,
        "execute_pdf_task": 4,
        "execute_excel_task": 4,
        "execute_ocr_task": 4,
        "execute_web_search_task": 4,
        "execute_code_interpreter_task": 4,
        "execute_data_extraction_task": 4,
        "execute_problem_solver_task": 4,
        "execute_document_task": 4,
        "auto_synthesis": 5,
        "aggregate_results": 6,
        "synthesize_research": 7,
        "agentic_debate": 8,
        "handle_error": 3,
        "human_review": 3,
        "__end__": 9,
    }
    for n in nodes:
        n["layer"] = layer_map.get(n["id"], 5)  # type: ignore[assignment]

    chain_groups = [
        {
            "id": "chain_pdf_ocr",
            "label": "PDF → OCR Chain",
            "color": "#f97316",
            "nodes": ["execute_pdf_task", "execute_ocr_task"],
            "description": "PDF agent automatically chains to OCR when images are extracted from the document.",
        },
        {
            "id": "chain_ocr_excel",
            "label": "OCR → Excel Chain",
            "color": "#84cc16",
            "nodes": ["execute_ocr_task", "execute_excel_task"],
            "description": "OCR agent chains to Excel when tabular data is detected in the extracted content.",
        },
        {
            "id": "chain_web_excel",
            "label": "WebSearch → Excel Chain",
            "color": "#38bdf8",
            "nodes": ["execute_web_search_task", "execute_excel_task"],
            "description": "Web Search agent chains to Excel when a CSV file is generated from search results.",
        },
        {
            "id": "chain_code_ocr",
            "label": "Code → OCR Chain",
            "color": "#c084fc",
            "nodes": ["execute_code_interpreter_task", "execute_ocr_task"],
            "description": "Code Interpreter chains to OCR when charts or images are produced by the code execution.",
        },
        {
            "id": "observer_ocr",
            "label": "OCR Observer",
            "color": "#fb923c",
            "nodes": ["execute_ocr_task", "auto_synthesis"],
            "description": "Event-driven: ocr_results key triggers auto-synthesis before regular aggregation.",
        },
        {
            "id": "observer_web",
            "label": "WebSearch Observer",
            "color": "#60a5fa",
            "nodes": ["execute_web_search_task", "auto_synthesis"],
            "description": "Event-driven: web_findings key triggers auto-synthesis before regular aggregation.",
        },
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "chain_groups": chain_groups,
        "entry_point": "initialize",
        "end_node": "__end__",
        "categories": {
            "core":      {"label": "Core Orchestration",  "color": "#6366f1"},
            "subagent":  {"label": "Sub-Agent Execution", "color": "#64748b"},
            "synthesis": {"label": "Synthesis / Debate",  "color": "#0ea5e9"},
            "control":   {"label": "Control Flow",        "color": "#f59e0b"},
            "terminal":  {"label": "Terminal",            "color": "#374151"},
        },
        "edge_types": {
            "direct":      {"label": "Direct edge",           "style": "solid",  "color": "#6b7280"},
            "conditional": {"label": "Conditional routing",   "style": "dashed", "color": "#9ca3af"},
            "chain":       {"label": "Chain execution",       "style": "chain",  "color": "#f97316"},
        },
    }


# ============================================================================
# LANGGRAPH CODE PARSER API
# ============================================================================

class LangGraphParseRequest(BaseModel):
    code: str = Field(..., description="Raw Python source code containing a LangGraph StateGraph definition")
    filename: str = Field("", description="Optional filename hint (used for display only)")


@app.post("/api/langgraph/parse")
async def parse_langgraph_code(data: LangGraphParseRequest):
    """
    Parse arbitrary LangGraph Python code and return a visual graph topology.

    Handles all common LangGraph patterns:
      - StateGraph(State) construction
      - workflow.add_node("name", fn)
      - workflow.set_entry_point("name")
      - workflow.add_edge("src", "dst")                    — direct edges
      - workflow.add_edge("src", END)                      — terminal edges
      - workflow.add_conditional_edges(                    — conditional fan-out
            "src", router_fn, {"key": "dst", ...})
      - Inline lambda router:  lambda s: "a" if … else "b"
      - Router function bodies: if/elif/else → condition labels on each branch
      - Synthetic DECISION nodes inserted for every conditional split
      - graph.compile(...)                                 — recognised but not rendered
      - Multiple graph variable names (workflow / graph / app / builder / sg)
    """
    code = data.code.strip()
    if not code:
        raise HTTPException(status_code=422, detail="Empty code submitted")

    # ── Colour palette for auto-assigned colours ──────────────────────────
    _PALETTE = [
        "#6366f1", "#3b82f6", "#22c55e", "#f59e0b", "#ef4444",
        "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6", "#f97316",
        "#84cc16", "#a855f7", "#0ea5e9", "#64748b", "#10b981",
    ]

    def _color_for(name: str) -> str:
        idx = int(hashlib.md5(name.encode()).hexdigest(), 16) % len(_PALETTE)
        return _PALETTE[idx]

    # ── Known agent-type colour map (matches designer palette) ───────────
    _AGENT_COLORS: Dict[str, str] = {
        "web_search": "#3b82f6",
        "pdf": "#ef4444",
        "excel": "#22c55e",
        "ocr": "#f59e0b",
        "code_interpreter": "#8b5cf6",
        "data_extraction": "#06b6d4",
        "problem_solver": "#ec4899",
        "document": "#14b8a6",
        "mermaid": "#a855f7",
        # core orchestration
        "initialize": "#6366f1",
        "select_task": "#6366f1",
        "analyze_task": "#6366f1",
        "breakdown_task": "#6366f1",
        "aggregate_results": "#6366f1",
        "synthesize_research": "#0ea5e9",
        "agentic_debate": "#0ea5e9",
        "auto_synthesis": "#0ea5e9",
        "handle_error": "#ef4444",
        "human_review": "#f59e0b",
    }

    # ── Category inference ────────────────────────────────────────────────
    def _infer_category(name: str) -> str:
        n = name.lower()
        if any(k in n for k in ("synthesize", "synthesis", "debate", "auto_synth")):
            return "synthesis"
        if any(k in n for k in ("error", "review", "human", "checkpoint")):
            return "control"
        if n in ("__end__", "end"):
            return "terminal"
        if any(k in n for k in ("execute_", "agent", "search", "pdf", "excel", "ocr",
                                  "code_interp", "data_extract", "document", "mermaid",
                                  "problem_solver")):
            return "subagent"
        if any(k in n for k in ("initialize", "init", "select_task", "analyze_task",
                                  "breakdown", "aggregate")):
            return "core"
        return "task"

    # ── Layer heuristic (top-down ordering) ──────────────────────────────
    _LAYER_HINTS: Dict[str, int] = {
        "initialize": 0, "init": 0,
        "select_task": 1,
        "analyze_task": 2,
        "breakdown_task": 3, "breakdown": 3,
        "handle_error": 3, "human_review": 3,
        "aggregate_results": 6, "aggregate": 6,
        "synthesize_research": 7, "synthesize": 7,
        "agentic_debate": 8, "debate": 8,
        "__end__": 9, "end": 9,
    }

    # ── AST-based extraction ──────────────────────────────────────────────
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Python syntax error at line {exc.lineno}: {exc.msg}",
        )

    # Detect all variable names assigned a StateGraph instance
    graph_vars: set[str] = set()
    _GRAPH_CONSTRUCTORS = {"StateGraph", "MessageGraph", "Graph"}

    class _GraphVarFinder(ast.NodeVisitor):
        def visit_Assign(self, node):  # type: ignore[override]
            if isinstance(node.value, ast.Call):
                func = node.value.func
                fname = ""
                if isinstance(func, ast.Name):
                    fname = func.id
                elif isinstance(func, ast.Attribute):
                    fname = func.attr
                if fname in _GRAPH_CONSTRUCTORS:
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            graph_vars.add(t.id)
            self.generic_visit(node)

    _GraphVarFinder().visit(tree)

    # Fallback: common names used for StateGraph variables
    _FALLBACK_VAR_NAMES = {"workflow", "graph", "app", "builder", "sg", "wf", "g"}
    if not graph_vars:
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute):
                    obj = call.func.value
                    method = call.func.attr
                    if method in ("add_node", "add_edge", "add_conditional_edges", "set_entry_point"):
                        if isinstance(obj, ast.Name):
                            graph_vars.add(obj.id)
        if not graph_vars:
            graph_vars = _FALLBACK_VAR_NAMES

    # ── Collect all top-level function definitions for router analysis ────
    # Maps function name → list of (condition_str, return_value) tuples
    router_functions: Dict[str, Dict[str, Any]] = {}

    def _extract_router_branches(fn_node: ast.FunctionDef) -> Dict[str, Any]:
        """
        Walk a router function body and extract if/elif/else branch conditions
        mapped to their return values. Returns:
          {
            "branches": [{"condition": str, "returns": str}, ...],
            "default_return": str | None,
            "source": str  (unparsed body snippet)
          }
        """
        branches = []
        default_ret = None

        def _returns_in(stmts) -> Optional[str]:
            """Find the first Return value string inside a statement list."""
            for s in stmts:
                if isinstance(s, ast.Return):
                    if s.value is not None:
                        try:
                            return ast.unparse(s.value).strip("\"'")
                        except Exception:
                            return None
                # recurse one level for nested ifs
                elif isinstance(s, ast.If):
                    ret = _returns_in(s.body)
                    if ret:
                        return ret
            return None

        for stmt in fn_node.body:
            if isinstance(stmt, ast.If):
                _walk_if_chain(stmt, branches)
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                # Bare return at function root = default / unconditional
                try:
                    default_ret = ast.unparse(stmt.value).strip("\"'")
                except Exception:
                    pass

        # If no branches but there's a default return, record it
        if not branches and default_ret:
            branches.append({"condition": "default", "returns": default_ret})

        try:
            source_snippet = ast.unparse(fn_node)[:300]
        except Exception:
            source_snippet = ""

        return {
            "branches": branches,
            "default_return": default_ret,
            "source": source_snippet,
        }

    def _walk_if_chain(if_node: ast.If, branches: list) -> None:
        """Recursively walk if / elif / else chain and append branch dicts."""
        try:
            cond_str = ast.unparse(if_node.test)
        except Exception:
            cond_str = "<condition>"

        # Shorten long condition strings
        if len(cond_str) > 60:
            cond_str = cond_str[:57] + "…"

        # Find return value inside this branch body
        ret = None
        for s in if_node.body:
            if isinstance(s, ast.Return) and s.value is not None:
                try:
                    ret = ast.unparse(s.value).strip("\"'")
                except Exception:
                    ret = None
                break

        if ret:
            branches.append({"condition": cond_str, "returns": ret})

        # Handle elif (orelse is a single If node) and else (orelse is a list)
        if if_node.orelse:
            if len(if_node.orelse) == 1 and isinstance(if_node.orelse[0], ast.If):
                _walk_if_chain(if_node.orelse[0], branches)
            else:
                # else branch
                else_ret = None
                for s in if_node.orelse:
                    if isinstance(s, ast.Return) and s.value is not None:
                        try:
                            else_ret = ast.unparse(s.value).strip("\"'")
                        except Exception:
                            pass
                        break
                if else_ret:
                    branches.append({"condition": "else", "returns": else_ret})

    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            router_functions[fn.name] = _extract_router_branches(fn)  # type: ignore[arg-type]

    # ── Collect all method calls on graph vars ────────────────────────────
    raw_nodes: Dict[str, Dict[str, Any]] = {}
    raw_edges: list[Dict[str, Any]] = []
    entry_point: str = ""
    end_aliases: set[str] = {"END", "__end__", "end"}

    def _str_val(node) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        return ""

    def _get_dict_mapping(node) -> Dict[str, str]:
        result = {}
        if not isinstance(node, ast.Dict):
            return result
        for k, v in zip(node.keys, node.values):
            key = _str_val(k)   # type: ignore[arg-type]
            val = _str_val(v)   # type: ignore[arg-type]
            if key:
                result[key] = val
        return result

    edge_id_counter = 0

    def _new_edge_id() -> str:
        nonlocal edge_id_counter
        edge_id_counter += 1
        return f"e{edge_id_counter}"

    def _ensure_node(name: str) -> None:
        if name not in raw_nodes:
            raw_nodes[name] = {
                "id": name,
                "label": name.replace("_", " ").title(),
                "category": _infer_category(name),
                "color": _AGENT_COLORS.get(name, _color_for(name)),
                "description": f"Node: {name}",
            }

    # ── Pending conditional edges — resolved after all nodes collected ────
    # Each entry: (src_node_id, router_name, branch_map, lambda_body)
    pending_conditionals: list[tuple] = []

    for stmt in ast.walk(tree):
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            continue
        call = stmt.value
        if not isinstance(call.func, ast.Attribute):
            continue
        obj = call.func.value
        method = call.func.attr
        if not (isinstance(obj, ast.Name) and obj.id in graph_vars):
            continue

        args = call.args
        kwargs = {kw.arg: kw.value for kw in call.keywords}

        # ── add_node("name", fn) ──────────────────────────────────────────
        if method == "add_node":
            if args:
                name = _str_val(args[0])
                if name:
                    handler_name = ""
                    if len(args) >= 2:
                        h = args[1]
                        if isinstance(h, ast.Attribute):
                            handler_name = f"{ast.unparse(h.value)}.{h.attr}"
                        elif isinstance(h, ast.Name):
                            handler_name = h.id
                        elif isinstance(h, ast.Lambda):
                            handler_name = "<lambda>"
                    _ensure_node(name)
                    if handler_name:
                        raw_nodes[name]["description"] = f"Handler: {handler_name}"

        # ── set_entry_point("name") ───────────────────────────────────────
        elif method == "set_entry_point":
            if args:
                ep = _str_val(args[0])
                if ep:
                    entry_point = ep
                    _ensure_node(ep)

        # ── add_edge("src", "dst" | END) ─────────────────────────────────
        elif method == "add_edge":
            if len(args) >= 2:
                src = _str_val(args[0])
                dst = _str_val(args[1])
                if src and dst:
                    _ensure_node(src)
                    _ensure_node(dst)
                    is_end_edge = dst in end_aliases
                    raw_edges.append({
                        "id": _new_edge_id(),
                        "source": src,
                        "target": "__end__" if is_end_edge else dst,
                        "type": "direct",
                        "label": "",
                    })
                    if is_end_edge:
                        _ensure_node("__end__")

        # ── add_conditional_edges("src", router, {"key":"dst", ...}) ──────
        elif method == "add_conditional_edges":
            if not args:
                continue
            src = _str_val(args[0])
            if not src:
                continue
            _ensure_node(src)

            # Router function name / lambda body
            router_name = ""
            lambda_body = ""
            if len(args) >= 2:
                r = args[1]
                if isinstance(r, ast.Name):
                    router_name = r.id
                elif isinstance(r, ast.Attribute):
                    router_name = r.attr
                elif isinstance(r, ast.Lambda):
                    try:
                        lambda_body = ast.unparse(r.body)[:80]
                        router_name = f"λ: {lambda_body}"
                    except Exception:
                        router_name = "<lambda>"

            # Branch mapping dict
            branch_map: Dict[str, str] = {}
            if len(args) >= 3:
                branch_map = _get_dict_mapping(args[2])
            elif "path_map" in kwargs:
                branch_map = _get_dict_mapping(kwargs["path_map"])

            pending_conditionals.append((src, router_name, branch_map, lambda_body))

    # ── Resolve conditional edges — insert decision nodes ─────────────────
    # For each add_conditional_edges call we insert a synthetic DECISION node
    # between the source and all its conditional targets so the diagram shows
    # a clear fork point (diamond shape in the frontend).
    decision_nodes_created: Dict[str, str] = {}   # src → decision_node_id

    for src, router_name, branch_map, lambda_body in pending_conditionals:
        # Create a unique decision node id for this source
        decision_id = f"__decision__{src}"
        if decision_id not in raw_nodes:
            short_router = router_name.split("(")[0].strip()
            if short_router.startswith("λ:"):
                short_router = "λ condition"
            display_label = short_router if short_router else "Route"
            # Build condition summary from router function body if available
            condition_info = router_functions.get(router_name, {})
            branches_desc = ""
            if condition_info and condition_info.get("branches"):
                branches_desc = "; ".join(
                    f"{b['condition']} → {b['returns']}"
                    for b in condition_info["branches"][:6]
                )
            elif lambda_body:
                branches_desc = lambda_body

            raw_nodes[decision_id] = {
                "id": decision_id,
                "label": display_label[:18] if display_label else "Route",
                "category": "condition",
                "color": "#f59e0b",
                "description": (
                    f"Router: {router_name or 'conditional'}"
                    + (f"\n{branches_desc}" if branches_desc else "")
                ),
                "router_name": router_name,
                "router_branches": condition_info.get("branches", []) if condition_info else [],
                "lambda_body": lambda_body,
            }

        decision_nodes_created[src] = decision_id

        # Edge from source → decision node (direct)
        raw_edges.append({
            "id": _new_edge_id(),
            "source": src,
            "target": decision_id,
            "type": "direct",
            "label": "",
        })

        if branch_map:
            # Enrich branch labels with condition text from router function body
            router_info = router_functions.get(router_name, {})
            branch_conditions: Dict[str, str] = {}
            if router_info and router_info.get("branches"):
                for b in router_info["branches"]:
                    ret = b.get("returns", "")
                    cond = b.get("condition", "")
                    if ret and ret in branch_map.values():
                        # Map destination name → condition string
                        branch_conditions[ret] = cond

            for cond_key, dst in branch_map.items():
                is_end = dst in end_aliases
                real_dst = "__end__" if is_end else dst
                _ensure_node(real_dst)

                # Build the best possible label:
                # 1. condition expression from router body  2. branch map key
                label = cond_key if cond_key not in end_aliases else "complete"
                condition_text = branch_conditions.get(real_dst, "")
                raw_edges.append({
                    "id": _new_edge_id(),
                    "source": decision_id,
                    "target": real_dst,
                    "type": "conditional",
                    "label": label,
                    "condition": condition_text,
                    "router": router_name,
                })
        else:
            # No branch map — single conditional edge with condition label
            raw_edges.append({
                "id": _new_edge_id(),
                "source": decision_id,
                "target": "__unknown__",
                "type": "conditional",
                "label": router_name or "conditional",
                "condition": "",
                "router": router_name,
            })
            _ensure_node("__unknown__")

    # ── If no nodes found at all, try regex fallback ──────────────────────
    if not raw_nodes:
        _re_add_node = re.compile(r'\.add_node\(\s*["\'](\w+)["\']')
        _re_add_edge = re.compile(r'\.add_edge\(\s*["\'](\w+)["\']\s*,\s*["\']?(\w+)["\']?')
        _re_entry    = re.compile(r'\.set_entry_point\(\s*["\'](\w+)["\']')
        _re_cond     = re.compile(r'\.add_conditional_edges\(\s*["\'](\w+)["\']\s*,\s*[^,]+,\s*\{([^}]+)\}')
        for m in _re_add_node.finditer(code):
            _ensure_node(m.group(1))
        for m in _re_add_edge.finditer(code):
            src, dst = m.group(1), m.group(2)
            _ensure_node(src); _ensure_node(dst)
            raw_edges.append({
                "id": _new_edge_id(), "source": src,
                "target": "__end__" if dst in end_aliases else dst,
                "type": "direct", "label": "",
            })
        for m in _re_entry.finditer(code):
            entry_point = m.group(1)
        for m in _re_cond.finditer(code):
            src = m.group(1)
            _ensure_node(src)
            decision_id = f"__decision__{src}"
            if decision_id not in raw_nodes:
                raw_nodes[decision_id] = {
                    "id": decision_id, "label": "Route",
                    "category": "condition", "color": "#f59e0b",
                    "description": f"Conditional routing from {src}",
                    "router_name": "", "router_branches": [], "lambda_body": "",
                }
            raw_edges.append({"id": _new_edge_id(), "source": src, "target": decision_id, "type": "direct", "label": ""})
            for pair in re.finditer(r'["\'](\w+)["\']\s*:\s*["\']?(\w+)["\']?', m.group(2)):
                k, v = pair.group(1), pair.group(2)
                real_dst = "__end__" if v in end_aliases else v
                _ensure_node(real_dst)
                raw_edges.append({
                    "id": _new_edge_id(), "source": decision_id, "target": real_dst,
                    "type": "conditional", "label": k, "condition": "", "router": "",
                })

    if not raw_nodes:
        raise HTTPException(
            status_code=422,
            detail="No LangGraph nodes found. Make sure the code uses StateGraph.add_node(), "
                   "add_edge(), or add_conditional_edges().",
        )

    # ── Ensure __end__ node exists if referenced ──────────────────────────
    has_end_edges = any(e["target"] == "__end__" for e in raw_edges)
    if has_end_edges and "__end__" not in raw_nodes:
        _ensure_node("__end__")
    if "__end__" in raw_nodes:
        raw_nodes["__end__"]["label"] = "END"
        raw_nodes["__end__"]["category"] = "terminal"
        raw_nodes["__end__"]["color"] = "#374151"

    # ── Auto-detect entry point if not found via set_entry_point ─────────
    if not entry_point:
        all_targets = {e["target"] for e in raw_edges}
        candidates = [nid for nid in raw_nodes if nid not in all_targets and nid != "__end__"
                      and not nid.startswith("__decision__")]
        if candidates:
            for c in ("initialize", "init", "start", "intake"):
                if c in candidates:
                    entry_point = c
                    break
            if not entry_point:
                entry_point = candidates[0]

    # ── Assign layers (BFS from entry point) ─────────────────────────────
    adj: Dict[str, list] = defaultdict(list)
    for e in raw_edges:
        adj[e["source"]].append(e["target"])

    depth: Dict[str, int] = {}
    if entry_point and entry_point in raw_nodes:
        queue: deque = deque([(entry_point, 0)])
        while queue:
            node_id, d = queue.popleft()
            if node_id in depth:
                continue
            depth[node_id] = d
            for nxt in adj.get(node_id, []):
                if nxt not in depth:
                    queue.append((nxt, d + 1))

    for nid, nm in raw_nodes.items():
        layer = _LAYER_HINTS.get(nid, depth.get(nid, 5))
        nm["layer"] = layer  # type: ignore[assignment]

    # ── Infer chain groups ────────────────────────────────────────────────
    chain_groups: list[Dict[str, Any]] = []
    _chain_keywords = ("image", "ocr", "csv", "excel", "table", "chart", "chain", "found", "created")
    seen_chains: set[str] = set()
    for e in raw_edges:
        if e.get("type") == "conditional" and e.get("label"):
            lbl = e["label"].lower()
            if any(k in lbl for k in _chain_keywords):
                pair_key = f"{e['source']}→{e['target']}"
                if pair_key not in seen_chains:
                    seen_chains.add(pair_key)
                    cg_id = f"chain_{e['source']}_{e['target']}"
                    chain_groups.append({
                        "id": cg_id,
                        "label": f"{raw_nodes.get(e['source'], {}).get('label', e['source'])} → "
                                 f"{raw_nodes.get(e['target'], {}).get('label', e['target'])}",
                        "color": "#f97316",
                        "nodes": [e["source"], e["target"]],
                        "description": f"Conditional chain: {e['label']}",
                    })

    # ── Build final response ──────────────────────────────────────────────
    node_list = list(raw_nodes.values())
    present_categories = {n["category"] for n in node_list}
    cat_defs = {
        "core":      {"label": "Core Orchestration",   "color": "#6366f1"},
        "subagent":  {"label": "Sub-Agent Execution",  "color": "#64748b"},
        "synthesis": {"label": "Synthesis / Debate",   "color": "#0ea5e9"},
        "control":   {"label": "Control Flow",         "color": "#f59e0b"},
        "condition": {"label": "Condition / Router",   "color": "#f59e0b"},
        "terminal":  {"label": "Terminal",             "color": "#374151"},
        "task":      {"label": "Task Node",            "color": "#6b7280"},
    }

    parse_summary = {
        "nodes_found": len(node_list),
        "edges_found": len(raw_edges),
        "entry_point": entry_point,
        "graph_vars": list(graph_vars),
        "filename": data.filename or "(pasted code)",
        "decision_nodes": len(decision_nodes_created),
        "router_functions": list(router_functions.keys()),
    }

    return {
        "nodes": node_list,
        "edges": raw_edges,
        "chain_groups": chain_groups,
        "entry_point": entry_point,
        "end_node": "__end__",
        "source": "parsed",
        "parse_summary": parse_summary,
        "categories": {k: v for k, v in cat_defs.items() if k in present_categories},
        "edge_types": {
            "direct":      {"label": "Direct edge",         "style": "solid",  "color": "#6b7280"},
            "conditional": {"label": "Conditional branch",  "style": "dashed", "color": "#f59e0b"},
            "chain":       {"label": "Chain execution",     "style": "chain",  "color": "#f97316"},
        },
    }


@app.get("/api/langgraph/workflow-source")
async def get_langgraph_workflow_source():
    """Return the raw source of the main workflow.py file so the UI can load it into the parser."""
    candidates = [
        Path(__file__).parent.parent / "task_manager" / "core" / "workflow.py",
        Path(__file__).parent.parent / "task_manager" / "workflow.py",
    ]
    for p in candidates:
        if p.exists():
            return {"source": p.read_text(encoding="utf-8"), "path": str(p)}
    raise HTTPException(status_code=404, detail="workflow.py not found in expected locations")


# ============================================================================
# ENTRYPOINT
# ============================================================================

def start_server(host: str = "127.0.0.1", port: int = 8550):
    """Start the UI server."""
    import uvicorn
    ai_logger.log_info(
        f"Server starting on {host}:{port}",
        category=LogCategory.SYSTEM,
        tags=["startup", "server"],
        metadata={"host": host, "port": port},
    )
    print(f"\n{'='*60}")
    print(f"  AceTaskAgent UI Server")
    print(f"  Open: http://{host}:{port}")
    print(f"  AI Chat: enabled (DeepSeek)")
    print(f"  Logs: {config.get('logging.ai_context_file', './logs/ai_context_log.jsonl')}")
    print(f"{'='*60}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
