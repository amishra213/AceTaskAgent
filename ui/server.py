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
import time
import httpx
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
