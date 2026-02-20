"""
Workflow Store - Persistence and management of workflow definitions.

Manages workflow templates, versions, and instances with JSON file-based storage.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class WorkflowNode:
    """A node in a workflow graph."""
    id: str
    type: str  # agent type: pdf, excel, ocr, web_search, code_interpreter, etc.
    label: str
    x: float = 0.0
    y: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorkflowEdge:
    """A connection between two workflow nodes."""
    id: str
    source: str  # source node ID
    target: str  # target node ID
    condition: str = ""  # optional routing condition
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition with nodes and edges."""
    id: str
    name: str
    description: str = ""
    nodes: List[WorkflowNode] = field(default_factory=list)
    edges: List[WorkflowEdge] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    tags: List[str] = field(default_factory=list)
    status: str = "draft"  # draft, active, archived

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "tags": self.tags,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowDefinition":
        nodes = [WorkflowNode(**n) for n in data.get("nodes", [])]
        edges = [WorkflowEdge(**e) for e in data.get("edges", [])]
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            nodes=nodes,
            edges=edges,
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            version=data.get("version", 1),
            tags=data.get("tags", []),
            status=data.get("status", "draft"),
        )


# ============================================================================
# WORKFLOW STORE
# ============================================================================

class WorkflowStore:
    """File-based persistence for workflow definitions."""

    def __init__(self, storage_dir: str = "temp_folder/workflows"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._load_all()

    def _load_all(self):
        """Load all workflow definitions from disk."""
        for path in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                wf = WorkflowDefinition.from_dict(data)
                self._workflows[wf.id] = wf
            except Exception:
                pass  # Skip corrupted files

    def _save(self, workflow: WorkflowDefinition):
        """Persist a workflow to disk."""
        path = self.storage_dir / f"{workflow.id}.json"
        path.write_text(json.dumps(workflow.to_dict(), indent=2), encoding="utf-8")

    def create(self, name: str, description: str = "", nodes: Optional[List[dict]] = None,
               edges: Optional[List[dict]] = None, tags: Optional[List[str]] = None) -> WorkflowDefinition:
        """Create a new workflow definition."""
        wf = WorkflowDefinition(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            nodes=[WorkflowNode(**n) for n in (nodes or [])],
            edges=[WorkflowEdge(**e) for e in (edges or [])],
            tags=tags or [],
        )
        self._workflows[wf.id] = wf
        self._save(wf)
        return wf

    def get(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)

    def list_all(self, status: Optional[str] = None) -> List[WorkflowDefinition]:
        """List all workflows, optionally filtered by status."""
        workflows = list(self._workflows.values())
        if status:
            workflows = [w for w in workflows if w.status == status]
        return sorted(workflows, key=lambda w: w.updated_at, reverse=True)

    def update(self, workflow_id: str, **kwargs) -> Optional[WorkflowDefinition]:
        """Update a workflow definition."""
        wf = self._workflows.get(workflow_id)
        if not wf:
            return None

        for key, value in kwargs.items():
            if key == "nodes":
                wf.nodes = [WorkflowNode(**n) if isinstance(n, dict) else n for n in value]
            elif key == "edges":
                wf.edges = [WorkflowEdge(**e) if isinstance(e, dict) else e for e in value]
            elif hasattr(wf, key):
                setattr(wf, key, value)

        wf.updated_at = datetime.now().isoformat()
        wf.version += 1
        self._save(wf)
        return wf

    def delete(self, workflow_id: str) -> bool:
        """Delete a workflow definition."""
        if workflow_id not in self._workflows:
            return False
        del self._workflows[workflow_id]
        path = self.storage_dir / f"{workflow_id}.json"
        if path.exists():
            path.unlink()
        return True

    def duplicate(self, workflow_id: str, new_name: Optional[str] = None) -> Optional[WorkflowDefinition]:
        """Duplicate a workflow."""
        original = self._workflows.get(workflow_id)
        if not original:
            return None
        return self.create(
            name=new_name or f"{original.name} (Copy)",
            description=original.description,
            nodes=[n.to_dict() for n in original.nodes],
            edges=[e.to_dict() for e in original.edges],
            tags=original.tags.copy(),
        )

    def get_templates(self) -> List[Dict[str, Any]]:
        """Return built-in workflow templates."""
        return [
            {
                "id": "template-research",
                "name": "Deep Research Workflow",
                "description": "Web search → Data extraction → Document generation",
                "nodes": [
                    {"id": "n1", "type": "web_search", "label": "Web Search", "x": 100, "y": 200,
                     "config": {"search_depth": "deep", "max_results": 20}},
                    {"id": "n2", "type": "data_extraction", "label": "Extract Data", "x": 350, "y": 200,
                     "config": {"format": "structured"}},
                    {"id": "n3", "type": "problem_solver", "label": "Analyze", "x": 600, "y": 200,
                     "config": {"analysis_type": "comprehensive"}},
                    {"id": "n4", "type": "document", "label": "Generate Report", "x": 850, "y": 200,
                     "config": {"output_format": "markdown"}},
                ],
                "edges": [
                    {"id": "e1", "source": "n1", "target": "n2"},
                    {"id": "e2", "source": "n2", "target": "n3"},
                    {"id": "e3", "source": "n3", "target": "n4"},
                ],
                "tags": ["research", "template"],
            },
            {
                "id": "template-document-pipeline",
                "name": "Document Processing Pipeline",
                "description": "PDF → OCR → Excel → Report",
                "nodes": [
                    {"id": "n1", "type": "pdf", "label": "Parse PDF", "x": 100, "y": 200,
                     "config": {"extract_tables": True}},
                    {"id": "n2", "type": "ocr", "label": "OCR Images", "x": 350, "y": 120,
                     "config": {"engine": "tesseract"}},
                    {"id": "n3", "type": "excel", "label": "Process Data", "x": 350, "y": 300,
                     "config": {"operations": ["analyze", "chart"]}},
                    {"id": "n4", "type": "document", "label": "Generate Report", "x": 600, "y": 200,
                     "config": {"output_format": "markdown"}},
                ],
                "edges": [
                    {"id": "e1", "source": "n1", "target": "n2"},
                    {"id": "e2", "source": "n1", "target": "n3"},
                    {"id": "e3", "source": "n2", "target": "n4"},
                    {"id": "e4", "source": "n3", "target": "n4"},
                ],
                "tags": ["document", "template"],
            },
            {
                "id": "template-code-analysis",
                "name": "Code Analysis Workflow",
                "description": "Problem solving → Code interpretation → Report",
                "nodes": [
                    {"id": "n1", "type": "problem_solver", "label": "Analyze Problem", "x": 100, "y": 200,
                     "config": {"mode": "analysis"}},
                    {"id": "n2", "type": "code_interpreter", "label": "Run Code", "x": 400, "y": 200,
                     "config": {"language": "python"}},
                    {"id": "n3", "type": "document", "label": "Write Report", "x": 700, "y": 200,
                     "config": {"output_format": "markdown"}},
                ],
                "edges": [
                    {"id": "e1", "source": "n1", "target": "n2"},
                    {"id": "e2", "source": "n2", "target": "n3"},
                ],
                "tags": ["code", "analysis", "template"],
            },
        ]
