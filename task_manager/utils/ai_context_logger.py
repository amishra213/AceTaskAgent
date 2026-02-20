"""
AI-Context Structured Logging Framework

Provides structured JSON logging designed for AI agent consumption.
Each log entry is a self-contained JSON object with rich context including:
- Timestamp, level, source module, function, line number
- Structured tags and categories for filtering
- Correlation IDs for request tracing
- Performance metrics
- Error context with stack traces
- User session context

Logs are written to a JSONL file (one JSON object per line) that AI agents
can read to understand system state, diagnose issues, and provide assistance.
"""

import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
import traceback
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Deque
from enum import Enum


class LogCategory(str, Enum):
    """Categories for structured log messages."""
    SYSTEM = "system"
    WORKFLOW = "workflow"
    EXECUTION = "execution"
    AGENT = "agent"
    API = "api"
    UI = "ui"
    CONFIG = "config"
    SEARCH = "search"
    LLM = "llm"
    FILE_IO = "file_io"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USER_ACTION = "user_action"
    AI_CHAT = "ai_chat"
    ERROR = "error"
    DEBUG = "debug"


class StructuredLogEntry:
    """A single structured log entry for AI consumption."""
    
    def __init__(
        self,
        level: str,
        message: str,
        category: str = "system",
        source_module: str = "",
        source_function: str = "",
        source_line: int = 0,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_info: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None,
    ):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.level = level.upper()
        self.message = message
        self.category = category
        self.source_module = source_module
        self.source_function = source_function
        self.source_line = source_line
        self.correlation_id = correlation_id or ""
        self.session_id = session_id or ""
        self.tags = tags or []
        self.metadata = metadata or {}
        self.error_info = error_info
        self.performance = performance
        self.user_context = user_context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        entry = {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "category": self.category,
            "source": {
                "module": self.source_module,
                "function": self.source_function,
                "line": self.source_line,
            },
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "tags": self.tags,
            "metadata": self.metadata,
        }
        if self.error_info:
            entry["error"] = self.error_info
        if self.performance:
            entry["performance"] = self.performance
        if self.user_context:
            entry["user_context"] = self.user_context
        return entry
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AIContextLogger:
    """
    Structured logging system designed for AI agent consumption.
    
    Writes structured JSON log entries to a JSONL file and maintains
    an in-memory ring buffer of recent entries that can be queried
    by the AI chat assistant for context.
    
    Usage:
        logger = AIContextLogger.get_instance()
        logger.log_info("Workflow started", category="workflow",
                       tags=["start"], metadata={"workflow_id": "abc"})
    """
    
    _instance: Optional["AIContextLogger"] = None
    _lock = threading.Lock()
    
    def __init__(
        self,
        log_file: str = "./logs/ai_context_log.jsonl",
        json_log_file: str = "./logs/ace_agent_structured.jsonl",
        max_memory_entries: int = 500,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.log_file = log_file
        self.json_log_file = json_log_file
        self.max_memory_entries = max_memory_entries
        self._memory_buffer: Deque[Dict[str, Any]] = deque(maxlen=max_memory_entries)
        self._session_id = str(uuid.uuid4())[:8]
        self._correlation_ids: Dict[str, str] = {}
        
        if self.enabled:
            # Ensure log directories exist
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            Path(json_log_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Write session start marker
            self._write_entry(StructuredLogEntry(
                level="INFO",
                message=f"AI Context Logger session started (session_id={self._session_id})",
                category=LogCategory.SYSTEM,
                session_id=self._session_id,
                tags=["session_start"],
                metadata={"pid": os.getpid(), "platform": sys.platform}
            ))
    
    @classmethod
    def get_instance(cls, **kwargs) -> "AIContextLogger":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        cls._instance = None
    
    # ------------------------------------------------------------------
    # Core logging methods
    # ------------------------------------------------------------------
    
    def log(
        self,
        level: str,
        message: str,
        category: str = "system",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        error_info: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        source_override: Optional[Dict[str, str]] = None,
    ) -> None:
        """Write a structured log entry."""
        if not self.enabled:
            return
        
        # Auto-detect source from call stack
        frame = sys._getframe(2) if not source_override else None
        source_module = source_override.get("module", "") if source_override else (
            frame.f_globals.get("__name__", "") if frame else ""
        )
        source_function = source_override.get("function", "") if source_override else (
            frame.f_code.co_name if frame else ""
        )
        source_line = int(source_override.get("line", 0)) if source_override else (
            frame.f_lineno if frame else 0
        )
        
        entry = StructuredLogEntry(
            level=level,
            message=message,
            category=category,
            source_module=source_module,
            source_function=source_function,
            source_line=source_line,
            correlation_id=correlation_id,
            session_id=self._session_id,
            tags=tags,
            metadata=metadata,
            error_info=error_info,
            performance=performance,
            user_context=user_context,
        )
        
        self._write_entry(entry)
    
    def log_debug(self, message: str, **kwargs) -> None:
        """Log a DEBUG message."""
        self.log("DEBUG", message, **kwargs)
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log an INFO message."""
        self.log("INFO", message, **kwargs)
    
    def log_warning(self, message: str, **kwargs) -> None:
        """Log a WARNING message."""
        self.log("WARNING", message, **kwargs)
    
    def log_error(self, message: str, exc: Optional[Exception] = None, **kwargs) -> None:
        """Log an ERROR message, optionally capturing exception details."""
        if exc:
            kwargs.setdefault("error_info", {})
            kwargs["error_info"].update({
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
            })
            kwargs.setdefault("category", LogCategory.ERROR)
        self.log("ERROR", message, **kwargs)
    
    def log_critical(self, message: str, **kwargs) -> None:
        """Log a CRITICAL message."""
        self.log("CRITICAL", message, **kwargs)
    
    # ------------------------------------------------------------------
    # Specialized logging helpers
    # ------------------------------------------------------------------
    
    def log_api_call(
        self,
        endpoint: str,
        method: str = "GET",
        status_code: Optional[int] = None,
        duration_ms: Optional[float] = None,
        request_summary: Optional[str] = None,
        response_summary: Optional[str] = None,
        error: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log an API call with structured metadata."""
        level = "ERROR" if error else ("WARNING" if status_code and status_code >= 400 else "INFO")
        self.log(
            level=level,
            message=f"API {method} {endpoint} -> {status_code or 'pending'}",
            category=LogCategory.API,
            tags=["api_call", method.lower()],
            metadata={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "request_summary": request_summary,
                "response_summary": response_summary,
            },
            performance={"duration_ms": duration_ms} if duration_ms else None,
            error_info={"message": error} if error else None,
            correlation_id=correlation_id,
        )
    
    def log_llm_interaction(
        self,
        provider: str,
        model: str,
        prompt_summary: str,
        response_summary: Optional[str] = None,
        tokens_used: Optional[int] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log an LLM API interaction."""
        level = "ERROR" if error else "INFO"
        self.log(
            level=level,
            message=f"LLM [{provider}/{model}] call {'failed' if error else 'completed'}",
            category=LogCategory.LLM,
            tags=["llm_call", provider],
            metadata={
                "provider": provider,
                "model": model,
                "prompt_summary": prompt_summary[:200] if prompt_summary else "",
                "response_summary": response_summary[:200] if response_summary else "",
                "tokens_used": tokens_used,
            },
            performance={"duration_ms": duration_ms} if duration_ms else None,
            error_info={"message": error} if error else None,
            correlation_id=correlation_id,
        )
    
    def log_workflow_event(
        self,
        event: str,
        workflow_id: str,
        workflow_name: str = "",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log a workflow lifecycle event."""
        self.log(
            level="INFO",
            message=f"Workflow {event}: {workflow_name or workflow_id}",
            category=LogCategory.WORKFLOW,
            tags=["workflow", event],
            metadata={
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                **(details or {}),
            },
            correlation_id=correlation_id,
        )
    
    def log_execution_event(
        self,
        event: str,
        execution_id: str,
        task_name: str = "",
        progress: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log a task execution event."""
        level = "ERROR" if error else "INFO"
        self.log(
            level=level,
            message=f"Execution {event}: {task_name or execution_id}",
            category=LogCategory.EXECUTION,
            tags=["execution", event],
            metadata={
                "execution_id": execution_id,
                "task_name": task_name,
                "progress": progress,
                **(details or {}),
            },
            error_info={"message": error} if error else None,
            correlation_id=correlation_id,
        )
    
    def log_user_action(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a user action in the UI."""
        self.log(
            level="INFO",
            message=f"User action: {action}",
            category=LogCategory.USER_ACTION,
            tags=["user_action"],
            metadata=details or {},
        )
    
    def log_agent_activity(
        self,
        agent_type: str,
        activity: str,
        details: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log sub-agent activity."""
        level = "ERROR" if error else "INFO"
        self.log(
            level=level,
            message=f"Agent [{agent_type}] {activity}",
            category=LogCategory.AGENT,
            tags=["agent", agent_type],
            metadata={
                "agent_type": agent_type,
                "activity": activity,
                **(details or {}),
            },
            performance={"duration_ms": duration_ms} if duration_ms else None,
            error_info={"message": error} if error else None,
            correlation_id=correlation_id,
        )
    
    def log_performance_metric(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a performance metric."""
        level = "INFO" if success else "WARNING"
        self.log(
            level=level,
            message=f"Performance: {operation} {'completed' if success else 'slow/failed'} in {duration_ms:.1f}ms",
            category=LogCategory.PERFORMANCE,
            tags=["performance", "metric"],
            metadata={"operation": operation, **(metadata or {})},
            performance={
                "duration_ms": duration_ms,
                "success": success,
            },
        )
    
    # ------------------------------------------------------------------
    # Correlation ID management
    # ------------------------------------------------------------------
    
    def start_correlation(self, name: str = "request") -> str:
        """Generate and return a new correlation ID for tracing."""
        cid = f"{name}-{uuid.uuid4().hex[:12]}"
        self._correlation_ids[name] = cid
        return cid
    
    def get_correlation_id(self, name: str = "request") -> Optional[str]:
        """Get the current correlation ID for a given name."""
        return self._correlation_ids.get(name)
    
    # ------------------------------------------------------------------
    # AI context query methods (used by AI chat assistant)
    # ------------------------------------------------------------------
    
    def get_recent_logs(
        self,
        limit: int = 50,
        level: Optional[str] = None,
        category: Optional[str] = None,
        search_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent log entries for AI context.
        
        Args:
            limit: Maximum number of entries to return
            level: Filter by level (e.g., "ERROR", "WARNING")
            category: Filter by category
            search_text: Search text in message
            
        Returns:
            List of log entry dicts (most recent first)
        """
        entries = list(self._memory_buffer)
        
        if level:
            entries = [e for e in entries if e.get("level") == level.upper()]
        if category:
            entries = [e for e in entries if e.get("category") == category]
        if search_text:
            search_lower = search_text.lower()
            entries = [e for e in entries if search_lower in e.get("message", "").lower()]
        
        # Return most recent first
        return list(reversed(entries[-limit:]))
    
    def get_error_summary(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent errors for AI troubleshooting context."""
        return self.get_recent_logs(limit=limit, level="ERROR")
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """
        Generate a health summary from recent logs for AI context.
        """
        entries = list(self._memory_buffer)
        total = len(entries)
        errors = sum(1 for e in entries if e.get("level") == "ERROR")
        warnings = sum(1 for e in entries if e.get("level") == "WARNING")
        
        # Category breakdown
        categories: Dict[str, int] = {}
        for e in entries:
            cat = e.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        # Recent errors
        recent_errors = [
            {"message": e["message"], "timestamp": e["timestamp"], "category": e.get("category")}
            for e in entries if e.get("level") == "ERROR"
        ][-5:]
        
        return {
            "session_id": self._session_id,
            "total_log_entries": total,
            "error_count": errors,
            "warning_count": warnings,
            "category_breakdown": categories,
            "recent_errors": recent_errors,
            "health_status": "healthy" if errors == 0 else ("degraded" if errors < 5 else "unhealthy"),
        }
    
    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    
    def _write_entry(self, entry: StructuredLogEntry) -> None:
        """Write entry to file and memory buffer."""
        entry_dict = entry.to_dict()
        
        # Add to memory buffer
        self._memory_buffer.append(entry_dict)
        
        # Write to JSONL file
        try:
            json_line = json.dumps(entry_dict, default=str)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
            # Also write to the structured JSON log file
            with open(self.json_log_file, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
        except Exception:
            pass  # Don't let logging errors break the application


# --------------------------------------------------------------------------
# Module-level convenience functions
# --------------------------------------------------------------------------

def get_ai_logger() -> AIContextLogger:
    """Get the global AIContextLogger instance."""
    return AIContextLogger.get_instance()


def ai_log_info(message: str, **kwargs) -> None:
    """Quick info log to AI context."""
    get_ai_logger().log_info(message, **kwargs)


def ai_log_error(message: str, exc: Optional[Exception] = None, **kwargs) -> None:
    """Quick error log to AI context."""
    get_ai_logger().log_error(message, exc=exc, **kwargs)


def ai_log_warning(message: str, **kwargs) -> None:
    """Quick warning log to AI context."""
    get_ai_logger().log_warning(message, **kwargs)
