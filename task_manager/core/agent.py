"""
Agent module - Main TaskManagerAgent implementation
"""

from typing import Literal, List, Optional, Dict, Any, Tuple, cast
from datetime import datetime
import json
import hashlib

from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from task_manager.models import AgentState, TaskStatus, Task, PlanNode, BlackboardEntry, HistoryEntry
from task_manager.models.messages import AgentExecutionRequest, AgentExecutionResponse
from task_manager.config import AgentConfig
from task_manager.utils.logger import get_logger
from task_manager.utils.prompt_builder import PromptBuilder
from task_manager.utils.rate_limiter import global_rate_limiter
from task_manager.utils.input_context import InputContext
from task_manager.utils.temp_manager import TempDataManager
from task_manager.utils.redis_cache import RedisCacheManager
from task_manager.utils.execution_tracer import ExecutionTracer
from task_manager.utils.exceptions import (
    InvalidParameterError,
    MissingDependencyError
)
from task_manager.sub_agents import (
    PDFAgent, ExcelAgent, OCRImageAgent, WebSearchAgent, 
    CodeInterpreterAgent, DataExtractionAgent, ProblemSolverAgent, DocumentAgent
)
from .workflow import WorkflowBuilder
from .master_planner import MasterPlanner, PlanStatus, BlackboardType

logger = get_logger(__name__)


class TaskManagerAgent:
    """
    LangGraph-based agent for recursive task management and execution.
    
    This agent can handle any complex objective by:
    1. Breaking it down recursively into manageable subtasks
    2. Executing tasks in parallel where possible
    3. Managing state and checkpointing progress
    4. Recovering from failures automatically
    """
    
    def __init__(
        self,
        objective: str,
        config: Optional[AgentConfig] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Initialize the Task Manager Agent.
        
        Args:
            objective: The main goal/objective to accomplish
            config: AgentConfig instance with all settings
            metadata: Additional context (e.g., list of items to process)
        """
        self.objective = objective
        self.config = config or AgentConfig()
        self.metadata = metadata or {}
        
        logger.info(f"Initializing TaskManagerAgent with objective: {objective[:100]}...")
        logger.debug(f"Configuration: {self.config.to_dict()}")
        
        # Initialize Execution Tracer for comprehensive debugging
        self.tracer = ExecutionTracer(enable_detailed_logging=True)
        logger.info("[TRACER] Execution tracer initialized for detailed workflow diagnostics")
        
        # Initialize Health Checker for detecting hangs and failures
        from task_manager.utils.state_validator import HealthChecker, StateValidator
        self.health_checker = HealthChecker(
            max_iterations=self.config.max_iterations,
            timeout_seconds=600  # 10 minutes per node
        )
        self.state_validator = StateValidator()
        logger.info("[HEALTH] Health checker initialized")
        
        # Initialize Temp Data Manager first - for organizing WIP data
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_manager = TempDataManager(
            self.config.folders.temp_path,
            session_id=f"agent_{session_id}"
        )
        logger.info(f"Temp data manager initialized: {self.config.folders.temp_path}")
        self.metadata['temp_folder'] = str(self.config.folders.temp_path)
        self.metadata['session_id'] = self.temp_manager.session_id
        
        # Initialize Input Context - scan input folder for user-provided data
        self.input_context = InputContext(self.config.folders.input_path, auto_scan=True)
        if self.input_context:
            logger.info(f"Input context loaded: {len(self.input_context)} files found")
            self.metadata['input_folder'] = str(self.config.folders.input_path)
        else:
            logger.info("No input files found in input folder")
        
        # Initialize Data Extraction Agent - for intelligent context extraction
        # This replaces raw input context with relevance-filtered extractions
        self.data_extraction_agent = DataExtractionAgent(cache_extractions=True)
        
        # Extract relevant data from input folder based on objective
        if len(self.input_context) > 0:
            logger.info("Extracting relevant data from input files...")
            extracted_data = self.data_extraction_agent.extract_relevant_data(
                input_folder=self.config.folders.input_path,
                objective=objective,
                temp_folder=self.config.folders.temp_path,
                max_files=15,
                max_content_per_file=4000,
                max_total_content=40000
            )
            self.metadata['input_context'] = extracted_data
            logger.info(f"Extracted {extracted_data.get('files_processed', 0)} relevant files, "
                       f"{extracted_data.get('total_content_size', 0)} chars")
        else:
            self.metadata['input_context'] = None
        
        # Initialize LLM based on provider
        self.llm = self._initialize_llm()
        
        # Initialize Master Planner for sophisticated planning and coordination
        self.master_planner = MasterPlanner(llm=self.llm)
        
        # Initialize tools list
        self.tools = ["web_search"] if self.config.enable_search else []
        
        # Initialize sub-agents for file operations
        self.pdf_agent = PDFAgent()
        self.excel_agent = ExcelAgent()
        self.ocr_image_agent = OCRImageAgent(llm=self.llm)
        self.web_search_agent = WebSearchAgent()
        self.code_interpreter_agent = CodeInterpreterAgent(llm=self.llm)
        self.document_agent = DocumentAgent()
        # Note: data_extraction_agent already initialized above for context building
        
        # Initialize ProblemSolverAgent for LLM-based error analysis and human input interpretation
        self.problem_solver_agent = ProblemSolverAgent(llm_client=self.llm)
        
        # Initialize TaskRelayAgent for centralized task orchestration
        from task_manager.core.task_relay_agent import TaskRelayAgent
        self.task_relay_agent = TaskRelayAgent(enable_tracing=True)
        self.task_relay_agent.register_agent("pdf_agent", self.pdf_agent)
        self.task_relay_agent.register_agent("excel_agent", self.excel_agent)
        self.task_relay_agent.register_agent("ocr_image_agent", self.ocr_image_agent)
        self.task_relay_agent.register_agent("web_search_agent", self.web_search_agent)
        self.task_relay_agent.register_agent("code_interpreter_agent", self.code_interpreter_agent)
        self.task_relay_agent.register_agent("document_agent", self.document_agent)
        self.task_relay_agent.register_agent("data_extraction_agent", self.data_extraction_agent)
        self.task_relay_agent.register_agent("problem_solver_agent", self.problem_solver_agent)
        logger.info("[RELAY] Task relay agent initialized and all sub-agents registered")
        
        # Initialize Redis cache manager for task result caching
        self.cache = RedisCacheManager()
        if self.cache.redis_available:
            logger.info("[CACHE] Redis cache enabled - task results will be cached")
        else:
            logger.warning("[CACHE] Redis unavailable - caching disabled")
        
        # Initialize state
        self.initial_state = self._create_initial_state()
        
        # Build workflow graph
        workflow_builder = WorkflowBuilder(self)
        self.workflow = workflow_builder.build()
        
        # Add memory/checkpointing
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        logger.debug(f"Agent initialized with max_iterations={self.config.max_iterations}")
        logger.debug("Sub-agents initialized: PDFAgent, ExcelAgent, OCRImageAgent, WebSearchAgent, CodeInterpreterAgent, DataExtractionAgent, DocumentAgent, ProblemSolverAgent")
    
    
    def _initialize_llm(self):
        """
        Initialize LLM based on provider configuration.
        
        Returns:
            Initialized LLM instance
        """
        provider = self.config.llm.provider.lower()
        
        try:
            if provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                
                kwargs = {"model": self.config.llm.model_name, "temperature": self.config.llm.temperature}
                if self.config.llm.api_key:
                    kwargs["api_key"] = self.config.llm.api_key
                kwargs.update(self.config.llm.extra_params)
                return ChatAnthropic(**kwargs)
            
            elif provider == "openai":
                from langchain_openai import ChatOpenAI
                
                kwargs = {"model": self.config.llm.model_name, "temperature": self.config.llm.temperature}
                if self.config.llm.api_key:
                    kwargs["api_key"] = self.config.llm.api_key
                kwargs.update(self.config.llm.extra_params)
                return ChatOpenAI(**kwargs)
            
            elif provider == "google":
                # Check if using native provider SDK
                if self.config.llm.use_native_sdk:
                    from task_manager.utils.llm_client import LLMClient
                    
                    return LLMClient(
                        api_key=self.config.llm.api_key,
                        model=self.config.llm.model_name,
                        temperature=self.config.llm.temperature,
                        api_version=self.config.llm.api_version,
                        use_native_sdk=True,
                        api_base_url=self.config.llm.api_base_url,
                        api_endpoint_path=self.config.llm.api_endpoint_path
                    )
                else:
                    # Use LangChain wrapper (existing behavior)
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    
                    kwargs = {"model": self.config.llm.model_name, "temperature": self.config.llm.temperature}
                    if self.config.llm.api_key:
                        kwargs["google_api_key"] = self.config.llm.api_key
                    kwargs.update(self.config.llm.extra_params)
                    return ChatGoogleGenerativeAI(**kwargs)
            
            elif provider == "groq":
                # Groq uses OpenAI-compatible API
                try:
                    from langchain_groq import ChatGroq
                    
                    kwargs = {"model": self.config.llm.model_name, "temperature": self.config.llm.temperature}
                    if self.config.llm.api_key:
                        kwargs["api_key"] = self.config.llm.api_key
                    kwargs.update(self.config.llm.extra_params)
                    return ChatGroq(**kwargs)
                except ImportError:
                    # Fall back to OpenAI client with Groq base URL
                    from langchain_openai import ChatOpenAI
                    
                    kwargs = {
                        "model": self.config.llm.model_name,
                        "temperature": self.config.llm.temperature,
                        "base_url": "https://api.groq.com/openai/v1"
                    }
                    if self.config.llm.api_key:
                        kwargs["api_key"] = self.config.llm.api_key
                    kwargs.update(self.config.llm.extra_params)
                    return ChatOpenAI(**kwargs)
            
            elif provider == "deepseek":
                # DeepSeek uses OpenAI-compatible API
                from langchain_openai import ChatOpenAI
                
                # Use configured base_url or api_base_url if provided, otherwise default
                base_url = self.config.llm.base_url or self.config.llm.api_base_url or "https://api.deepseek.com/v1"
                
                kwargs = {
                    "model": self.config.llm.model_name,
                    "temperature": self.config.llm.temperature,
                    "base_url": base_url
                }
                if self.config.llm.api_key:
                    kwargs["api_key"] = self.config.llm.api_key
                kwargs.update(self.config.llm.extra_params)
                return ChatOpenAI(**kwargs)
            
            elif provider == "local":
                from langchain_community.llms import Ollama
                
                kwargs = {
                    "model": self.config.llm.model_name,
                    "base_url": self.config.llm.base_url or "http://localhost:11434",
                    "temperature": self.config.llm.temperature
                }
                kwargs.update(self.config.llm.extra_params)
                return Ollama(**kwargs)
            
            else:
                raise InvalidParameterError(
                    parameter_name="provider",
                    message=f"Unsupported LLM provider: {provider}. "
                    f"Supported providers: anthropic, openai, google, groq, deepseek, local"
                )
        except ImportError as e:
            raise MissingDependencyError(
                package_name=f"langchain-{provider}",
                install_command=f"pip install langchain-{provider}",
                purpose=f"{provider} LLM provider"
            ) from e
    
    
    def _rate_limited_invoke(self, messages: List[Any], **kwargs) -> Any:
        """
        Invoke LLM with global rate limiting applied.
        
        This wrapper ensures all LLM calls respect the configured rate limits.
        
        Args:
            messages: List of LangChain messages to send
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLM response
        """
        import time
        
        # Log request details
        logger.debug(f"[LLM] Preparing LLM invocation")
        logger.debug(f"[LLM]   Provider: {self.config.llm.provider}")
        logger.debug(f"[LLM]   Model: {self.config.llm.model_name}")
        logger.debug(f"[LLM]   Messages: {len(messages)} items")
        logger.debug(f"[LLM]   Temperature: {self.config.llm.temperature}")
        
        # Calculate total message content length for debugging
        total_content_length = sum(
            len(getattr(msg, 'content', str(msg))) for msg in messages
        )
        logger.debug(f"[LLM]   Total content size: {total_content_length} chars")
        
        # Apply rate limiting before making the request
        wait_time = global_rate_limiter.wait()
        if wait_time > 0:
            logger.warning(f"[LLM] Rate limiter delayed request by {wait_time:.2f}s")
        
        # Invoke LLM and measure latency
        start_time = time.time()
        try:
            logger.debug(f"[LLM] Sending request to LLM endpoint...")
            response = self.llm.invoke(messages, **kwargs)
            
            latency = time.time() - start_time
            response_length = len(getattr(response, 'content', str(response)))
            
            logger.debug(f"[LLM] ✓ Response received")
            logger.debug(f"[LLM]   Latency: {latency:.2f}s")
            logger.debug(f"[LLM]   Response size: {response_length} chars")
            
            return response
            
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"[LLM] ✗ LLM invocation failed after {latency:.2f}s")
            logger.error(f"[LLM]   Error type: {type(e).__name__}")
            logger.error(f"[LLM]   Error message: {str(e)}")
            raise
    
    
    def _generate_cache_key(
        self, 
        task_description: str, 
        agent_type: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a deterministic cache key based on task content.
        
        Creates a hash from task description and parameters to ensure:
        - Same task description = same cache key (cache hit)
        - Different parameters = different cache key (cache isolation)
        - Deterministic across multiple runs
        
        Args:
            task_description: The task description text
            agent_type: Type of agent (web_search, ocr, pdf, etc.)
            parameters: Optional task parameters to include in hash
            
        Returns:
            Cache key string in format: "task_<agent>_<hash>"
            
        Example:
            key = _generate_cache_key(
                "Search for Karnataka districts",
                "web_search",
                {"max_results": 10}
            )
            # Returns: "task_web_search_a3f5c8d9..."
        """
        # Normalize task description (lowercase, strip whitespace)
        normalized_desc = task_description.lower().strip()
        
        # Create hashable content
        hash_content = f"{agent_type}:{normalized_desc}"
        
        # Include parameters if provided (for cache isolation)
        if parameters:
            # Sort keys for deterministic hash
            param_str = json.dumps(parameters, sort_keys=True)
            hash_content += f":{param_str}"
        
        # Generate SHA256 hash (first 16 chars for readability)
        hash_digest = hashlib.sha256(hash_content.encode('utf-8')).hexdigest()[:16]
        
        # Format: task_<agent>_<hash>
        cache_key = f"task_{agent_type}_{hash_digest}"
        
        logger.debug(f"[CACHE KEY] Generated: {cache_key} for '{task_description[:50]}...'")
        
        return cache_key
    
    
    def _generate_entry_id(self, entry: BlackboardEntry) -> str:
        """
        Generate a unique ID for a blackboard entry based on its content.
        
        Args:
            entry: The blackboard entry to generate an ID for
            
        Returns:
            A unique string ID for the entry
        """
        # Create a deterministic hash from entry attributes
        id_content = f"{entry.get('source_agent')}:{entry.get('source_task_id')}:{entry.get('entry_type')}"
        entry_id = hashlib.md5(id_content.encode('utf-8')).hexdigest()[:12]
        return entry_id
    
    def _create_initial_state(self) -> AgentState:
        """
        Create and return the initial agent state with Blackboard pattern and deep hierarchy support.
        
        Initializes:
        - plan: Empty, will be created during initialization
        - blackboard: Empty, agents will populate with findings (with nested entry support)
        - history: Empty, will track execution steps
        - next_step: Set to "initialize" to start the workflow
        - current_depth: Starts at 0 (root level)
        - depth_limit: Prevents infinite recursion
        - parent_context: Inherits context from parent tasks
        - input_context: User-provided data files from input folder
        """
        return {
            "objective": self.objective,
            # ===== MASTER PLANNER FIELDS =====
            "plan": [],  # Will be populated by _initialize
            "blackboard": [],  # Will be populated by agents
            "history": [],  # Will track execution steps
            "next_step": "initialize",  # Entry point
            # ===== INPUT CONTEXT =====
            "input_context": self.metadata.get('input_context'),  # User-provided files
            # ===== DEEP HIERARCHY SUPPORT =====
            "current_depth": 0,  # Current depth in hierarchy
            "depth_limit": 5,  # Maximum recursion depth (default 5)
            "parent_context": None,  # Context from parent level
            # ===== LEGACY COMPATIBILITY FIELDS =====
            "tasks": [],
            "active_task_id": "",
            "completed_task_ids": [],
            "failed_task_ids": [],
            "results": {},
            "metadata": self.metadata,
            "iteration_count": 0,
            "max_iterations": self.config.max_iterations,
            "requires_human_review": False,
            "human_feedback": "",
            "human_review_context": {}  # Context for human review workflows
        }
    
    
    # ========================================================================
    # NODE IMPLEMENTATIONS - ENHANCED WITH BLACKBOARD PATTERN
    # ========================================================================
    
    def _initialize(self, state: AgentState) -> AgentState:
        """
        Initialize agent with root task and create initial hierarchical plan.
        
        This node:
        1. Creates the root task from objective
        2. Uses Master Planner to create hierarchical task plan
        3. Initializes empty blackboard and history
        4. Sets next_step for non-linear routing
        """
        logger.info("="*60)
        logger.info("INITIALIZING TASK MANAGER AGENT WITH MASTER PLANNER")
        logger.info("="*60)
        logger.info(f"Objective: {state['objective']}")
        logger.info(f"Max Iterations: {state['max_iterations']}")
        logger.info("="*60)
        
        # Create root task (legacy compatibility)
        root_task = Task(
            id="1",
            description=state['objective'],
            status=TaskStatus.PENDING,
            parent_id=None,
            depth=0,
            context="Root objective",
            result=None,
            error=None,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        # Create hierarchical plan using Master Planner
        logger.info("Creating hierarchical task plan...")
        plan = self.master_planner.create_initial_plan(
            state['objective'],
            state['metadata']
        )
        
        logger.info(f"Plan created with {len(plan)} nodes")
        for pnode in plan:
            logger.debug(
                f"  └─ [{pnode['task_id']}] Depth {pnode['depth']}: "
                f"{pnode['description'][:60]}..."
            )
        
        # Initialize empty blackboard and history
        initial_blackboard: List[BlackboardEntry] = []
        initial_history: List[HistoryEntry] = []
        
        # Record initialization in history
        initial_history = self.master_planner.record_step(
            initial_history,
            step_name="initialize",
            agent="master_planner",
            task_id="1",
            outcome={"plan_nodes": len(plan)},
            duration_seconds=0.0
        )
        
        return {
            **state,
            # Legacy fields
            "tasks": [root_task],
            "active_task_id": "1",
            # New Blackboard pattern fields
            "plan": plan,
            "blackboard": initial_blackboard,
            "history": initial_history,
            "next_step": "select_task"
        }
    
    
    def _select_next_task(self, state: AgentState) -> AgentState:
        """Select next pending task to process."""
        
        # Immediate debug output
        print(">>> ENTERED _select_next_task", flush=True)
        logger.info(">>> ENTERED _select_next_task")
        
        # Health checks to detect issues early
        logger.info("[HEALTH] Running pre-execution health checks...")
        try:
            # Validate state integrity
            is_valid, errors = self.state_validator.validate_state_integrity(state)
            if not is_valid:
                logger.error(f"[HEALTH] ✗ State integrity issues detected:")
                for error in errors:
                    logger.error(f"  • {error}")
            else:
                logger.info("[HEALTH] ✓ State integrity valid")
            
            # Check active task validity
            active_valid, active_error = self.state_validator.validate_active_task(state)
            if not active_valid and active_error:
                logger.warning(f"[HEALTH] ⚠️  {active_error}")
            
            # Check for duplicates
            duplicates = self.state_validator.check_task_duplication(state)
            if duplicates:
                logger.warning(f"[HEALTH] ⚠️  Duplicate task IDs in completed_task_ids: {duplicates}")
            
            # Check iteration limit
            exceeded, msg = self.health_checker.check_iteration_limit(state.get('iteration_count', 0))
            if exceeded:
                logger.error(f"[HEALTH] ✗ {msg}")
                return {
                    **state,
                    "requires_human_review": True,
                    "human_feedback": f"Maximum iterations reached: {msg}"
                }
            
            # Check task progress
            progressing, progress_msg = self.health_checker.check_task_progress(state)
            if not progressing and progress_msg:
                logger.warning(f"[HEALTH] ⚠️  {progress_msg}")
        
        except Exception as health_error:
            logger.error(f"[HEALTH] Error during health checks: {str(health_error)}")
            logger.log_exception("Health check exception:", health_error)
            # Continue despite health check errors
        
        # Check state health
        try:
            state_keys = list(state.keys())
            tasks_count = len(state.get('tasks', []))
            iteration = state.get('iteration_count', 0)
            blackboard_size = len(state.get('blackboard', []))
            print(f">>> [STATE CHECK] Keys: {len(state_keys)}, Tasks: {tasks_count}, Iter: {iteration}, Blackboard: {blackboard_size}", flush=True)
            logger.debug(f"[STATE CHECK] State keys: {len(state_keys)}, tasks: {tasks_count}, iteration: {iteration}, blackboard: {blackboard_size}")
        except Exception as e:
            logger.error(f"[STATE CHECK] Error checking state: {e}")
            print(f">>> [STATE CHECK ERROR] {e}", flush=True)
        
        # Log task hierarchy summary periodically (every 5 iterations)
        if state['iteration_count'] % 5 == 0 and state['iteration_count'] > 0:
            self._log_task_hierarchy_summary(state)
        
        logger.info("=" * 80)
        logger.info("[SELECT TASK] Finding next task to process")
        logger.info("=" * 80)
        
        tasks = state['tasks']
        completed = set(state['completed_task_ids'])
        failed = set(state['failed_task_ids'])
        
        # Calculate progress percentage
        total_tasks = len(tasks)
        completed_count = len(completed)
        progress_pct = (completed_count * 100 // total_tasks) if total_tasks > 0 else 0
        
        logger.info(f"[SELECT] Total tasks: {len(tasks)}")
        logger.info(f"[SELECT] Completed: {len(completed)} ({progress_pct}%)")
        logger.info(f"[SELECT] Failed: {len(failed)}")
        
        # Find pending tasks (not completed, not failed, and status is PENDING)
        # NOTE: BROKEN_DOWN tasks are NOT selected - they are intermediate states
        # We only select leaf tasks that are ready to execute
        pending_tasks = [
            t for t in tasks 
            if t['id'] not in completed 
            and t['id'] not in failed
            and t.get('status') == TaskStatus.PENDING  # Use .get() for safety
        ]
        
        logger.info(f"[SELECT] Pending tasks: {len(pending_tasks)}")
        
        # Log all pending task IDs for tracking
        if pending_tasks:
            pending_ids = [t['id'] for t in pending_tasks]
            logger.info(f"[SELECT] Pending task IDs (before sort): {pending_ids}")
        else:
            # Additional debugging if no pending tasks found
            logger.warning("[SELECT] No PENDING tasks found, checking task statuses:")
            status_counts = {}
            for t in tasks:
                status = t.get('status', 'UNKNOWN')
                status_counts[str(status)] = status_counts.get(str(status), 0) + 1
            for status, count in status_counts.items():
                logger.warning(f"[SELECT]   {status}: {count} tasks")
            
            # Check if there are tasks that aren't completed or failed
            unfinished_tasks = [
                t for t in tasks 
                if t['id'] not in completed 
                and t['id'] not in failed
            ]
            if unfinished_tasks:
                logger.warning(f"[SELECT] Found {len(unfinished_tasks)} unfinished tasks (not completed/failed)")
                for t in unfinished_tasks[:5]:  # Log first 5
                    logger.warning(f"[SELECT]   Task {t['id']}: status={t.get('status')}, desc={t.get('description', 'N/A')[:50]}...")
        
        if not pending_tasks:
            logger.info("[SELECT] No pending tasks found - workflow will check completion")
            logger.info("=" * 80)
            return {**state, "active_task_id": ""}
        
        # SORT pending tasks by hierarchical numeric ID (e.g., 1.1.1, 1.1.2, 1.2.1)
        def parse_task_id(task_id: str) -> List[int]:
            """Parse task ID into numeric components for proper sorting."""
            try:
                # Extract numeric parts: "1.1.2" -> [1, 1, 2]
                parts = task_id.split('.')
                return [int(p) for p in parts if p.isdigit()]
            except (ValueError, AttributeError):
                logger.debug(f"[SELECT] Could not parse task ID: {task_id}")
                return [999999]  # Put unparseable IDs at the end
        
        # Sort by numeric hierarchy
        pending_tasks.sort(key=lambda t: parse_task_id(t['id']))
        
        sorted_ids = [t['id'] for t in pending_tasks]
        logger.info(f"[SELECT] Pending task IDs (after sort): {sorted_ids}")
        
        # Select first pending task after sorting
        next_task = pending_tasks[0]
        logger.info(f"[SELECT] Selected task {next_task['id']}: {next_task['description'][:80]}...")
        logger.info(f"[SELECT] Task depth: {next_task.get('depth', 0)}")
        logger.info(f"[SELECT] Task status: {next_task.get('status')}")
        logger.info(f"[SELECT] Iteration: {state['iteration_count']} → {state['iteration_count'] + 1}")
        logger.info(f"[SELECT] Next: analyze_task → determine breakdown or execute")
        logger.info("=" * 80)
        
        return {
            **state,
            "active_task_id": next_task['id'],
            "iteration_count": state['iteration_count'] + 1
        }
    
    
    def _analyze_task(self, state: AgentState) -> AgentState:
        """
        Analyze current task using LLM to decide next action.
        
        Enhanced with:
        - Comprehensive error recovery using ProblemSolverAgent
        - Automatic retry with adjusted prompts on failure
        - Human review workflow for persistent failures
        - Robust result validation
        """
        # IMMEDIATE LOGGING - to detect if this method is even being called
        print(">>> ENTERED _analyze_task", flush=True)
        logger.info(">>> ENTERED _analyze_task")
        
        # Record entry state snapshot
        self.tracer.record_state_snapshot("analyze_task", dict(state), phase="entry")
        
        task_id = state['active_task_id']
        logger.info(f"[ANALYZE] Analyzing task: {task_id}")
        self.tracer.record_routing_decision("analyze_task", task_id, "entry", "Starting analysis")
        
        if not task_id:
            logger.warning("[ANALYZE] No active_task_id - returning state unchanged")
            self.tracer.record_routing_decision("analyze_task", "", "error", "No active task ID")
            return state
        
        try:
            task = next(t for t in state['tasks'] if t['id'] == task_id)
        except StopIteration:
            logger.error(f"[ANALYZE] Task {task_id} not found in tasks list!")
            logger.error(f"[ANALYZE] Available tasks: {[t['id'] for t in state['tasks']]}")
            self.tracer.record_routing_decision("analyze_task", task_id, "error", "Task not found in tasks list")
            return state
        
        logger.info(f"[ANALYZE] Found task {task_id}: {task.get('description', '')[:80]}")
        
        # DEPTH LIMIT ENFORCEMENT: Force execution at depth 2 or higher for research tasks
        current_depth = task.get('depth', 0)
        depth_limit = state.get('depth_limit', 5)
        task_description = task.get('description', '').lower()
        
        # Check if this is a research/search type task
        research_keywords = ['search', 'research', 'find', 'identify', 'list', 'trends', 
                            'sources', 'information', 'data', 'latest', 'current', 'industry']
        is_research_task = any(keyword in task_description for keyword in research_keywords)
        
        # Force execution at depth 2 for research tasks, depth 3 for others
        force_execution_depth = 2 if is_research_task else 3
        
        if current_depth >= force_execution_depth:
            logger.warning(f"[ANALYZE] Task at depth {current_depth} ({'research' if is_research_task else 'general'}) - FORCING EXECUTION to prevent infinite breakdown")
            # Force execute_web_search_task execution for research tasks or execute_problem_solver_task for analysis
            if is_research_task:
                action_name = "execute_web_search_task"
                file_op = {
                    "type": "web_search",
                    "operation": "search",
                    "parameters": {
                        "query": task.get('description', ''),
                        "num_results": 5
                    }
                }
            else:
                action_name = "execute_problem_solver_task"
                file_op = None
            
            analysis = {
                "action": action_name,
                "reasoning": f"Task at depth {current_depth} - forcing execution to prevent excessive decomposition",
                "subtasks": None,
                "search_query": task.get('description', '') if is_research_task else None,
                "file_operation": file_op,
                "estimated_complexity": "medium",
                "requires_human_review": False
            }
            
            logger.info(f"[ANALYZE] Forced analysis action: {analysis['action']}")
            
            updated_task: Task = {
                **task,
                "status": TaskStatus.ANALYZING,
                "result": analysis,
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks: List[Task] = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            logger.info(f"[ANALYZE] ✓ Task {task_id} forced to execute - returning updated state")
            
            result_state: AgentState = {
                **state,
                "tasks": updated_tasks,
                "requires_human_review": False
            }
            
            self.tracer.record_state_snapshot("analyze_task", dict(result_state), phase="exit", notes=f"Forced execution at depth {current_depth}")
            return result_state
        
        # Build analysis prompt using PromptBuilder (with input context)
        analysis_prompt = PromptBuilder.build_analysis_prompt(
            objective=state['objective'],
            task=task,
            metadata=state['metadata'],
            current_depth=current_depth,
            depth_limit=depth_limit,
            input_context=state.get('input_context')
        )
        
        logger.debug(f"[ANALYZE] Built analysis prompt ({len(analysis_prompt)} chars)")
        
        # Track analysis attempts for retry logic
        max_analysis_retries = 2
        last_error = None
        
        for attempt in range(1, max_analysis_retries + 1):
            try:
                logger.info(f"[ANALYZE] Analysis attempt {attempt}/{max_analysis_retries}")
                
                response = self._rate_limited_invoke([
                    SystemMessage(content="You are a task analysis expert. Respond only with valid JSON."),
                    HumanMessage(content=analysis_prompt)
                ])
                
                # Extract content from response (handle different response types)
                response_content = response.content if hasattr(response, 'content') else str(response)
                if isinstance(response_content, list):
                    # If content is a list, join it
                    response_content = "".join(str(item) for item in response_content)
                response_content = str(response_content)
                
                logger.debug(f"[ANALYZE] LLM response ({len(response_content)} chars): {response_content[:200]}")
                
                # Parse response
                analysis = self._parse_json_response(response_content)
                
                # VALIDATE analysis result
                if not isinstance(analysis, dict):
                    raise ValueError(f"Analysis result is not a dict: {type(analysis)}")
                
                if 'action' not in analysis:
                    raise ValueError("Analysis missing 'action' field")
                
                if 'reasoning' not in analysis:
                    raise ValueError("Analysis missing 'reasoning' field")
                
                # Validate that action is meaningful
                valid_actions = ['breakdown', 'execute_task', 'execute_pdf_task', 'execute_excel_task', 
                               'execute_ocr_task', 'execute_web_search_task', 'execute_code_interpreter_task',
                               'execute_data_extraction_task', 'execute_problem_solver_task', 'execute_document_task']
                if analysis['action'] not in valid_actions:
                    raise ValueError(f"Invalid action '{analysis['action']}'. Valid actions: {valid_actions}")
                
                logger.info(f"[ANALYZE] Result: {analysis['action']} - {analysis['reasoning']}")
                logger.info(f"[ANALYZE] ✓ Analysis validation passed")
                self.tracer.record_routing_decision("analyze_task", task_id, analysis['action'], analysis['reasoning'], {"analysis": analysis})
                
                # Update task with analysis
                updated_task: Task = {
                    **task,
                    "status": TaskStatus.ANALYZING,
                    "result": analysis,
                    "updated_at": datetime.now().isoformat()
                }
                
                # Update tasks list
                updated_tasks: List[Task] = [
                    updated_task if t['id'] == task_id else t
                    for t in state['tasks']
                ]
                
                logger.info(f"[ANALYZE] ✓ Task {task_id} analyzed successfully - returning updated state")
                
                result_state = {
                    **state,
                    "tasks": updated_tasks,
                    "requires_human_review": analysis.get('requires_human_review', False)
                }
                
                self.tracer.record_state_snapshot("analyze_task", dict(result_state), phase="exit")
                return result_state
                
            except Exception as llm_error:
                last_error = llm_error
                logger.warning(f"[ANALYZE] Attempt {attempt} failed: {type(llm_error).__name__}: {str(llm_error)}")
                logger.log_exception("Analysis attempt exception:", llm_error)
                
                if attempt < max_analysis_retries:
                    logger.warning(f"[ANALYZE] Retrying analysis (attempt {attempt + 1}/{max_analysis_retries})...")
                    # Continue to next attempt
                    continue
                else:
                    # All retries exhausted
                    logger.error(f"[ANALYZE] ✗ All {max_analysis_retries} analysis attempts failed")
                    break
        
        # If we get here, all analysis attempts failed - invoke error recovery workflow
        logger.error(f"[ANALYZE] ✗ Analysis failed after {max_analysis_retries} attempts: {str(last_error)}")
        logger.error(f"[ANALYZE] Invoking LLM-based error diagnosis and human review workflow")
        
        # Use ProblemSolverAgent to diagnose the analysis failure
        error_analysis = None
        suggested_solutions = None
        
        try:
            logger.info(f"[ANALYZE] Invoking ProblemSolverAgent for error diagnosis...")
            
            error_context = {
                'task_id': task_id,
                'task_description': task.get('description', ''),
                'analysis_prompt': analysis_prompt[:500],  # Include first 500 chars of prompt
                'error_message': str(last_error)
            }
            
            # Diagnose the error
            diagnose_request: AgentExecutionRequest = {
                "task_id": f"diagnose_analysis_{task_id}",
                "task_description": "Diagnose analysis failure",
                "task_type": "atomic",
                "operation": "diagnose_error",
                "parameters": {
                    "error_message": str(last_error),
                    "task_context": error_context,
                    "agent_type": "analysis"
                },
                "input_data": {},
                "temp_folder": str(self.config.folders.temp_path),
                "output_folder": str(self.config.folders.output_path),
                "cache_enabled": False,
                "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                "relevant_entries": [],
                "max_retries": 1
            }
            
            diagnose_response = cast(AgentExecutionResponse, self.problem_solver_agent.execute_task(diagnose_request))
            error_analysis = diagnose_response.get('result', {}) if diagnose_response['success'] else None
            
            if error_analysis:
                logger.info(f"[ANALYZE] Error diagnosis: category={error_analysis.get('error_category')}")
                logger.info(f"[ANALYZE] Solution suggestion: {error_analysis.get('solution_prompt', 'Unknown')[:100]}...")
            
        except Exception as e:
            logger.warning(f"[ANALYZE] ProblemSolverAgent diagnosis failed: {str(e)}")
            logger.log_exception("Error diagnosis exception:", e)
        
        # Mark task as FAILED with error analysis for human review
        updated_task: Task = {
            **task,
            "status": TaskStatus.FAILED,
            "error": f"Analysis failed after {max_analysis_retries} attempts: {str(last_error)}",
            "result": {
                "error": str(last_error),
                "error_analysis": error_analysis,
                "error_type": type(last_error).__name__
            },
            "updated_at": datetime.now().isoformat()
        }
        
        updated_tasks: List[Task] = [
            updated_task if t['id'] == task_id else t
            for t in state['tasks']
        ]
        
        logger.warning(f"[ANALYZE] Task {task_id} marked as FAILED for human review")
        
        # Return state with task marked as failed and requires_human_review set
        # This will trigger the error recovery workflow in routing
        result_state = {
            **state,
            "tasks": updated_tasks,
            "failed_task_ids": [task_id],  # Add to failed tasks
            "active_task_id": "",  # Clear active task to trigger selection
            "requires_human_review": True  # Flag for human review workflow
        }
        
        self.tracer.record_state_snapshot("analyze_task", dict(result_state), phase="exit", notes=f"Analysis failed and marked for human review")
        return result_state
    
    
    def _breakdown_task(self, state: AgentState) -> AgentState:
        """Break down task into subtasks."""
        task_id = state['active_task_id']
        
        # TRACKING: Log state on entry
        entry_completed_list = state.get('completed_task_ids', [])
        logger.warning(f"🔍 [TASK_ID_TRACKER] BREAKDOWN ENTRY | task_id='{task_id}' | completed_task_ids size: {len(entry_completed_list)} | unique: {len(set(entry_completed_list))}")
        
        task = next(t for t in state['tasks'] if t['id'] == task_id)
        analysis = task.get('result') if isinstance(task, dict) else task['result']
        
        # Ensure analysis is a dict and has required keys
        if not isinstance(analysis, dict):
            logger.error(f"[BREAKDOWN] Invalid analysis format for task {task_id}")
            return state
        
        logger.info("=" * 80)
        logger.info(f"[BREAKDOWN] TASK DECOMPOSITION INITIATED")
        logger.info("=" * 80)
        logger.info(f"Parent Task: {task_id}")
        logger.info(f"Description: {task['description'][:100]}...")
        logger.info(f"Depth Level: {task.get('depth', 0)}")
        
        subtasks = []
        subtasks_list = analysis.get('subtasks', []) or []
        
        # Check if subtasks are empty - this is a problem!
        if not subtasks_list:
            logger.error(f"[BREAKDOWN] ✗ FAILED: No subtasks provided in analysis for task {task_id}")
            logger.error(f"[BREAKDOWN] Analysis was: {analysis}")
            
            # Treat as failed breakdown - mark task as failed
            updated_task: Task = {
                **task,
                "status": TaskStatus.FAILED,
                "error": "Breakdown was requested but no subtasks were provided",
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks: List[Task] = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            # NOTE: failed_task_ids uses operator.add annotation, so only return NEW items to add
            return {
                **state,
                "tasks": updated_tasks,
                "failed_task_ids": [task_id],  # Only the new task ID - LangGraph will concatenate
                "active_task_id": ""
            }
        
        logger.info(f"[BREAKDOWN] Creating {len(subtasks_list)} subtasks...")
        logger.info("-" * 80)
        
        for idx, subtask_desc in enumerate(subtasks_list, 1):
            subtask = Task(
                id=f"{task_id}.{idx}",
                description=subtask_desc,
                status=TaskStatus.PENDING,
                parent_id=task_id,
                depth=task['depth'] + 1,
                context=f"Parent: {task['description']}",
                result=None,
                error=None,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            subtasks.append(subtask)
            logger.info(f"  [{idx}/{len(subtasks_list)}] {subtask['id']}")
            logger.info(f"       Description: {subtask_desc[:70]}...")
            logger.info(f"       Status: {subtask['status']}")
            logger.info(f"       Depth: {subtask['depth']}")
        
        # Update parent task status
        updated_task = {
            **task,
            "status": TaskStatus.BROKEN_DOWN,
            "updated_at": datetime.now().isoformat()
        }
        
        updated_tasks = [
            updated_task if t['id'] == task_id else t
            for t in state['tasks']
        ]
        
        logger.info("-" * 80)
        logger.info(f"[BREAKDOWN] ✓ SUCCESS: Created {len(subtasks)} subtasks")
        logger.info(f"[BREAKDOWN] Parent task {task_id} marked as BROKEN_DOWN")
        logger.info("=" * 80)
        
        # TRACKING: Log task ID addition for debugging
        current_completed = state.get('completed_task_ids', [])
        logger.warning(f"🔍 [TASK_ID_TRACKER] BREAKDOWN RETURN | Task_id='{task_id}' marked as BROKEN_DOWN | NOT adding to completed_task_ids | Location: _breakdown_task:658")
        
        # IMPORTANT: Do NOT add BROKEN_DOWN tasks to completed_task_ids
        # A parent task is only truly complete when ALL its subtasks are complete
        # This will be handled by checking subtask completion in _check_completion
        return_dict = {
            **state,
            "tasks": updated_tasks + subtasks  # type: ignore
            # NOT adding to completed_task_ids - parent will be completed when all subtasks are done
        }
        logger.warning(f"🔍 [TASK_ID_TRACKER] BREAKDOWN RETURN | Returning dict WITHOUT adding to completed_task_ids | Parent will complete when all subtasks finish")
        return return_dict  # type: ignore
    
    
    def _execute_task(self, state: AgentState) -> AgentState:
        """Execute a specific task (e.g., web search, computation)."""
        task_id = state['active_task_id']
        task = next(t for t in state['tasks'] if t['id'] == task_id)
        analysis = task.get('result') if isinstance(task, dict) else task['result']
        
        # Ensure analysis is a dict
        if not isinstance(analysis, dict):
            analysis = {}
        
        logger.info(f"[EXECUTE] Executing task {task_id}...")
        
        try:
            result_data = {}
            
            # Check if search is needed
            search_query = analysis.get('search_query')
            if search_query and self.config.enable_search:
                logger.info(f"[EXECUTE] Searching: {search_query}")
                
                # Execute search using LLM with tools
                search_prompt = PromptBuilder.build_search_prompt(
                    search_query=search_query,
                    task_description=task['description'],
                    objective=state['objective']
                )
                
                response = self._rate_limited_invoke([
                    SystemMessage(content="You are a data extraction assistant."),
                    HumanMessage(content=search_prompt)
                ])
                
                result_data = {
                    "query": search_query,
                    "findings": response.content,
                    "source": "web_search"
                }
            else:
                # Direct execution without search
                result_data = {
                    "task": task['description'],
                    "status": "completed_without_search",
                    "note": "Task completed based on existing knowledge"
                }
            
            # Update task with results
            updated_task = {
                **task,
                "status": TaskStatus.COMPLETED,
                "result": result_data,
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            logger.info(f"[EXECUTE] ✓ Task {task_id} completed")
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "results": {
                    **state['results'],
                    task_id: result_data
                }
            }
            
        except Exception as e:
            logger.error(f"[EXECUTE] ✗ Error: {str(e)}")
            
            updated_task = {
                **task,
                "status": TaskStatus.FAILED,
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "failed_task_ids": [task_id]
            }
    
    
    def _filter_relevant_blackboard_entries(
        self,
        blackboard: list[BlackboardEntry],
        task_id: str,
        agent_name: str = ""
    ) -> list[BlackboardEntry]:
        """
        Filter blackboard entries relevant to the current task.
        
        This provides contextual data to agents so they can use information
        from upstream tasks (sibling tasks, parent tasks, etc.).
        
        Args:
            blackboard: Full blackboard from state
            task_id: Current task ID
            agent_name: Agent name for logging
            
        Returns:
            Filtered list of relevant blackboard entries
        """
        if not blackboard:
            return []
        
        task_prefix = '.'.join(task_id.split('.')[:-1]) if '.' in task_id else task_id
        
        def is_relevant_entry(entry: BlackboardEntry, current_task_id: str, prefix: str) -> bool:
            """Check if blackboard entry is relevant to current task."""
            relevant_to = entry.get('relevant_to', [])
            
            # No specific relevance - available to all tasks
            if not relevant_to:
                return True
            
            # Directly relevant to this task
            if current_task_id in relevant_to:
                return True
            
            # Include data from sibling/related tasks
            for rel_task in relevant_to:
                # Same parent (sibling tasks) - e.g., 1.1.1 and 1.1.2 data available to 1.1.3
                rel_prefix = '.'.join(rel_task.split('.')[:-1]) if '.' in rel_task else rel_task
                if rel_prefix == prefix and prefix:  # Only if we have a valid prefix
                    return True
                
                # Parent task data - e.g., 1.1 data available to 1.1.3
                if current_task_id.startswith(rel_task + '.'):
                    return True
                
                # Child task data - e.g., 1.1.3 data available to 1.1
                if rel_task.startswith(current_task_id + '.'):
                    return True
            
            return False
        
        relevant_entries = [
            e for e in blackboard
            if is_relevant_entry(e, task_id, task_prefix)
        ]
        
        if agent_name:
            logger.debug(f"[{agent_name}] Blackboard filtering: {len(blackboard)} total → {len(relevant_entries)} relevant")
            if blackboard and not relevant_entries:
                logger.warning(f"[{agent_name}] ⚠️  All {len(blackboard)} blackboard entries filtered out for task {task_id}")
        
        return relevant_entries
    
    
    def _execute_pdf_task(self, state: AgentState) -> AgentState:
        """
        Execute a PDF file operation task using PDFAgent.
        
        Enhanced with:
        - File pointer tracking for cross-agent workflows
        - Chain execution detection (images found → OCR)
        - Blackboard updates for findings
        """
        task_id = state['active_task_id']
        task = next(t for t in state['tasks'] if t['id'] == task_id)
        analysis = task.get('result') if isinstance(task, dict) else task['result']
        
        # Ensure analysis is a dict
        if not isinstance(analysis, dict):
            analysis = {}
        
        logger.info("=" * 80)
        logger.info(f"[PDF AGENT] EXECUTION STARTED")
        logger.info("=" * 80)
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Description: {task['description'][:100]}...")
        logger.info(f"Depth Level: {task.get('depth', 0)}")
        
        try:
            # Extract file operation details from analysis
            file_operation = analysis.get('file_operation', {})
            if not file_operation or file_operation.get('type') != 'pdf':
                logger.error(f"[PDF AGENT] ✗ Invalid PDF operation specification")
                return {
                    **state,
                    "tasks": [
                        {
                            **task,
                            "status": TaskStatus.FAILED,
                            "error": "Invalid PDF operation specification",
                            "updated_at": datetime.now().isoformat()
                        } if t['id'] == task_id else t
                        for t in state['tasks']
                    ],  # type: ignore
                    "failed_task_ids": [task_id]
                }
            
            operation = file_operation.get('operation', 'read')
            parameters = file_operation.get('parameters', {})
            
            logger.info(f"[PDF AGENT] Operation: {operation}")
            logger.info(f"[PDF AGENT] File: {parameters.get('file_path', 'N/A')[:80]}...")
            logger.info(f"[PDF AGENT] Parameters: {parameters}")
            logger.info("-" * 80)
            
            # Filter relevant blackboard entries for context
            blackboard = state.get('blackboard', [])
            relevant_entries = self._filter_relevant_blackboard_entries(blackboard, task_id, "PDF AGENT")
            
            # Create standardized AgentExecutionRequest
            logger.info(f"[PDF AGENT] Calling PDFAgent.execute_task() with standardized request...")
            logger.info(f"[PDF AGENT] Providing {len(relevant_entries)} relevant blackboard entries for context")
            request: AgentExecutionRequest = {
                "task_id": task_id,
                "task_description": task['description'],
                "task_type": "atomic",
                "operation": operation,
                "parameters": parameters,
                "input_data": file_operation,
                "temp_folder": str(self.config.folders.temp_path),
                "output_folder": str(self.config.folders.output_path),
                "cache_enabled": True,
                "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                "relevant_entries": [self._generate_entry_id(e) for e in relevant_entries],
                "max_retries": 3
            }
            
            # Execute using standardized interface
            response: AgentExecutionResponse = self.pdf_agent.execute_task(request=request)
            
            # Extract legacy-compatible result_data from standardized response
            result_data = {
                'success': response['success'],
                'summary': response['result'].get('summary', ''),
                'findings': response['result'].get('findings', {}),
                **response['result']  # Include all other result fields
            }
            
            # Update task with results
            updated_task = {
                **task,
                "status": TaskStatus.COMPLETED if response['success'] else TaskStatus.FAILED,
                "result": result_data,
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            logger.info("-" * 80)
            logger.info(f"[PDF AGENT] ✓ SUCCESS: Task {task_id} completed")
            logger.info(f"[PDF AGENT] Status: {updated_task['status']}")
            logger.info(f"[PDF AGENT] Result Summary: {result_data.get('summary', 'N/A')[:100]}...")
            
            # Merge blackboard entries from agent response with task-specific entry
            agent_blackboard_entries = response.get('blackboard_entries', [])
            
            # Create additional blackboard entry with findings and file pointers
            blackboard_entry: BlackboardEntry = {
                "entry_type": "pdf_extraction_result",
                "source_agent": "pdf_agent",
                "source_task_id": task_id,
                "content": {
                    "operation": operation,
                    "success": result_data.get('success'),
                    "summary": result_data.get('summary', ''),
                    "findings": result_data.get('findings', {}),
                    "execution_time_ms": response.get('execution_time_ms', 0)
                },
                "timestamp": datetime.now().isoformat(),
                "relevant_to": [task_id],
                "depth_level": 0,
                "file_pointers": {},  # Will populate based on results
                "chain_next_agents": []  # Will populate if chain conditions met
            }
            
            # Check for extracted images (chain to OCR condition)
            extracted_images = result_data.get('findings', {}).get('extracted_images', [])
            if extracted_images:
                logger.info(f"[PDF AGENT] 🔗 CHAIN DETECTION: Found {len(extracted_images)} images")
                logger.info(f"[PDF AGENT] 🔗 Scheduling OCR Agent for downstream processing")
                blackboard_entry["file_pointers"] = {  # type: ignore
                    "ocr_agent": ",".join(extracted_images)
                }
                blackboard_entry["chain_next_agents"] = ["ocr_agent"]  # type: ignore
            
            # Add source file path for traceability
            source_file = parameters.get('file_path', '')
            if source_file:
                blackboard_entry["source_file_path"] = source_file  # type: ignore
            
            logger.info("=" * 80)
            
            # NOTE: blackboard uses operator.add annotation, so only return NEW items to add
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "results": {
                    **state['results'],
                    task_id: result_data
                },
                "blackboard": [blackboard_entry]  # Only new entry - LangGraph will concatenate
            }
        
        except Exception as e:
            logger.error(f"[PDF AGENT] ✗ FAILED: Error executing task")
            logger.error(f"[PDF AGENT] Exception: {str(e)}")
            logger.info("=" * 80)
            
            updated_task = {
                **task,
                "status": TaskStatus.FAILED,
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "failed_task_ids": [task_id]
            }
    
    
    def _execute_excel_task(self, state: AgentState) -> AgentState:
        """
        Execute an Excel file operation task using ExcelAgent.
        
        Enhanced with:
        - File pointer consumption from OCR/WebSearch agents
        - Data processing with awareness of source agent
        - Blackboard updates with processed results
        - Traceability of data lineage through chain
        """
        task_id = state['active_task_id']
        task = next(t for t in state['tasks'] if t['id'] == task_id)
        analysis = task.get('result') if isinstance(task, dict) else task['result']
        
        # Ensure analysis is a dict
        if not isinstance(analysis, dict):
            analysis = {}
        
        logger.info("=" * 80)
        logger.info(f"[EXCEL AGENT] EXECUTION STARTED")
        logger.info("=" * 80)
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Description: {task['description'][:100]}...")
        logger.info(f"Depth Level: {task.get('depth', 0)}")
        
        try:
            # Check for data from previous agents via file pointers
            source_data_type = None
            source_task_from_chain = None
            
            blackboard = state.get('blackboard', [])
            logger.info(f"[EXCEL AGENT] Scanning blackboard ({len(blackboard)} entries) for chain data...")
            
            for entry in reversed(blackboard[-10:]):  # Check last 10 entries
                if entry.get('chain_next_agents'):
                    if "excel_agent" in entry.get('chain_next_agents', []):
                        source_data_type = entry.get('source_agent', 'unknown')
                        source_task_from_chain = entry.get('source_task_id')
                        logger.info(f"[EXCEL AGENT] 🔗 CHAIN INPUT DETECTED")
                        logger.info(f"[EXCEL AGENT] 🔗 Source Agent: {source_data_type}")
                        logger.info(f"[EXCEL AGENT] 🔗 Source Task: {source_task_from_chain}")
                        break
            
            # Extract file operation details from analysis
            file_operation = analysis.get('file_operation', {})
            if not file_operation or file_operation.get('type') != 'excel':
                logger.error(f"[EXCEL AGENT] ✗ Invalid Excel operation specification")
                return {
                    **state,
                    "tasks": [
                        {
                            **task,
                            "status": TaskStatus.FAILED,
                            "error": "Invalid Excel operation specification",
                            "updated_at": datetime.now().isoformat()
                        } if t['id'] == task_id else t
                        for t in state['tasks']
                    ],  # type: ignore
                    "failed_task_ids": [task_id]
                }
            
            operation = file_operation.get('operation', 'read')
            parameters = file_operation.get('parameters', {})
            
            # Add context about data source
            if source_data_type:
                parameters['source_agent'] = source_data_type
                parameters['source_task'] = source_task_from_chain
            
            # ENHANCEMENT: Generate sample filename with task_id and timestamp if not provided
            # This prevents errors when LLM doesn't provide proper file paths
            if operation == 'create' and not parameters.get('file_path') and not parameters.get('output_path'):
                # Generate safe filename from task_id and timestamp
                safe_task_id = task_id.replace('.', '_')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                default_filename = f"excel_output_{safe_task_id}_{timestamp}.xlsx"
                # Use output_path (not temp_path) for final Excel files
                parameters['output_path'] = str(self.config.folders.output_path / default_filename)
                logger.info(f"[EXCEL AGENT] Generated default filename in output folder: {default_filename}")
                logger.info(f"[EXCEL AGENT] Full path: {parameters['output_path']}")
            
            # Also ensure folder_path uses output_folder if not specified
            if operation == 'create' and not parameters.get('folder_path'):
                parameters['folder_path'] = str(self.config.folders.output_path)
                logger.info(f"[EXCEL AGENT] Using output folder: {parameters['folder_path']}")
            
            logger.info(f"[EXCEL AGENT] Operation: {operation}")
            logger.info(f"[EXCEL AGENT] File: {parameters.get('file_path', parameters.get('output_path', 'N/A'))[:80]}...")
            logger.info(f"[EXCEL AGENT] Parameters: {parameters}")
            logger.info("-" * 80)
            
            # Filter relevant blackboard entries for context
            blackboard = state.get('blackboard', [])
            relevant_entries = self._filter_relevant_blackboard_entries(blackboard, task_id, "EXCEL AGENT")
            
            # Create standardized AgentExecutionRequest
            logger.info(f"[EXCEL AGENT] Calling ExcelAgent.execute_task() with standardized request...")
            logger.info(f"[EXCEL AGENT] Providing {len(relevant_entries)} relevant blackboard entries for context")
            request: AgentExecutionRequest = {
                "task_id": task_id,
                "task_description": task['description'],
                "task_type": "atomic",
                "operation": operation,
                "parameters": parameters,
                "input_data": file_operation,
                "temp_folder": str(self.config.folders.temp_path),
                "output_folder": str(self.config.folders.output_path),
                "cache_enabled": True,
                "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                "relevant_entries": [self._generate_entry_id(e) for e in relevant_entries],
                "max_retries": 3
            }
            
            # Execute using standardized interface - pass as single positional argument
            # ExcelAgent.execute_task expects: execute_task(request) NOT execute_task(request=request)
            response = cast(AgentExecutionResponse, self.excel_agent.execute_task(request))
            
            # Extract legacy-compatible result_data from standardized response
            result_data = {
                'success': response['success'],
                'rows_processed': response['result'].get('rows_processed', 0),
                'output_file': response['result'].get('output_file', ''),
                'findings': response['result'].get('findings', {}),
                **response['result']  # Include all other result fields
            }
            
            # Update task with results
            updated_task = {
                **task,
                "status": TaskStatus.COMPLETED if response['success'] else TaskStatus.FAILED,
                "result": result_data,
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            logger.info("-" * 80)
            logger.info(f"[EXCEL AGENT] ✓ SUCCESS: Task {task_id} completed")
            logger.info(f"[EXCEL AGENT] Status: {updated_task['status']}")
            logger.info(f"[EXCEL AGENT] Rows Processed: {result_data.get('rows_processed', 0)}")
            
            # Merge blackboard entries from agent response
            agent_blackboard_entries = response.get('blackboard_entries', [])
            
            # Create additional blackboard entry with final processed results
            blackboard_entry: BlackboardEntry = {
                "entry_type": "excel_processing_result",
                "source_agent": "excel_agent",
                "source_task_id": task_id,
                "parent_task_id": source_task_from_chain,  # type: ignore
                "content": {
                    "operation": operation,
                    "success": result_data.get('success'),
                    "rows_processed": result_data.get('rows_processed', 0),
                    "output_file": result_data.get('output_file', ''),
                    "findings": result_data.get('findings', {}),
                    "execution_time_ms": response.get('execution_time_ms', 0)
                },
                "timestamp": datetime.now().isoformat(),
                "relevant_to": [task_id],
                "depth_level": 0,
                "file_pointers": {},  # type: ignore
                "chain_next_agents": []  # type: ignore
            }
            
            # Track generated CSV file for potential next agent
            if result_data.get('output_file'):
                blackboard_entry["file_pointers"] = {  # type: ignore
                    "downstream": result_data.get('output_file')
                }
                logger.info(f"[EXCEL AGENT] Generated Output File: {result_data.get('output_file')}")
            
            # Add data lineage - trace back through chain
            blackboard_entry["content"]["chain_trace"] = {  # type: ignore
                "source_agent": source_data_type,
                "source_task": source_task_from_chain,
                "processing_task": task_id
            }
            
            logger.info("=" * 80)
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "results": {
                    **state['results'],
                    task_id: result_data
                },
                "blackboard": [blackboard_entry]  # Only new entry - LangGraph will concatenate
            }
        
        except Exception as e:
            logger.error(f"[EXCEL AGENT] ✗ FAILED: Error executing task")
            logger.error(f"[EXCEL AGENT] Exception: {str(e)}")
            logger.error(f"[EXCEL AGENT] Exception Type: {type(e).__name__}")
            logger.info("=" * 80)
            
            # ============================================================
            # ERROR RECOVERY: Use ProblemSolverAgent for error analysis
            # ============================================================
            error_analysis = None
            suggested_solutions = None
            
            try:
                logger.info(f"[EXCEL AGENT] Invoking ProblemSolverAgent for error analysis...")
                
                task_context = {
                    'task_id': task_id,
                    'description': task['description'],
                    'operation': file_operation.get('operation', 'unknown') if 'file_operation' in locals() else 'unknown',
                    'parameters': parameters if 'parameters' in locals() else {},
                }
                
                # Diagnose the error
                diagnose_request: AgentExecutionRequest = {
                    "task_id": f"diagnose_{task_id}",
                    "task_description": "Diagnose Excel agent error",
                    "task_type": "atomic",
                    "operation": "diagnose_error",
                    "parameters": {
                        "error_message": str(e),
                        "task_context": task_context,
                        "agent_type": "excel"
                    },
                    "input_data": {},
                    "temp_folder": str(self.config.folders.temp_path),
                    "output_folder": str(self.config.folders.output_path),
                    "cache_enabled": False,
                    "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                    "relevant_entries": [],
                    "max_retries": 1
                }
                
                diagnose_response = cast(AgentExecutionResponse, self.problem_solver_agent.execute_task(diagnose_request))
                error_analysis = diagnose_response.get('result', {}) if diagnose_response['success'] else None
                
                if error_analysis:
                    logger.info(f"[EXCEL AGENT] Error diagnosis: category={error_analysis.get('error_category')}")
                
                # Get solution suggestions
                solution_request: AgentExecutionRequest = {
                    "task_id": f"solution_{task_id}",
                    "task_description": "Get solution for Excel agent error",
                    "task_type": "atomic",
                    "operation": "get_solution",
                    "parameters": {
                        "error_message": str(e),
                        "task_context": task_context,
                        "agent_type": "excel"
                    },
                    "input_data": {},
                    "temp_folder": str(self.config.folders.temp_path),
                    "output_folder": str(self.config.folders.output_path),
                    "cache_enabled": False,
                    "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                    "relevant_entries": [],
                    "max_retries": 1
                }
                
                solution_response = cast(AgentExecutionResponse, self.problem_solver_agent.execute_task(solution_request))
                suggested_solutions = solution_response.get('result', {}) if solution_response['success'] else None
                
                if suggested_solutions:
                    logger.info(f"[EXCEL AGENT] Suggested solution type: {suggested_solutions.get('solution_type')}")
                    
            except Exception as err_e:
                logger.warning(f"[EXCEL AGENT] ProblemSolverAgent analysis failed: {str(err_e)}")
            
            # Store error details with AI analysis in the task
            updated_task = {
                **task,
                "status": TaskStatus.FAILED,
                "error": str(e),
                "error_type": type(e).__name__,
                "result": {
                    "error": str(e),
                    "error_analysis": error_analysis,
                    "suggested_solutions": suggested_solutions,
                    "error_context": {
                        "operation": file_operation.get('operation', 'unknown') if 'file_operation' in locals() else 'unknown',
                        "parameters": parameters if 'parameters' in locals() else {},
                        "agent": "excel_agent"
                    }
                },
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            # Set requires_human_review to trigger human review workflow
            # The human_review node will display the AI analysis and solutions
            logger.warning(f"[EXCEL AGENT] Setting requires_human_review=True to trigger human review workflow")
            logger.warning(f"[EXCEL AGENT] AI error analysis and solutions have been prepared for human review")
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "failed_task_ids": [task_id],
                "requires_human_review": True,  # Trigger human review workflow
                "active_task_id": task_id  # Keep task active for human review
            }
    
    
    def _execute_ocr_task(self, state: AgentState) -> AgentState:
        """
        Execute an OCR/Image operation task using OCRImageAgent.
        
        Enhanced with:
        - File pointer tracking from previous agents (PDF found images)
        - Extracted table detection for Excel chain
        - Blackboard updates with extracted content
        """
        task_id = state['active_task_id']
        task = next(t for t in state['tasks'] if t['id'] == task_id)
        analysis = task.get('result') if isinstance(task, dict) else task['result']
        
        # Ensure analysis is a dict
        if not isinstance(analysis, dict):
            analysis = {}
        
        logger.info("=" * 80)
        logger.info(f"[OCR AGENT] EXECUTION STARTED")
        logger.info("=" * 80)
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Description: {task['description'][:100]}...")
        logger.info(f"Depth Level: {task.get('depth', 0)}")
        
        try:
            # Check for file pointers from previous agent (e.g., PDF agent)
            source_images = []
            source_agent_type = None
            source_task_id = None
            
            blackboard = state.get('blackboard', [])
            logger.info(f"[OCR AGENT] Scanning blackboard ({len(blackboard)} entries) for chain data...")
            
            for entry in reversed(blackboard[-10:]):  # Check last 10 blackboard entries
                if entry.get('source_agent') == 'pdf_agent' and 'file_pointers' in entry:
                    # Get image paths from PDF agent's file pointers
                    image_files = entry.get('file_pointers', {}).get('ocr_agent', '')
                    if image_files:
                        source_images = image_files.split(',')
                        source_agent_type = entry.get('source_agent', 'unknown')
                        source_task_id = entry.get('source_task_id')
                        
                        logger.info(f"[OCR AGENT] 🔗 CHAIN INPUT DETECTED")
                        logger.info(f"[OCR AGENT] 🔗 Source Agent: {source_agent_type}")
                        logger.info(f"[OCR AGENT] 🔗 Source Task: {source_task_id}")
                        logger.info(f"[OCR AGENT] 🔗 Images to Process: {len(source_images)}")
                        break
            
            # Extract file operation details from analysis
            file_operation = analysis.get('file_operation', {})
            if not file_operation or file_operation.get('type') != 'ocr':
                logger.error(f"[OCR AGENT] ✗ Invalid OCR operation specification")
                return {
                    **state,
                    "tasks": [
                        {
                            **task,
                            "status": TaskStatus.FAILED,
                            "error": "Invalid OCR operation specification",
                            "updated_at": datetime.now().isoformat()
                        } if t['id'] == task_id else t
                        for t in state['tasks']
                    ],  # type: ignore
                    "failed_task_ids": [task_id]
                }
            
            operation = file_operation.get('operation', 'ocr_image')
            parameters = file_operation.get('parameters', {})
            
            # If we have source images from file pointers, use them
            if source_images:
                parameters['image_paths'] = source_images
            
            # ============================================================
            # CACHE CHECK: Look for cached OCR result
            # ============================================================
            # Use image paths as part of cache key for uniqueness
            image_paths = parameters.get('image_paths', [])
            cache_key = self._generate_cache_key(
                task_description=task.get('description', ''),
                agent_type="ocr",
                parameters={"operation": operation, "image_count": len(image_paths)}
            )
            
            cached_result = self.cache.get_cached_result(cache_key)
            
            if cached_result:
                logger.info("=" * 80)
                logger.info(f"[CACHE HIT] 🎯 Using cached OCR result")
                logger.info(f"[CACHE HIT] Cache Key: {cache_key}")
                logger.info(f"[CACHE HIT] Cached At: {cached_result['timestamp']}")
                logger.info(f"[CACHE HIT] TTL Remaining: {cached_result['ttl']} seconds")
                logger.info("=" * 80)
                
                # Use cached output data
                result_data = cached_result['output']
                task_success = result_data.get('success', True)
                
                # Mark task as completed with cached result
                updated_task = {
                    **task,
                    "status": TaskStatus.COMPLETED if task_success else TaskStatus.FAILED,
                    "result": result_data,
                    "updated_at": datetime.now().isoformat(),
                    "cached": True
                }
                
                updated_tasks = [
                    updated_task if t['id'] == task_id else t
                    for t in state['tasks']
                ]
                
                # Create blackboard entry for cached OCR result
                blackboard_entry: BlackboardEntry = {
                    "entry_type": "ocr_extraction_result_cached",
                    "source_agent": "ocr_agent_cache",
                    "source_task_id": task_id,
                    "parent_task_id": None,  # type: ignore
                    "content": {
                        "operation": operation,
                        "success": result_data.get('success'),
                        "text_extracted": result_data.get('text', ''),
                        "findings": result_data.get('findings', {}),
                        "cache_hit": True,
                        "cached_at": cached_result['timestamp']
                    },
                    "timestamp": datetime.now().isoformat(),
                    "relevant_to": [task_id],
                    "depth_level": 0,
                    "file_pointers": {},  # type: ignore
                    "chain_next_agents": []  # type: ignore
                }
                
                logger.info(f"[OCR AGENT] ✓ Task {task_id} completed from cache")
                
                return {
                    **state,
                    "tasks": updated_tasks,  # type: ignore
                    "results": {
                        **state['results'],
                        task_id: result_data
                    },
                    "blackboard": [blackboard_entry],
                    "last_updated_key": "ocr_results"  # Trigger observer even for cached results
                }
            
            # ============================================================
            # CACHE MISS: Execute OCR normally
            # ============================================================
            logger.info(f"[CACHE MISS] No cached OCR result found - executing OCR")
            
            logger.info(f"[OCR AGENT] Operation: {operation}")
            logger.info(f"[OCR AGENT] Image Count: {len(parameters.get('image_paths', []))}")
            logger.info(f"[OCR AGENT] Parameters: {parameters}")
            logger.info("-" * 80)
            
            # Filter relevant blackboard entries for context
            blackboard = state.get('blackboard', [])
            relevant_entries = self._filter_relevant_blackboard_entries(blackboard, task_id, "OCR AGENT")
            
            # Create standardized AgentExecutionRequest
            logger.info(f"[OCR AGENT] Calling OCRImageAgent.execute_task() with standardized request...")
            logger.info(f"[OCR AGENT] Providing {len(relevant_entries)} relevant blackboard entries for context")
            request: AgentExecutionRequest = {
                "task_id": task_id,
                "task_description": task['description'],
                "task_type": "atomic",
                "operation": operation,
                "parameters": parameters,
                "input_data": file_operation,
                "temp_folder": str(self.config.folders.temp_path),
                "output_folder": str(self.config.folders.output_path),
                "cache_enabled": True,
                "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                "relevant_entries": [self._generate_entry_id(e) for e in relevant_entries],
                "max_retries": 3
            }
            
            # Execute using standardized interface
            response = cast(AgentExecutionResponse, self.ocr_image_agent.execute_task(request=request))
            
            # Extract legacy-compatible result_data from standardized response
            result_data = {
                'success': response['success'],
                'text': response['result'].get('text', ''),
                'findings': response['result'].get('findings', {}),
                **response['result']  # Include all other result fields
            }
            
            # Update task with results
            updated_task = {
                **task,
                "status": TaskStatus.COMPLETED if response['success'] else TaskStatus.FAILED,
                "result": result_data,
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            logger.info("-" * 80)
            logger.info(f"[OCR AGENT] ✓ SUCCESS: Task {task_id} completed")
            logger.info(f"[OCR AGENT] Status: {updated_task['status']}")
            logger.info(f"[OCR AGENT] Text Extracted: {len(result_data.get('text', ''))} characters")
            
            # Create blackboard entry with findings and chain markers
            blackboard_entry: BlackboardEntry = {
                "entry_type": "ocr_extraction_result",
                "source_agent": "ocr_agent",
                "source_task_id": task_id,
                "parent_task_id": None,  # type: ignore
                "content": {
                    "operation": operation,
                    "success": result_data.get('success'),
                    "text_extracted": result_data.get('text', ''),
                    "findings": result_data.get('findings', {})
                },
                "timestamp": datetime.now().isoformat(),
                "relevant_to": [task_id],
                "depth_level": 0,
                "file_pointers": {},  # type: ignore
                "chain_next_agents": []  # type: ignore
            }
            
            # Check for extracted tables (chain to Excel condition)
            extracted_table = result_data.get('findings', {}).get('extracted_table', [])
            if extracted_table:
                logger.info(f"[OCR EXECUTE] Found table data - marking for Excel chain")
                # Store table data as file pointer for Excel agent
                blackboard_entry["file_pointers"] = {  # type: ignore
                    "excel_agent": "table_data_extracted"
                }
                blackboard_entry["chain_next_agents"] = ["excel_agent"]  # type: ignore
            
            # ============================================================
            # CACHE STORAGE: Store successful OCR result in Redis cache
            # ============================================================
            task_success = result_data.get('success', False)
            if task_success:
                cache_success = self.cache.cache_task_result(
                    task_id=cache_key,
                    input_data={
                        "description": task.get('description', ''),
                        "operation": operation,
                        "image_paths": parameters.get('image_paths', []),
                        "image_count": len(parameters.get('image_paths', []))
                    },
                    output_data=result_data,
                    agent_type="ocr"
                )
                
                if cache_success:
                    logger.info(f"[CACHE STORAGE] ✓ OCR result cached: {cache_key}")
                else:
                    logger.debug(f"[CACHE STORAGE] Cache storage skipped or failed")
            else:
                logger.debug(f"[CACHE STORAGE] Skipping cache for failed OCR task")
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "results": {
                    **state['results'],
                    task_id: result_data
                },
                "blackboard": [blackboard_entry],  # Only new entry - LangGraph will concatenate
                "last_updated_key": "ocr_results"  # Observer trigger for auto-synthesis
            }
        
        except Exception as e:
            logger.error(f"[OCR EXECUTE] ✗ Error: {str(e)}")
            
            updated_task = {
                **task,
                "status": TaskStatus.FAILED,
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "failed_task_ids": [task_id]
            }
    
    
    def _execute_web_search_task(self, state: AgentState) -> AgentState:
        """
        Execute a web search operation task using WebSearchAgent.
        
        Enhanced with:
        - CSV file generation tracking for chain to Excel
        - Blackboard updates with search findings
        - File pointer setup for downstream agents
        - Automatic deep_search detection for comprehensive extraction
        """
        print(">>> ENTERED _execute_web_search_task node", flush=True)
        logger.info(">>> ENTERED _execute_web_search_task node")
        logger.info("🔹 ENTERED _execute_web_search_task node")
        
        task_id = state['active_task_id']
        logger.info(f"🔹 Processing task_id: {task_id}")
        
        # DEBUG: Log task details
        task = next(t for t in state['tasks'] if t['id'] == task_id)
        logger.debug(f"[DEBUG] Task found: id={task['id']}, status={task.get('status')}, description={task.get('description', 'N/A')[:80]}")
        logger.debug(f"[DEBUG] Task depth: {task.get('depth', 0)}, Task type: {task.get('type', 'N/A')}")
        
        analysis = task.get('result') if isinstance(task, dict) else task['result']
        
        # Ensure analysis is a dict
        if not isinstance(analysis, dict):
            analysis = {}
            logger.debug("[DEBUG] Analysis was not a dict, initialized to empty dict")
        
        logger.info("=" * 80)
        logger.info(f"[WEB SEARCH AGENT] EXECUTION STARTED")
        logger.info("=" * 80)
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Description: {task['description'][:100]}...")
        logger.info(f"Depth Level: {task.get('depth', 0)}")
        
        try:
            # Extract operation and parameters - handle None values properly
            file_operation = analysis.get('file_operation') or {}
            operation = file_operation.get('type', 'search') if isinstance(file_operation, dict) else 'search'
            params = file_operation.get('parameters', {}) if isinstance(file_operation, dict) else {}
            
            # Ensure params is a dict (could be None)
            if not isinstance(params, dict):
                params = {}
            
            task_description = task.get('description', '').lower()
            
            # Ensure query is always set - use task description as fallback
            if not params.get('query'):
                params['query'] = task.get('description', '')
            
            # ============================================================
            # CACHE CHECK: Look for cached result before executing
            # ============================================================
            cache_key = self._generate_cache_key(
                task_description=task.get('description', ''),
                agent_type="web_search",
                parameters={"query": params.get('query', ''), "operation": operation}
            )
            
            cached_result = self.cache.get_cached_result(cache_key)
            
            if cached_result:
                logger.info("=" * 80)
                logger.info(f"[CACHE HIT] 🎯 Using cached web search result")
                logger.info(f"[CACHE HIT] Cache Key: {cache_key}")
                logger.info(f"[CACHE HIT] Cached At: {cached_result['timestamp']}")
                logger.info(f"[CACHE HIT] TTL Remaining: {cached_result['ttl']} seconds")
                logger.info("=" * 80)
                
                # Use cached output data
                result_data = cached_result['output']
                task_success = result_data.get('success', True)
                
                # Mark task as completed with cached result
                updated_task = {
                    **task,
                    "status": TaskStatus.COMPLETED if task_success else TaskStatus.FAILED,
                    "result": result_data,
                    "updated_at": datetime.now().isoformat(),
                    "cached": True  # Flag to indicate this was from cache
                }
                
                updated_tasks = [
                    updated_task if t['id'] == task_id else t
                    for t in state['tasks']
                ]
                
                # Create blackboard entry for cached result
                blackboard_entry: BlackboardEntry = {
                    "entry_type": "web_search_result_cached",
                    "source_agent": "web_search_agent_cache",
                    "source_task_id": task_id,
                    "content": {
                        "operation": operation,
                        "success": result_data.get('success'),
                        "query": params.get('query', ''),
                        "results_count": result_data.get('results_count', 0),
                        "summary": result_data.get('summary', ''),
                        "cache_hit": True,
                        "cached_at": cached_result['timestamp']
                    },
                    "timestamp": datetime.now().isoformat(),
                    "relevant_to": [task_id],
                    "depth_level": 0,
                    "file_pointers": {},  # type: ignore
                    "chain_next_agents": []  # type: ignore
                }
                
                logger.info(f"[WEB SEARCH AGENT] ✓ Task {task_id} completed from cache")
                
                return {
                    **state,
                    "tasks": updated_tasks,  # type: ignore
                    "results": {
                        **state['results'],
                        task_id: result_data
                    },
                    "blackboard": [blackboard_entry],
                    "last_updated_key": "web_findings"  # Trigger observer even for cached results
                }
            
            # ============================================================
            # CACHE MISS: Execute task normally
            # ============================================================
            logger.info(f"[CACHE MISS] No cached result found - executing web search")
            
            # Intelligent operation selection: use deep_search for:
            # 1. Tasks with specific URLs that need data extraction
            # 2. Tasks mentioning "list", "retrieve", "extract", "find all", "collect"
            # 3. Tasks that seem to need comprehensive research
            deep_search_indicators = [
                'list of', 'retrieve', 'extract', 'find all', 'collect',
                'get all', 'gather', 'scrape all', 'fetch all', 'districts',
                'data from', 'information from', 'details from'
            ]
            
            has_url = params.get('url') or params.get('target_url')
            needs_deep_search = any(indicator in task_description for indicator in deep_search_indicators)
            
            # Auto-upgrade to deep_search if conditions are met and operation is basic search/scrape
            if operation in ['search', 'scrape', 'smart_scrape'] and (has_url or needs_deep_search):
                original_operation = operation
                operation = 'deep_search'
                
                # Migrate parameters for deep_search
                if has_url:
                    params['target_url'] = params.get('url') or params.get('target_url')
                if not params.get('query'):
                    # Use task description as query if not provided
                    params['query'] = task.get('description', '')
                
                logger.info(f"[WEB SEARCH AGENT] Auto-upgraded operation: {original_operation} -> deep_search")
                logger.info(f"[WEB SEARCH AGENT] Reason: {'Has target URL' if has_url else 'Needs comprehensive extraction'}")
            
            logger.info(f"[WEB SEARCH AGENT] Operation: {operation}")
            logger.info(f"[WEB SEARCH AGENT] Query: {str(params.get('query', 'N/A'))[:80]}...")
            logger.info(f"[WEB SEARCH AGENT] Parameters: {params}")
            logger.info("-" * 80)
            
            # Filter relevant blackboard entries for context
            blackboard = state.get('blackboard', [])
            relevant_entries = self._filter_relevant_blackboard_entries(blackboard, task_id, "WEB SEARCH AGENT")
            
            # Create standardized AgentExecutionRequest
            logger.info(f"[WEB SEARCH AGENT] Calling WebSearchAgent.execute_task() with standardized request...")
            logger.info(f"[WEB SEARCH AGENT] Providing {len(relevant_entries)} relevant blackboard entries for context")
            request: AgentExecutionRequest = {
                "task_id": task_id,
                "task_description": task['description'],
                "task_type": "atomic",
                "operation": operation,
                "parameters": params,
                "input_data": params,
                "temp_folder": str(self.config.folders.temp_path),
                "output_folder": str(self.config.folders.output_path),
                "cache_enabled": True,
                "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                "relevant_entries": [self._generate_entry_id(e) for e in relevant_entries],
                "max_retries": 3
            }
            
            # Execute using standardized interface (pass as positional arg to support dual-format handling)
            logger.debug(f"[WEB SEARCH AGENT] Request object created")
            logger.debug(f"[WEB SEARCH AGENT] Request keys: {list(request.keys())}")
            logger.debug(f"[WEB SEARCH AGENT] Request operation: {request.get('operation')}, parameters keys: {list(request.get('parameters', {}).keys())}")
            logger.debug(f"[WEB SEARCH AGENT] About to call execute_task with positional argument (request dict)")
            
            try:
                response = cast(AgentExecutionResponse, self.web_search_agent.execute_task(request))
                logger.debug(f"[WEB SEARCH AGENT] execute_task() returned successfully")
                logger.debug(f"[WEB SEARCH AGENT] Response type: {type(response)}")
                logger.debug(f"[WEB SEARCH AGENT] Response keys: {list(response.keys()) if isinstance(response, dict) else 'N/A'}")
            except Exception as web_search_ex:
                logger.error(f"[WEB SEARCH AGENT] ✗ EXCEPTION during execute_task(): {str(web_search_ex)}")
                logger.error(f"[WEB SEARCH AGENT] Exception type: {type(web_search_ex).__name__}")
                logger.error(f"[WEB SEARCH AGENT] Exception details: {web_search_ex}")
                import traceback
                logger.error(f"[WEB SEARCH AGENT] Traceback:\n{traceback.format_exc()}")
                
                # Create failure response
                response = {
                    'success': False,
                    'result': {
                        'error': str(web_search_ex),
                        'exception_type': type(web_search_ex).__name__,
                        'summary': '',
                        'results_count': 0
                    }
                }
            
            # Extract legacy-compatible result_data from standardized response
            result_data = {
                'success': response['success'],
                'summary': response['result'].get('summary', ''),
                'results_count': response['result'].get('results_count', 0),
                **response['result']  # Include all other result fields
            }
            
            # NORMALIZATION: For regular search operations, convert 'results' to 'extracted_data' and 'findings'
            # This ensures consistency with deep_search output format for downstream processing
            if operation == 'search' and 'results' in result_data and 'extracted_data' not in result_data:
                # Convert search results to extracted_data format
                search_results = result_data.get('results', [])
                
                # Create extracted_data from snippets (with fallback fields in case of naming differences)
                result_data['extracted_data'] = []
                for r in search_results:
                    # Try snippet field first, then fall back to body, then title
                    snippet = r.get('snippet', '') or r.get('body', '') or r.get('title', '')
                    if isinstance(snippet, str) and snippet.strip():
                        result_data['extracted_data'].append(snippet)
                
                # Create findings from full result objects  
                result_data['findings'] = []
                for r in search_results:
                    snippet = r.get('snippet', '') or r.get('body', '') or r.get('title', '')
                    # Only include findings that have at least a URL or title
                    if r.get('url', '') or r.get('title', ''):
                        result_data['findings'].append({
                            'url': r.get('url', ''),
                            'title': r.get('title', ''),
                            'relevance_score': 0.8,  # Regular search results are considered highly relevant
                            'text_preview': (snippet[:500] if isinstance(snippet, str) else str(snippet)[:500])
                        })
                
                # Generate summary if empty
                if not result_data.get('summary'):
                    result_data['summary'] = {
                        'query': result_data.get('query', ''),
                        'total_results': len(search_results),
                        'top_sources': [r.get('url', '') for r in search_results[:5] if r.get('url', '')],
                        'snippet_count': len(result_data['extracted_data'])
                    }
                
                logger.debug(f"[WEB SEARCH AGENT] Normalized search results: {len(result_data['extracted_data'])} items, {len(result_data['findings'])} findings")
                
                # CRITICAL: Ensure we have data after normalization
                if result_data.get('results_count', 0) > 0 and not result_data.get('extracted_data', []):
                    logger.warning(f"[WEB SEARCH AGENT] WARNING: results_count={result_data.get('results_count')} but extracted_data is empty after normalization!")
                    logger.warning(f"[WEB SEARCH AGENT] Sample result structure: {search_results[0] if search_results else 'N/A'}")
                    # Try alternate approach - if all else fails, use stringified results
                    if search_results:
                        result_data['extracted_data'] = [str(r) for r in search_results]
                        result_data['findings'] = [{'url': str(idx), 'title': str(r), 'relevance_score': 0.8} for idx, r in enumerate(search_results)]
            
            # Update task with results
            updated_task = {
                **task,
                "status": TaskStatus.COMPLETED if response['success'] else TaskStatus.FAILED,
                "result": result_data,
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            logger.info("-" * 80)
            
            # Log appropriately based on actual status
            task_success = result_data.get('success', False)
            if task_success:
                logger.info(f"[WEB SEARCH AGENT] ✓ SUCCESS: Task {task_id} completed")
            else:
                logger.warning(f"[WEB SEARCH AGENT] ✗ FAILED: Task {task_id} failed")
                if result_data.get('error'):
                    logger.warning(f"[WEB SEARCH AGENT] Error: {result_data.get('error')}")
            
            # Log result_data details for debugging
            logger.info(f"[WEB SEARCH AGENT] Result Data Keys: {list(result_data.keys())}")
            logger.info(f"[WEB SEARCH AGENT] Success Flag: {result_data.get('success')}")
            logger.info(f"[WEB SEARCH AGENT] Task Status Set To: {updated_task['status']}")
            logger.info(f"[WEB SEARCH AGENT] Status Enum Value: {updated_task['status'].value if hasattr(updated_task['status'], 'value') else 'N/A'}")
            
            # CRITICAL CHECK: Ensure successful search actually has results
            if task_success and operation == 'search':
                results_count = result_data.get('results_count', 0)
                extracted_count = len(result_data.get('extracted_data', []))
                findings_count = len(result_data.get('findings', []))
                
                if results_count > 0 and extracted_count == 0 and findings_count == 0:
                    logger.error(f"[WEB SEARCH AGENT] CRITICAL: Search returned results_count={results_count} but NO extracted_data or findings!")
                    logger.error(f"[WEB SEARCH AGENT] Result structure: {json.dumps({k: type(v).__name__ for k, v in result_data.items()}, indent=2)}")
                    # Mark as failed since data extraction failed
                    updated_task['status'] = TaskStatus.FAILED
                    updated_task['error'] = "Search returned results but data extraction failed"
                    task_success = False
            
            # Log retry information if available
            retry_count = result_data.get('retry_count', 0)
            if retry_count > 0:
                logger.info(f"[WEB SEARCH AGENT] Retries: {retry_count}")
                attempts = result_data.get('attempts', [])
                for att in attempts:
                    logger.info(f"[WEB SEARCH AGENT]   Attempt {att.get('attempt')}: '{att.get('query')}' -> {att.get('results_count')} results")
            
            # Log results based on operation type
            if operation == 'deep_search':
                pages_visited = result_data.get('pages_visited', 0)
                extracted_count = len(result_data.get('extracted_data', []))
                summary = result_data.get('summary', {})
                logger.info(f"[WEB SEARCH AGENT] Deep Search Results:")
                logger.info(f"[WEB SEARCH AGENT]   - Pages Visited: {pages_visited}")
                logger.info(f"[WEB SEARCH AGENT]   - Relevant Pages: {summary.get('relevant_pages_found', 0)}")
                logger.info(f"[WEB SEARCH AGENT]   - Extracted Items: {extracted_count}")
            else:
                results_count = result_data.get('results_count', 0)
                logger.info(f"[WEB SEARCH AGENT] Results Found: {results_count}")
                
                # Warn if no results found
                if results_count == 0 and task_success:
                    logger.warning(f"[WEB SEARCH AGENT] ⚠ WARNING: Task marked successful but 0 results found")
            
            # Create blackboard entry with findings and chain markers
            blackboard_entry: BlackboardEntry = {
                "entry_type": "web_search_result",
                "source_agent": "web_search_agent",
                "source_task_id": task_id,
                "content": {
                    "operation": operation,
                    "success": result_data.get('success'),
                    "query": params.get('query', ''),
                    "results_count": result_data.get('results_count', 0),
                    "summary": result_data.get('summary', ''),
                    # Include deep_search specific data
                    "pages_visited": result_data.get('pages_visited', 0),
                    "extracted_data": result_data.get('extracted_data', [])[:50],  # Limit for blackboard
                    "findings": result_data.get('findings', [])[:10],  # Top findings
                },
                "timestamp": datetime.now().isoformat(),
                "relevant_to": [task_id],
                "depth_level": 0,
                "file_pointers": {},  # type: ignore
                "chain_next_agents": []  # type: ignore
            }
            
            # Check for CSV file generation (chain to Excel condition)
            output_files = result_data.get('output', {}).get('generated_files', {})
            csv_file = output_files.get('csv')
            
            if csv_file:
                logger.info(f"[WEB SEARCH AGENT] 🔗 CHAIN DETECTION: Generated CSV file")
                logger.info(f"[WEB SEARCH AGENT] 🔗 File: {csv_file}")
                logger.info(f"[WEB SEARCH AGENT] 🔗 Scheduling Excel Agent for downstream processing")
                blackboard_entry["file_pointers"] = {  # type: ignore
                    "excel_agent": csv_file
                }
                blackboard_entry["chain_next_agents"] = ["excel_agent"]  # type: ignore
            
            logger.info("=" * 80)
            logger.info(f"[WEB SEARCH AGENT] 📊 EXECUTION COMPLETE - Returning state")
            logger.info(f"[WEB SEARCH AGENT] Task {task_id} final status: {updated_task['status']}")
            logger.info(f"[WEB SEARCH AGENT] Blackboard entry type: {blackboard_entry['entry_type']}")
            logger.info(f"[WEB SEARCH AGENT] Next step: aggregate_results → completion check")
            logger.info("=" * 80)
            
            # ============================================================
            # CACHE STORAGE: Store successful result in Redis cache
            # ============================================================
            if task_success:
                cache_success = self.cache.cache_task_result(
                    task_id=cache_key,
                    input_data={
                        "description": task.get('description', ''),
                        "query": params.get('query', ''),
                        "operation": operation,
                        "parameters": params
                    },
                    output_data=result_data,
                    agent_type="web_search"
                )
                
                if cache_success:
                    logger.info(f"[CACHE STORAGE] ✓ Result cached: {cache_key}")
                else:
                    logger.debug(f"[CACHE STORAGE] Cache storage skipped or failed")
            else:
                logger.debug(f"[CACHE STORAGE] Skipping cache for failed task")
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "results": {
                    **state['results'],
                    task_id: result_data
                },
                "blackboard": [blackboard_entry],  # Only new entry - LangGraph will concatenate
                "last_updated_key": "web_findings"  # Observer trigger for auto-synthesis
            }
        
        except Exception as e:
            logger.error(f"[WEB SEARCH AGENT] ✗ FAILED: Error executing task")
            logger.error(f"[WEB SEARCH AGENT] Exception: {str(e)}")
            logger.error(f"[WEB SEARCH AGENT] Exception Type: {type(e).__name__}")
            logger.error(f"[WEB SEARCH AGENT] Task ID: {task_id}")
            logger.error(f"[WEB SEARCH AGENT] Operation: {operation}")
            logger.error(f"[WEB SEARCH AGENT] Parameters: {params}")
            
            # Get full traceback
            import traceback
            logger.error(f"[WEB SEARCH AGENT] Traceback:\n{traceback.format_exc()}")
            logger.info("=" * 80)
            
            updated_task = {
                **task,
                "status": TaskStatus.FAILED,
                "error": str(e),
                "error_type": type(e).__name__,
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "failed_task_ids": [task_id]
            }
    
    
    def _execute_code_interpreter_task(self, state: AgentState) -> AgentState:
        """
        Execute a code interpreter operation task using CodeInterpreterAgent.
        
        Enhanced with:
        - Code generation from natural language requests
        - Subprocess execution with output capture
        - Generated file tracking for output
        - Blackboard updates with computational results
        """
        task_id = state['active_task_id']
        task = next(t for t in state['tasks'] if t['id'] == task_id)
        analysis = task.get('result') if isinstance(task, dict) else task['result']
        
        # Ensure analysis is a dict
        if not isinstance(analysis, dict):
            analysis = {}
        
        logger.info("=" * 80)
        logger.info(f"[CODE INTERPRETER AGENT] EXECUTION STARTED")
        logger.info("=" * 80)
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Description: {task['description'][:100]}...")
        logger.info(f"Depth Level: {task.get('depth', 0)}")
        
        try:
            # Extract operation and parameters - handle None values properly
            file_operation = analysis.get('file_operation') or {}
            operation = file_operation.get('type', 'execute_analysis') if isinstance(file_operation, dict) else 'execute_analysis'
            params = file_operation.get('parameters', {}) if isinstance(file_operation, dict) else {}
            
            logger.info(f"[CODE INTERPRETER AGENT] Operation: {operation}")
            logger.info(f"[CODE INTERPRETER AGENT] Request: {params.get('request', 'N/A')[:80]}...")
            logger.info(f"[CODE INTERPRETER AGENT] Parameters: {params}")
            logger.info("-" * 80)
            
            # Filter relevant blackboard entries for context
            blackboard = state.get('blackboard', [])
            relevant_entries = self._filter_relevant_blackboard_entries(blackboard, task_id, "CODE INTERPRETER AGENT")
            
            # Create standardized AgentExecutionRequest
            logger.info(f"[CODE INTERPRETER AGENT] Calling CodeInterpreterAgent.execute_task() with standardized request...")
            logger.info(f"[CODE INTERPRETER AGENT] Providing {len(relevant_entries)} relevant blackboard entries for context")
            request: AgentExecutionRequest = {
                "task_id": task_id,
                "task_description": task['description'],
                "task_type": "atomic",
                "operation": operation,
                "parameters": params,
                "input_data": params,
                "temp_folder": str(self.config.folders.temp_path),
                "output_folder": str(self.config.folders.output_path),
                "cache_enabled": True,
                "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                "relevant_entries": [self._generate_entry_id(e) for e in relevant_entries],
                "max_retries": 3
            }
            
            # Execute using standardized interface
            response: AgentExecutionResponse = self.code_interpreter_agent.execute_task(request=request)
            
            # Extract legacy-compatible result_data from standardized response
            result_data = {
                'success': response['success'],
                'output': response['result'].get('output', ''),
                'generated_code': response['result'].get('generated_code', ''),
                'execution_time': response.get('execution_time_ms', 0) / 1000.0,
                'generated_files': response['result'].get('generated_files', {}),
                'output_directory': response['result'].get('output_directory', ''),
                'libraries_used': response['result'].get('libraries_used', []),
                **response['result']  # Include all other result fields
            }
            
            # Update task with results
            updated_task = {
                **task,
                "status": TaskStatus.COMPLETED if response['success'] else TaskStatus.FAILED,
                "result": result_data,
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            logger.info("-" * 80)
            logger.info(f"[CODE INTERPRETER AGENT] ✓ SUCCESS: Task {task_id} completed")
            logger.info(f"[CODE INTERPRETER AGENT] Status: {updated_task['status']}")
            logger.info(f"[CODE INTERPRETER AGENT] Execution Time: {result_data.get('execution_time', 0):.2f}s")
            
            # Create blackboard entry with computational results
            generated_files = result_data.get('generated_files', {})
            all_files = []
            for file_list in generated_files.values():
                if isinstance(file_list, list):
                    all_files.extend(file_list)
            
            blackboard_entry: BlackboardEntry = {
                "entry_type": "computational_results",
                "source_agent": "code_interpreter_agent",
                "source_task_id": task_id,
                "content": {
                    "operation": operation,
                    "request": params.get('request', ''),
                    "success": result_data.get('success'),
                    "output": result_data.get('output', ''),
                    "generated_code": result_data.get('generated_code', ''),
                    "execution_time": result_data.get('execution_time', 0),
                    "libraries_used": result_data.get('libraries_used', [])
                },
                "timestamp": datetime.now().isoformat(),
                "relevant_to": [task_id],
                "depth_level": 0,
                "file_pointers": {  # type: ignore
                    "output_directory": result_data.get('output_directory', ''),
                    "generated_files": generated_files
                },
                "chain_next_agents": []  # type: ignore
            }
            
            # Check for generated charts/images for downstream processing
            images = generated_files.get('images', []) or []
            charts = generated_files.get('charts', []) or []
            
            if images or charts:
                logger.info(f"[CODE INTERPRETER AGENT] 🔗 CHAIN DETECTION: Generated {len(images)} images and {len(charts)} charts")
                logger.info(f"[CODE INTERPRETER AGENT] 🔗 Scheduling OCR Agent for downstream processing")
                blackboard_entry["chain_next_agents"] = ["ocr_image_agent"]  # type: ignore
            
            logger.info("=" * 80)
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "results": {
                    **state['results'],
                    task_id: result_data
                },
                "blackboard": [blackboard_entry]  # Only new entry - LangGraph will concatenate
            }
        
        except Exception as e:
            logger.error(f"[CODE INTERPRETER AGENT] ✗ FAILED: Error executing task")
            logger.error(f"[CODE INTERPRETER AGENT] Exception: {str(e)}")
            logger.info("=" * 80)
            
            updated_task = {
                **task,
                "status": TaskStatus.FAILED,
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "failed_task_ids": [task_id]
            }
    
    
    def _execute_data_extraction_task(self, state: AgentState) -> AgentState:
        """
        Execute a data extraction task using DataExtractionAgent.
        
        This agent intelligently extracts relevant data from input/temp files
        based on the task description, avoiding large context overhead.
        
        Capabilities:
        - Extract relevant content from input files
        - Search for specific data across files
        - Summarize file contents
        - Get file previews
        """
        task_id = state['active_task_id']
        task = next(t for t in state['tasks'] if t['id'] == task_id)
        analysis = task.get('result') if isinstance(task, dict) else task['result']
        
        # Ensure analysis is a dict
        if not isinstance(analysis, dict):
            analysis = {}
        
        logger.info("=" * 80)
        logger.info(f"[DATA EXTRACTION AGENT] Starting task: {task_id}")
        logger.info(f"[DATA EXTRACTION AGENT] Description: {task.get('description', 'N/A')[:100]}")
        
        try:
            # Get file operation details
            file_operation = analysis.get('file_operation') or {}
            operation = file_operation.get('operation', 'extract_relevant_data')
            parameters = file_operation.get('parameters', {})
            
            # Use task description as extraction objective
            extraction_objective = task.get('description', state['objective'])
            
            # Get folder paths from metadata
            input_folder = state.get('metadata', {}).get('input_folder', './input_folder')
            temp_folder = state.get('metadata', {}).get('temp_folder', './temp_folder')
            
            # Prepare parameters for standardized interface
            if 'input_folder' not in parameters:
                parameters['input_folder'] = input_folder
            if 'objective' not in parameters and operation == 'extract_relevant_data':
                parameters['objective'] = extraction_objective
            if 'temp_folder' not in parameters:
                parameters['temp_folder'] = temp_folder
            
            logger.info(f"[DATA EXTRACTION AGENT] Operation: {operation}")
            logger.info(f"[DATA EXTRACTION AGENT] Parameters: {parameters}")
            logger.info("-" * 80)
            
            # Filter relevant blackboard entries for context
            blackboard = state.get('blackboard', [])
            relevant_entries = self._filter_relevant_blackboard_entries(blackboard, task_id, "DATA EXTRACTION AGENT")
            
            # Create standardized AgentExecutionRequest
            logger.info(f"[DATA EXTRACTION AGENT] Calling DataExtractionAgent.execute_task() with standardized request...")
            logger.info(f"[DATA EXTRACTION AGENT] Providing {len(relevant_entries)} relevant blackboard entries for context")
            request: AgentExecutionRequest = {
                "task_id": task_id,
                "task_description": task['description'],
                "task_type": "atomic",
                "operation": operation,
                "parameters": parameters,
                "input_data": parameters,
                "temp_folder": str(self.config.folders.temp_path),
                "output_folder": str(self.config.folders.output_path),
                "cache_enabled": True,
                "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                "relevant_entries": [self._generate_entry_id(e) for e in relevant_entries],
                "max_retries": 3
            }
            
            # Execute using standardized interface
            response = cast(AgentExecutionResponse, self.data_extraction_agent.execute_task(request=request))
            
            logger.info(f"[DATA EXTRACTION AGENT] Operation '{operation}' completed")
            
            # Extract legacy-compatible result_data from standardized response
            result_data = {
                "operation": operation,
                "success": response['success'],
                "data": response['result']
            }
            
            updated_task = {
                **task,
                "status": TaskStatus.COMPLETED if response['success'] else TaskStatus.FAILED,
                "result": result_data,
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            # Create blackboard entry with extracted data summary
            result = response.get('result', {})
            blackboard_entry: BlackboardEntry = {
                "entry_type": BlackboardType.DATA_POINT,
                "source_agent": "data_extraction_agent",
                "source_task_id": task_id,
                "parent_task_id": task.get('parent_id'),
                "content": {
                    "operation": operation,
                    "files_processed": result.get('files_processed', 0) if result else 0,
                    "total_content_size": result.get('total_content_size', 0) if result else 0,
                    "summary": result.get('summary', '') if result else '',
                    "extractions_count": len(result.get('extractions', [])) if result else 0,
                    "objective": extraction_objective[:200],
                    "input_folder": input_folder
                },
                "timestamp": datetime.now().isoformat(),
                "relevant_to": [task_id],
                "depth_level": task.get('depth', 0)
            }
            
            logger.info(f"[DATA EXTRACTION AGENT] ✓ Task completed successfully")
            logger.info(f"[DATA EXTRACTION AGENT] Files processed: {result.get('files_processed', 0) if result else 0}")
            logger.info("=" * 80)
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "results": {
                    **state['results'],
                    task_id: result_data
                },
                "blackboard": [blackboard_entry]  # Only new entry - LangGraph will concatenate
            }
        
        except Exception as e:
            logger.error(f"[DATA EXTRACTION AGENT] ✗ FAILED: Error executing task")
            logger.error(f"[DATA EXTRACTION AGENT] Exception: {str(e)}")
            logger.info("=" * 80)
            
            updated_task = {
                **task,
                "status": TaskStatus.FAILED,
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "failed_task_ids": [task_id]
            }
    
    
    def _execute_document_task(self, state: AgentState) -> AgentState:
        """
        Execute a document file operation task using DocumentAgent.
        
        Capabilities:
        - Create and format DOCX files with rich formatting
        - Create TXT files with structured content
        - Handle content with headings, lists, paragraphs
        - Automatic file naming with timestamps
        - File path management with output folder defaults
        """
        task_id = state['active_task_id']
        task = next(t for t in state['tasks'] if t['id'] == task_id)
        analysis = task.get('result') if isinstance(task, dict) else task['result']
        
        # Ensure analysis is a dict
        if not isinstance(analysis, dict):
            analysis = {}
        
        logger.info("=" * 80)
        logger.info(f"[DOCUMENT AGENT] EXECUTION STARTED")
        logger.info("=" * 80)
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Description: {task['description'][:100]}...")
        logger.info(f"Depth Level: {task.get('depth', 0)}")
        
        try:
            # Extract file operation details from analysis
            file_operation = analysis.get('file_operation', {})
            
            # VALIDATION: Check if file_operation is valid
            # Accepts type as "document" or "docx"/"txt" for backwards compatibility
            operation_type = file_operation.get('type', '')
            valid_types = ['document', 'docx', 'txt']
            
            if not file_operation or operation_type not in valid_types:
                logger.error(f"[DOCUMENT AGENT] ✗ Invalid document operation specification")
                logger.error(f"[DOCUMENT AGENT]   Received type: '{operation_type}'")
                logger.error(f"[DOCUMENT AGENT]   Valid types: {valid_types}")
                logger.error(f"[DOCUMENT AGENT]   Full file_operation: {file_operation}")
                logger.error(f"[DOCUMENT AGENT]   Full analysis: {analysis}")
                
                # RECOVERY: Attempt to recover with ProblemSolverAgent
                logger.info("[DOCUMENT AGENT] Attempting recovery with ProblemSolverAgent...")
                
                error_context = {
                    "task_id": task_id,
                    "task_description": task['description'],
                    "received_type": operation_type,
                    "received_operation": file_operation.get('operation', 'unknown'),
                    "full_analysis": analysis
                }
                
                recovery_request: AgentExecutionRequest = {
                    "task_id": f"recover_document_{task_id}",
                    "task_description": f"Fix invalid document operation for: {task['description']}",
                    "task_type": "analysis",
                    "operation": "diagnose_error",
                    "parameters": {
                        "error_message": f"Document task failed: invalid operation specification (type='{operation_type}')",
                        "task_context": error_context,
                        "agent_type": "document_task",
                        "suggestions": [
                            "File operation should have type='document' (or 'docx'/'txt' for legacy)",
                            "Operation should be 'create_docx', 'create_txt', 'append_docx', 'append_txt', 'read_docx', or 'read_txt'",
                            "Parameters should include 'content' and optionally 'title', 'file_path'",
                            "For create_docx: {\"content\": \"...\", \"title\": \"...\"}",
                            "For create_txt: {\"content\": \"...\", \"encoding\": \"utf-8\"}"
                        ]
                    },
                    "input_data": {},
                    "temp_folder": str(self.config.folders.temp_path),
                    "output_folder": str(self.config.folders.output_path),
                    "cache_enabled": False,
                    "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                    "relevant_entries": [],
                    "max_retries": 1
                }
                
                try:
                    diagnose_response = cast(
                        AgentExecutionResponse,
                        self.problem_solver_agent.execute_task(recovery_request)
                    )
                    
                    if diagnose_response.get('success'):
                        recovery_result = diagnose_response.get('result', {})
                        logger.info(f"[DOCUMENT AGENT] Recovery analysis: {recovery_result}")
                        
                        # Request human review instead of silent failure
                        logger.info(f"[DOCUMENT AGENT] Requesting human review for task {task_id}")
                        
                        return {
                            **state,
                            "requires_human_review": True,
                            "human_review_context": {
                                "reason": f"Document task {task_id} requires human intervention to fix invalid operation specification",
                                "error": f"Invalid document operation type: '{operation_type}'. Expected: {valid_types}",
                                "task_id": task_id,
                                "task_description": task['description'],
                                "error_details": error_context,
                                "recovery_suggestions": recovery_result,
                                "original_analysis": analysis
                            }
                        }
                    else:
                        logger.error("[DOCUMENT AGENT] Recovery analysis also failed - proceeding with failure")
                        
                except Exception as recovery_error:
                    logger.error(f"[DOCUMENT AGENT] Recovery attempt failed: {str(recovery_error)}")
                    logger.log_exception("Document recovery exception:", recovery_error)
                
                # If recovery failed or wasn't attempted, mark task as failed with context
                updated_task = {
                    **task,
                    "status": TaskStatus.FAILED,
                    "error": f"Invalid document operation specification - type must be one of {valid_types}",
                    "error_details": error_context,
                    "updated_at": datetime.now().isoformat()
                }
                
                updated_tasks = [
                    updated_task if t['id'] == task_id else t
                    for t in state['tasks']
                ]
                
                return {
                    **state,
                    "tasks": updated_tasks,  # type: ignore
                    "failed_task_ids": [task_id]
                }
            
            operation = file_operation.get('operation', 'create')
            parameters = file_operation.get('parameters', {})
            
            # Generate default filename if not provided
            if operation == 'create' and not parameters.get('file_path'):
                safe_task_id = task_id.replace('.', '_')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                doc_type = file_operation.get('type', 'docx')
                default_filename = f"document_{safe_task_id}_{timestamp}.{doc_type}"
                parameters['file_path'] = str(self.config.folders.output_path / default_filename)
                logger.info(f"[DOCUMENT AGENT] Generated default filename: {default_filename}")
            
            logger.info(f"[DOCUMENT AGENT] Operation: {operation}")
            logger.info(f"[DOCUMENT AGENT] Type: {file_operation.get('type')}")
            logger.info(f"[DOCUMENT AGENT] File: {parameters.get('file_path', 'N/A')[:80]}...")
            logger.info(f"[DOCUMENT AGENT] Parameters: {parameters}")
            logger.info("-" * 80)
            
            # Filter relevant blackboard entries for context
            blackboard = state.get('blackboard', [])
            relevant_entries = self._filter_relevant_blackboard_entries(blackboard, task_id, "DOCUMENT AGENT")
            
            # Create standardized AgentExecutionRequest
            logger.info(f"[DOCUMENT AGENT] Calling DocumentAgent.execute_task() with standardized request...")
            logger.info(f"[DOCUMENT AGENT] Providing {len(relevant_entries)} relevant blackboard entries for context")
            request: AgentExecutionRequest = {
                "task_id": task_id,
                "task_description": task['description'],
                "task_type": "atomic",
                "operation": operation,
                "parameters": parameters,
                "input_data": file_operation,
                "temp_folder": str(self.config.folders.temp_path),
                "output_folder": str(self.config.folders.output_path),
                "cache_enabled": True,
                "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                "relevant_entries": [self._generate_entry_id(e) for e in relevant_entries],
                "max_retries": 3
            }
            
            # Execute using standardized interface
            response = cast(AgentExecutionResponse, self.document_agent.execute_task(request=request))
            
            # Extract legacy-compatible result_data from standardized response
            result_data = {
                'success': response['success'],
                'file_path': response['result'].get('file_path', ''),
                'file_size': response['result'].get('file_size', 0),
                'document_type': response['result'].get('document_type', ''),
                'summary': response['result'].get('summary', ''),
                **response['result']
            }
            
            # Update task with results
            updated_task = {
                **task,
                "status": TaskStatus.COMPLETED if response['success'] else TaskStatus.FAILED,
                "result": result_data,
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            logger.info("-" * 80)
            logger.info(f"[DOCUMENT AGENT] ✓ SUCCESS: Task {task_id} completed")
            logger.info(f"[DOCUMENT AGENT] Status: {updated_task['status']}")
            logger.info(f"[DOCUMENT AGENT] File: {result_data.get('file_path', 'N/A')}")
            logger.info(f"[DOCUMENT AGENT] Size: {result_data.get('file_size', 0)} bytes")
            logger.info("=" * 80)
            
            # Create blackboard entry with document metadata
            blackboard_entry: BlackboardEntry = {
                "entry_type": "document_creation_result",
                "source_agent": "document_agent",
                "source_task_id": task_id,
                "content": {
                    "operation": operation,
                    "success": result_data.get('success'),
                    "file_path": result_data.get('file_path', ''),
                    "file_size": result_data.get('file_size', 0),
                    "document_type": result_data.get('document_type', ''),
                    "execution_time_ms": response.get('execution_time_ms', 0)
                },
                "timestamp": datetime.now().isoformat(),
                "relevant_to": [task_id],
                "depth_level": 0,
                "file_pointers": {}
            }
            
            # Add source file path for traceability
            output_file = parameters.get('file_path', '')
            if output_file:
                blackboard_entry["source_file_path"] = output_file  # type: ignore
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "results": {
                    **state['results'],
                    task_id: result_data
                },
                "blackboard": [blackboard_entry]  # Only new entry - LangGraph will concatenate
            }
        
        except Exception as e:
            logger.error(f"[DOCUMENT AGENT] ✗ FAILED: Error executing task")
            logger.error(f"[DOCUMENT AGENT] Exception type: {type(e).__name__}")
            logger.error(f"[DOCUMENT AGENT] Exception message: {str(e)}")
            logger.log_exception("Document agent exception:", e)
            logger.info("=" * 80)
            
            updated_task = {
                **task,
                "status": TaskStatus.FAILED,
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "failed_task_ids": [task_id]
            }
    
    
    def _execute_problem_solver_task(self, state: AgentState) -> AgentState:
        """
        Execute an analysis/problem-solving task using ProblemSolverAgent.
        
        This handles tasks that require deep analysis, synthesis, or comparison of data,
        such as "Analyze each trend comprehensively" or "Compare findings".
        
        The problem solver analyzes blackboard data and existing task results
        to provide insights and conclusions.
        """
        task_id = state['active_task_id']
        task = next(t for t in state['tasks'] if t['id'] == task_id)
        analysis = task.get('result') if isinstance(task, dict) else task['result']
        
        # Ensure analysis is a dict
        if not isinstance(analysis, dict):
            analysis = {}
        
        logger.info("=" * 80)
        logger.info(f"[PROBLEM SOLVER] EXECUTION STARTED")
        logger.info("=" * 80)
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Description: {task['description'][:100]}...")
        logger.info(f"Depth Level: {task.get('depth', 0)}")
        
        try:
            # Build analysis prompt using existing blackboard data
            objective = state.get('objective', '')
            blackboard = state.get('blackboard', [])
            
            # Extract relevant blackboard entries for this task
            # For analysis tasks, be more inclusive to capture upstream data
            task_prefix = '.'.join(task_id.split('.')[:-1]) if '.' in task_id else task_id
            
            def is_relevant_entry(entry: BlackboardEntry, current_task_id: str, prefix: str) -> bool:
                """Check if blackboard entry is relevant to current analysis task."""
                relevant_to = entry.get('relevant_to', [])
                
                # No specific relevance - available to all tasks
                if not relevant_to:
                    return True
                
                # Directly relevant to this task
                if current_task_id in relevant_to:
                    return True
                
                # For analysis tasks, include data from sibling/related tasks
                # E.g., task 1.1.3 should see data from 1.1.1 and 1.1.2
                for rel_task in relevant_to:
                    # Same parent (sibling tasks)
                    rel_prefix = '.'.join(rel_task.split('.')[:-1]) if '.' in rel_task else rel_task
                    if rel_prefix == prefix:
                        return True
                    
                    # Parent task data
                    if current_task_id.startswith(rel_task + '.'):
                        return True
                    
                    # Child task data (include data from subtasks)
                    if rel_task.startswith(current_task_id + '.'):
                        return True
                
                return False
            
            relevant_entries = [
                e for e in blackboard 
                if is_relevant_entry(e, task_id, task_prefix)
            ]
            
            logger.info(f"[PROBLEM SOLVER] Total blackboard entries: {len(blackboard)}")
            logger.info(f"[PROBLEM SOLVER] Filtered relevant entries: {len(relevant_entries)}")
            logger.info(f"[PROBLEM SOLVER] Task ID for filtering: {task_id}")
            logger.info(f"[PROBLEM SOLVER] Task prefix for filtering: {task_prefix}")
            
            # Log sample of what was filtered
            if blackboard and not relevant_entries:
                logger.warning(f"[PROBLEM SOLVER] ⚠️  All {len(blackboard)} entries were filtered out!")
                sample_entries = blackboard[:3]
                for idx, entry in enumerate(sample_entries, 1):
                    logger.warning(f"[PROBLEM SOLVER]   Sample entry {idx}: relevant_to={entry.get('relevant_to', [])} source={entry.get('source_agent', 'unknown')}")
            elif relevant_entries:
                sample_sources = list(set([e.get('source_agent', 'unknown') for e in relevant_entries]))
                logger.info(f"[PROBLEM SOLVER] Data sources in relevant entries: {sample_sources}")
            
            # Create analysis request for problem solver
            # Build data for analysis separately to avoid hashable type issues
            analysis_data = [
                {
                    'source': e.get('source_agent', 'unknown'),
                    'type': e.get('entry_type', 'unknown'),
                    'content': e.get('content', {})
                }
                for e in relevant_entries[:10]  # Limit to last 10 entries for clarity
            ]
            
            analysis_prompt = f"""You are a problem solver tasked with comprehensive analysis.

OBJECTIVE: {objective}

TASK: {task['description']}

CONTEXT:
{json.dumps(state.get('metadata', {}), indent=2)}

AVAILABLE DATA FOR ANALYSIS:
{json.dumps(analysis_data, indent=2)}

Your task:
1. Analyze the available data comprehensively
2. Identify key insights, patterns, and relationships
3. Compare different data sources and findings
4. Synthesize into clear conclusions
5. Flag any contradictions or areas needing clarification

Respond with JSON:
{{
  "analysis": "Detailed analysis of the data",
  "key_insights": ["insight 1", "insight 2", ...],
  "patterns_identified": ["pattern 1", "pattern 2", ...],
  "contradictions": ["contradiction 1", ...] or [],
  "recommendations": ["recommendation 1", "recommendation 2", ...],
  "confidence_level": "HIGH/MEDIUM/LOW"
}}"""
            
            logger.info("[PROBLEM SOLVER] Building analysis request...")
            
            # Create execution request for task relay
            # Convert BlackboardEntry objects to dicts for the request
            blackboard_as_dicts: list[dict[str, Any]] = [
                dict(entry) if isinstance(entry, dict) else {**entry}  # type: ignore
                for entry in blackboard
            ]
            relevant_entries_as_dicts: list[dict[str, Any]] = [
                dict(entry) if isinstance(entry, dict) else {**entry}  # type: ignore
                for entry in relevant_entries
            ]
            
            logger.info(f"[PROBLEM SOLVER] Passing {len(relevant_entries_as_dicts)} relevant entries for analysis")
            logger.info(f"[PROBLEM SOLVER] Data sources: {list(set([e.get('source_agent', 'unknown') for e in relevant_entries_as_dicts]))}")
            
            request: AgentExecutionRequest = {
                "task_id": task_id,
                "task_description": task['description'],
                "task_type": "analysis",
                "operation": "analyze_task",
                "parameters": {
                    "analysis_type": "comprehensive",
                    "data_sources": [e.get('source_agent', 'unknown') for e in relevant_entries],
                    "depth_level": task.get('depth', 0),
                    "objective": objective,
                    "blackboard_entries": relevant_entries_as_dicts  # Pass blackboard entries in parameters
                },
                "input_data": {
                    "objective": objective,
                    "blackboard_entries": relevant_entries_as_dicts  # Also in input_data for backward compatibility
                },
                "temp_folder": state.get('metadata', {}).get('temp_folder', ''),
                "output_folder": state.get('metadata', {}).get('output_folder', ''),
                "cache_enabled": state.get('metadata', {}).get('cache_enabled', True),
                "blackboard": blackboard_as_dicts,
                "relevant_entries": [e.get('id', '') for e in relevant_entries],
                "max_retries": 2
            }
            
            logger.info("[PROBLEM SOLVER] Invoking ProblemSolverAgent...")
            response_result = self.problem_solver_agent.execute_task(request)
            
            # Handle both dict and AgentExecutionResponse formats
            response: AgentExecutionResponse | dict[str, Any]
            if isinstance(response_result, dict):
                response = response_result
            else:
                response = response_result  # type: ignore
            
            logger.info(f"[PROBLEM SOLVER] ✓ Agent execution completed")
            logger.info(f"[PROBLEM SOLVER]   Success: {response.get('success')}")
            logger.info(f"[PROBLEM SOLVER]   Execution time: {response.get('execution_time_ms', 0)}ms")
            
            # Extract results
            success = response.get('success', False)
            result_data = response.get('result', {})
            error = response.get('error', None)
            
            if success:
                # Create blackboard entry for analysis results
                analysis_entry: BlackboardEntry = {
                    "entry_type": "problem_solver_analysis",
                    "source_agent": "problem_solver_agent",
                    "source_task_id": task_id,
                    "content": result_data,
                    "timestamp": datetime.now().isoformat(),
                    "relevant_to": [task_id],
                    "depth_level": task.get('depth', 0),
                    "file_pointers": {},  # type: ignore
                    "chain_next_agents": []  # type: ignore
                }
                
                logger.info("[PROBLEM SOLVER] Analysis results saved to blackboard")
                
                # Update task as completed
                updated_task: Task = {
                    **task,
                    "status": TaskStatus.COMPLETED,
                    "result": result_data,
                    "updated_at": datetime.now().isoformat()
                }
            else:
                logger.warning(f"[PROBLEM SOLVER] Analysis reported failure: {error}")
                # Create error entry on blackboard
                analysis_entry: BlackboardEntry = {
                    "entry_type": "problem_solver_error",
                    "source_agent": "problem_solver_agent",
                    "source_task_id": task_id,
                    "content": {"error": error},
                    "timestamp": datetime.now().isoformat(),
                    "relevant_to": [task_id],
                    "depth_level": task.get('depth', 0),
                    "file_pointers": {},  # type: ignore
                    "chain_next_agents": []  # type: ignore
                }
                
                updated_task: Task = {
                    **task,
                    "status": TaskStatus.FAILED,
                    "error": error,
                    "result": result_data,
                    "updated_at": datetime.now().isoformat()
                }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            logger.info("=" * 80)
            logger.info(f"[PROBLEM SOLVER] EXECUTION COMPLETED")
            logger.info("=" * 80)
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "results": {
                    **state['results'],
                    task_id: result_data
                },
                "blackboard": [analysis_entry]  # Only new entry - LangGraph will concatenate
            }
        
        except Exception as e:
            logger.error(f"[PROBLEM SOLVER] ✗ FAILED: Error executing task")
            logger.error(f"[PROBLEM SOLVER] Exception type: {type(e).__name__}")
            logger.error(f"[PROBLEM SOLVER] Exception message: {str(e)}")
            logger.log_exception("Problem solver exception:", e)
            logger.info("=" * 80)
            
            updated_task = {
                **task,
                "status": TaskStatus.FAILED,
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            return {
                **state,
                "tasks": updated_tasks,  # type: ignore
                "failed_task_ids": [task_id]
            }
    
    
    def _aggregate_results(self, state: AgentState) -> AgentState:
        """Aggregate results from completed task."""
        task_id = state['active_task_id']
        
        # TRACKING: Log state on entry
        entry_completed_list = state.get('completed_task_ids', [])
        logger.warning(f"🔍 [TASK_ID_TRACKER] AGGREGATE ENTRY | task_id='{task_id}' | completed_task_ids size: {len(entry_completed_list)} | unique: {len(set(entry_completed_list))}")
        
        logger.info("=" * 80)
        logger.info(f"[AGGREGATE] Processing task {task_id}")
        logger.info("=" * 80)
        
        # Get task details for logging
        tasks = state.get('tasks', [])
        current_task = next((t for t in tasks if t.get('id') == task_id), None)
        if current_task:
            logger.info(f"[AGGREGATE] Task description: {current_task.get('description', 'N/A')[:80]}...")
            logger.info(f"[AGGREGATE] Task status: {current_task.get('status')}")
        
        # Get current count for logging (DETECT DUPLICATION BUG)
        current_completed_list = state.get('completed_task_ids', [])
        current_completed_unique = len(set(current_completed_list))
        current_completed_total = len(current_completed_list)
        
        # Log unique vs total to detect duplication
        logger.info(f"[AGGREGATE] Completed task IDs before adding: {current_completed_unique} unique, {current_completed_total} total")
        if current_completed_total > current_completed_unique:
            logger.warning(f"[AGGREGATE] ⚠️  DUPLICATION DETECTED: {current_completed_total - current_completed_unique} duplicate IDs in completed_task_ids list!")
        
        # FIX: Properly handle task completion tracking
        # - BROKEN_DOWN tasks are NOT added to completed_task_ids (they're intermediate states)
        # - FAILED tasks are added to failed_task_ids
        # - COMPLETED tasks are added to completed_task_ids
        task_status = current_task.get('status') if current_task else None
        already_completed = task_id in current_completed_list
        already_failed = task_id in state.get('failed_task_ids', [])
        
        if already_completed:
            logger.info(f"[AGGREGATE] Task {task_id} already in completed list - skipping duplicate addition")
        elif already_failed:
            logger.info(f"[AGGREGATE] Task {task_id} already in failed list - skipping")
        elif task_status == TaskStatus.BROKEN_DOWN:
            logger.info(f"[AGGREGATE] Task {task_id} is BROKEN_DOWN - not counting as complete (parent completes when all subtasks finish)")
        else:
            logger.info(f"[AGGREGATE] Adding task {task_id} to completed/failed list based on status: {task_status}")
        
        logger.info(f"[AGGREGATE] Next: routing to synthesis/continue/complete check")
        logger.info("=" * 80)
        
        # NOTE: completed_task_ids and failed_task_ids use operator.add annotation
        # LangGraph will automatically concatenate with existing lists
        # Only add if not already present and not a BROKEN_DOWN task
        
        # Handle failed tasks - route to human review for resolution
        if task_status == TaskStatus.FAILED and not already_failed:
            # Log the failure details for debugging
            error_msg = 'No error message provided'
            if current_task:
                error_msg = current_task.get('error', 'No error message provided')
                result = current_task.get('result', {})
                
                logger.error("=" * 80)
                logger.error(f"⚠️  TASK FAILED: {task_id}")
                logger.error("=" * 80)
                logger.error(f"Description: {current_task.get('description', 'N/A')[:150]}")
                logger.error(f"Error: {error_msg}")
                if isinstance(result, dict):
                    logger.error(f"Result Success: {result.get('success', 'N/A')}")
                    logger.error(f"Result Error: {result.get('error', 'N/A')}")
                logger.error("=" * 80)
            
            # Route to human review for failed tasks
            logger.warning(f"🔍 [TASK_ID_TRACKER] AGGREGATE RETURN | FAILED task_id='{task_id}' requires human review | Location: _aggregate_results:1788")
            logger.warning(f"[AGGREGATE] Routing FAILED task {task_id} to human review for resolution")
            return {
                **state,
                "requires_human_review": True,
                "human_feedback": f"Task {task_id} failed: {str(error_msg)[:200]}"
            }
        
        # Handle successful task completion (not BROKEN_DOWN, not FAILED)
        if not already_completed and not already_failed and task_status != TaskStatus.BROKEN_DOWN and task_status != TaskStatus.FAILED:
            # TRACKING: Log task ID addition for debugging
            logger.warning(f"🔍 [TASK_ID_TRACKER] AGGREGATE RETURN | Adding task_id='{task_id}' to completed_task_ids | Status: {task_status} | Location: _aggregate_results:1798")
            return_dict = {
                **state,
                "completed_task_ids": [task_id]  # Only the new task ID - NOT the full list
            }
            logger.warning(f"🔍 [TASK_ID_TRACKER] AGGREGATE RETURN | Returning dict with completed_task_ids={return_dict.get('completed_task_ids', [])} | LangGraph will use operator.add to concatenate")
            return return_dict  # type: ignore
        else:
            # Don't add duplicate or BROKEN_DOWN tasks
            logger.warning(f"🔍 [TASK_ID_TRACKER] AGGREGATE RETURN | SKIPPING task_id='{task_id}' | Already completed: {already_completed} | Already failed: {already_failed} | Is BROKEN_DOWN: {task_status == TaskStatus.BROKEN_DOWN} | Location: _aggregate_results:1807")
            return {
                **state
            }
    
    
    def _synthesize_research(self, state: AgentState) -> AgentState:
        """
        Synthesize multi-level research findings by analyzing entire blackboard.
        
        This node:
        1. Analyzes all blackboard entries across hierarchy levels
        2. Compares WebSearch findings against PDF/Excel data
        3. Identifies and flags numerical contradictions
        4. Synthesizes into a professional research brief
        5. Routes to human review if conflicts found
        
        Triggers only after all tasks in plan are complete.
        """
        logger.info("="*70)
        logger.info("SYNTHESIZING MULTI-LEVEL RESEARCH FINDINGS")
        logger.info("="*70)
        
        try:
            objective = state['objective']
            blackboard = state.get('blackboard', [])
            metadata = state.get('metadata', {})
            input_context = state.get('input_context')
            
            logger.info(f"Analyzing {len(blackboard)} blackboard entries from all agents and levels")
            
            # If no blackboard entries, nothing to synthesize
            if not blackboard:
                logger.info("No findings to synthesize")
                return state
            
            # Build synthesis prompt (with input context)
            synthesis_prompt = PromptBuilder.build_synthesis_prompt(
                objective=objective,
                blackboard_entries=blackboard,
                objective_context=metadata,
                input_context=input_context
            )
            
            # Call LLM for synthesis
            logger.info("Calling LLM to synthesize findings...")
            response = self._rate_limited_invoke([
                SystemMessage(content="You are a research synthesis expert. Analyze blackboard findings and provide comprehensive research synthesis. Respond only with valid JSON."),
                HumanMessage(content=synthesis_prompt)
            ])
            
            # Extract content
            response_content = response.content if hasattr(response, 'content') else str(response)
            if isinstance(response_content, list):
                response_content = "".join(str(item) for item in response_content)
            response_content = str(response_content)
            
            # Parse synthesis result
            synthesis_result = self._parse_json_response(response_content)
            
            logger.info("Synthesis complete")
            logger.info(f"Found {len(synthesis_result.get('contradictions', []))} contradictions")
            logger.info(f"Confidence level: {synthesis_result.get('confidence_level', 'UNKNOWN')}")
            
            # Extract contradictions
            contradictions = synthesis_result.get('contradictions', [])
            has_critical = any(c.get('severity') == 'CRITICAL' for c in contradictions)
            
            # Create synthesis blackboard entry
            synthesis_entry: BlackboardEntry = {
                "entry_type": "synthesis_result",
                "source_agent": "synthesis_node",
                "source_task_id": "synthesis",
                "content": {
                    "objective": objective,
                    "synthesis": synthesis_result.get('synthesis', ''),
                    "executive_summary": synthesis_result.get('executive_summary', ''),
                    "key_findings": synthesis_result.get('key_findings', []),
                    "contradictions": contradictions,
                    "confidence_level": synthesis_result.get('confidence_level', 'LOW'),
                    "data_sources": synthesis_result.get('data_sources', {})
                },
                "timestamp": datetime.now().isoformat(),
                "relevant_to": ["synthesis"],
                "depth_level": 0,
                "file_pointers": {},  # type: ignore
                "chain_next_agents": []  # type: ignore
            }
            
            # Log contradictions
            if contradictions:
                logger.warning(f"[SYNTHESIS] Found {len(contradictions)} data contradictions:")
                for contradiction in contradictions:
                    logger.warning(f"  • {contradiction.get('metric', 'unknown')}: {contradiction.get('severity')} - {contradiction.get('explanation', 'N/A')[:100]}")
            
            # Determine if human review needed
            requires_review = synthesis_result.get('requires_human_review', False)
            review_reason = synthesis_result.get('human_review_reason')
            
            if requires_review or has_critical:
                logger.warning(f"[SYNTHESIS] Flagging for human review")
                if review_reason:
                    logger.warning(f"Reason: {review_reason}")
                
                return {
                    **state,
                    "blackboard": [synthesis_entry],  # Only new entry - LangGraph will concatenate
                    "requires_human_review": True,
                    "human_feedback": f"Conflict Warning: {review_reason}" if review_reason else "Conflict Warning: Research synthesis found data contradictions requiring review"
                }
            else:
                logger.info(f"[SYNTHESIS] Research synthesis complete - no conflicts requiring review")
                
                return {
                    **state,
                    "blackboard": [synthesis_entry],  # Only new entry - LangGraph will concatenate
                    "requires_human_review": False,
                    "human_feedback": ""
                }
        
        except Exception as e:
            logger.error(f"[SYNTHESIS] Error during synthesis: {str(e)}")
            logger.log_exception("Synthesis exception:", e)
            
            return {
                **state,
                "requires_human_review": True,
                "human_feedback": f"Synthesis Error: {str(e)}"
            }
    
    
    def _agentic_debate(self, state: AgentState) -> AgentState:
        """
        Agentic Debate Node: Consensus-based validation of conflicting research findings.
        
        This node implements a robust debate mechanism where two distinct LLM personas
        systematically evaluate contradictory evidence to reach consensus:
        
        1. Fact-Checker (conservative persona): Questions assumptions, requires strong evidence
        2. Lead Researcher (inferential persona): Considers context, makes reasoned inferences
        
        Process:
        1. Extract contradictions from synthesis result
        2. Spawn both personas with the conflicting data
        3. Have them debate the validity of each contradiction
        4. Record the debate arguments and positions
        5. Determine consensus (which source is more reliable)
        6. Update final report based on consensus
        
        Only triggers if contradiction_score > 0.7 from synthesis node.
        
        Args:
            state: Current agent state with synthesis results
            
        Returns:
            Updated state with debate outcomes recorded in blackboard and history
        """
        logger.info("="*70)
        logger.info("AGENTIC DEBATE NODE: CONSENSUS-BASED CONFLICT RESOLUTION")
        logger.info("="*70)
        
        try:
            # Extract synthesis result from blackboard
            blackboard = state.get('blackboard', [])
            synthesis_entry = next(
                (e for e in reversed(blackboard) if e.get('source_agent') == 'synthesis_node'),
                None
            )
            
            if not synthesis_entry:
                logger.info("[DEBATE] No synthesis entry found - skipping debate")
                return state
            
            synthesis_content = synthesis_entry.get('content', {})
            contradictions = synthesis_content.get('contradictions', [])
            
            if not contradictions:
                logger.info("[DEBATE] No contradictions found - skipping debate")
                return state
            
            # Calculate contradiction score (0-1 scale based on severity and count)
            severity_weights = {'CRITICAL': 0.4, 'HIGH': 0.3, 'MEDIUM': 0.2, 'LOW': 0.1}
            total_score = sum(severity_weights.get(c.get('severity', 'LOW'), 0.1) for c in contradictions)
            contradiction_score = min(total_score, 1.0)  # Cap at 1.0
            
            logger.info(f"[DEBATE] Contradiction score: {contradiction_score:.2f} (trigger threshold: 0.7)")
            
            # Only proceed if contradiction score > 0.7
            if contradiction_score <= 0.7:
                logger.info(f"[DEBATE] Score {contradiction_score:.2f} below threshold - no debate needed")
                return state
            
            logger.warning(f"[DEBATE] Contradiction score {contradiction_score:.2f} exceeds threshold - initiating debate")
            
            # Get objective and findings
            objective = state['objective']
            key_findings = synthesis_content.get('key_findings', [])
            
            # Create debate prompt for both personas
            debate_data = {
                "objective": objective,
                "key_findings": key_findings,
                "contradictions": contradictions,
                "original_blackboard": blackboard
            }
            
            import json
            debate_data_str = json.dumps(debate_data, indent=2)
            
            # PERSONA 1: Fact-Checker (Conservative)
            logger.info("[DEBATE] Spawning Fact-Checker persona (conservative approach)...")
            fact_checker_prompt = f"""You are a FACT-CHECKER - a conservative, evidence-focused analyst.

Your role: Question assumptions, demand strong evidence, prioritize data reliability.

CONFLICTING FINDINGS TO EVALUATE:
{debate_data_str}

Your task:
1. Review each contradiction listed above
2. For each contradiction, assess:
   - Is the conflicting data reliable? What's the evidence quality?
   - Could different methodologies explain the difference?
   - Which source is more likely to be accurate?
3. State your position on each conflict (which data should we trust?)
4. Identify any sources that seem unreliable
5. Provide your overall assessment

Format your response as:
FACT-CHECKER POSITION:
[Your detailed analysis and conclusions about which data sources are reliable]

CONTRADICTIONS ASSESSMENT:
[For each contradiction: metric, your judgment on reliability, recommended source]

CONFIDENCE LEVEL: [HIGH/MEDIUM/LOW]
REASONING: [Why you have this confidence level]"""
            
            fact_checker_response = self._rate_limited_invoke([
                SystemMessage(content="You are a rigorous fact-checker. Demand evidence. Question assumptions. Respond in clear, professional language."),
                HumanMessage(content=fact_checker_prompt)
            ])
            
            fact_checker_analysis = fact_checker_response.content if hasattr(fact_checker_response, 'content') else str(fact_checker_response)
            fact_checker_analysis = str(fact_checker_analysis)
            logger.info("[DEBATE] Fact-Checker analysis complete")
            
            # PERSONA 2: Lead Researcher (Inferential)
            logger.info("[DEBATE] Spawning Lead Researcher persona (inferential approach)...")
            lead_researcher_prompt = f"""You are a LEAD RESEARCHER - an expert who considers context and makes informed inferences.

Your role: Weigh evidence in context, consider methodological differences, make reasoned judgments.

CONFLICTING FINDINGS TO EVALUATE:
{debate_data_str}

Your task:
1. Review each contradiction listed above
2. For each contradiction, assess:
   - What contextual factors might explain the differences?
   - Are the data sources measuring the same thing?
   - What's the most likely explanation for the conflict?
3. State your position on each conflict (what does the evidence suggest?)
4. Identify which sources provide the most reliable overall picture
5. Provide your overall assessment

Format your response as:
LEAD RESEARCHER POSITION:
[Your detailed analysis and conclusions about what the evidence actually shows]

CONTRADICTIONS ASSESSMENT:
[For each contradiction: metric, your contextual interpretation, recommended approach]

CONFIDENCE LEVEL: [HIGH/MEDIUM/LOW]
REASONING: [Why you have this confidence level]"""
            
            lead_researcher_response = self._rate_limited_invoke([
                SystemMessage(content="You are a lead researcher with deep domain expertise. Consider context. Make informed judgments. Respond in clear, professional language."),
                HumanMessage(content=lead_researcher_prompt)
            ])
            
            lead_researcher_analysis = lead_researcher_response.content if hasattr(lead_researcher_response, 'content') else str(lead_researcher_response)
            lead_researcher_analysis = str(lead_researcher_analysis)
            logger.info("[DEBATE] Lead Researcher analysis complete")
            
            # CONSENSUS DETERMINATION
            logger.info("[DEBATE] Determining consensus between personas...")
            consensus_prompt = f"""You are a neutral arbiter synthesizing two expert analyses.

FACT-CHECKER ANALYSIS:
{fact_checker_analysis}

LEAD RESEARCHER ANALYSIS:
{lead_researcher_analysis}

ORIGINAL CONTRADICTIONS:
{json.dumps(contradictions, indent=2)}

Your task:
1. Compare the two analyses
2. Where they agree, note strong consensus
3. Where they disagree, identify the core disagreement
4. For each contradiction, determine what the consensus position should be
5. Provide a final verdict on data reliability and recommended findings

Format your response as JSON:
{{
  "consensus_summary": "Overall agreement between personas",
  "agreements": ["point of strong consensus 1", "point of strong consensus 2"],
  "disagreements": ["point of disagreement 1 and why", "point of disagreement 2 and why"],
  "contradiction_resolutions": [
    {{
      "metric": "the contradicted metric",
      "consensus_position": "Which source should we trust and why",
      "confidence": "HIGH/MEDIUM/LOW"
    }}
  ],
  "final_recommendation": "What the evidence ultimately shows",
  "human_review_still_needed": true or false,
  "reason_for_human_review": "if still needed"
}}"""
            
            consensus_response = self._rate_limited_invoke([
                SystemMessage(content="You are a neutral arbiter. Synthesize both perspectives fairly. Respond only with valid JSON."),
                HumanMessage(content=consensus_prompt)
            ])
            
            consensus_content = consensus_response.content if hasattr(consensus_response, 'content') else str(consensus_response)
            consensus_content = str(consensus_content)
            
            # Parse consensus result
            consensus_result = self._parse_json_response(consensus_content)
            
            logger.info("[DEBATE] Consensus determination complete")
            logger.info(f"[DEBATE] Consensus agreements: {len(consensus_result.get('agreements', []))}")
            logger.info(f"[DEBATE] Consensus disagreements: {len(consensus_result.get('disagreements', []))}")
            
            # Create debate outcome blackboard entry
            debate_entry: BlackboardEntry = {
                "entry_type": "debate_outcome",
                "source_agent": "agentic_debate_node",
                "source_task_id": "agentic_debate",
                "content": {
                    "contradiction_score": contradiction_score,
                    "fact_checker_position": fact_checker_analysis,
                    "lead_researcher_position": lead_researcher_analysis,
                    "consensus_summary": consensus_result.get('consensus_summary', ''),
                    "agreements": consensus_result.get('agreements', []),
                    "disagreements": consensus_result.get('disagreements', []),
                    "contradiction_resolutions": consensus_result.get('contradiction_resolutions', []),
                    "final_recommendation": consensus_result.get('final_recommendation', ''),
                    "human_review_still_needed": consensus_result.get('human_review_still_needed', False)
                },
                "timestamp": datetime.now().isoformat(),
                "relevant_to": ["contradiction_resolution"],
                "depth_level": 0,
                "file_pointers": {},  # type: ignore
                "chain_next_agents": []  # type: ignore
            }
            
            # Log debate outcomes
            logger.info("[DEBATE] DEBATE OUTCOMES:")
            if consensus_result.get('agreements'):
                logger.info("  Strong Consensus Points:")
                for agreement in consensus_result.get('agreements', []):
                    logger.info(f"    ✓ {agreement}")
            
            if consensus_result.get('disagreements'):
                logger.info("  Points of Disagreement:")
                for disagreement in consensus_result.get('disagreements', []):
                    logger.info(f"    ✗ {disagreement}")
            
            logger.info(f"  Final Recommendation: {consensus_result.get('final_recommendation', 'N/A')[:150]}")
            
            # Update state with debate results
            requires_review = consensus_result.get('human_review_still_needed', False)
            
            return {
                **state,
                "blackboard": [debate_entry],  # Only new entry - LangGraph will concatenate
                "requires_human_review": requires_review,
                "human_feedback": f"Debate Resolution: {consensus_result.get('reason_for_human_review', 'Consensus reached')}" if requires_review else f"Consensus: {consensus_result.get('final_recommendation', 'Research validated')[:200]}"
            }
        
        except Exception as e:
            logger.error(f"[DEBATE] Error during agentic debate: {str(e)}")
            logger.log_exception("Debate exception:", e)
            
            return {
                **state,
                "requires_human_review": True,
                "human_feedback": f"Debate Error: {str(e)}"
            }
    
    
    def _auto_synthesis(self, state: AgentState) -> AgentState:
        """
        Auto-Synthesis Observer Node: Event-driven reactive analysis.
        
        This node implements the Observer pattern, automatically triggering when
        specific state changes occur (OCR results or web search findings).
        
        Purpose:
        - Provides immediate context-aware analysis when new data arrives
        - Creates preliminary insights before full synthesis
        - Enables reactive workflows where data triggers analysis
        - Reduces latency between data collection and insight generation
        
        Triggered by:
        - state['last_updated_key'] == 'ocr_results': OCR agent extracted text/tables
        - state['last_updated_key'] == 'web_findings': WebSearch agent found data
        
        Workflow:
        1. Detect which type of data triggered this node
        2. Extract relevant blackboard entries
        3. Perform focused analysis on the new data
        4. Post preliminary insights to blackboard
        5. Route back to aggregate_results to continue normal flow
        
        Args:
            state: Current agent state with last_updated_key set
            
        Returns:
            Updated state with auto-synthesis entry in blackboard
        """
        logger.info("="*70)
        logger.info("🔔 AUTO-SYNTHESIS OBSERVER: Event-Driven Analysis Triggered")
        logger.info("="*70)
        
        try:
            trigger_key = state.get('last_updated_key', '')
            objective = state.get('objective', '')
            blackboard = state.get('blackboard', [])
            
            logger.info(f"[AUTO-SYNTHESIS] Trigger: {trigger_key}")
            logger.info(f"[AUTO-SYNTHESIS] Blackboard entries: {len(blackboard)}")
            
            # Determine trigger type and extract relevant data
            if trigger_key == 'ocr_results':
                logger.info("[AUTO-SYNTHESIS] 🖼️ OCR Results detected - analyzing extracted content")
                
                # Find most recent OCR entry
                ocr_entries = [e for e in reversed(blackboard) if e.get('source_agent') == 'ocr_agent']
                if not ocr_entries:
                    logger.warning("[AUTO-SYNTHESIS] No OCR entries found despite trigger")
                    return state
                
                latest_ocr = ocr_entries[0]
                ocr_content = latest_ocr.get('content', {})
                
                analysis_prompt = f"""You are analyzing freshly extracted OCR data in real-time.

OBJECTIVE: {objective}

OCR EXTRACTION RESULTS:
- Text Extracted: {len(ocr_content.get('text_extracted', ''))} characters
- Success: {ocr_content.get('success')}
- Findings: {json.dumps(ocr_content.get('findings', {}), indent=2)}

Your task:
1. Identify key information extracted from the image/document
2. Assess relevance to the objective
3. Flag any important tables, numbers, or structured data
4. Note any quality issues or extraction limitations
5. Suggest immediate next steps (e.g., "extracted table should be analyzed by Excel agent")

Respond with JSON:
{{
  "summary": "Brief summary of what was extracted",
  "key_insights": ["insight 1", "insight 2"],
  "data_quality": "HIGH/MEDIUM/LOW with explanation",
  "structured_data_found": true or false,
  "suggested_next_actions": ["action 1", "action 2"],
  "relevance_to_objective": "How this data helps achieve the objective"
}}"""
            
            elif trigger_key == 'web_findings':
                logger.info("[AUTO-SYNTHESIS] 🌐 Web Search Results detected - analyzing findings")
                
                # Find most recent web search entry
                web_entries = [e for e in reversed(blackboard) if e.get('source_agent') == 'web_search_agent']
                if not web_entries:
                    logger.warning("[AUTO-SYNTHESIS] No web search entries found despite trigger")
                    return state
                
                latest_web = web_entries[0]
                web_content = latest_web.get('content', {})
                
                analysis_prompt = f"""You are analyzing freshly collected web search data in real-time.

OBJECTIVE: {objective}

WEB SEARCH RESULTS:
- Query: {web_content.get('query', '')}
- Results Count: {web_content.get('results_count', 0)}
- Pages Visited: {web_content.get('pages_visited', 0)}
- Summary: {web_content.get('summary', '')}
- Top Findings: {json.dumps(web_content.get('findings', [])[:5], indent=2)}

Your task:
1. Assess the quality and relevance of search results
2. Identify key data points, metrics, or facts found
3. Note any gaps or areas needing deeper investigation
4. Compare against objective requirements
5. Suggest follow-up searches or data extraction needs

Respond with JSON:
{{
  "summary": "Brief summary of search findings",
  "key_data_points": ["data point 1", "data point 2"],
  "credibility_assessment": "Assessment of source quality",
  "coverage_gaps": ["gap 1", "gap 2"],
  "suggested_next_actions": ["action 1", "action 2"],
  "relevance_to_objective": "How these findings help achieve the objective"
}}"""
            
            else:
                logger.warning(f"[AUTO-SYNTHESIS] Unknown trigger: {trigger_key}")
                return state
            
            # Call LLM for reactive analysis
            logger.info("[AUTO-SYNTHESIS] Calling LLM for event-driven analysis...")
            response = self._rate_limited_invoke([
                SystemMessage(content="You are a reactive analysis system that provides immediate insights when new data arrives. Respond only with valid JSON."),
                HumanMessage(content=analysis_prompt)
            ])
            
            # Extract and parse response
            response_content = response.content if hasattr(response, 'content') else str(response)
            if isinstance(response_content, list):
                response_content = "".join(str(item) for item in response_content)
            response_content = str(response_content)
            
            analysis_result = self._parse_json_response(response_content)
            
            logger.info("[AUTO-SYNTHESIS] ✓ Reactive analysis complete")
            logger.info(f"[AUTO-SYNTHESIS] Key insights: {len(analysis_result.get('key_insights', analysis_result.get('key_data_points', [])))}")
            
            # Create auto-synthesis blackboard entry
            auto_synthesis_entry: BlackboardEntry = {
                "entry_type": "auto_synthesis_result",
                "source_agent": "auto_synthesis_observer",
                "source_task_id": "auto_synthesis",
                "content": {
                    "trigger": trigger_key,
                    "objective": objective,
                    "analysis": analysis_result,
                    "timestamp": datetime.now().isoformat()
                },
                "timestamp": datetime.now().isoformat(),
                "relevant_to": ["reactive_analysis"],
                "depth_level": 0,
                "file_pointers": {},  # type: ignore
                "chain_next_agents": []  # type: ignore
            }
            
            logger.info("[AUTO-SYNTHESIS] Posting reactive insights to blackboard")
            logger.info(f"[AUTO-SYNTHESIS] Summary: {analysis_result.get('summary', 'N/A')[:150]}")
            logger.info("="*70)
            
            return {
                **state,
                "blackboard": [auto_synthesis_entry],  # Only new entry - LangGraph will concatenate
                "last_updated_key": None  # Clear trigger to prevent re-firing
            }
        
        except Exception as e:
            logger.error(f"[AUTO-SYNTHESIS] Error during auto-synthesis: {str(e)}")
            logger.log_exception("Auto-synthesis exception:", e)
            
            # Don't fail the workflow - just log and continue
            return {
                **state,
                "last_updated_key": None  # Clear trigger
            }
    
    
    def _handle_error(self, state: AgentState) -> AgentState:
        """
        Handle task errors with retry logic and LLM-based error analysis.
        
        Uses ProblemSolverAgent to analyze errors and potentially suggest solutions
        before routing to human review.
        """
        task_id = state['active_task_id']
        
        if not task_id:
            logger.warning("[ERROR] Handling error with no active task ID")
            # Just clear active_task_id and increment iteration
            return {
                **state,
                "active_task_id": "",
                "iteration_count": state['iteration_count'] + 1
            }
        
        logger.warning(f"[ERROR] Handling error for task {task_id}")
        
        # Get the failed task for analysis
        current_task = next((t for t in state['tasks'] if t['id'] == task_id), None)
        
        # Use ProblemSolverAgent to analyze the error and generate solutions
        error_analysis = None
        suggested_solutions = None
        
        if current_task:
            try:
                error_msg = current_task.get('error', 'Task failed during processing')
                task_description = current_task.get('description', '')
                task_result = current_task.get('result')
                task_context = {
                    'task_id': task_id,
                    'description': task_description,
                    'result': task_result,
                    'metadata': state.get('metadata', {}),
                    'action': task_result.get('action') if isinstance(task_result, dict) else None
                }
                
                # Determine agent type from the task action
                agent_type = None
                if isinstance(task_result, dict):
                    action = task_result.get('action', '')
                    if 'excel' in action.lower():
                        agent_type = 'excel'
                    elif 'pdf' in action.lower():
                        agent_type = 'pdf'
                    elif 'web' in action.lower() or 'search' in action.lower():
                        agent_type = 'web_search'
                    elif 'ocr' in action.lower():
                        agent_type = 'ocr'
                    elif 'code' in action.lower():
                        agent_type = 'code'
                
                logger.info(f"[ERROR] Analyzing error with ProblemSolverAgent for task {task_id}")
                
                # Diagnose the error using standardized interface
                diagnose_request: AgentExecutionRequest = {
                    "task_id": f"diagnose_{task_id}",
                    "task_description": "Diagnose error for task",
                    "task_type": "atomic",
                    "operation": "diagnose_error",
                    "parameters": {
                        "error_message": str(error_msg),
                        "task_context": task_context,
                        "agent_type": agent_type
                    },
                    "input_data": {},
                    "temp_folder": str(self.config.folders.temp_path),
                    "output_folder": str(self.config.folders.output_path),
                    "cache_enabled": False,
                    "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                    "relevant_entries": [],
                    "max_retries": 1
                }
                
                diagnose_response = cast(AgentExecutionResponse, self.problem_solver_agent.execute_task(diagnose_request))
                error_analysis = diagnose_response.get('result', {}) if diagnose_response['success'] else None
                
                if error_analysis:
                    logger.info(f"[ERROR] Error diagnosis: category={error_analysis.get('error_category')}")
                    logger.info(f"[ERROR] Solution prompt: {error_analysis.get('solution_prompt', 'Unknown')[:100]}...")
                
                # Generate solution using standardized interface
                solution_request: AgentExecutionRequest = {
                    "task_id": f"solution_{task_id}",
                    "task_description": "Get solution for task error",
                    "task_type": "atomic",
                    "operation": "get_solution",
                    "parameters": {
                        "error_message": str(error_msg),
                        "task_context": task_context,
                        "agent_type": agent_type
                    },
                    "input_data": {},
                    "temp_folder": str(self.config.folders.temp_path),
                    "output_folder": str(self.config.folders.output_path),
                    "cache_enabled": False,
                    "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                    "relevant_entries": [],
                    "max_retries": 1
                }
                
                solution_response = cast(AgentExecutionResponse, self.problem_solver_agent.execute_task(solution_request))
                suggested_solutions = solution_response.get('result', {}) if solution_response['success'] else None
                
                if suggested_solutions:
                    logger.info(f"[ERROR] Solution type: {suggested_solutions.get('solution_type')}")
                    logger.info(f"[ERROR] Suggested action: {suggested_solutions.get('suggested_action', 'N/A')[:80]}...")
                        
            except Exception as e:
                logger.warning(f"[ERROR] ProblemSolverAgent analysis failed: {str(e)}")
                logger.log_exception("ProblemSolverAgent analysis exception:", e)
        
        # Mark task as failed in the tasks list with enhanced error info
        updated_tasks = []
        for task in state['tasks']:
            if task['id'] == task_id:
                task['status'] = TaskStatus.FAILED
                task['result'] = {
                    'error': task.get('error', 'Task failed during processing'),
                    'error_analysis': error_analysis,
                    'suggested_solutions': suggested_solutions
                }
                updated_tasks.append(task)
            else:
                updated_tasks.append(task)
        
        logger.warning(f"[ERROR] Task {task_id} marked as failed with error analysis")
        
        # NOTE: failed_task_ids uses operator.add annotation, so only return NEW items to add
        return {
            **state,
            "tasks": updated_tasks,
            "failed_task_ids": [task_id],  # Only the new task ID - LangGraph will concatenate
            "active_task_id": "",  # Clear active task
            "iteration_count": state['iteration_count'] + 1
        }
    
    
    def _request_human_review(self, state: AgentState) -> AgentState:
        """
        Request human review for failed tasks or complex decisions.
        
        Uses ProblemSolverAgent for:
        - Displaying error analysis and suggested solutions
        - Interpreting natural language human input into structured format
        
        For failed tasks, provides three options:
        1. Restart task with updated user input/context
        2. Provide human input as task output (for dependent tasks)
        3. Ignore/skip the failed task
        """
        logger.warning("!"*60)
        logger.warning("HUMAN REVIEW REQUIRED")
        logger.warning("!"*60)
        
        task_id = state.get('active_task_id')
        current_task = None
        is_failed_task = False
        error_analysis = None
        suggested_solutions = None
        
        if task_id:
            task = next((t for t in state.get('tasks', []) if t['id'] == task_id), None)
            if task:
                current_task = task
                is_failed_task = task.get('status') == TaskStatus.FAILED
                logger.warning(f"Active task: {task_id}")
                logger.warning(f"Task description: {task.get('description', 'N/A')[:100]}...")
                logger.warning(f"Task status: {task.get('status')}")
                if is_failed_task:
                    logger.warning(f"Task error: {task.get('error', 'Unknown error')}")
                    # Extract error analysis from task result if available
                    result = task.get('result', {})
                    if isinstance(result, dict):
                        error_analysis = result.get('error_analysis')
                        suggested_solutions = result.get('suggested_solutions')
                if task.get('result'):
                    logger.warning(f"Analysis result: {task.get('result')}")
        
        logger.warning("!"*60)
        
        # Interactive human review
        print()
        print("=" * 70)
        if is_failed_task:
            print("FAILED TASK - HUMAN REVIEW REQUIRED")
        else:
            print("HUMAN REVIEW REQUIRED")
        print("=" * 70)
        
        if current_task:
            print()
            print(f"Task ID: {task_id}")
            print(f"Task: {current_task.get('description', 'Unknown')}")
            
            if is_failed_task:
                error = current_task.get('error', 'Unknown error')
                print(f"\n❌ FAILURE REASON: {error}")
                result = current_task.get('result', {})
                if isinstance(result, dict) and result.get('error'):
                    print(f"   Details: {result.get('error')}")
                
                # Display AI error diagnosis if available
                if error_analysis:
                    print()
                    print("🤖 AI ERROR DIAGNOSIS:")
                    print(f"   Category: {error_analysis.get('error_category', 'Unknown')}")
                    print(f"   Solution Hint: {error_analysis.get('solution_prompt', 'Unknown')[:80]}...")
                
                # Display suggested solutions if available
                if suggested_solutions:
                    print()
                    print("💡 SUGGESTED SOLUTION:")
                    print(f"   Type: {suggested_solutions.get('solution_type', 'N/A')}")
                    print(f"   Action: {suggested_solutions.get('suggested_action', 'N/A')[:70]}...")
                    print(f"   Confidence: {suggested_solutions.get('confidence', 'N/A')}")
                    if suggested_solutions.get('requires_human_input'):
                        print(f"   Human Input Needed: {suggested_solutions.get('human_input_prompt', 'Please provide additional information')[:60]}...")
                    if suggested_solutions.get('alternative_approaches'):
                        print("   Alternatives:")
                        for i, alt in enumerate(suggested_solutions.get('alternative_approaches', [])[:2], 1):
                            print(f"     {i}. {alt[:60]}...")
            
            result = current_task.get('result')
            if result and not is_failed_task:
                if isinstance(result, dict):
                    print()
                    print("Analysis Result:")
                    for key, value in result.items():
                        if key not in ['subtasks', 'error_analysis', 'suggested_solutions']:
                            print(f"  {key}: {str(value)[:80]}")
                        elif key == 'subtasks':
                            if isinstance(value, list):
                                print(f"  {key}: {len(value)} subtasks created")
        
        print()
        if is_failed_task:
            print("Options for Failed Task:")
            print("  1. Restart task with updated context/input")
            print("  2. Provide manual output (use your input as task result)")
            print("  3. Ignore and continue (mark as skipped)")
            print("  4. Abort workflow")
        else:
            print("Options:")
            print("  1. Continue with execution")
            print("  2. Provide additional guidance")
            print("  3. Skip this task")
            print("  4. Abort workflow")
        print()
        
        updated_state: AgentState = dict(state)  # type: ignore
        
        while True:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                if is_failed_task:
                    # Option 1: Restart task with updated context
                    print()
                    print("Provide updated context/input for task retry:")
                    print("(Enter multiple lines, type 'DONE' on a new line when finished)")
                    print("(Your input will be interpreted by AI for optimal formatting)")
                    lines = []
                    while True:
                        line = input()
                        if line.strip().upper() == 'DONE':
                            break
                        lines.append(line)
                    
                    raw_input = '\n'.join(lines).strip()
                    
                    if raw_input:
                        logger.info(f"[HUMAN REVIEW] Processing human input for task {task_id}")
                        
                        # Use ProblemSolverAgent to interpret and format human input
                        try:
                            # Get agent type from task result
                            agent_type = None
                            action = ''
                            if current_task:
                                task_result = current_task.get('result')
                                if isinstance(task_result, dict):
                                    action = task_result.get('action', '')
                                    if 'excel' in action.lower():
                                        agent_type = 'excel'
                                    elif 'pdf' in action.lower():
                                        agent_type = 'pdf'
                                    elif 'web' in action.lower() or 'search' in action.lower():
                                        agent_type = 'web_search'
                                    elif 'ocr' in action.lower():
                                        agent_type = 'ocr'
                                    elif 'code' in action.lower():
                                        agent_type = 'code'
                            
                            # Interpret the human input using standardized interface
                            interpret_request: AgentExecutionRequest = {
                                "task_id": f"interpret_{task_id}",
                                "task_description": "Interpret human input",
                                "task_type": "atomic",
                                "operation": "interpret_human_input",
                                "parameters": {
                                    "human_input": raw_input,
                                    "target_format": agent_type or 'json',
                                    "task_context": {
                                        'task_id': task_id,
                                        'task_description': current_task.get('description') if current_task else '',
                                        'error': current_task.get('error') if current_task else None,
                                        'original_action': action if agent_type else None
                                    }
                                },
                                "input_data": {"human_input": raw_input},
                                "temp_folder": str(self.config.folders.temp_path),
                                "output_folder": str(self.config.folders.output_path),
                                "cache_enabled": True,
                                "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                                "relevant_entries": [],
                                "max_retries": 2
                            }
                            interpret_response = cast(AgentExecutionResponse, self.problem_solver_agent.execute_task(interpret_request))
                            interpreted = interpret_response.get('result', {}) if interpret_response['success'] else None
                            
                            if interpreted and interpreted.get('success'):
                                logger.info(f"[HUMAN REVIEW] Interpreted input: confidence={interpreted.get('confidence')}")
                                
                                # Format for the specific agent if applicable
                                if agent_type:
                                    task_result = current_task.get('result', {}) if current_task else {}
                                    operation = task_result.get('operation') if isinstance(task_result, dict) else None
                                    format_request: AgentExecutionRequest = {
                                        "task_id": f"format_{task_id}",
                                        "task_description": "Format data for agent",
                                        "task_type": "atomic",
                                        "operation": "format_for_agent",
                                        "parameters": {
                                            "data": interpreted.get('parsed_data', raw_input),
                                            "agent_type": agent_type,
                                            "operation": operation
                                        },
                                        "input_data": {"data": interpreted.get('parsed_data', raw_input)},
                                        "temp_folder": str(self.config.folders.temp_path),
                                        "output_folder": str(self.config.folders.output_path),
                                        "cache_enabled": True,
                                        "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                                        "relevant_entries": [],
                                        "max_retries": 2
                                    }
                                    format_response = cast(AgentExecutionResponse, self.problem_solver_agent.execute_task(format_request))
                                    formatted = format_response.get('result', {}) if format_response['success'] else interpreted.get('parsed_data', raw_input)
                                    updated_context = json.dumps(formatted) if isinstance(formatted, dict) else str(formatted)
                                    logger.info(f"[HUMAN REVIEW] Formatted for {agent_type} agent")
                                else:
                                    updated_context = interpreted.get('parsed_data', raw_input)
                                    if isinstance(updated_context, dict):
                                        updated_context = json.dumps(updated_context)
                                
                                print(f"🤖 AI interpreted your input (confidence: {interpreted.get('confidence', 'N/A')})")
                            else:
                                updated_context = raw_input
                                logger.info(f"[HUMAN REVIEW] Using raw input (interpretation unavailable)")
                                
                        except Exception as e:
                            logger.warning(f"[HUMAN REVIEW] Input interpretation failed: {str(e)}, using raw input")
                            updated_context = raw_input
                        
                        logger.info(f"[HUMAN REVIEW] Restarting task {task_id} with updated context")
                        
                        # Reset task to PENDING status with updated context
                        updated_tasks = []
                        for t in state['tasks']:
                            if t['id'] == task_id:
                                updated_task = {
                                    **t,
                                    "status": TaskStatus.PENDING,
                                    "error": None,
                                    "human_context": updated_context,
                                    "human_raw_input": raw_input,
                                    "retry_count": t.get('retry_count', 0) + 1,
                                    "updated_at": datetime.now().isoformat()
                                }
                                updated_tasks.append(updated_task)
                                logger.info(f"[HUMAN REVIEW] Task {task_id} reset to PENDING with new context (retry #{updated_task['retry_count']})")
                            else:
                                updated_tasks.append(t)
                        
                        # Update metadata with human context
                        updated_state['tasks'] = updated_tasks  # type: ignore
                        updated_state['metadata'] = {
                            **state.get('metadata', {}),
                            f'human_context_{task_id}': updated_context,
                            f'human_raw_input_{task_id}': raw_input
                        }
                        updated_state['requires_human_review'] = False
                        
                        print(f"✓ Task {task_id} will be retried with your context")
                        break
                    else:
                        print("No context provided. Please try again or choose another option.")
                        continue
                else:
                    # Normal flow: continue with execution
                    print()
                    print("Proceeding with execution...")
                    updated_state['requires_human_review'] = False
                    break
                
            elif choice == "2":
                if is_failed_task:
                    # Option 2: Provide manual output as task result
                    print()
                    print("Provide the output/result for this task:")
                    print("(This will be used as the task's result for dependent tasks)")
                    print("(Enter multiple lines, type 'DONE' on a new line when finished)")
                    print("(Your input will be interpreted by AI for optimal formatting)")
                    lines = []
                    while True:
                        line = input()
                        if line.strip().upper() == 'DONE':
                            break
                        lines.append(line)
                    
                    raw_output = '\n'.join(lines).strip()
                    
                    if raw_output:
                        logger.info(f"[HUMAN REVIEW] Processing manual output for task {task_id}")
                        
                        # Use ProblemSolverAgent to interpret and format the output
                        formatted_output = raw_output
                        try:
                            interpret_output_request: AgentExecutionRequest = {
                                "task_id": f"interpret_output_{task_id}",
                                "task_description": "Interpret human output",
                                "task_type": "atomic",
                                "operation": "interpret_human_input",
                                "parameters": {
                                    "human_input": raw_output,
                                    "target_format": 'json',
                                    "task_context": {
                                        'task_id': task_id,
                                        'task_description': current_task.get('description') if current_task else '',
                                        'purpose': 'task_output',
                                        'for_dependent_tasks': True
                                    }
                                },
                                "input_data": {"human_input": raw_output},
                                "temp_folder": str(self.config.folders.temp_path),
                                "output_folder": str(self.config.folders.output_path),
                                "cache_enabled": True,
                                "blackboard": cast(list[dict[str, Any]], state.get('blackboard', [])),
                                "relevant_entries": [],
                                "max_retries": 2
                            }
                            interpret_output_response = cast(AgentExecutionResponse, self.problem_solver_agent.execute_task(interpret_output_request))
                            interpreted = interpret_output_response.get('result', {}) if interpret_output_response['success'] else None
                            
                            if interpreted and interpreted.get('success'):
                                formatted_output = interpreted.get('parsed_data', raw_output)
                                if isinstance(formatted_output, dict):
                                    # Keep as dict for structured output
                                    logger.info(f"[HUMAN REVIEW] Output interpreted as structured data")
                                    print(f"🤖 AI interpreted your output (confidence: {interpreted.get('confidence', 'N/A')})")
                                else:
                                    formatted_output = str(formatted_output)
                        except Exception as e:
                            logger.warning(f"[HUMAN REVIEW] Output interpretation failed: {str(e)}, using raw output")
                            formatted_output = raw_output
                        
                        logger.info(f"[HUMAN REVIEW] Using manual output for task {task_id}")
                        
                        # Mark task as COMPLETED with human-provided output
                        updated_tasks = []
                        for t in state['tasks']:
                            if t['id'] == task_id:
                                updated_task = {
                                    **t,
                                    "status": TaskStatus.COMPLETED,
                                    "error": None,
                                    "result": {
                                        "success": True,
                                        "output": formatted_output,
                                        "raw_output": raw_output,
                                        "human_provided": True,
                                        "original_error": t.get('error', 'Unknown')
                                    },
                                    "updated_at": datetime.now().isoformat()
                                }
                                updated_tasks.append(updated_task)
                                logger.info(f"[HUMAN REVIEW] Task {task_id} marked as COMPLETED with human output")
                            else:
                                updated_tasks.append(t)
                        
                        # Add to completed tasks and update results
                        updated_state['tasks'] = updated_tasks  # type: ignore
                        updated_state['results'] = {
                            **state.get('results', {}),
                            task_id: {
                                "success": True,
                                "output": formatted_output,
                                "raw_output": raw_output,
                                "human_provided": True
                            }
                        }
                        updated_state['completed_task_ids'] = [task_id]
                        updated_state['requires_human_review'] = False
                        
                        print(f"✓ Task {task_id} marked as completed with your output")
                        print("  Dependent tasks can now use this result")
                        break
                    else:
                        print("No output provided. Please try again or choose another option.")
                        continue
                else:
                    # Normal flow: provide guidance
                    print()
                    guidance = input("Provide guidance (e.g., data sources, specific requirements): ").strip()
                    if guidance:
                        updated_state['metadata'] = {**state.get('metadata', {}), 'human_guidance': guidance}
                        updated_state['requires_human_review'] = False
                        print("Guidance recorded. Continuing...")
                        break
                    else:
                        print("No guidance provided. Please try again or choose another option.")
                        continue
                
            elif choice == "3":
                # Option 3: Ignore/skip the task
                print()
                if is_failed_task:
                    print("Ignoring failed task and continuing...")
                    logger.info(f"[HUMAN REVIEW] Ignoring failed task {task_id}")
                else:
                    print("Skipping this task...")
                    logger.info(f"[HUMAN REVIEW] Skipping task {task_id}")
                
                if task_id:
                    # Add to failed_task_ids to mark as handled
                    logger.warning(f"🔍 [TASK_ID_TRACKER] HUMAN_REVIEW adding task_id='{task_id}' to failed (user ignore) | Location: _request_human_review")
                    updated_state['failed_task_ids'] = [task_id]
                    updated_state['requires_human_review'] = False
                break
                
            elif choice == "4":
                # Option 4: Abort workflow
                print()
                print("Aborting execution...")
                logger.warning(f"[HUMAN REVIEW] User aborted workflow")
                # Mark all remaining tasks as completed to end workflow
                all_tasks = state.get('tasks', [])
                completed = set(state.get('completed_task_ids', []))
                tasks_to_add = []
                for task in all_tasks:
                    if task['id'] not in completed:
                        tasks_to_add.append(task['id'])
                logger.warning(f"🔍 [TASK_ID_TRACKER] HUMAN_REVIEW adding {len(tasks_to_add)} task_ids (abort all) | Tasks: {tasks_to_add} | Location: _request_human_review")
                updated_state['completed_task_ids'] = tasks_to_add
                updated_state['requires_human_review'] = False
                break
            
            else:
                print()
                print("Invalid choice. Please enter 1-4.")
        
        print()
        print("=" * 70)
        print()
        
        return updated_state
    
    
    # ========================================================================
    # ROUTING FUNCTIONS
    # ========================================================================
    
    def _route_with_chain_execution(self, state: AgentState) -> Literal["breakdown", "execute_task", "execute_pdf_task", "execute_excel_task", "execute_ocr_task", "execute_web_search_task", "execute_code_interpreter_task", "execute_data_extraction_task", "execute_document_task", "execute_problem_solver_task", "handle_error", "review"]:
        """
        Enhanced routing with chain execution support for cross-agent workflows.
        
        Routing logic with chain execution awareness:
        1. If task requires breakdown → breakdown
        2. If human review needed → human_review
        3. If specific agent action → route to execute_*_task
        4. Check for chain execution conditions:
           - PDF output → Can chain to OCR if images found
           - OCR output → Can chain to Excel if tables extracted
           - WebSearch output → Can chain to Excel if CSV created
        5. Otherwise error handling
        
        This enables seamless agent-to-agent handoffs with file pointers
        and blackboard state sharing, bypassing the select_task node
        for immediate chain execution when appropriate.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node to execute
        """
        task_id = state['active_task_id']
        logger.info(f"[ROUTE ENTRY] task_id={task_id}")
        
        # Record entry
        self.tracer.record_state_snapshot("_route_with_chain_execution", dict(state), phase="entry")
        
        # Handle case where no more tasks are pending
        # This happens after breakdown fails or all tasks are completed
        if not task_id:
            logger.info("[ROUTE] No active task - all pending tasks completed")
            logger.warning("[ROUTE] ⚠️  ANOMALY: routing to handle_error when no active task (should not happen)")
            # Don't route to handle_error; instead wait for next select_task
            # The workflow will handle completion check through aggregate_results
            self.tracer.record_routing_decision("_route_with_chain_execution", "", "handle_error", "No active task ID")
            return "handle_error"  # Will be caught by select_task loop
        
        task = next((t for t in state['tasks'] if t['id'] == task_id), None)
        
        if not task:
            logger.error(f"[ROUTE] Task {task_id} not found in tasks list")
            logger.error(f"[ROUTE]   Available task IDs: {[t['id'] for t in state['tasks']]}")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "handle_error", "Task not found in tasks list")
            return "handle_error"
        
        if task['status'] == TaskStatus.FAILED:
            logger.warning(f"[ROUTE] Task {task_id} already marked as FAILED, handling error")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "handle_error", "Task already failed")
            return "handle_error"
        
        analysis = task.get('result') if isinstance(task, dict) else task['result']
        if not isinstance(analysis, dict):
            logger.error(f"[ROUTE] Task {task_id} has invalid analysis format (not dict): {type(analysis)}")
            logger.error(f"[ROUTE]   Task result: {analysis}")
            analysis = {}
        
        action = analysis.get('action', 'handle_error')
        
        logger.info(f"[ROUTE] 🔥 DEBUG: Original action from analysis: '{action}'")
        logger.info(f"[ROUTE]   Task description: {task.get('description', '')[:100]}")
        logger.info(f"[ROUTE]   Task status: {task.get('status')}")
        
        # Normalize action names - handle variations from LLM responses
        # Map shorthand actions to their full execute_* equivalents
        action_mapping = {
            "web_search_task": "execute_web_search_task",
            "web_search": "execute_web_search_task",
            "search": "execute_web_search_task",
            "pdf_task": "execute_pdf_task",
            "pdf": "execute_pdf_task",
            "excel_task": "execute_excel_task",
            "excel": "execute_excel_task",
            "ocr_task": "execute_ocr_task",
            "ocr": "execute_ocr_task",
            "image_task": "execute_ocr_task",
            "code_interpreter_task": "execute_code_interpreter_task",
            "code_task": "execute_code_interpreter_task",
            "code": "execute_code_interpreter_task",
            "data_extraction_task": "execute_data_extraction_task",
            "data_extraction": "execute_data_extraction_task",
            "extract_data": "execute_data_extraction_task",
            "document_task": "execute_document_task",
            "document": "execute_document_task",
            "docx": "execute_document_task",
            "create_docx": "execute_document_task",
            "write_docx": "execute_document_task",
            "create_document": "execute_document_task",
            "write_document": "execute_document_task",
            "problem_solving": "execute_problem_solver_task",
            "problem_solver": "execute_problem_solver_task",
            "analysis": "execute_problem_solver_task",
            "analyze": "execute_problem_solver_task",
            "execute": "execute_task",
        }
        action = action_mapping.get(action, action)
        
        logger.info(f"[ROUTE] 🔥 DEBUG: Mapped action: '{action}'")
        logger.info(f"[ROUTE]   All available routes: breakdown, execute_task, execute_pdf_task, execute_excel_task, execute_ocr_task, execute_web_search_task, execute_code_interpreter_task, execute_data_extraction_task, execute_document_task, handle_error, review")
        
        # Priority 1: Human review (if required and not breakdown)
        if state.get('requires_human_review', False) and action != "breakdown":
            logger.info(f"[ROUTE] Requiring human review for task {task_id}")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "review", "Human review required")
            return "review"
        
        # Priority 2: Breakdown decomposition
        if action == "breakdown":
            logger.info(f"[ROUTE] Routing to breakdown for task {task_id}")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "breakdown", "Breakdown requested")
            return "breakdown"
        
        # Priority 3: Execute-specific agent tasks
        if action == "execute_task":
            logger.info(f"[ROUTE] Routing to generic execute for task {task_id}")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "execute_task", "Generic execution")
            return "execute_task"
        elif action == "execute_pdf_task":
            logger.info(f"[ROUTE] Routing to PDF task executor for {task_id}")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "execute_pdf_task", "PDF execution")
            return "execute_pdf_task"
        elif action == "execute_excel_task":
            logger.info(f"[ROUTE] Routing to Excel task executor for {task_id}")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "execute_excel_task", "Excel execution")
            return "execute_excel_task"
        elif action == "execute_ocr_task":
            logger.info(f"[ROUTE] Routing to OCR task executor for {task_id}")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "execute_ocr_task", "OCR execution")
            return "execute_ocr_task"
        elif action == "execute_web_search_task":
            logger.info(f"[ROUTE] Routing to WebSearch task executor for {task_id}")
            logger.warning(f"[ROUTE] 🔥 DEBUG: About to return 'execute_web_search_task' for task {task_id}")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "execute_web_search_task", "Web search execution")
            print(f">>> [ROUTE] Returning 'execute_web_search_task' for task {task_id}", flush=True)
            return "execute_web_search_task"
        elif action == "execute_code_interpreter_task":
            logger.info(f"[ROUTE] Routing to Code Interpreter task executor for {task_id}")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "execute_code_interpreter_task", "Code interpreter execution")
            return "execute_code_interpreter_task"
        elif action == "execute_data_extraction_task":
            logger.info(f"[ROUTE] Routing to Data Extraction task executor for {task_id}")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "execute_data_extraction_task", "Data extraction execution")
            return "execute_data_extraction_task"
        elif action == "execute_problem_solver_task":
            logger.info(f"[ROUTE] Routing to Problem Solver task executor for {task_id}")
            logger.info(f"[ROUTE]   Task: {task.get('description', '')[:100]}")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "execute_problem_solver_task", "Problem solver execution")
            return "execute_problem_solver_task"
        elif action == "execute_document_task":
            logger.info(f"[ROUTE] Routing to Document task executor for {task_id}")
            logger.info(f"[ROUTE]   🔥 DEBUG: Routing to execute_document_task")
            logger.info(f"[ROUTE]   Task: {task.get('description', '')[:100]}")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "execute_document_task", "Document execution")
            print(f">>> [ROUTE] Returning 'execute_document_task' for task {task_id}", flush=True)
            return "execute_document_task"
        else:
            logger.error(f"[ROUTE] ✗ UNKNOWN ACTION: Task {task_id}")
            logger.error(f"[ROUTE]   Action requested: '{action}'")
            logger.error(f"[ROUTE]   Analysis: {analysis}")
            available_agents = [
                'execute_pdf_task', 'execute_excel_task', 'execute_ocr_task',
                'execute_web_search_task', 'execute_code_interpreter_task',
                'execute_data_extraction_task', 'execute_problem_solver_task', 'execute_document_task'
            ]
            logger.error(f"[ROUTE]   Available actions: {available_agents}")
            logger.error(f"[ROUTE]   Routing to error handler for resolution")
            self.tracer.record_routing_decision("_route_with_chain_execution", task_id, "handle_error", f"Unknown action: {action}")
            return "handle_error"
    
    
    def _log_task_hierarchy_summary(self, state: AgentState) -> None:
        """
        Log a comprehensive summary of all tasks organized by hierarchy.
        
        This provides visibility into:
        - Task tree structure (parent-child relationships)
        - Status of each task (PENDING, COMPLETED, FAILED, BROKEN_DOWN)
        - Progress at each depth level
        - Which tasks remain to be processed
        
        Args:
            state: Current agent state
        """
        tasks = state.get('tasks', [])
        completed = set(state.get('completed_task_ids', []))
        failed = set(state.get('failed_task_ids', []))
        
        if not tasks:
            return
        
        # Organize tasks by depth
        tasks_by_depth: Dict[int, List[Task]] = {}
        for task in tasks:
            depth = task.get('depth', 0)
            if depth not in tasks_by_depth:
                tasks_by_depth[depth] = []
            tasks_by_depth[depth].append(task)
        
        # Log the hierarchy
        logger.info("=" * 80)
        logger.info("📊 [TASK HIERARCHY SUMMARY]")
        logger.info("=" * 80)
        logger.info(f"Total Tasks: {len(tasks)}")
        logger.info(f"Completed: {len(completed)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Pending: {len([t for t in tasks if t['id'] not in completed and t['id'] not in failed and t['status'] == TaskStatus.PENDING])}")
        logger.info("-" * 80)
        
        # Log each depth level
        for depth in sorted(tasks_by_depth.keys()):
            depth_tasks = tasks_by_depth[depth]
            logger.info(f"\n📍 DEPTH {depth} ({len(depth_tasks)} tasks)")
            logger.info("-" * 80)
            
            for task in sorted(depth_tasks, key=lambda t: t['id']):
                task_id = task['id']
                description = task.get('description', 'No description')[:60]
                status = task.get('status', TaskStatus.PENDING)
                
                # Determine status symbol
                if task_id in completed:
                    symbol = "✓"
                    status_str = "COMPLETED"
                elif task_id in failed:
                    symbol = "✗"
                    status_str = "FAILED"
                elif status == TaskStatus.BROKEN_DOWN:
                    symbol = "🔀"
                    status_str = "BROKEN_DOWN"
                elif status == TaskStatus.PENDING:
                    symbol = "⏳"
                    status_str = "PENDING"
                elif status == TaskStatus.EXECUTING:
                    symbol = "⚙️"
                    status_str = "EXECUTING"
                elif status == TaskStatus.ANALYZING:
                    symbol = "🔍"
                    status_str = "ANALYZING"
                else:
                    symbol = "❓"
                    status_str = str(status)
                
                # Indent based on depth
                indent = "  " * depth
                logger.info(f"{indent}{symbol} [{task_id}] {status_str}: {description}...")
        
        logger.info("=" * 80)
    
    
    def _log_final_execution_summary(self, state: AgentState) -> None:
        """
        Log a comprehensive final execution summary.
        
        This provides:
        - Overall completion statistics
        - Time/iteration usage
        - Task completion breakdown by depth
        - Failed tasks (if any)
        - Blackboard entry summary
        
        Args:
            state: Final agent state
        """
        tasks = state.get('tasks', [])
        completed = set(state.get('completed_task_ids', []))
        failed = set(state.get('failed_task_ids', []))
        blackboard = state.get('blackboard', [])
        
        # Calculate statistics
        total_tasks = len(tasks)
        total_completed = len(completed)
        total_failed = len(failed)
        total_pending = len([t for t in tasks if t['id'] not in completed and t['id'] not in failed and t['status'] == TaskStatus.PENDING])
        
        # Count by depth
        tasks_by_depth: Dict[int, Dict[str, int]] = {}
        for task in tasks:
            depth = task.get('depth', 0)
            if depth not in tasks_by_depth:
                tasks_by_depth[depth] = {'total': 0, 'completed': 0, 'failed': 0, 'pending': 0}
            
            tasks_by_depth[depth]['total'] += 1
            if task['id'] in completed:
                tasks_by_depth[depth]['completed'] += 1
            elif task['id'] in failed:
                tasks_by_depth[depth]['failed'] += 1
            elif task['status'] == TaskStatus.PENDING:
                tasks_by_depth[depth]['pending'] += 1
        
        # Log summary
        logger.info("=" * 80)
        logger.info("📋 FINAL EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Objective: {state.get('objective', 'N/A')[:100]}...")
        logger.info(f"Iterations Used: {state.get('iteration_count', 0)}/{state.get('max_iterations', 0)}")
        logger.info("-" * 80)
        logger.info(f"Total Tasks: {total_tasks}")
        logger.info(f"  ✓ Completed: {total_completed} ({100*total_completed//total_tasks if total_tasks > 0 else 0}%)")
        logger.info(f"  ✗ Failed: {total_failed}")
        logger.info(f"  ⏳ Pending: {total_pending}")
        logger.info("-" * 80)
        logger.info("Tasks by Depth Level:")
        for depth in sorted(tasks_by_depth.keys()):
            stats = tasks_by_depth[depth]
            logger.info(f"  Depth {depth}: {stats['total']} tasks "
                       f"(✓{stats['completed']} ✗{stats['failed']} ⏳{stats['pending']})")
        logger.info("-" * 80)
        logger.info(f"Blackboard Entries: {len(blackboard)}")
        
        # Show blackboard entry types
        if blackboard:
            entry_types: Dict[str, int] = {}
            for entry in blackboard:
                entry_type = entry.get('entry_type', 'unknown')
                entry_types[entry_type] = entry_types.get(entry_type, 0) + 1
            
            logger.info("Blackboard Entry Types:")
            for entry_type, count in sorted(entry_types.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {entry_type}: {count}")
        
        # Show failed tasks if any
        if failed:
            logger.info("-" * 80)
            logger.info(f"Failed Tasks ({len(failed)}):")
            failed_tasks = [t for t in tasks if t['id'] in failed]
            for task in failed_tasks[:10]:  # Show first 10
                logger.info(f"  ✗ [{task['id']}] {task.get('description', 'N/A')[:60]}...")
            if len(failed_tasks) > 10:
                logger.info(f"  ... and {len(failed_tasks) - 10} more")
        
        logger.info("=" * 80)
    
    
    def _check_completion(self, state: AgentState) -> Literal["continue", "complete", "max_iterations"]:
        """Check if all tasks are completed or max iterations reached."""
        
        # Log comprehensive task hierarchy
        self._log_task_hierarchy_summary(state)
        
        logger.info("=" * 80)
        logger.info("[COMPLETION CHECK] Evaluating workflow state")
        logger.info("=" * 80)
        
        # Check iteration limit
        if state['iteration_count'] >= state['max_iterations']:
            # Count pending tasks
            tasks = state['tasks']
            completed = set(state['completed_task_ids'])
            failed = set(state['failed_task_ids'])
            
            pending_tasks = [
                t for t in tasks 
                if t['id'] not in completed 
                and t['id'] not in failed
                and t['status'] == TaskStatus.PENDING
            ]
            
            logger.warning(f"[COMPLETION CHECK] Max iterations ({state['max_iterations']}) reached!")
            logger.warning(f"[COMPLETION CHECK] Pending tasks remaining: {len(pending_tasks)}")
            logger.info(f"[COMPLETION CHECK] Iteration: {state['iteration_count']}/{state['max_iterations']}")
            logger.info(f"[COMPLETION CHECK] Completed: {len(completed)}, Failed: {len(failed)}, Pending: {len(pending_tasks)}")
            
            # Ask user if they want to continue
            print()
            print("=" * 70)
            print("⚠️  MAX ITERATIONS REACHED")
            print("=" * 70)
            print(f"Current iterations: {state['iteration_count']}/{state['max_iterations']}")
            print(f"Tasks status:")
            print(f"  - Completed: {len(completed)}")
            print(f"  - Failed: {len(failed)}")
            print(f"  - Pending: {len(pending_tasks)}")
            print()
            
            if pending_tasks:
                print("There are still pending tasks to complete.")
                print()
                print("Options:")
                print("  1. Continue with more iterations (recommended)")
                print("  2. Stop now and save current progress")
                print()
                
                try:
                    choice = input("Your choice (1 or 2): ").strip()
                    
                    if choice == "1":
                        # Ask for new limit
                        print()
                        current_limit = state['max_iterations']
                        suggested_limit = current_limit + 500
                        
                        new_limit_input = input(f"Enter new iteration limit (suggested: {suggested_limit}): ").strip()
                        
                        if new_limit_input:
                            try:
                                new_limit = int(new_limit_input)
                                if new_limit > current_limit:
                                    logger.info(f"[COMPLETION CHECK] Increasing iteration limit from {current_limit} to {new_limit}")
                                    state['max_iterations'] = new_limit
                                    print(f"✓ Iteration limit increased to {new_limit}")
                                    print("Continuing execution...")
                                    print()
                                    logger.info("[COMPLETION CHECK] Decision: continue → select_task")
                                    logger.info("=" * 80)
                                    return "continue"
                                else:
                                    print(f"⚠️  New limit must be greater than {current_limit}. Stopping.")
                            except ValueError:
                                print("⚠️  Invalid number. Stopping.")
                        else:
                            # Use suggested limit
                            logger.info(f"[COMPLETION CHECK] Increasing iteration limit from {current_limit} to {suggested_limit}")
                            state['max_iterations'] = suggested_limit
                            print(f"✓ Iteration limit increased to {suggested_limit}")
                            print("Continuing execution...")
                            print()
                            logger.info("[COMPLETION CHECK] Decision: continue → select_task")
                            logger.info("=" * 80)
                            return "continue"
                    
                    print("Stopping execution...")
                    print()
                    
                except (KeyboardInterrupt, EOFError):
                    print()
                    print("Stopping execution...")
                    print()
            
            logger.info("[COMPLETION CHECK] Decision: max_iterations → END")
            logger.info("=" * 80)
            return "max_iterations"
        
        # Check if any pending tasks remain
        tasks = state['tasks']
        completed = set(state['completed_task_ids'])
        failed = set(state['failed_task_ids'])
        
        # DETECT DUPLICATION BUG
        completed_list = state['completed_task_ids']
        if len(completed_list) > len(completed):
            duplicates = len(completed_list) - len(completed)
            logger.warning(f"[COMPLETION CHECK] ⚠️  DUPLICATION DETECTED: {duplicates} duplicate IDs in completed_task_ids (total: {len(completed_list)}, unique: {len(completed)})")
        
        pending_tasks = [
            t for t in tasks 
            if t['id'] not in completed 
            and t['id'] not in failed
            and t['status'] == TaskStatus.PENDING
        ]
        
        logger.info(f"[COMPLETION CHECK] Current iteration: {state['iteration_count']}/{state['max_iterations']}")
        logger.info(f"[COMPLETION CHECK] Total tasks: {len(tasks)}")
        logger.info(f"[COMPLETION CHECK] Completed: {len(completed)} tasks")
        logger.info(f"[COMPLETION CHECK] Failed: {len(failed)} tasks")
        logger.info(f"[COMPLETION CHECK] Pending: {len(pending_tasks)} tasks")
        
        if not pending_tasks:
            logger.info("="*60)
            logger.info("[COMPLETION CHECK] ✓ ALL TASKS COMPLETED!")
            logger.info(f"[COMPLETION CHECK] Total tasks: {len(tasks)}")
            logger.info(f"[COMPLETION CHECK] Completed: {len(completed)}")
            logger.info(f"[COMPLETION CHECK] Failed: {len(failed)}")
            logger.info("[COMPLETION CHECK] Decision: complete → END")
            logger.info("="*60)
            return "complete"
        
        # List pending task IDs for debugging
        pending_ids = [t['id'] for t in pending_tasks]
        logger.info(f"[COMPLETION CHECK] Pending task IDs: {pending_ids[:10]}" + (" ..." if len(pending_ids) > 10 else ""))
        logger.info(f"[COMPLETION CHECK] Decision: continue → select_task")
        logger.info(f"[COMPLETION CHECK] [PROGRESS] {len(completed)} completed, {len(pending_tasks)} pending, {len(failed)} failed")
        logger.info("=" * 80)
        return "continue"
    
    
    # ========================================================================
    # PUBLIC INTERFACE
    # ========================================================================
    
    def run(self, thread_id: str = "default") -> AgentState:
        """
        Run the agent to completion.
        
        Args:
            thread_id: Unique identifier for this execution (for checkpointing)
        
        Returns:
            Final state with all results
        """
        from langchain_core.runnables.config import RunnableConfig
        
        # Calculate recursion limit: each task iteration involves ~6 graph nodes
        # For safety, use max_iterations * 10 to accommodate complex workflows
        recursion_limit = max(self.config.max_iterations * 10, 5000)
        
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit
        }
        
        logger.info(f"Starting Task Manager Agent")
        logger.info(f"Thread ID: {thread_id}")
        logger.info(f"Max Task Iterations: {self.config.max_iterations}")
        logger.info(f"LangGraph Recursion Limit: {recursion_limit} (allows ~{recursion_limit // 6} graph cycles)")
        logger.warning(f"⚠️  Note: LangGraph counts EVERY node execution, not just task iterations")
        
        final_state: AgentState = self.initial_state
        
        try:
            stream_count = 0
            last_node_name = "NONE"
            stream_iteration = 0
            for state in self.app.stream(self.initial_state, config):
                stream_iteration += 1
                stream_count += 1
                # state is a dict with node name as key
                try:
                    logger.debug(f"[STREAM ITERATION {stream_iteration}] State keys: {list(state.keys())}")
                    node_name = list(state.keys())[0]
                    last_node_name = node_name
                    logger.info(f"🔷 GRAPH NODE EXECUTED: {node_name}")
                    print(f"🔷 GRAPH NODE EXECUTED: {node_name}", flush=True)  # Also print to stdout with flush
                    
                    result = list(state.values())[0]
                    logger.debug(f"[STREAM] Node {node_name} result type: {type(result)}")
                    
                    if isinstance(result, dict):
                        final_state = result  # type: ignore
                        
                        # Log current progress after each node
                        iteration = final_state.get('iteration_count', 0)
                        total_tasks = len(final_state.get('tasks', []))
                        completed = len(set(final_state.get('completed_task_ids', [])))
                        failed = len(set(final_state.get('failed_task_ids', [])))
                        logger.debug(f"[PROGRESS] Iter {iteration}/{self.config.max_iterations} | Tasks: {total_tasks} | Completed: {completed} | Failed: {failed}")
                    else:
                        logger.warning(f"[STREAM] Node {node_name} returned non-dict result: {type(result)}")
                        
                    # Check if we're approaching recursion limit
                    if stream_count % 100 == 0:
                        logger.warning(f"⚠️  Graph nodes executed: {stream_count}/{recursion_limit}")
                        
                except Exception as e:
                    logger.error(f"[STREAM EXCEPTION] Iteration {stream_iteration}: Error processing state update: {str(e)}")
                    logger.log_exception("State update exception:", e)
                    import traceback
                    logger.error(f"[STREAM EXCEPTION TRACEBACK]\n{traceback.format_exc()}")
                    # Continue to next state
            
            logger.info(f"[STREAM COMPLETED] Loop exited naturally after {stream_iteration} iterations")
            
            # If we exit the loop normally, log it
            logger.info(f"🔷 GRAPH STREAM ENDED NORMALLY after {stream_count} nodes")
            logger.info(f"🔷 LAST NODE EXECUTED: {last_node_name}")
            print(f"🔷 GRAPH STREAM ENDED NORMALLY after {stream_count} nodes", flush=True)
            print(f"🔷 LAST NODE EXECUTED: {last_node_name}", flush=True)
                    
        except StopIteration:
            logger.info("🔷 GRAPH STREAM STOPPED (StopIteration)")
            print("🔷 GRAPH STREAM STOPPED (StopIteration)", flush=True)
        except GeneratorExit:
            logger.warning("🔷 GRAPH STREAM GENERATOR EXIT")
            print("🔷 GRAPH STREAM GENERATOR EXIT", flush=True)
        except KeyboardInterrupt:
            logger.warning("🔷 GRAPH STREAM INTERRUPTED BY USER")
            print("🔷 GRAPH STREAM INTERRUPTED BY USER", flush=True)
            raise  # Re-raise to allow proper cleanup
        except Exception as e:
            # Check for recursion limit errors
            error_str = str(e).lower()
            if "recursion" in error_str or "maximum" in error_str:
                logger.error("=" * 80)
                logger.error("🚨 LANGGRAPH RECURSION LIMIT EXCEEDED")
                logger.error("=" * 80)
                logger.error(f"Graph nodes executed: {stream_count}")
                logger.error(f"Recursion limit: {recursion_limit}")
                logger.error(f"Task iterations: {final_state.get('iteration_count', 0)}")
                logger.error("=" * 80)
                logger.error("SOLUTION: The workflow involves too many graph nodes.")
                logger.error("Each task iteration uses ~6 graph nodes (select → analyze → execute → aggregate → check).")
                logger.error(f"Consider reducing max_iterations or simplifying task breakdown.")
                logger.error("=" * 80)
                print(f"🚨 RECURSION LIMIT EXCEEDED: {stream_count} nodes / {recursion_limit} limit", flush=True)
            else:
                logger.error("=" * 80)
                logger.error("🚨 CRITICAL ERROR IN WORKFLOW EXECUTION")
                logger.error("=" * 80)
                logger.error(f"Exception Type: {type(e).__name__}")
                logger.error(f"Exception: {str(e)}")
                logger.log_exception("Workflow execution exception:", e)
                logger.error("=" * 80)
                logger.error("Workflow terminated unexpectedly. Returning last known state.")
                logger.error("=" * 80)
                print(f"🚨 CRITICAL ERROR: {type(e).__name__}: {str(e)}", flush=True)
        
        # Log final task hierarchy summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("🏁 WORKFLOW EXECUTION COMPLETED")
        logger.info("=" * 80)
        self._log_task_hierarchy_summary(final_state)
        self._log_final_execution_summary(final_state)
        
        # Save results to output folder
        try:
            self._save_results_to_file(final_state)
        except Exception as e:
            logger.error(f"Error saving results to file: {str(e)}")
        
        return final_state
    
    
    def resume(self, state: AgentState, thread_id: str = "default") -> AgentState:
        """
        Resume execution from a given state (e.g., after human review).
        
        Args:
            state: Current agent state to resume from
            thread_id: Unique identifier for this execution (for checkpointing)
        
        Returns:
            Final state with all results
        """
        from langchain_core.runnables.config import RunnableConfig
        
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"Resuming Task Manager Agent from state")
        logger.info(f"Thread ID: {thread_id}")
        
        final_state: AgentState = state
        
        for state_update in self.app.stream(state, config):
            # state is a dict with node name as key
            result = list(state_update.values())[0]
            if isinstance(result, dict):
                final_state = result  # type: ignore
        
        return final_state
    
    
    def get_results_summary(self, state: AgentState) -> dict:
        """Generate a summary of all results."""
        summary = {
            "objective": state['objective'],
            "total_tasks": len(state['tasks']),
            "completed_tasks": len(state['completed_task_ids']),
            "failed_tasks": len(state['failed_task_ids']),
            "iterations_used": state['iteration_count'],
            "results": state['results'],
            "task_tree": self._build_task_tree(state['tasks'])
        }
        
        return summary
    
    
    def _save_results_to_file(self, state: AgentState) -> None:
        """
        Save final results to output folder.
        
        Creates a comprehensive output file with:
        - Research findings
        - Synthesis results
        - Blackboard entries
        - Task execution details
        
        Args:
            state: Final agent state
        """
        from pathlib import Path
        from datetime import datetime
        
        # Use configured output folder
        output_folder = self.config.folders.output_path
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_folder / f"research_results_{timestamp}.md"
        
        logger.info(f"Saving results to {output_file}")
        
        # Build comprehensive output
        content = []
        content.append(f"# Research Results\n")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        content.append(f"**Objective:** {state['objective']}\n\n")
        
        # Add executive summary from synthesis if available
        blackboard = state.get('blackboard', [])
        synthesis_entry = next(
            (e for e in reversed(blackboard) if e.get('source_agent') == 'synthesis_node'),
            None
        )
        
        if synthesis_entry:
            synthesis_content = synthesis_entry.get('content', {})
            content.append("## Executive Summary\n\n")
            content.append(f"{synthesis_content.get('executive_summary', 'No summary available')}\n\n")
            
            content.append("## Key Findings\n\n")
            for finding in synthesis_content.get('key_findings', []):
                content.append(f"- {finding}\n")
            content.append("\n")
            
            # Add contradictions if any
            contradictions = synthesis_content.get('contradictions', [])
            if contradictions:
                content.append("## Data Quality Issues\n\n")
                for contradiction in contradictions:
                    content.append(f"**{contradiction.get('metric', 'Unknown')}** ({contradiction.get('severity', 'MEDIUM')})\n")
                    content.append(f"- {contradiction.get('explanation', 'No explanation')}\n")
                    sources = contradiction.get('sources', {})
                    for source, value in sources.items():
                        content.append(f"  - {source}: {value}\n")
                    content.append("\n")
        
        # Add detailed findings from each agent
        content.append("## Detailed Findings\n\n")
        
        # Group by agent
        agent_findings = {}
        for entry in blackboard:
            agent = entry.get('source_agent', 'unknown')
            if agent not in agent_findings:
                agent_findings[agent] = []
            agent_findings[agent].append(entry)
        
        for agent, findings in agent_findings.items():
            if agent == 'synthesis_node':
                continue  # Already included above
            
            content.append(f"### {agent.replace('_', ' ').title()}\n\n")
            for entry in findings[:5]:  # Limit to first 5 per agent
                entry_content = entry.get('content', {})
                if isinstance(entry_content, dict):
                    if 'summary' in entry_content:
                        content.append(f"- {entry_content.get('summary', 'N/A')}\n")
                    elif 'findings' in entry_content:
                        findings_data = entry_content.get('findings', {})
                        if isinstance(findings_data, dict):
                            for key, value in list(findings_data.items())[:3]:
                                content.append(f"- **{key}:** {str(value)[:100]}\n")
                        else:
                            content.append(f"- {str(findings_data)[:200]}\n")
            content.append("\n")
        
        # Add execution statistics
        content.append("## Execution Statistics\n\n")
        content.append(f"- **Total Tasks:** {len(state['tasks'])}\n")
        content.append(f"- **Completed:** {len(state['completed_task_ids'])}\n")
        content.append(f"- **Failed:** {len(state['failed_task_ids'])}\n")
        content.append(f"- **Iterations:** {state['iteration_count']}\n")
        content.append(f"- **Blackboard Entries:** {len(blackboard)}\n\n")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(''.join(content))
        
        logger.info(f"✓ Results saved to {output_file}")
        print()
        print("="*70)
        print(f"RESULTS SAVED TO: {output_file.absolute()}")
        print("="*70)
    
    
    def _build_task_tree(self, tasks: List[Task]) -> dict:
        """Build hierarchical task tree for visualization."""
        tree = {}
        
        for task in tasks:
            tree[task['id']] = {
                "description": task['description'],
                "status": task['status'],
                "parent": task['parent_id'],
                "depth": task['depth']
            }
        
        return tree
    
    
    # ========================================================================
    # MASTER PLANNER METHODS - NEW BLACKBOARD ARCHITECTURE
    # ========================================================================
    
    def _plan_task(self, state: AgentState) -> AgentState:
        """
        Create or refine plan based on current blackboard state.
        
        This node:
        1. Reviews existing plan and blackboard findings
        2. Refines plan based on discovered information
        3. Updates plan node statuses based on readiness
        """
        logger.info("[PLANNER] Reviewing and refining task plan...")
        
        plan = state.get('plan', [])
        blackboard = state.get('blackboard', [])
        
        # Find next ready task
        next_task = self.master_planner.get_next_ready_task(plan, blackboard)
        
        if next_task:
            logger.info(f"[PLANNER] Next ready task: {next_task['task_id']} - {next_task['description'][:60]}...")
        else:
            logger.info("[PLANNER] No tasks currently ready to execute")
        
        return {
            **state,
            "next_step": "select_task" if next_task else "aggregate_results"
        }
    
    
    def _query_blackboard(self, state: AgentState, query_type: Optional[str] = None, agent_source: Optional[str] = None) -> List[BlackboardEntry]:
        """
        Query blackboard for relevant findings.
        
        This is a utility method agents can use to discover relevant findings.
        
        Args:
            state: Current agent state
            query_type: Type of finding to search for
            agent_source: Source agent to filter by
            
        Returns:
            List of matching blackboard entries
        """
        blackboard = state.get('blackboard', [])
        return self.master_planner.query_blackboard(
            blackboard,
            entry_type=query_type,
            source_agent=agent_source
        )
    
    
    def _post_finding(
        self,
        state: AgentState,
        entry_type: str,
        content: dict,
        source_agent: str,
        source_task_id: str,
        relevant_to: Optional[List[str]] = None
    ) -> AgentState:
        """
        Post a finding to the blackboard from an agent.
        
        Args:
            state: Current agent state
            entry_type: Type of finding (web_evidence, data_point, etc.)
            content: The actual finding
            source_agent: Which agent posted this
            source_task_id: Which task produced this
            relevant_to: Task IDs that might find this useful
            
        Returns:
            Updated state with new blackboard entry
        """
        blackboard = state.get('blackboard', [])
        
        updated_blackboard = self.master_planner.post_finding(
            blackboard,
            entry_type=entry_type,
            content=content,
            source_agent=source_agent,
            source_task_id=source_task_id,
            relevant_to=relevant_to or []
        )
        
        return {
            **state,
            "blackboard": updated_blackboard
        }
    
    
    def _record_execution_step(
        self,
        state: AgentState,
        step_name: str,
        agent: str,
        task_id: str,
        outcome: dict,
        duration_seconds: float = 0.0,
        error: Optional[str] = None
    ) -> AgentState:
        """
        Record a completed step in execution history.
        
        Args:
            state: Current agent state
            step_name: Name of the step
            agent: Which agent executed
            task_id: Which task
            outcome: Result of the step
            duration_seconds: How long it took
            error: Any error that occurred
            
        Returns:
            Updated state with new history entry
        """
        history = state.get('history', [])
        
        updated_history = self.master_planner.record_step(
            history,
            step_name=step_name,
            agent=agent,
            task_id=task_id,
            outcome=outcome,
            duration_seconds=duration_seconds,
            error=error
        )
        
        return {
            **state,
            "history": updated_history
        }
    
    
    def _get_blackboard_summary(self, state: AgentState) -> Dict[str, Any]:
        """
        Get a summary of what's on the blackboard.
        
        Useful for understanding what information agents have discovered.
        
        Returns:
            Summary of blackboard entries by type
        """
        blackboard = state.get('blackboard', [])
        
        summary = {}
        for entry in blackboard:
            entry_type = entry.get('entry_type', 'unknown')
            if entry_type not in summary:
                summary[entry_type] = []
            summary[entry_type].append({
                'agent': entry.get('source_agent'),
                'task': entry.get('source_task_id'),
                'timestamp': entry.get('timestamp'),
                'relevant_to': len(entry.get('relevant_to', []))
            })
        
        logger.info(f"[BLACKBOARD] Summary: {len(blackboard)} entries across {len(summary)} types")
        return summary
    
    
    def _get_execution_summary(self, state: AgentState) -> Dict[str, Any]:
        """
        Get summary of execution history.
        
        Returns:
            Statistics about steps executed, success rate, total time, etc.
        """
        history = state.get('history', [])
        return self.master_planner.get_execution_summary(history)
    
    
    # ========================================================================
    # DEEP HIERARCHY METHODS - Context & Depth Management
    # ========================================================================
    
    def _post_nested_finding(
        self,
        state: AgentState,
        entry_type: str,
        content: dict,
        source_agent: str,
        source_task_id: str,
        parent_task_id: str,
        depth_level: int,
        relevant_to: Optional[List[str]] = None
    ) -> AgentState:
        """
        Post a finding to the blackboard with parent task tagging.
        
        Used when a sub-task posts findings organized under its parent.
        Enables hierarchical organization of knowledge.
        
        Args:
            state: Current agent state
            entry_type: Type of finding
            content: The finding data
            source_agent: Which agent posted this
            source_task_id: Which task produced this
            parent_task_id: Parent task for organization
            depth_level: Hierarchical depth
            relevant_to: Task IDs that might use this
            
        Returns:
            Updated state with new nested blackboard entry
        """
        blackboard = state.get('blackboard', [])
        
        updated_blackboard = self.master_planner.post_nested_finding(
            blackboard,
            entry_type=entry_type,
            content=content,
            source_agent=source_agent,
            source_task_id=source_task_id,
            parent_task_id=parent_task_id,
            depth_level=depth_level,
            relevant_to=relevant_to or []
        )
        
        return {
            **state,
            "blackboard": updated_blackboard
        }
    
    
    def _extract_parent_context(self, state: AgentState, parent_task_id: str) -> Dict[str, Any]:
        """
        Extract context from a parent task to guide child execution.
        
        Gathers all findings nested under parent and creates summary.
        
        Args:
            state: Current agent state
            parent_task_id: Parent task to extract from
            
        Returns:
            Context dictionary with parent's findings
        """
        plan = state.get('plan', [])
        blackboard = state.get('blackboard', [])
        
        return self.master_planner.extract_context_from_parent(
            plan,
            parent_task_id,
            blackboard
        )
    
    
    def _check_depth_limit(self, state: AgentState) -> bool:
        """
        Check if current depth exceeds the limit.
        
        Prevents infinite recursion in hierarchical decomposition.
        
        Args:
            state: Current agent state
            
        Returns:
            True if within limit, False if exceeded
        """
        current_depth = state.get('current_depth', 0)
        depth_limit = state.get('depth_limit', 5)
        
        return self.master_planner.check_depth_limit(current_depth, depth_limit)
    
    
    def _add_context_to_task(
        self,
        state: AgentState,
        task_id: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """
        Add context to a plan node for hierarchical execution.
        
        Args:
            state: Current agent state
            task_id: Task to update
            context: Context to add
            
        Returns:
            Updated state with context added
        """
        plan = state.get('plan', [])
        
        updated_plan = self.master_planner.add_context_to_plan_node(
            plan,
            task_id,
            context
        )
        
        return {
            **state,
            "plan": updated_plan
        }
    
    
    def _establish_hierarchy(
        self,
        state: AgentState,
        parent_id: str,
        child_ids: List[str]
    ) -> AgentState:
        """
        Establish parent-child relationships in the plan.
        
        Args:
            state: Current agent state
            parent_id: Parent task ID
            child_ids: List of child task IDs
            
        Returns:
            Updated state with relationships established
        """
        plan = state.get('plan', [])
        
        updated_plan = self.master_planner.establish_parent_child_relationship(
            plan,
            parent_id,
            child_ids
        )
        
        return {
            **state,
            "plan": updated_plan
        }
    
    
    def _get_hierarchy_structure(self, state: AgentState) -> Dict[int, int]:
        """
        Get distribution of tasks across hierarchy depths.
        
        Useful for understanding the plan structure.
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary mapping depth → count of tasks
        """
        plan = state.get('plan', [])
        return self.master_planner.get_hierarchy_depth_distribution(plan)
    
    
    def _query_blackboard_by_parent(
        self,
        state: AgentState,
        parent_task_id: str
    ) -> List[BlackboardEntry]:
        """
        Query blackboard for findings under a parent task.
        
        Args:
            state: Current agent state
            parent_task_id: Parent to query
            
        Returns:
            List of findings under parent
        """
        blackboard = state.get('blackboard', [])
        return self.master_planner.query_blackboard_by_parent(blackboard, parent_task_id)
    
    
    def _query_blackboard_by_depth(
        self,
        state: AgentState,
        depth_level: int
    ) -> List[BlackboardEntry]:
        """
        Query blackboard for findings at a specific depth.
        
        Args:
            state: Current agent state
            depth_level: Depth to query
            
        Returns:
            List of findings at depth
        """
        blackboard = state.get('blackboard', [])
        return self.master_planner.query_blackboard_by_depth(blackboard, depth_level)
    
    
    # ========================================================================
    # REFLECTION & DYNAMIC TASK INJECTION - Wrapper methods
    # ========================================================================
    
    def _evaluate_finding_depth(
        self,
        state: AgentState,
        task_id: str
    ) -> Dict[str, Any]:
        """
        Agent wrapper: Evaluate if completed task's findings are shallow.
        
        Calls master_planner.evaluate_finding_depth() to assess whether
        findings from a completed task require deeper investigation.
        
        Args:
            state: Current agent state
            task_id: ID of the completed task
            
        Returns:
            Evaluation result with depth assessment and drill areas
        """
        plan = state.get('plan', [])
        task = next((t for t in plan if t['task_id'] == task_id), None)
        
        if not task:
            logger.error(f"Task {task_id} not found during depth evaluation")
            return {
                "task_id": task_id,
                "is_shallow": False,
                "depth_assessment": "unknown",
                "confidence": 0.0,
                "reasoning": "Task not found",
                "suggested_drill_areas": [],
                "recommended_agent_types": []
            }
        
        # Get task outcome from results
        outcome = state.get('results', {}).get(task_id, {})
        
        evaluation = self.master_planner.evaluate_finding_depth(
            task_id=task_id,
            finding_content=outcome,
            task_description=task.get('description', ''),
            metadata=state.get('metadata', {})
        )
        
        logger.debug(f"Evaluated task {task_id}: {evaluation['depth_assessment']}")
        return evaluation
    
    
    def _create_deep_dive_subtasks(
        self,
        state: AgentState,
        parent_task_id: str,
        evaluation: Dict[str, Any]
    ) -> AgentState:
        """
        Agent wrapper: Create deep dive subtasks if findings are shallow.
        
        Updates state's plan with new Level 3 tasks focused on specific
        areas identified in the evaluation.
        
        Args:
            state: Current agent state
            parent_task_id: The Level 2 task needing deeper analysis
            evaluation: The depth evaluation results
            
        Returns:
            Updated state with new deep dive tasks in plan
        """
        plan = state.get('plan', [])
        parent_task = next((t for t in plan if t['task_id'] == parent_task_id), None)
        
        if not parent_task:
            logger.error(f"Parent task {parent_task_id} not found")
            return state
        
        # Create deep dive subtasks
        updated_plan = self.master_planner.create_deep_dive_subtasks(
            plan,
            parent_task_id,
            evaluation,
            parent_task.get('description', '')
        )
        
        logger.info(
            f"Injected {len(updated_plan) - len(plan)} deep dive tasks "
            f"under {parent_task_id}"
        )
        
        return {
            **state,
            "plan": updated_plan  # type: ignore
        }
    
    
    def _reflect_and_inject(
        self,
        state: AgentState,
        task_id: str
    ) -> Tuple[AgentState, bool]:
        """
        Agent wrapper: Execute full reflection step after task completion.
        
        Workflow:
        1. Evaluate if completed task's findings are sufficiently deep
        2. If shallow, dynamically create 2-3 Level 3 deep dive subtasks
        3. Update plan with new tasks and mark them as READY
        4. Return indication of whether tasks were injected
        
        Args:
            state: Current agent state
            task_id: The task that just completed
            
        Returns:
            Tuple of (updated_state, new_tasks_injected)
        """
        plan = state.get('plan', [])
        outcome = state.get('results', {}).get(task_id, {})
        
        # Perform reflection and task injection
        updated_plan, injected = self.master_planner.reflect_and_inject(
            state, task_id, outcome
        )
        
        if injected:
            logger.info(f"Deep dive tasks injected after completing {task_id}")
            
            updated_state = {
                **state,
                "plan": updated_plan  # type: ignore
            }
            return updated_state, True  # type: ignore
        
        return state, False
    
    
    @staticmethod
    def _parse_json_response(content: str) -> dict:
        """
        Parse JSON response from LLM, handling markdown formatting and errors.
        
        Robust parser that:
        - Extracts JSON from markdown code blocks
        - Handles incomplete JSON with fallback
        - Validates required fields
        - Provides helpful error messages
        
        Args:
            content: Raw response content from LLM
            
        Returns:
            Parsed JSON dictionary with valid structure
            
        Raises:
            ValueError: If JSON cannot be parsed or lacks required fields
        """
        if not content or not isinstance(content, str):
            raise ValueError(f"Invalid content type: {type(content)}, expected string")
        
        content = content.strip()
        
        # Remove markdown code blocks
        if content.startswith("```json"):
            parts = content.split("```json")
            if len(parts) > 1:
                content = parts[1].split("```")[0].strip()
        elif content.startswith("```"):
            parts = content.split("```")
            if len(parts) > 2:
                content = parts[1].strip()
        
        if not content:
            raise ValueError("Content is empty after removing markdown")
        
        try:
            parsed = json.loads(content)
            
            # Validate that result is a dict
            if not isinstance(parsed, dict):
                raise ValueError(f"Parsed result is not a dict: {type(parsed)}")
            
            return parsed
            
        except json.JSONDecodeError as e:
            # Try to provide helpful error message
            error_msg = f"Failed to parse JSON: {str(e)}"
            logger.error(f"[PARSE] {error_msg}")
            logger.error(f"[PARSE] Content preview: {content[:200]}...")
            
            # Try to recover by finding first { and last }
            first_brace = content.find('{')
            last_brace = content.rfind('}')
            
            if first_brace >= 0 and last_brace > first_brace:
                logger.warning("[PARSE] Attempting to recover by extracting JSON object...")
                recovered_content = content[first_brace:last_brace + 1]
                try:
                    parsed = json.loads(recovered_content)
                    logger.info("[PARSE] ✓ Successfully recovered JSON from response")
                    return parsed
                except json.JSONDecodeError as e2:
                    logger.error(f"[PARSE] Recovery failed: {str(e2)}")
                    raise ValueError(f"Could not parse JSON even after recovery attempt: {str(e2)}")
            else:
                raise ValueError(f"Could not find JSON object boundaries in response: {str(e)}")

    def _detect_chain_execution(self, agent_output: dict) -> List[str]:
        """
        Detect if agent output should trigger chain execution to next agent.
        
        Checks for indicators that another agent should execute next:
        - PDF agent finds images → OCR agent should run
        - Web search creates CSV → Excel agent should process
        - OCR produces text → Analysis agent should evaluate
        
        Args:
            agent_output: The output from current agent's analysis
            
        Returns:
            List of next agents to execute (empty if no chain needed)
        """
        chain_agents = []
        
        # Check what was generated/found
        findings = agent_output.get('findings', {})
        has_images = findings.get('extracted_images', []) or agent_output.get('images_found', False)
        has_csv = findings.get('csv_generated', False) or agent_output.get('generated_files', {}).get('csv')
        has_text_data = findings.get('extracted_text', "") or agent_output.get('text_output')
        
        # If images found → OCR should analyze them
        if has_images:
            chain_agents.append("ocr_task")
        
        # If CSV created → Excel should process it
        if has_csv:
            chain_agents.append("excel_task")
        
        # If structured text extracted → May trigger analysis
        if has_text_data:
            # Check if text contains table/financial data
            text_sample = str(has_text_data)[:200].lower()
            if any(keyword in text_sample for keyword in ['table', 'price', 'cost', 'amount', 'total', 'revenue']):
                if "excel_task" not in chain_agents:
                    chain_agents.append("excel_task")
        
        return chain_agents
    
    def _post_file_pointer(self, file_path: str, file_type: str, 
                          created_by_agent: str, source_task_id: str,
                          state: AgentState) -> None:
        """
        Post a file pointer to the blackboard for cross-agent workflow handoff.
        
        Enables agents to discover files generated by other agents.
        Example: Excel agent finds CSV created by Web Search agent.
        
        Args:
            file_path: Path to the file (relative or absolute)
            file_type: Type of file (csv, json, pdf, image, text, etc.)
            created_by_agent: Name of agent that created this file
            source_task_id: Task ID that created the file
            state: Current agent state (to update blackboard)
        """
        from datetime import datetime
        
        file_pointer: BlackboardEntry = {
            "entry_type": f"file_pointer_{file_type}",
            "source_agent": created_by_agent,
            "source_task_id": source_task_id,
            "content": {
                "file_path": file_path,
                "file_type": file_type,
                "file_size_bytes": None,  # Could be computed if needed
                "accessible": True,
                "description": f"Generated by {created_by_agent}"
            },
            "timestamp": datetime.now().isoformat(),
            "relevant_to": [],  # Will be populated by planning agents
            "source_file_path": file_path,
            "chain_next_agents": []  # Populated by chain detection
        }
        
        # Add to blackboard
        state["blackboard"].append(file_pointer)