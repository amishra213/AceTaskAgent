"""
Agent module - Main TaskManagerAgent implementation
"""

from typing import Literal, List, Optional, Dict, Any, Tuple
from datetime import datetime
import json
import hashlib

from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from task_manager.models import AgentState, TaskStatus, Task, PlanNode, BlackboardEntry, HistoryEntry
from task_manager.config import AgentConfig
from task_manager.utils.logger import get_logger
from task_manager.utils.prompt_builder import PromptBuilder
from task_manager.utils.rate_limiter import global_rate_limiter
from task_manager.utils.input_context import InputContext
from task_manager.utils.temp_manager import TempDataManager
from task_manager.utils.redis_cache import RedisCacheManager
from task_manager.sub_agents import (
    PDFAgent, ExcelAgent, OCRImageAgent, WebSearchAgent, 
    CodeInterpreterAgent, DataExtractionAgent
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
        # Note: data_extraction_agent already initialized above for context building
        
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
        logger.debug("Sub-agents initialized: PDFAgent, ExcelAgent, OCRImageAgent, WebSearchAgent, CodeInterpreterAgent")
    
    
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
                raise ValueError(
                    f"Unsupported LLM provider: {provider}. "
                    f"Supported providers: anthropic, openai, google, groq, local"
                )
        except ImportError as e:
            raise ImportError(
                f"Missing package for {provider} provider. "
                f"Install with: pip install langchain-{provider}"
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
            "human_feedback": ""
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
        
        # Find pending tasks (not completed, not failed)
        pending_tasks = [
            t for t in tasks 
            if t['id'] not in completed 
            and t['id'] not in failed
            and t['status'] == TaskStatus.PENDING
        ]
        
        logger.info(f"[SELECT] Pending tasks: {len(pending_tasks)}")
        
        # Log all pending task IDs for tracking
        if pending_tasks:
            pending_ids = [t['id'] for t in pending_tasks]
            logger.info(f"[SELECT] Pending task IDs: {pending_ids}")
        
        if not pending_tasks:
            logger.info("[SELECT] No pending tasks found - workflow will check completion")
            logger.info("=" * 80)
            return {**state, "active_task_id": ""}
        
        # Select first pending task (could implement priority logic here)
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
        """Analyze current task using LLM to decide next action."""
        task_id = state['active_task_id']
        
        if not task_id:
            return state
        
        task = next(t for t in state['tasks'] if t['id'] == task_id)
        
        logger.info(f"[ANALYZE] Analyzing task {task_id}...")
        
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
            # Force web_search_task execution for research tasks
            analysis = {
                "action": "web_search_task",
                "reasoning": f"Task at depth {current_depth} - forcing execution to prevent excessive decomposition",
                "subtasks": None,
                "search_query": task.get('description', ''),
                "file_operation": {
                    "type": "web_search",
                    "operation": "search",
                    "parameters": {
                        "query": task.get('description', ''),
                        "num_results": 5
                    }
                },
                "estimated_complexity": "medium",
                "requires_human_review": False
            }
            
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
            
            return {
                **state,
                "tasks": updated_tasks,
                "requires_human_review": False
            }
        
        # Build analysis prompt using PromptBuilder (with input context)
        analysis_prompt = PromptBuilder.build_analysis_prompt(
            objective=state['objective'],
            task=task,
            metadata=state['metadata'],
            current_depth=current_depth,
            depth_limit=depth_limit,
            input_context=state.get('input_context')
        )
        
        try:
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
            
            # Parse response
            analysis = self._parse_json_response(response_content)
            
            logger.info(f"[ANALYZE] Result: {analysis['action']} - {analysis['reasoning']}")
            
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
            
            return {
                **state,
                "tasks": updated_tasks,
                "requires_human_review": analysis.get('requires_human_review', False)
            }
            
        except Exception as e:
            logger.error(f"[ANALYZE] Error: {str(e)}")
            
            # Update task with error
            updated_task: Task = {
                **task,
                "status": TaskStatus.FAILED,
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }
            
            updated_tasks: List[Task] = [
                updated_task if t['id'] == task_id else t
                for t in state['tasks']
            ]
            
            return {
                **state,
                "tasks": updated_tasks,
                "failed_task_ids": [task_id]
            }
    
    
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
            
            # Execute the operation via PDF Agent
            logger.info(f"[PDF AGENT] Calling PDFAgent.execute_task()...")
            result_data = self.pdf_agent.execute_task(operation, parameters)
            
            # Update task with results
            updated_task = {
                **task,
                "status": TaskStatus.COMPLETED if result_data.get('success') else TaskStatus.FAILED,
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
            
            # Create blackboard entry with findings and file pointers
            blackboard_entry: BlackboardEntry = {
                "entry_type": "pdf_extraction_result",
                "source_agent": "pdf_agent",
                "source_task_id": task_id,
                "content": {
                    "operation": operation,
                    "success": result_data.get('success'),
                    "summary": result_data.get('summary', ''),
                    "findings": result_data.get('findings', {})
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
            
            logger.info(f"[EXCEL AGENT] Operation: {operation}")
            logger.info(f"[EXCEL AGENT] File: {parameters.get('file_path', 'N/A')[:80]}...")
            logger.info(f"[EXCEL AGENT] Parameters: {parameters}")
            logger.info("-" * 80)
            
            # Execute the operation via Excel Agent
            logger.info(f"[EXCEL AGENT] Calling ExcelAgent.execute_task()...")
            result_data = self.excel_agent.execute_task(operation, parameters)
            
            # Update task with results
            updated_task = {
                **task,
                "status": TaskStatus.COMPLETED if result_data.get('success') else TaskStatus.FAILED,
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
            
            # Create blackboard entry with final processed results
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
                    "findings": result_data.get('findings', {})
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
            
            # Execute the operation via OCR Image Agent
            logger.info(f"[OCR AGENT] Calling OCRImageAgent.run_operation()...")
            result_data = self.ocr_image_agent.run_operation(operation, parameters)
            
            # Update task with results
            updated_task = {
                **task,
                "status": TaskStatus.COMPLETED if result_data.get('success') else TaskStatus.FAILED,
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
        logger.info("🔹 ENTERED _execute_web_search_task node")
        
        task_id = state['active_task_id']
        logger.info(f"🔹 Processing task_id: {task_id}")
        
        task = next(t for t in state['tasks'] if t['id'] == task_id)
        analysis = task.get('result') if isinstance(task, dict) else task['result']
        
        # Ensure analysis is a dict
        if not isinstance(analysis, dict):
            analysis = {}
        
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
            
            # Execute the operation via Web Search Agent
            logger.info(f"[WEB SEARCH AGENT] Calling WebSearchAgent.execute_task()...")
            result_data = self.web_search_agent.execute_task(operation, params)
            
            # Update task with results
            updated_task = {
                **task,
                "status": TaskStatus.COMPLETED if result_data.get('success') else TaskStatus.FAILED,
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
            
            # Execute the operation via Code Interpreter Agent
            logger.info(f"[CODE INTERPRETER AGENT] Calling CodeInterpreterAgent.execute_task()...")
            result_data = self.code_interpreter_agent.execute_task(operation, params)
            
            # Update task with results
            updated_task = {
                **task,
                "status": TaskStatus.COMPLETED if result_data.get('success') else TaskStatus.FAILED,
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
            operation = file_operation.get('operation', 'extract_relevant')
            parameters = file_operation.get('parameters', {})
            
            # Use task description as extraction objective
            extraction_objective = task.get('description', state['objective'])
            
            # Get folder paths from metadata
            input_folder = state.get('metadata', {}).get('input_folder', './input_folder')
            temp_folder = state.get('metadata', {}).get('temp_folder', './temp_folder')
            
            result = None
            
            if operation == 'extract_relevant':
                # Main extraction operation - get relevant data
                result = self.data_extraction_agent.extract_relevant_data(
                    input_folder=input_folder,
                    objective=extraction_objective,
                    keywords=parameters.get('keywords'),
                    temp_folder=temp_folder,
                    max_files=parameters.get('max_files', 15),
                    max_content_per_file=parameters.get('max_content_per_file', 4000),
                    max_total_content=parameters.get('max_total_content', 40000),
                    file_types=parameters.get('file_types')
                )
                
            elif operation == 'search_content':
                # Search for specific content
                result = self.data_extraction_agent.search_in_files(
                    input_folder=input_folder,
                    search_query=parameters.get('query', extraction_objective),
                    file_types=parameters.get('file_types'),
                    max_results=parameters.get('max_results', 10)
                )
                
            elif operation == 'get_file_preview':
                # Get preview of a specific file
                file_path = parameters.get('file_path')
                if file_path:
                    result = self.data_extraction_agent.get_file_preview(
                        file_path=file_path,
                        max_chars=parameters.get('max_chars', 2000)
                    )
                else:
                    result = {"success": False, "error": "No file_path provided"}
                    
            elif operation == 'summarize_folder':
                # Get folder summary
                result = self.data_extraction_agent.summarize_input_folder(
                    input_folder=input_folder,
                    objective=extraction_objective
                )
                
            else:
                # Default to extract_relevant
                result = self.data_extraction_agent.extract_relevant_data(
                    input_folder=input_folder,
                    objective=extraction_objective,
                    temp_folder=temp_folder
                )
            
            logger.info(f"[DATA EXTRACTION AGENT] Operation '{operation}' completed")
            
            # Update task with result
            result_data = {
                "operation": operation,
                "success": result.get('success', True) if result else False,
                "data": result
            }
            
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
            
            # Create blackboard entry with extracted data summary
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
        
        # Handle failed tasks separately - add to failed_task_ids instead of completed_task_ids
        if task_status == TaskStatus.FAILED and not already_failed:
            logger.warning(f"🔍 [TASK_ID_TRACKER] AGGREGATE RETURN | Adding FAILED task_id='{task_id}' to failed_task_ids | Location: _aggregate_results:1788")
            return {
                **state,
                "failed_task_ids": [task_id]  # Add to failed list, not completed list
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
            logger.exception(e)
            
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
            logger.exception(e)
            
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
            logger.exception(e)
            
            # Don't fail the workflow - just log and continue
            return {
                **state,
                "last_updated_key": None  # Clear trigger
            }
    
    
    def _handle_error(self, state: AgentState) -> AgentState:
        """Handle task errors with retry logic."""
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
        
        # Mark task as failed in the tasks list
        updated_tasks = []
        for task in state['tasks']:
            if task['id'] == task_id:
                task['status'] = TaskStatus.FAILED
                task['result'] = {'error': 'Task failed during processing'}
                updated_tasks.append(task)
            else:
                updated_tasks.append(task)
        
        logger.warning(f"[ERROR] Task {task_id} marked as failed")
        
        # NOTE: failed_task_ids uses operator.add annotation, so only return NEW items to add
        return {
            **state,
            "tasks": updated_tasks,
            "failed_task_ids": [task_id],  # Only the new task ID - LangGraph will concatenate
            "active_task_id": "",  # Clear active task
            "iteration_count": state['iteration_count'] + 1
        }
    
    
    def _request_human_review(self, state: AgentState) -> AgentState:
        """Request human review for complex decisions - Try websearch first"""
        logger.warning("!"*60)
        logger.warning("HUMAN REVIEW REQUIRED")
        logger.warning("!"*60)
        logger.warning("The agent needs human input to proceed.")
        
        task_id = state.get('active_task_id')
        current_task = None
        if task_id:
            task = next((t for t in state.get('tasks', []) if t['id'] == task_id), None)
            if task:
                current_task = task
                logger.warning(f"Active task: {task_id}")
                logger.warning(f"Task description: {task.get('description', 'N/A')[:100]}...")
                if task.get('result'):
                    logger.warning(f"Analysis result: {task.get('result')}")
        
        logger.warning("!"*60)
        
        # Try websearch first
        if self.config.enable_search and current_task:
            logger.info("Attempting to search for task information...")
            try:
                from langchain_community.tools import DuckDuckGoSearchRun
                search = DuckDuckGoSearchRun()
                search_query = current_task.get('description', '')[:100]
                result = search.run(search_query)
                
                if result and len(result.strip()) > 0:
                    logger.info("Search results found. Proceeding with execution using search data...")
                    # Add search results to task metadata
                    updated_state: AgentState = dict(state)  # type: ignore
                    updated_state['metadata'] = {
                        **state.get('metadata', {}),
                        'search_results': result,
                        'search_performed': True
                    }
                    return updated_state
            except Exception as e:
                logger.debug(f"Web search attempt failed: {str(e)}. Proceeding to human input...")
        
        # Interactive human review if search didn't help
        print()
        print("=" * 70)
        print("HUMAN REVIEW REQUIRED")
        print("=" * 70)
        
        if current_task:
            print()
            print(f"Task: {current_task.get('description', 'Unknown')[:100]}")
            
            result = current_task.get('result')
            if result:
                if isinstance(result, dict):
                    print()
                    print("Analysis Result:")
                    for key, value in result.items():
                        if key != 'subtasks':
                            print(f"  {key}: {str(value)[:80]}")
                        else:
                            if isinstance(value, list):
                                print(f"  {key}: {len(value)} subtasks created")
        
        print()
        print("Options:")
        print("  1. Continue with execution")
        print("  2. Provide additional guidance")
        print("  3. Skip this task")
        print("  4. Abort")
        print()
        
        updated_state: AgentState = dict(state)  # type: ignore
        
        while True:
            choice = input("Enter your choice (1-4): ").strip()
            
            # Update state based on choice
            
            if choice == "1":
                print()
                print("Proceeding with execution...")
                break
                
            elif choice == "2":
                print()
                guidance = input("Provide guidance (e.g., data sources, specific requirements): ").strip()
                if guidance:
                    updated_state['metadata'] = {**state.get('metadata', {}), 'human_guidance': guidance}
                    print("Guidance recorded. Continuing...")
                    break
                else:
                    print("No guidance provided. Please try again or choose another option.")
                    continue
                
            elif choice == "3":
                print()
                print("Skipping this task...")
                if task_id:
                    completed = set(state.get('completed_task_ids', []))
                    if task_id not in completed:
                        # TRACKING: Log task ID addition for debugging
                        logger.warning(f"🔍 [TASK_ID_TRACKER] HUMAN_REVIEW adding task_id='{task_id}' (user skip) | Current list size: {len(state.get('completed_task_ids', []))} | Location: _request_human_review:2302")
                        # NOTE: completed_task_ids uses operator.add annotation
                        # Only return NEW task IDs, NOT the full list - LangGraph concatenates
                        updated_state['completed_task_ids'] = [task_id]
                break
                
            elif choice == "4":
                print()
                print("Aborting execution...")
                # Mark all remaining tasks as completed to end workflow
                all_tasks = state.get('tasks', [])
                completed = set(state.get('completed_task_ids', []))
                tasks_to_add = []
                for task in all_tasks:
                    if task['id'] not in completed:
                        tasks_to_add.append(task['id'])
                # TRACKING: Log task ID addition for debugging
                logger.warning(f"🔍 [TASK_ID_TRACKER] HUMAN_REVIEW adding {len(tasks_to_add)} task_ids (abort all) | Tasks: {tasks_to_add} | Current list size: {len(state.get('completed_task_ids', []))} | Location: _request_human_review:2322")
                # NOTE: completed_task_ids uses operator.add annotation
                # Only return NEW task IDs, NOT the full list - LangGraph concatenates
                updated_state['completed_task_ids'] = tasks_to_add
                break
            
            else:
                print()
                print("Invalid choice. Please enter 1-4.")
                continue
        
        return updated_state  # type: ignore
    
    
    # ========================================================================
    # ROUTING FUNCTIONS
    # ========================================================================
    
    def _route_with_chain_execution(self, state: AgentState) -> Literal["breakdown", "execute_task", "execute_pdf_task", "execute_excel_task", "execute_ocr_task", "execute_web_search_task", "execute_code_interpreter_task", "execute_data_extraction_task", "handle_error", "review"]:
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
        
        # Handle case where no more tasks are pending
        # This happens after breakdown fails or all tasks are completed
        if not task_id:
            logger.info("[ROUTE] No active task - all pending tasks completed")
            # Don't route to handle_error; instead wait for next select_task
            # The workflow will handle completion check through aggregate_results
            return "handle_error"  # Will be caught by select_task loop
        
        task = next((t for t in state['tasks'] if t['id'] == task_id), None)
        
        if not task:
            logger.error(f"[ROUTE] Task {task_id} not found in tasks list")
            return "handle_error"
        
        if task['status'] == TaskStatus.FAILED:
            logger.warning(f"[ROUTE] Task {task_id} already marked as FAILED, handling error")
            return "handle_error"
        
        analysis = task.get('result') if isinstance(task, dict) else task['result']
        if not isinstance(analysis, dict):
            logger.error(f"[ROUTE] Task {task_id} has invalid analysis format (not dict)")
            analysis = {}
        
        action = analysis.get('action', 'handle_error')
        
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
            "execute": "execute_task",
        }
        action = action_mapping.get(action, action)
        
        # Priority 1: Human review (if required and not breakdown)
        if state.get('requires_human_review', False) and action != "breakdown":
            logger.info(f"[ROUTE] Requiring human review for task {task_id}")
            return "review"
        
        # Priority 2: Breakdown decomposition

        if action == "breakdown":
            logger.info(f"[ROUTE] Routing to breakdown for task {task_id}")
            return "breakdown"
        
        # Priority 3: Execute-specific agent tasks
        if action == "execute_task":
            logger.info(f"[ROUTE] Routing to generic execute for task {task_id}")
            return "execute_task"
        elif action == "execute_pdf_task":
            logger.info(f"[ROUTE] Routing to PDF task executor for {task_id}")
            return "execute_pdf_task"
        elif action == "execute_excel_task":
            logger.info(f"[ROUTE] Routing to Excel task executor for {task_id}")
            return "execute_excel_task"
        elif action == "execute_ocr_task":
            logger.info(f"[ROUTE] Routing to OCR task executor for {task_id}")
            return "execute_ocr_task"
        elif action == "execute_web_search_task":
            logger.info(f"[ROUTE] Routing to WebSearch task executor for {task_id}")
            return "execute_web_search_task"
        elif action == "execute_code_interpreter_task":
            logger.info(f"[ROUTE] Routing to Code Interpreter task executor for {task_id}")
            return "execute_code_interpreter_task"
        elif action == "execute_data_extraction_task":
            logger.info(f"[ROUTE] Routing to Data Extraction task executor for {task_id}")
            return "execute_data_extraction_task"
        else:
            logger.error(f"[ROUTE] Unknown action '{action}' for task {task_id}")
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
        
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.config.max_iterations + 100  # Set high limit to allow workflow to continue
        }
        
        logger.info(f"Starting Task Manager Agent")
        logger.info(f"Thread ID: {thread_id}")
        logger.info(f"Recursion limit set to: {self.config.max_iterations + 100}")
        
        final_state: AgentState = self.initial_state
        
        try:
            stream_count = 0
            for state in self.app.stream(self.initial_state, config):
                stream_count += 1
                # state is a dict with node name as key
                try:
                    node_name = list(state.keys())[0]
                    logger.info(f"🔷 GRAPH NODE EXECUTED: {node_name}")
                    print(f"🔷 GRAPH NODE EXECUTED: {node_name}", flush=True)  # Also print to stdout with flush
                    
                    result = list(state.values())[0]
                    if isinstance(result, dict):
                        final_state = result  # type: ignore
                except Exception as e:
                    logger.error(f"Error processing state update: {str(e)}")
                    logger.exception(e)
                    # Continue to next state
            
            # If we exit the loop normally, log it
            logger.info(f"🔷 GRAPH STREAM ENDED NORMALLY after {stream_count} nodes")
            print(f"🔷 GRAPH STREAM ENDED NORMALLY after {stream_count} nodes", flush=True)
                    
        except StopIteration:
            logger.info("🔷 GRAPH STREAM STOPPED (StopIteration)")
            print("🔷 GRAPH STREAM STOPPED (StopIteration)", flush=True)
        except GeneratorExit:
            logger.warning("🔷 GRAPH STREAM GENERATOR EXIT")
            print("🔷 GRAPH STREAM GENERATOR EXIT", flush=True)
        except KeyboardInterrupt:
            logger.warning("🔷 GRAPH STREAM INTERRUPTED BY USER")
            print("🔷 GRAPH STREAM INTERRUPTED BY USER", flush=True)
        except Exception as e:
            logger.error("=" * 80)
            logger.error("🚨 CRITICAL ERROR IN WORKFLOW EXECUTION")
            logger.error("=" * 80)
            logger.error(f"Exception Type: {type(e).__name__}")
            logger.error(f"Exception: {str(e)}")
            logger.exception(e)
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
        Parse JSON response from LLM, handling markdown formatting.
        
        Args:
            content: Raw response content from LLM
            
        Returns:
            Parsed JSON dictionary
        """
        content = content.strip()
        
        # Remove markdown if present
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)
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