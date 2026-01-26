"""
Master Planner Module - Sophisticated planning and coordination for multi-agent systems

This module implements the Master Planner architecture with:
1. Hierarchical task planning from objectives
2. Blackboard pattern for shared agent findings
3. Dynamic routing based on plan and blackboard state
4. Cross-agent communication and dependency management
"""

from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json
from enum import Enum

from task_manager.models import AgentState, PlanNode, BlackboardEntry, HistoryEntry
from task_manager.utils.logger import get_logger
from task_manager.utils.rate_limiter import global_rate_limiter

logger = get_logger(__name__)


class PlanStatus(str, Enum):
    """Status of a plan node."""
    PENDING = "pending"
    READY = "ready"  # All dependencies satisfied
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Waiting on blackboard data


class BlackboardType(str, Enum):
    """Categories of information stored on blackboard."""
    WEB_EVIDENCE = "web_evidence"  # Findings from web search
    DATA_POINT = "data_point"  # Extracted numerical/categorical data
    EXTRACTED_TABLE = "extracted_table"  # Structured table data
    DOCUMENT_CONTENT = "document_content"  # PDF or document text
    IMAGE_TEXT = "image_text"  # Text extracted from images
    HYPOTHESIS = "hypothesis"  # Agent proposed theory/connection
    ANALYSIS = "analysis"  # Analyzed findings
    RELATIONSHIP = "relationship"  # Connection between data points


class MasterPlanner:
    """
    Orchestrates sophisticated multi-agent task planning and execution.
    
    Key Features:
    - Converts objectives into hierarchical task plans
    - Manages blackboard as shared knowledge base
    - Routes tasks based on plan and available blackboard data
    - Enables cross-agent communication and discovery
    - Supports non-linear execution flows
    """
    
    def __init__(self, llm=None):
        """
        Initialize the Master Planner.
        
        Args:
            llm: Optional LLM instance for advanced planning (Google Gemini, etc.)
        """
        self.llm = llm
        logger.debug("Master Planner initialized")
    
    
    def create_initial_plan(self, objective: str, metadata: Dict[str, Any]) -> List[PlanNode]:
        """
        Create initial hierarchical task plan from objective.
        
        This converts a high-level objective into a structured plan with:
        - Hierarchical decomposition (depth levels)
        - Task dependencies
        - Priority ordering
        - Estimated effort
        
        Args:
            objective: The main goal/objective
            metadata: Additional context
            
        Returns:
            List of PlanNode representing the task hierarchy
        """
        logger.info("=" * 80)
        logger.info("MASTER PLANNER: CREATING INITIAL TASK PLAN")
        logger.info("=" * 80)
        logger.info(f"Objective: {objective[:100]}...")
        logger.info(f"Metadata: {metadata}")
        
        # Root plan node (no dependencies)
        root_plan = PlanNode(
            task_id="plan_0",
            parent_id=None,
            depth=0,
            description=objective,
            status=PlanStatus.READY.value,
            priority=1,
            dependency_task_ids=[],  # Root has no dependencies
            estimated_effort="medium"
        )
        
        logger.info(f"[PLAN] Created root task: plan_0")
        logger.info(f"[PLAN] Root Status: {root_plan['status']}")
        
        plan = [root_plan]
        
        # If LLM available, use it for sophisticated planning
        if self.llm:
            try:
                sub_plans = self._llm_create_subplan(objective, metadata)
                plan.extend(sub_plans)
                logger.info(f"LLM-generated plan with {len(sub_plans)} sub-tasks")
            except Exception as e:
                logger.warning(f"LLM planning failed, using heuristic: {e}")
                plan.extend(self._heuristic_create_subplan(objective))
        else:
            plan.extend(self._heuristic_create_subplan(objective))
        
        logger.info(f"Created plan with {len(plan)} nodes, max depth: {max(p['depth'] for p in plan)}")
        return plan
    
    
    def _heuristic_create_subplan(self, objective: str) -> List[PlanNode]:
        """
        Create sub-plan using heuristic rules (no LLM).
        
        Detects keywords in objective to suggest tasks with graph-based dependencies:
        - "search" / "find" → web_search task
        - "extract" / "read" → pdf/ocr task
        - "analyze" / "calculate" → excel task
        - "compare" / "correlate" → analysis task depends on preceding tasks
        """
        logger.info("[PLAN] Using HEURISTIC planning strategy")
        logger.info("-" * 80)
        
        sub_plans = []
        objective_lower = objective.lower()
        
        task_id_counter = 1
        priority_counter = 1
        created_task_ids = ["plan_0"]  # Root is always created
        
        # Rule 1: Search/Find keywords
        if any(kw in objective_lower for kw in ["search", "find", "look for", "get", "fetch", "online"]):
            task_id_counter += 1
            task_id = f"plan_{task_id_counter}"
            logger.info(f"[PLAN] ✓ DETECTED: Search/Find pattern")
            logger.info(f"[PLAN]   Creating: {task_id} - Conduct web search and gather online evidence")
            sub_plans.append(PlanNode(
                task_id=task_id,
                parent_id="plan_0",
                depth=1,
                description="Conduct web search and gather online evidence",
                status=PlanStatus.PENDING.value,
                priority=priority_counter,
                dependency_task_ids=[],  # No cross-task dependencies
                estimated_effort="low"
            ))
            created_task_ids.append(task_id)
            priority_counter += 1
        
        # Rule 2: Extract/Read keywords
        if any(kw in objective_lower for kw in ["extract", "read", "parse", "document", "pdf", "file"]):
            task_id_counter += 1
            task_id = f"plan_{task_id_counter}"
            logger.info(f"[PLAN] ✓ DETECTED: Extract/Read pattern")
            logger.info(f"[PLAN]   Creating: {task_id} - Extract and parse document content")
            sub_plans.append(PlanNode(
                task_id=task_id,
                parent_id="plan_0",
                depth=1,
                description="Extract and parse document content",
                status=PlanStatus.PENDING.value,
                priority=priority_counter,
                dependency_task_ids=[],  # No cross-task dependencies
                estimated_effort="medium"
            ))
            created_task_ids.append(task_id)
            priority_counter += 1
        
        # Rule 3: Analyze/Calculate keywords
        if any(kw in objective_lower for kw in ["analyze", "calculate", "compute", "data", "spreadsheet", "excel"]):
            task_id_counter += 1
            task_id = f"plan_{task_id_counter}"
            logger.info(f"[PLAN] ✓ DETECTED: Analyze/Calculate pattern")
            logger.info(f"[PLAN]   Creating: {task_id} - Analyze and process data")
            sub_plans.append(PlanNode(
                task_id=task_id,
                parent_id="plan_0",
                depth=1,
                description="Analyze and process data",
                status=PlanStatus.PENDING.value,
                priority=priority_counter,
                dependency_task_ids=[],  # No cross-task dependencies
                estimated_effort="medium"
            ))
            created_task_ids.append(task_id)
            priority_counter += 1
        
        # Rule 4: Synthesis/Comparison (depends on preceding tasks)
        if any(kw in objective_lower for kw in ["compare", "correlate", "synthesize", "combine", "summary", "summarize"]):
            task_id_counter += 1
            task_id = f"plan_{task_id_counter}"
            # This task depends on all previously created tasks (graph-based dependency)
            deps = [t for t in created_task_ids if t != "plan_0"]
            logger.info(f"[PLAN] ✓ DETECTED: Synthesis/Comparison pattern")
            logger.info(f"[PLAN]   Creating: {task_id} - Synthesize and correlate findings")
            logger.info(f"[PLAN]   Dependencies: {deps}")
            sub_plans.append(PlanNode(
                task_id=task_id,
                parent_id="plan_0",
                depth=1,
                description="Synthesize and correlate findings",
                status=PlanStatus.PENDING.value,
                priority=priority_counter,
                dependency_task_ids=deps,  # Graph-of-Thought: depends on all above tasks
                estimated_effort="high"
            ))
        
        logger.info("-" * 80)
        logger.info(f"[PLAN] Heuristic planning created {len(sub_plans)} sub-tasks")
        return sub_plans
    
    
    def _llm_create_subplan(self, objective: str, metadata: Dict[str, Any]) -> List[PlanNode]:
        """
        Use LLM to create sophisticated sub-plan with graph-based dependencies.
        
        Asks LLM to break down objective into structured tasks with:
        - Task descriptions and effort estimates
        - Priority ordering
        - Graph-based cross-task dependencies (not just parent-child)
        """
        if not self.llm:
            return []
        
        logger.info("[PLAN] Using LLM-based planning strategy")
        logger.info("-" * 80)
        
        from langchain_core.messages import HumanMessage, SystemMessage
        import json as json_lib
        
        # Convert metadata to JSON string
        metadata_str = json_lib.dumps(metadata, indent=2)
        
        prompt = f"""
        Create a detailed hierarchical task plan for this objective:
        
        Objective: {objective}
        
        Context: {metadata_str}
        
        Return a JSON array of tasks with this structure:
        [
            {{
                "task_id": "plan_1",
                "description": "Task description",
                "depth": 1,
                "priority": 1,
                "dependency_task_ids": ["plan_0"],  # Task IDs this depends on (cross-branch)
                "estimated_effort": "low|medium|high"
            }},
            ...
        ]
        
        Graph-of-Thought Planning:
        - Each task can depend on multiple OTHER tasks (not just parent)
        - For example: plan_1 (search), plan_2 (extract), plan_3 (synthesis) depends on both plan_1 and plan_2
        - Use dependency_task_ids to create non-linear workflows
        - All tasks have parent_id="plan_0" (root)
        
        Create 3-7 sub-tasks that break down the objective logically.
        Consider using these agents: web_search, pdf, excel, ocr
        """
        
        try:
            logger.info(f"[PLAN] Calling LLM to generate sophisticated task plan...")
            
            # Apply rate limiting before LLM call
            wait_time = global_rate_limiter.wait()
            if wait_time > 0:
                logger.debug(f"Rate limiter delayed LLM request by {wait_time:.2f}s")
            
            response = self.llm.invoke([
                SystemMessage(content="You are a task planning expert. Return only valid JSON."),
                HumanMessage(content=prompt)
            ])
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON from response
            import json
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                tasks_data = json.loads(json_str)
                
                logger.info(f"[PLAN] LLM returned {len(tasks_data)} task specifications")
                
                # Convert to PlanNode objects
                plan_nodes = []
                for i, task in enumerate(tasks_data):
                    task_id = f"plan_{i+1}"
                    node = PlanNode(
                        task_id=task_id,
                        parent_id="plan_0",
                        depth=1,
                        description=task.get('description', ''),
                        status=PlanStatus.PENDING.value,
                        priority=task.get('priority', i+1),
                        dependency_task_ids=task.get('dependency_task_ids', []),
                        estimated_effort=task.get('estimated_effort', 'medium')
                    )
                    plan_nodes.append(node)
                    logger.info(f"[PLAN]   {task_id} - {task.get('description', '')[:60]}...")
                    if task.get('dependency_task_ids'):
                        logger.info(f"[PLAN]     Dependencies: {task.get('dependency_task_ids')}")
                
                logger.info("-" * 80)
                logger.info(f"[PLAN] LLM-based planning created {len(plan_nodes)} sub-tasks with graph dependencies")
                return plan_nodes
        except Exception as e:
            logger.warning(f"[PLAN] LLM plan creation failed: {e}")
            logger.info("-" * 80)
        
        return []
    
    
    # ========================================================================
    # BLACKBOARD OPERATIONS
    # ========================================================================
    
    def post_finding(
        self,
        blackboard: List[BlackboardEntry],
        entry_type: BlackboardType | str,
        content: Dict[str, Any],
        source_agent: str,
        source_task_id: str,
        relevant_to: Optional[List[str]] = None
    ) -> List[BlackboardEntry]:
        """
        Post a finding to the blackboard.
        
        Any agent can post findings that other agents can discover and use.
        
        Args:
            blackboard: Current blackboard entries
            entry_type: Type of finding (web_evidence, data_point, etc.)
            content: The actual finding data
            source_agent: Which agent posted this
            source_task_id: Which task produced this
            relevant_to: Task IDs that might find this useful
            
        Returns:
            Updated blackboard
        """
        entry = BlackboardEntry(
            entry_type=str(entry_type),
            source_agent=source_agent,
            source_task_id=source_task_id,
            content=content,
            timestamp=datetime.now().isoformat(),
            relevant_to=relevant_to or []
        )
        
        blackboard = blackboard.copy() if blackboard else []
        blackboard.append(entry)
        
        logger.debug(f"[BLACKBOARD] Posting finding to blackboard")
        logger.debug(f"[BLACKBOARD]   Type: {entry_type}")
        logger.debug(f"[BLACKBOARD]   Source: {source_agent} (task {source_task_id})")
        logger.debug(f"[BLACKBOARD]   Content size: {len(str(content))} chars")
        logger.debug(f"[BLACKBOARD]   Relevant to: {len(relevant_to or [])} task(s)")
        logger.debug(f"[BLACKBOARD]   Blackboard size: {len(blackboard)} entries")
        
        logger.info(
            f"Posted {entry_type} to blackboard from {source_agent} "
            f"(relevant to {len(relevant_to or [])} tasks)"
        )
        
        return blackboard
    
    
    def post_nested_finding(
        self,
        blackboard: List[BlackboardEntry],
        entry_type: BlackboardType | str,
        content: Dict[str, Any],
        source_agent: str,
        source_task_id: str,
        parent_task_id: str,
        depth_level: int,
        relevant_to: Optional[List[str]] = None
    ) -> List[BlackboardEntry]:
        """
        Post a finding to the blackboard with parent task tagging for hierarchies.
        
        Used when a sub-task posts findings that should be organized under
        its parent task in the hierarchy.
        
        Args:
            blackboard: Current blackboard entries
            entry_type: Type of finding
            content: The actual finding data
            source_agent: Which agent posted this
            source_task_id: Which task produced this
            parent_task_id: Parent task ID for hierarchical organization
            depth_level: Depth in the hierarchy
            relevant_to: Task IDs that might find this useful
            
        Returns:
            Updated blackboard with nested entry
        """
        entry = BlackboardEntry(
            entry_type=str(entry_type),
            source_agent=source_agent,
            source_task_id=source_task_id,
            parent_task_id=parent_task_id,  # NEW: Organize under parent
            content=content,
            timestamp=datetime.now().isoformat(),
            relevant_to=relevant_to or [],
            depth_level=depth_level  # NEW: Track depth
        )
        
        blackboard = blackboard.copy() if blackboard else []
        blackboard.append(entry)
        
        logger.debug(f"[BLACKBOARD] Posting nested finding to blackboard")
        logger.debug(f"[BLACKBOARD]   Type: {entry_type}")
        logger.debug(f"[BLACKBOARD]   Source: {source_agent} (task {source_task_id})")
        logger.debug(f"[BLACKBOARD]   Parent: {parent_task_id}")
        logger.debug(f"[BLACKBOARD]   Depth: {depth_level}")
        logger.debug(f"[BLACKBOARD]   Content size: {len(str(content))} chars")
        logger.debug(f"[BLACKBOARD]   Blackboard size: {len(blackboard)} entries")
        
        logger.info(
            f"Posted nested {entry_type} to blackboard (depth {depth_level}) "
            f"under parent {parent_task_id} from {source_agent}"
        )
        
        return blackboard
    
    
    def query_blackboard_by_parent(
        self,
        blackboard: List[BlackboardEntry],
        parent_task_id: str
    ) -> List[BlackboardEntry]:
        """
        Query blackboard for all findings under a specific parent task.
        
        Useful for retrieving organized hierarchical findings from a parent task.
        
        Args:
            blackboard: Current blackboard entries
            parent_task_id: Parent task to query
            
        Returns:
            All findings nested under the parent task
        """
        results = blackboard or []
        results = [e for e in results if e.get('parent_task_id') == parent_task_id]
        
        logger.debug(f"Found {len(results)} findings under parent {parent_task_id}")
        return results
    
    
    def query_blackboard_by_depth(
        self,
        blackboard: List[BlackboardEntry],
        depth_level: int
    ) -> List[BlackboardEntry]:
        """
        Query blackboard for findings at a specific depth level.
        
        Useful for understanding work done at each hierarchical level.
        
        Args:
            blackboard: Current blackboard entries
            depth_level: Depth to query
            
        Returns:
            Findings from the specified depth
        """
        results = blackboard or []
        results = [e for e in results if e.get('depth_level', 0) == depth_level]
        
        logger.debug(f"Found {len(results)} findings at depth {depth_level}")
        return results
    
    
    def query_blackboard(
        self,
        blackboard: List[BlackboardEntry],
        entry_type: Optional[str] = None,
        source_agent: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> List[BlackboardEntry]:
        """
        Query blackboard for relevant findings.
        
        Args:
            blackboard: Current blackboard entries
            entry_type: Filter by type (web_evidence, data_point, etc.)
            source_agent: Filter by source agent
            task_id: Filter by tasks that marked this as relevant
            
        Returns:
            Filtered blackboard entries
        """
        logger.debug(f"[BLACKBOARD] Querying blackboard")
        logger.debug(f"[BLACKBOARD]   Total entries: {len(blackboard or [])}")
        logger.debug(f"[BLACKBOARD]   Filters: type={entry_type}, agent={source_agent}, task={task_id}")
        
        results = blackboard or []
        
        if entry_type:
            results = [e for e in results if e['entry_type'] == entry_type]
            logger.debug(f"[BLACKBOARD]   After type filter: {len(results)} entries")
        
        if source_agent:
            results = [e for e in results if e['source_agent'] == source_agent]
            logger.debug(f"[BLACKBOARD]   After agent filter: {len(results)} entries")
        
        if task_id:
            results = [e for e in results if task_id in e.get('relevant_to', [])]
            logger.debug(f"[BLACKBOARD]   After task filter: {len(results)} entries")
        
        logger.debug(f"[BLACKBOARD] ✓ Query complete: {len(results)} result(s)")
        
        logger.debug(
            f"Blackboard query: type={entry_type}, agent={source_agent}, "
            f"task={task_id} → {len(results)} results"
        )
        
        return results
    
    
    # ========================================================================
    # PLAN NAVIGATION
    # ========================================================================
    
    def get_next_ready_task(self, plan: List[PlanNode], blackboard: List[BlackboardEntry]) -> Optional[PlanNode]:
        """
        Get next task that is ready to execute.
        
        A task is ready when:
        1. Its status is PENDING
        2. All its dependencies are COMPLETED in blackboard/history
        3. Required blackboard data is available (if specified)
        
        Args:
            plan: The task plan
            blackboard: Current blackboard entries
            
        Returns:
            Next ready task, or None if none are ready
        """
        # Find all completed tasks
        completed_tasks = {e['source_task_id'] for e in blackboard 
                          if e.get('entry_type') == 'completion'}
        
        # Find pending tasks
        pending_tasks = [t for t in plan if t['status'] == PlanStatus.PENDING.value]
        
        # Sort by priority
        pending_tasks.sort(key=lambda t: t['priority'])
        
        # Return first task with all dependencies satisfied
        for task in pending_tasks:
            dependencies = task.get('dependencies', [])
            if all(dep in completed_tasks for dep in dependencies):
                logger.info(f"Next ready task: {task['task_id']} - {task['description'][:60]}...")
                return task
        
        logger.debug("No tasks currently ready to execute")
        return None
    
    
    def mark_task_complete(
        self,
        plan: List[PlanNode],
        task_id: str,
        outcome: Dict[str, Any]
    ) -> List[PlanNode]:
        """
        Mark a task as completed in the plan.
        
        Args:
            plan: The task plan
            task_id: Task to mark complete
            outcome: The result/outcome of the task
            
        Returns:
            Updated plan
        """
        updated_plan = []
        for task in plan:
            if task['task_id'] == task_id:
                task = dict(task)
                task['status'] = PlanStatus.COMPLETED.value
                updated_plan.append(task)
            else:
                updated_plan.append(task)
        
        logger.info(f"Marked task {task_id} as completed")
        return updated_plan
    
    
    # ========================================================================
    # HISTORY TRACKING
    # ========================================================================
    
    def record_step(
        self,
        history: List[HistoryEntry],
        step_name: str,
        agent: str,
        task_id: str,
        outcome: Dict[str, Any],
        duration_seconds: float = 0.0,
        error: Optional[str] = None
    ) -> List[HistoryEntry]:
        """
        Record a completed step in history.
        
        Args:
            history: Current history entries
            step_name: Name of the step
            agent: Which agent executed
            task_id: Which task
            outcome: Result of the step
            duration_seconds: How long it took
            error: Any error (if failed)
            
        Returns:
            Updated history
        """
        entry = HistoryEntry(
            timestamp=datetime.now().isoformat(),
            step_name=step_name,
            agent=agent,
            task_id=task_id,
            outcome=outcome,
            duration_seconds=duration_seconds,
            error=error
        )
        
        history = history.copy() if history else []
        history.append(entry)
        
        log_msg = f"Step {step_name} ({agent}) completed in {duration_seconds:.2f}s"
        if error:
            log_msg += f" with error: {error}"
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        return history
    
    
    def get_execution_summary(self, history: List[HistoryEntry]) -> Dict[str, Any]:
        """
        Get summary of execution history.
        
        Returns:
            Summary including total time, step count, success rate, etc.
        """
        if not history:
            return {
                "total_steps": 0,
                "successful_steps": 0,
                "failed_steps": 0,
                "total_duration": 0.0,
                "success_rate": 0.0
            }
        
        successful = len([h for h in history if not h.get('error')])
        failed = len(history) - successful
        total_duration = sum(h.get('duration_seconds', 0) for h in history)
        
        return {
            "total_steps": len(history),
            "successful_steps": successful,
            "failed_steps": failed,
            "total_duration": total_duration,
            "success_rate": successful / len(history) if history else 0.0,
            "first_step": history[0]['timestamp'] if history else None,
            "last_step": history[-1]['timestamp'] if history else None
        }
    
    
    # ========================================================================
    # DEEP HIERARCHY SUPPORT - Context & Depth Management
    # ========================================================================
    
    def extract_context_from_parent(
        self,
        plan: List[PlanNode],
        parent_task_id: str,
        blackboard: List[BlackboardEntry]
    ) -> Dict[str, Any]:
        """
        Extract key context from parent task for child execution.
        
        Gathers all findings posted to a parent task and creates a summary
        to guide child task execution. This enables child tasks to inherit
        the context and understanding from parent work.
        
        Args:
            plan: The task plan
            parent_task_id: Parent task to extract context from
            blackboard: Current blackboard entries
            
        Returns:
            Context dictionary with parent's findings and learnings
        """
        # Get parent task details
        parent_task = next((t for t in plan if t['task_id'] == parent_task_id), None)
        if not parent_task:
            return {}
        
        # Get all findings nested under parent
        parent_findings = self.query_blackboard_by_parent(blackboard, parent_task_id)
        
        # Extract content from findings
        extracted_content = {}
        for finding in parent_findings:
            entry_type = finding.get('entry_type', 'unknown')
            if entry_type not in extracted_content:
                extracted_content[entry_type] = []
            extracted_content[entry_type].append(finding.get('content', {}))
        
        # Create context summary
        context = {
            "parent_task_id": parent_task_id,
            "parent_description": parent_task.get('description', ''),
            "parent_status": parent_task.get('status', ''),
            "findings_by_type": extracted_content,
            "total_findings": len(parent_findings),
            "extraction_time": datetime.now().isoformat(),
        }
        
        logger.info(
            f"Extracted context from parent {parent_task_id}: "
            f"{len(parent_findings)} findings, {len(extracted_content)} types"
        )
        
        return context
    
    
    def check_depth_limit(self, current_depth: int, depth_limit: int) -> bool:
        """
        Check if current depth exceeds the limit.
        
        Prevents infinite recursion in hierarchical decomposition.
        
        Args:
            current_depth: Current depth in hierarchy
            depth_limit: Maximum allowed depth
            
        Returns:
            True if depth is within limit, False if exceeded
        """
        is_within_limit = current_depth < depth_limit
        
        if not is_within_limit:
            logger.warning(
                f"Depth limit exceeded: current={current_depth}, limit={depth_limit}"
            )
        
        return is_within_limit
    
    
    def add_context_to_plan_node(
        self,
        plan: List[PlanNode],
        task_id: str,
        context: Dict[str, Any]
    ) -> List[PlanNode]:
        """
        Add context to a plan node for hierarchical decomposition guidance.
        
        Args:
            plan: The task plan
            task_id: Task to update
            context: Context to add
            
        Returns:
            Updated plan with context
        """
        updated_plan = []
        for task in plan:
            if task['task_id'] == task_id:
                task = dict(task)
                task['context_summary'] = context
                updated_plan.append(task)
            else:
                updated_plan.append(task)
        
        logger.debug(f"Added context to task {task_id}")
        return updated_plan
    
    
    def establish_parent_child_relationship(
        self,
        plan: List[PlanNode],
        parent_id: str,
        child_ids: List[str]
    ) -> List[PlanNode]:
        """
        Establish parent-child relationships in the plan hierarchy.
        
        Updates parent task with list of child IDs for easy navigation.
        
        Args:
            plan: The task plan
            parent_id: Parent task ID
            child_ids: List of child task IDs
            
        Returns:
            Updated plan with established relationships
        """
        updated_plan = []
        for task in plan:
            if task['task_id'] == parent_id:
                task = dict(task)
                task['child_task_ids'] = child_ids
                updated_plan.append(task)
            else:
                updated_plan.append(task)
        
        logger.debug(f"Established {len(child_ids)} children for parent {parent_id}")
        return updated_plan
    
    
    def get_hierarchy_depth_distribution(
        self,
        plan: List[PlanNode]
    ) -> Dict[int, int]:
        """
        Get distribution of tasks across hierarchy depths.
        
        Useful for understanding task plan structure.
        
        Returns:
            Dictionary mapping depth → count of tasks at that depth
        """
        distribution = {}
        for task in plan:
            depth = task.get('depth', 0)
            distribution[depth] = distribution.get(depth, 0) + 1
        
        logger.debug(f"Hierarchy distribution: {distribution}")
        return distribution
    
    
    # ========================================================================
    # REFLECTION & DYNAMIC TASK INJECTION
    # ========================================================================
    
    def evaluate_finding_depth(
        self,
        task_id: str,
        finding_content: Dict[str, Any],
        task_description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate if a completed task's findings are sufficiently deep or require drilling deeper.
        
        Uses LLM to analyze findings and determine if:
        - Result is comprehensive and "deep" (no further analysis needed)
        - Result is "shallow" and requires deeper sub-task analysis
        
        Args:
            task_id: ID of the completed task
            finding_content: The findings/results from the completed task
            task_description: What the task was supposed to do
            metadata: Additional context for evaluation
            
        Returns:
            {
                "task_id": str,
                "is_shallow": bool,  # True = needs deeper analysis
                "depth_assessment": str,  # "deep" | "moderate" | "shallow"
                "confidence": float,  # 0.0-1.0
                "reasoning": str,  # Why this assessment
                "suggested_drill_areas": List[str],  # Areas to investigate deeper
                "recommended_agent_types": List[str]  # web_search, pdf, excel, etc.
            }
        """
        if not self.llm:
            logger.warning("LLM not available for depth evaluation, defaulting to moderate")
            return {
                "task_id": task_id,
                "is_shallow": False,
                "depth_assessment": "unknown",
                "confidence": 0.3,
                "reasoning": "No LLM available for evaluation",
                "suggested_drill_areas": [],
                "recommended_agent_types": []
            }
        
        from langchain_core.messages import HumanMessage, SystemMessage
        import json
        
        # Format finding content for evaluation
        finding_str = json.dumps(finding_content, indent=2)[:1000]  # Limit size
        
        evaluation_prompt = f"""
        A task has completed and produced findings. Evaluate the depth of analysis:
        
        Task ID: {task_id}
        Task Description: {task_description}
        
        Findings Summary (first 1000 chars):
        {finding_str}
        
        Assess if this finding is:
        - "deep": Comprehensive, well-analyzed, sufficient for conclusions
        - "moderate": Adequate but could use more detail
        - "shallow": Insufficient, needs deeper investigation
        
        Return a JSON object:
        {{
            "depth_assessment": "deep|moderate|shallow",
            "confidence": 0.0-1.0,
            "reasoning": "explanation of why",
            "suggested_drill_areas": ["area 1", "area 2", ...],
            "recommended_agent_types": ["web_search", "pdf", "excel", ...]
        }}
        
        Be specific about areas that need deeper investigation.
        """
        
        try:
            # Apply rate limiting before LLM call
            wait_time = global_rate_limiter.wait()
            if wait_time > 0:
                logger.debug(f"Rate limiter delayed LLM request by {wait_time:.2f}s")
            
            response = self.llm.invoke([
                SystemMessage(content="You are an analysis quality evaluator. Return only valid JSON."),
                HumanMessage(content=evaluation_prompt)
            ])
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                evaluation = json.loads(json_str)
                
                result = {
                    "task_id": task_id,
                    "is_shallow": evaluation.get("depth_assessment") in ["shallow", "moderate"],
                    "depth_assessment": evaluation.get("depth_assessment", "unknown"),
                    "confidence": float(evaluation.get("confidence", 0.5)),
                    "reasoning": evaluation.get("reasoning", ""),
                    "suggested_drill_areas": evaluation.get("suggested_drill_areas", []),
                    "recommended_agent_types": evaluation.get("recommended_agent_types", [])
                }
                
                logger.info(
                    f"Task {task_id} depth evaluation: {result['depth_assessment']} "
                    f"(confidence: {result['confidence']:.2f})"
                )
                
                if result["is_shallow"]:
                    logger.info(
                        f"Task {task_id} requires deeper analysis in areas: "
                        f"{', '.join(result['suggested_drill_areas'])}"
                    )
                
                return result
            else:
                # Could not parse JSON from LLM response
                logger.warning(f"Could not parse JSON from LLM response for task {task_id}")
                return {
                    "task_id": task_id,
                    "is_shallow": False,
                    "depth_assessment": "unknown",
                    "confidence": 0.0,
                    "reasoning": "Could not parse LLM response as JSON",
                    "suggested_drill_areas": [],
                    "recommended_agent_types": []
                }
            
        except Exception as e:
            logger.error(f"Failed to evaluate finding depth: {e}")
            # Fallback - return result with error info
            return {
                "task_id": task_id,
                "is_shallow": False,
                "depth_assessment": "unknown",
                "confidence": 0.0,
                "reasoning": f"Evaluation failed: {str(e)}",
                "suggested_drill_areas": [],
                "recommended_agent_types": []
            }
    
    
    def create_deep_dive_subtasks(
        self,
        plan: List[PlanNode],
        parent_task_id: str,
        evaluation: Dict[str, Any],
        parent_description: str = ""
    ) -> List[PlanNode]:
        """
        Dynamically create 2-3 Level 3 sub-tasks to drill deeper into a finding.
        
        When a Level 2 task is completed and evaluation shows it's shallow,
        this creates Level 3 (depth=3) tasks focused on the specific areas
        identified in the evaluation.
        
        Args:
            plan: Current plan
            parent_task_id: The Level 2 task that needs drilling
            evaluation: The depth evaluation results
            parent_description: Description of parent task
            
        Returns:
            Extended plan with new deep dive sub-tasks
        """
        if not evaluation.get("is_shallow"):
            logger.debug(f"Task {parent_task_id} evaluation shows sufficient depth, no drilling needed")
            return plan
        
        drill_areas = evaluation.get("suggested_drill_areas", [])
        agent_types = evaluation.get("recommended_agent_types", [])
        
        if not drill_areas:
            logger.warning(f"No drill areas suggested for {parent_task_id}, skipping deep dive")
            return plan
        
        # Get parent task for context
        parent_task = next((t for t in plan if t['task_id'] == parent_task_id), None)
        if not parent_task:
            logger.error(f"Parent task {parent_task_id} not found in plan")
            return plan
        
        # Create 2-3 deep dive tasks based on suggested areas
        new_tasks = []
        num_tasks = min(len(drill_areas), 3)  # Create 2-3 tasks max
        
        # Find next available task ID
        existing_ids = {t['task_id'] for t in plan}
        next_id = max(
            [int(t['task_id'].split('_')[-1]) for t in plan if '_' in t['task_id']],
            default=0
        ) + 1
        
        for i, drill_area in enumerate(drill_areas[:num_tasks]):
            task_id = f"plan_{next_id + i}"
            
            # Determine which agent to use based on recommendations
            agent_type = agent_types[i % len(agent_types)] if agent_types else "web_search"
            
            # Map agent types to task descriptions
            agent_task_map = {
                "web_search": f"Deep search for details on {drill_area}",
                "pdf": f"Extract from documents about {drill_area}",
                "excel": f"Analyze data related to {drill_area}",
                "ocr": f"Extract information about {drill_area} from images",
                "analysis": f"Detailed analysis of {drill_area}"
            }
            
            task_description = agent_task_map.get(
                agent_type,
                f"Deep dive investigation: {drill_area}"
            )
            
            # Create context-enriched description
            enriched_description = f"{task_description} (Parent: {parent_description[:50]}... Drill area: {drill_area})"
            
            deep_dive_task = PlanNode(
                task_id=task_id,
                parent_id=parent_task_id,
                depth=3,
                description=enriched_description,
                status=PlanStatus.PENDING.value,
                priority=1,  # High priority
                dependency_task_ids=[parent_task_id],  # Depends on parent completion
                estimated_effort="medium"
            )
            
            new_tasks.append(deep_dive_task)
            logger.info(
                f"Created deep dive task {task_id} (depth=3) under {parent_task_id}: {task_description}"
            )
        
        # Add new tasks to plan
        updated_plan = list(plan) + new_tasks
        
        logger.info(
            f"Injected {len(new_tasks)} Level 3 deep dive tasks under {parent_task_id}"
        )
        
        return updated_plan
    
    
    def reflect_and_inject(
        self,
        state: AgentState,
        task_id: str,
        outcome: Dict[str, Any]
    ) -> Tuple[List[PlanNode], bool]:
        """
        Reflection step: After a task completes, evaluate findings and inject deep dive tasks if needed.
        
        Workflow:
        1. Evaluate if findings are sufficiently deep
        2. If shallow, create Level 3 sub-tasks to drill deeper
        3. Update plan with new tasks
        4. Return indication of whether new tasks were injected
        
        Args:
            state: Current agent state
            task_id: The task that just completed
            outcome: The results/findings from the task
            
        Returns:
            Tuple of (updated_plan, new_tasks_injected)
        """
        plan = state.get('plan', [])
        task = next((t for t in plan if t['task_id'] == task_id), None)
        
        if not task:
            logger.error(f"Task {task_id} not found during reflection")
            return plan, False
        
        # Evaluate finding depth
        evaluation = self.evaluate_finding_depth(
            task_id=task_id,
            finding_content=outcome,
            task_description=task.get('description', ''),
            metadata=state.get('metadata', {})
        )
        
        # Determine if deep dive is needed
        if not evaluation.get("is_shallow"):
            logger.debug(f"Task {task_id} findings are sufficiently deep, no further drilling needed")
            return plan, False
        
        # Create deep dive subtasks
        updated_plan = self.create_deep_dive_subtasks(
            plan,
            task_id,
            evaluation,
            task.get('description', '')
        )
        
        injected = len(updated_plan) > len(plan)
        if injected:
            logger.info(f"Successfully injected {len(updated_plan) - len(plan)} deep dive tasks")
        
        return updated_plan, injected
    
    
    def get_ready_tasks(
        self,
        plan: List[PlanNode],
        blackboard: Optional[List[BlackboardEntry]] = None
    ) -> List[PlanNode]:
        """
        Get all tasks that are ready to execute (dependencies satisfied, status=READY).
        
        **Graph-of-Thought Enhanced**: Checks both tree-based (parent) and graph-based 
        (dependency_task_ids) dependencies before marking a task as ready.
        
        A task is ready when:
        1. Status is PENDING or READY (not BLOCKED, EXECUTING, FAILED)
        2. ALL dependencies in dependency_task_ids list are COMPLETED
        3. Parent task is COMPLETED (if parent_id exists)
        
        Prioritizes deep dive tasks (depth=3) created by reflection step.
        
        Args:
            plan: The task plan
            blackboard: Current blackboard findings (unused but kept for compatibility)
            
        Returns:
            List of ready tasks, prioritized by: depth (higher first), then priority
        """
        # Build a map of task IDs to their tasks for quick lookup
        task_map = {t.get('task_id'): t for t in plan}
        
        ready_tasks = []
        
        for task in plan:
            # Only consider tasks that are PENDING or READY
            task_status = task.get('status', PlanStatus.PENDING.value)
            if task_status not in [PlanStatus.PENDING.value, PlanStatus.READY.value]:
                continue
            
            # Check if parent task is completed (if parent exists)
            parent_id = task.get('parent_id')
            if parent_id:
                parent_task = task_map.get(parent_id)
                if not parent_task or parent_task.get('status') != PlanStatus.COMPLETED.value:
                    logger.debug(
                        f"Task {task.get('task_id')} blocked: parent {parent_id} not completed"
                    )
                    continue
            
            # Check if all graph dependencies are completed (Graph-of-Thought)
            dependency_ids = task.get('dependency_task_ids', [])
            if not dependency_ids:
                # No dependencies, task is ready
                ready_tasks.append(task)
            else:
                # All dependencies must be COMPLETED
                all_deps_done = all(
                    task_map.get(dep_id, {}).get('status') == PlanStatus.COMPLETED.value
                    for dep_id in dependency_ids
                )
                
                if all_deps_done:
                    logger.debug(
                        f"Task {task.get('task_id')} ready: all {len(dependency_ids)} "
                        f"dependencies completed"
                    )
                    ready_tasks.append(task)
                else:
                    incomplete_deps = [
                        dep_id for dep_id in dependency_ids
                        if task_map.get(dep_id, {}).get('status') != PlanStatus.COMPLETED.value
                    ]
                    logger.debug(
                        f"Task {task.get('task_id')} blocked: waiting on "
                        f"{len(incomplete_deps)} dependencies: {incomplete_deps}"
                    )
        
        # Sort by depth (higher depth = deep dive tasks = higher priority)
        # then by priority field
        ready_tasks.sort(
            key=lambda t: (-t.get('depth', 0), t.get('priority', 999)),
            reverse=False
        )
        
        logger.debug(
            f"Found {len(ready_tasks)} ready tasks out of {len(plan)} total "
            f"(deep dive and non-blocked first)"
        )
        return ready_tasks
    
    
    def determine_next_node(
        self,
        state: AgentState,
        current_node: str
    ) -> str:
        """
        Determine next node to execute based on plan, blackboard, and current state.
        
        **ENHANCED**: Prioritizes deep dive tasks (depth=3) injected by reflection step.
        
        Routing logic:
        1. Check for ready deep dive tasks (depth=3) - execute these FIRST
        2. Check for other ready Level 1/2 tasks
        3. Route based on task type (web_search, pdf, excel, etc.)
        
        This enables the analysis-after-breakdown pattern where shallow findings
        trigger automatic creation and prioritization of deeper investigation tasks.
        
        Args:
            state: Current agent state
            current_node: Name of current node
            
        Returns:
            Name of next node to execute
        """
        plan = state.get('plan', [])
        blackboard = state.get('blackboard', [])
        
        # Get ready tasks, prioritizing deep dive tasks
        ready_tasks = self.get_ready_tasks(plan, blackboard)
        
        # If there are ready deep dive tasks (depth=3), prioritize them
        deep_dive_tasks = [t for t in ready_tasks if t.get('depth') == 3]
        if deep_dive_tasks:
            active_task = deep_dive_tasks[0]
            logger.info(
                f"Prioritizing deep dive task {active_task['task_id']}: "
                f"{active_task['description'][:60]}..."
            )
        else:
            # Fall back to next ready task
            if not ready_tasks:
                return "select_task"
            active_task = ready_tasks[0]
        
        # Route based on task description/type
        task_desc = active_task.get('description', '').lower()
        
        if 'search' in task_desc or 'web' in task_desc:
            return "execute_web_search_task"
        elif 'pdf' in task_desc or 'document' in task_desc:
            return "execute_pdf_task"
        elif 'excel' in task_desc or 'spreadsheet' in task_desc or 'data' in task_desc:
            return "execute_excel_task"
        elif 'image' in task_desc or 'ocr' in task_desc or 'text extraction' in task_desc:
            return "execute_ocr_task"
        else:
            return "execute_task"    
    def query_file_pointers_by_agent(self, state: AgentState, agent_name: str) -> List[Dict[str, Any]]:
        """
        Find all files generated by a specific agent.
        
        Enables cross-agent discovery: "What files did WebSearch create?"
        
        Example use case:
        - Excel agent asks: What CSVs did WebSearch create?
        - Returns: [{"file_path": "data/results.csv", "file_type": "csv"}]
        
        Args:
            state: Current agent state with blackboard
            agent_name: Name of agent (pdf, excel, ocr, web_search)
            
        Returns:
            List of file pointers (dicts) created by that agent
        """
        blackboard = state.get('blackboard', [])
        file_pointers = []
        
        for entry in blackboard:
            # Check if this is a file pointer entry from the target agent
            if (entry.get('entry_type', '').startswith('file_pointer_') and 
                entry.get('source_agent') == agent_name):
                
                content = entry.get('content', {})
                file_pointers.append({
                    'file_path': content.get('file_path'),
                    'file_type': content.get('file_type'),
                    'source_task_id': entry.get('source_task_id'),
                    'timestamp': entry.get('timestamp'),
                    'accessible': content.get('accessible', True)
                })
        
        logger.debug(
            f"Found {len(file_pointers)} file pointers from agent '{agent_name}'"
        )
        return file_pointers
    
    def query_file_pointers_by_type(self, state: AgentState, file_type: str) -> List[Dict[str, Any]]:
        """
        Find all files of a specific type in the system.
        
        Enables type-based discovery: "What CSV files are available?"
        
        Example use case:
        - Excel agent needs: Find all CSV files
        - Returns: [{"file_path": "...", "source_agent": "web_search", ...}, ...]
        
        Args:
            state: Current agent state with blackboard
            file_type: Type of file to find (csv, json, pdf, image, text, etc.)
            
        Returns:
            List of file pointers matching the file type
        """
        blackboard = state.get('blackboard', [])
        file_pointers = []
        
        for entry in blackboard:
            # Check if this is a file pointer with matching type
            if entry.get('entry_type') == f'file_pointer_{file_type}':
                content = entry.get('content', {})
                file_pointers.append({
                    'file_path': content.get('file_path'),
                    'file_type': content.get('file_type'),
                    'created_by_agent': entry.get('source_agent'),
                    'source_task_id': entry.get('source_task_id'),
                    'timestamp': entry.get('timestamp')
                })
        
        logger.debug(
            f"Found {len(file_pointers)} file pointers of type '{file_type}'"
        )
        return file_pointers
    
    def query_chain_execution_status(self, state: AgentState) -> Dict[str, Any]:
        """
        Check the current chain execution status from blackboard.
        
        Returns metadata about what agents are chained together.
        
        Use case:
        - Understand workflow: "PDF→OCR→Excel" is active chain
        - Get files from previous agent in chain
        - Track execution path
        
        Args:
            state: Current agent state with blackboard
            
        Returns:
            Dict with chain_agents list and current execution status
        """
        blackboard = state.get('blackboard', [])
        current_analysis = state.get('current_analysis', {})
        
        # Get latest entry for chain metadata
        chain_agents = []
        if blackboard:
            latest_entry = blackboard[-1]
            chain_agents = latest_entry.get('chain_next_agents', [])
        
        return {
            'chain_agents': chain_agents,
            'current_agent': current_analysis.get('source_agent', 'unknown'),
            'files_in_chain': self._get_chain_files(blackboard),
            'total_chain_depth': len(chain_agents)
        }
    
    def _get_chain_files(self, blackboard: List[BlackboardEntry]) -> List[str]:
        """
        Get list of files being passed through the current chain.
        
        Args:
            blackboard: Current blackboard entries
            
        Returns:
            List of file paths in the chain
        """
        chain_files = []
        for entry in reversed(blackboard):
            # Stop if we hit an entry without chain metadata
            if not entry.get('chain_next_agents'):
                break
            
            source_file = entry.get('source_file_path')
            if source_file:
                chain_files.append(source_file)
        
        return chain_files