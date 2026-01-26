"""
Workflow module - LangGraph workflow construction and management
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from task_manager.models import AgentState, TaskStatus, Task
from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowBuilder:
    """
    Builds and manages the LangGraph workflow for recursive task orchestration.
    
    This class encapsulates all workflow graph construction logic,
    making the agent class cleaner and more maintainable.
    """
    
    def __init__(self, agent):
        """
        Initialize the workflow builder with reference to agent methods.
        
        Args:
            agent: The TaskManagerAgent instance
        """
        self.agent = agent
    
    def build(self) -> StateGraph:
        """
        Build the LangGraph workflow with all nodes and edges.
        
        Workflow:
        1. Initialize → Create root task
        2. Select Task → Pick next pending task
        3. Analyze Task → Decide: breakdown vs execute
        4. Breakdown → Create subtasks (cycles back to Select)
        5. Execute → Perform actual work
        6. Aggregate → Collect results
        7. Synthesize → Multi-level research synthesis with conflict detection
        8. Agentic Debate → Consensus validation if contradiction score > 0.7 (NEW)
        9. Check Completion → Continue or End
        
        Supports chain execution for cross-agent workflows:
        - PDF agent finds images → OCR agent runs directly (no select_task)
        - Web search creates CSV → Excel agent processes immediately
        - OCR extracts table → Excel analyzes the data
        - All completed tasks → Synthesis analyzes blackboard for contradictions
        - High contradictions → Agentic Debate spawns Fact-Checker & Lead Researcher personas
        
        Returns:
            Configured StateGraph instance
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("initialize", self.agent._initialize)
        workflow.add_node("select_task", self.agent._select_next_task)
        workflow.add_node("analyze_task", self.agent._analyze_task)
        workflow.add_node("breakdown_task", self.agent._breakdown_task)
        workflow.add_node("execute_task", self.agent._execute_task)
        workflow.add_node("execute_pdf_task", self.agent._execute_pdf_task)
        workflow.add_node("execute_excel_task", self.agent._execute_excel_task)
        workflow.add_node("execute_ocr_task", self.agent._execute_ocr_task)
        workflow.add_node("execute_web_search_task", self.agent._execute_web_search_task)
        workflow.add_node("execute_code_interpreter_task", self.agent._execute_code_interpreter_task)
        workflow.add_node("execute_data_extraction_task", self.agent._execute_data_extraction_task)
        workflow.add_node("aggregate_results", self.agent._aggregate_results)
        workflow.add_node("synthesize_research", self.agent._synthesize_research)
        workflow.add_node("agentic_debate", self.agent._agentic_debate)  # NEW: debate node for consensus validation
        workflow.add_node("auto_synthesis", self.agent._auto_synthesis)  # NEW: observer node for event-driven analysis
        workflow.add_node("handle_error", self.agent._handle_error)
        workflow.add_node("human_review", self.agent._request_human_review)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Add edges
        workflow.add_edge("initialize", "select_task")
        
        # Conditional routing after select_task
        # If no pending tasks, end the workflow; otherwise continue to analyze
        workflow.add_conditional_edges(
            "select_task",
            self._route_after_select_task,
            {
                "analyze_task": "analyze_task",
                "complete": END
            }
        )
        
        # Conditional routing after analysis
        # Uses enhanced routing that supports chain execution
        workflow.add_conditional_edges(
            "analyze_task",
            self.agent._route_with_chain_execution,  # NEW: enhanced router with chain support
            {
                "breakdown": "breakdown_task",
                "execute_task": "execute_task",
                "execute_pdf_task": "execute_pdf_task",
                "execute_excel_task": "execute_excel_task",
                "execute_ocr_task": "execute_ocr_task",
                "execute_web_search_task": "execute_web_search_task",
                "execute_code_interpreter_task": "execute_code_interpreter_task",
                "execute_data_extraction_task": "execute_data_extraction_task",
                "handle_error": "handle_error",
                "review": "human_review"
            }
        )
        
        # After breakdown, go back to select next task
        workflow.add_edge("breakdown_task", "select_task")
        
        # After execution, aggregate results
        workflow.add_edge("execute_task", "aggregate_results")
        
        # NEW: Chain execution edges - agents can trigger other agents directly
        
        # PDF → OCR: If PDF finds images, OCR analyzes them immediately
        workflow.add_conditional_edges(
            "execute_pdf_task",
            lambda state: "execute_ocr_task" if self._should_chain_to_ocr(state) else "aggregate_results",
            {
                "execute_ocr_task": "execute_ocr_task",
                "aggregate_results": "aggregate_results"
            }
        )
        
        # OCR → Excel: If OCR finds table data, Excel processes it
        # OCR → Auto-Synthesis: If OCR results trigger observer pattern
        workflow.add_conditional_edges(
            "execute_ocr_task",
            self._route_after_ocr_execution,
            {
                "auto_synthesis": "auto_synthesis",
                "execute_excel_task": "execute_excel_task",
                "aggregate_results": "aggregate_results"
            }
        )
        
        # WebSearch → Excel: If WebSearch created CSV, Excel processes it
        # WebSearch → Auto-Synthesis: If web findings trigger observer pattern
        workflow.add_conditional_edges(
            "execute_web_search_task",
            self._route_after_websearch_execution,
            {
                "auto_synthesis": "auto_synthesis",
                "execute_excel_task": "execute_excel_task",
                "aggregate_results": "aggregate_results"
            }
        )
        
        # CodeInterpreter → OCR: If CodeInterpreter created charts/images, OCR analyzes them
        workflow.add_conditional_edges(
            "execute_code_interpreter_task",
            lambda state: "execute_ocr_task" if self._should_chain_to_ocr(state) else "aggregate_results",
            {
                "execute_ocr_task": "execute_ocr_task",
                "aggregate_results": "aggregate_results"
            }
        )
        
        # Excel always goes to aggregate (final processing stage)
        workflow.add_edge("execute_excel_task", "aggregate_results")
        
        # Data extraction always goes to aggregate (context building stage)
        workflow.add_edge("execute_data_extraction_task", "aggregate_results")
        
        # Auto-synthesis always goes to aggregate (observer pattern completes)
        workflow.add_edge("auto_synthesis", "aggregate_results")
        
        # After aggregation, route to synthesis if all tasks complete
        # This enables multi-level research synthesis with conflict detection
        workflow.add_conditional_edges(
            "aggregate_results",
            self._route_to_synthesis_or_continue,
            {
                "synthesize": "synthesize_research",
                "continue": "select_task",
                "complete": END,
                "max_iterations": END
            }
        )
        
        # After synthesis, route to debate if contradiction score > 0.7, else check completion
        # The debate node will determine if contradictions warrant consensus validation
        workflow.add_conditional_edges(
            "synthesize_research",
            self._route_to_debate_or_completion,
            {
                "debate": "agentic_debate",
                "continue": "select_task",
                "complete": END,
                "max_iterations": END
            }
        )
        
        # After debate, check completion
        # Debate node's consensus result is recorded in blackboard and history
        workflow.add_conditional_edges(
            "agentic_debate",
            self.agent._check_completion,
            {
                "continue": "select_task",
                "complete": END,
                "max_iterations": END
            }
        )
        
        # Error handling
        workflow.add_edge("handle_error", "select_task")
        
        # Human review - after review, route based on task status
        # Check if task needs breakdown
        def route_after_human_review(state):
            task_id = state.get('active_task_id')
            if not task_id:
                return "select_task"
            
            task = next((t for t in state.get('tasks', []) if t['id'] == task_id), None)
            if not task:
                return "select_task"
            
            # If analysis suggests breakdown, go to breakdown
            analysis = task.get('result', {})
            if isinstance(analysis, dict) and analysis.get('action') == 'breakdown':
                return "breakdown"
            
            # Otherwise continue with selection
            return "select_task"
        
        workflow.add_conditional_edges(
            "human_review",
            route_after_human_review,
            {
                "breakdown": "breakdown_task",
                "select_task": "select_task"
            }
        )
        
        return workflow
    
    def _route_after_select_task(self, state: "AgentState") -> Literal["analyze_task", "complete"]:
        """
        Determine if workflow should continue to analyze_task or end.
        
        Routes to END when:
        1. No pending tasks remaining (all completed or failed)
        2. No active_task_id was set (nothing left to process)
        
        Args:
            state: Current agent state
            
        Returns:
            "analyze_task" to continue processing, "complete" to end workflow
        """
        active_task_id = state.get('active_task_id', '')
        
        # If select_task found no pending tasks, it returns empty active_task_id
        if not active_task_id:
            # Check if there are any pending tasks we might have missed
            tasks = state.get('tasks', [])
            completed = set(state.get('completed_task_ids', []))
            failed = set(state.get('failed_task_ids', []))
            
            pending_tasks = [
                t for t in tasks 
                if t['id'] not in completed 
                and t['id'] not in failed
                and t.get('status') == TaskStatus.PENDING
            ]
            
            if not pending_tasks:
                logger.info("[ROUTE AFTER SELECT] No pending tasks - routing to END")
                return "complete"
            else:
                # This shouldn't happen, but log it if it does
                logger.warning(f"[ROUTE AFTER SELECT] Found {len(pending_tasks)} pending tasks but no active_task_id set!")
                return "analyze_task"
        
        # Normal case: continue to analyze the selected task
        return "analyze_task"
    
    def _route_to_synthesis_or_continue(self, state: "AgentState") -> Literal["synthesize", "continue", "complete", "max_iterations"]:
        """
        Determine if synthesis should run or continue with regular flow.
        
        Triggers synthesis when:
        1. All tasks are complete
        2. Blackboard has findings to synthesize
        3. Multi-agent research has been conducted
        
        Args:
            state: Current agent state
            
        Returns:
            "synthesize" to run synthesis, else let _check_completion decide
        """
        logger.info("=" * 80)
        logger.info("[WORKFLOW ROUTING] aggregate_results → synthesis/continue/complete check")
        logger.info("=" * 80)
        
        # First check iteration limit and basic completion
        if state['iteration_count'] >= state['max_iterations']:
            logger.warning(f"[ROUTING] Max iterations reached: {state['iteration_count']}/{state['max_iterations']}")
            logger.info("[ROUTING] Decision: max_iterations → END")
            logger.info("=" * 80)
            return "max_iterations"
        
        # Check if all tasks completed
        tasks = state.get('tasks', [])
        completed = set(state.get('completed_task_ids', []))
        failed = set(state.get('failed_task_ids', []))
        
        logger.info(f"[ROUTING] Tasks overview:")
        logger.info(f"[ROUTING]   Total tasks: {len(tasks)}")
        logger.info(f"[ROUTING]   Completed: {len(completed)}")
        logger.info(f"[ROUTING]   Failed: {len(failed)}")
        
        # Get all non-root tasks
        non_root_tasks = [t for t in tasks if t.get('id') != '1']
        all_tasks_done = all(t['id'] in completed or t['id'] in failed for t in non_root_tasks)
        
        logger.info(f"[ROUTING]   Non-root tasks: {len(non_root_tasks)}")
        logger.info(f"[ROUTING]   All tasks done: {all_tasks_done}")
        
        # Check if synthesis should run
        # Only run if we have:
        # 1. Multiple agents have contributed findings (web_search, pdf, excel, ocr)
        # 2. Blackboard has entries to analyze
        blackboard = state.get('blackboard', [])
        logger.info(f"[ROUTING]   Blackboard entries: {len(blackboard)}")
        
        if all_tasks_done and len(blackboard) > 0:
            # Check if we have multi-agent findings
            agents_present = set()
            for entry in blackboard:
                agent = entry.get('source_agent', '')
                if agent and agent != 'synthesis_node':
                    agents_present.add(agent)
            
            logger.info(f"[ROUTING]   Agents present in blackboard: {agents_present}")
            
            # Synthesize if we have multiple agents
            if len(agents_present) >= 2:
                logger.info(f"[ROUTING] ✓ All tasks complete with {len(agents_present)} agents - routing to synthesis")
                logger.info("[ROUTING] Decision: synthesize → synthesize_research node")
                logger.info("=" * 80)
                return "synthesize"
            else:
                logger.info(f"[ROUTING] Only {len(agents_present)} agent(s) - not enough for synthesis")
        else:
            if not all_tasks_done:
                logger.info(f"[ROUTING] Tasks not all done - deferring to _check_completion")
            if len(blackboard) == 0:
                logger.info(f"[ROUTING] No blackboard entries - deferring to _check_completion")
        
        # Otherwise let standard completion check decide
        logger.info("[ROUTING] Calling _check_completion for final decision...")
        completion_result = self.agent._check_completion(state)
        logger.info(f"[ROUTING] Decision: {completion_result}")
        logger.info(f"[ROUTING] Decision type: {type(completion_result)}")
        logger.info("=" * 80)
        
        # Ensure we return a valid literal string, not something else
        if completion_result == "continue":
            logger.info("[ROUTING] Returning 'continue' → will route to select_task")
            return "continue"
        elif completion_result == "complete":
            logger.info("[ROUTING] Returning 'complete' → will route to END")
            return "complete"
        elif completion_result == "max_iterations":
            logger.info("[ROUTING] Returning 'max_iterations' → will route to END")
            return "max_iterations"
        else:
            logger.warning(f"[ROUTING] Unexpected completion result: {completion_result}, defaulting to continue")
            return "continue"
    
    def _route_to_debate_or_completion(self, state: "AgentState") -> Literal["debate", "continue", "complete", "max_iterations"]:
        """
        Determine if Agentic Debate should run after synthesis.
        
        Routes to debate when:
        1. Synthesis found contradictions with score > 0.7
        2. Multiple conflicting data sources need consensus
        3. Requires validation via Fact-Checker & Lead Researcher personas
        
        Args:
            state: Current agent state with synthesis results
            
        Returns:
            "debate" if contradictions warrant consensus validation, else route to completion check
        """
        # First check iteration limit
        if state['iteration_count'] >= state['max_iterations']:
            return "max_iterations"
        
        # Extract synthesis result from blackboard
        blackboard = state.get('blackboard', [])
        synthesis_entry = next(
            (e for e in reversed(blackboard) if e.get('source_agent') == 'synthesis_node'),
            None
        )
        
        if not synthesis_entry:
            # No synthesis, let completion check decide
            completion_result = self.agent._check_completion(state)
            return completion_result  # type: ignore
        
        # Calculate contradiction score
        synthesis_content = synthesis_entry.get('content', {})
        contradictions = synthesis_content.get('contradictions', [])
        
        if not contradictions:
            # No contradictions, let completion check decide
            completion_result = self.agent._check_completion(state)
            return completion_result  # type: ignore
        
        # Calculate score based on severity
        severity_weights = {'CRITICAL': 0.4, 'HIGH': 0.3, 'MEDIUM': 0.2, 'LOW': 0.1}
        total_score = sum(severity_weights.get(c.get('severity', 'LOW'), 0.1) for c in contradictions)
        contradiction_score = min(total_score, 1.0)  # Cap at 1.0
        
        logger.info(f"[ROUTING] Synthesis contradiction score: {contradiction_score:.2f}")
        
        # Route to debate if score exceeds threshold
        if contradiction_score > 0.7:
            logger.info(f"[ROUTING] Score {contradiction_score:.2f} > 0.7 threshold - routing to agentic_debate")
            return "debate"
        else:
            logger.info(f"[ROUTING] Score {contradiction_score:.2f} <= 0.7 threshold - skipping debate, checking completion")
            completion_result = self.agent._check_completion(state)
            return completion_result  # type: ignore
    
    def _should_chain_to_ocr(self, state: "AgentState") -> bool:
        """
        Determine if PDF agent output should automatically chain to OCR execution.
        
        Chain to OCR if:
        1. PDF agent just completed and found images
        2. Blackboard has file pointers for ocr_agent
        3. Current task indicates chain_next_agents includes ocr
        
        This bypasses select_task and runs OCR immediately on the PDF's extracted images.
        
        Args:
            state: Current agent state
            
        Returns:
            True if OCR should execute next
        """
        # Check the most recent task result
        active_task_id = state.get('active_task_id', '')
        tasks = state.get('tasks', [])
        current_task = next((t for t in tasks if t.get('id') == active_task_id), None)
        
        if not current_task:
            return False
        
        # Check if this is a PDF task that just completed
        result = current_task.get('result', {})
        if not isinstance(result, dict):
            return False
        
        findings = result.get('findings', {})
        if findings.get('images_found') or findings.get('extracted_images'):
            logger.debug(f"[CHAIN] PDF task {active_task_id} found images - enabling OCR chain")
            return True
        
        # Check blackboard for explicit chain markers
        blackboard = state.get('blackboard', [])
        for entry in reversed(blackboard[-5:]):  # Check last 5 entries
            chains = entry.get('chain_next_agents', [])
            if 'ocr_agent' in chains or 'execute_ocr_task' in chains:
                logger.debug(f"[CHAIN] Blackboard marks OCR as next agent")
                return True
        
        return False
    
    
    def _should_chain_to_excel(self, state: "AgentState") -> bool:
        """
        Determine if previous agent output should automatically chain to Excel execution.
        
        Chain to Excel if:
        1. OCR agent extracted table data
        2. WebSearch agent created CSV files
        3. Blackboard has file pointers for excel_agent
        4. Current task indicates chain_next_agents includes excel
        
        This enables immediate Excel processing without returning to select_task.
        
        Args:
            state: Current agent state
            
        Returns:
            True if Excel should execute next
        """
        # Check the most recent task result
        active_task_id = state.get('active_task_id', '')
        tasks = state.get('tasks', [])
        current_task = next((t for t in tasks if t.get('id') == active_task_id), None)
        
        if not current_task:
            return False
        
        result = current_task.get('result', {})
        if not isinstance(result, dict):
            return False
        
        # Check for table extraction from OCR
        findings = result.get('findings', {})
        if findings.get('extracted_table') or findings.get('table_found'):
            logger.debug(f"[CHAIN] OCR task {active_task_id} found table - enabling Excel chain")
            return True
        
        # Check for file generation from WebSearch
        output = result.get('output', {})
        if output.get('generated_files', {}).get('csv'):
            logger.debug(f"[CHAIN] WebSearch task {active_task_id} created CSV - enabling Excel chain")
            return True
        
        # Check blackboard for explicit chain markers
        blackboard = state.get('blackboard', [])
        for entry in reversed(blackboard[-5:]):
            chains = entry.get('chain_next_agents', [])
            if 'excel_agent' in chains or 'execute_excel_task' in chains:
                logger.debug(f"[CHAIN] Blackboard marks Excel as next agent")
                return True
        
        return False

    def _route_after_ocr_execution(self, state: "AgentState") -> Literal["auto_synthesis", "execute_excel_task", "aggregate_results"]:
        """
        Observer pattern routing after OCR execution.
        
        Priority routing:
        1. If last_updated_key == 'ocr_results' → auto_synthesis (observer trigger)
        2. If table data extracted → execute_excel_task (chain execution)
        3. Otherwise → aggregate_results (normal flow)
        
        This implements event-driven triggers where OCR results automatically
        trigger synthesis before continuing with normal workflow.
        
        Args:
            state: Current agent state
            
        Returns:
            Route destination node name
        """
        # Check for observer trigger first (highest priority)
        last_updated_key = state.get('last_updated_key', '')
        if last_updated_key == 'ocr_results':
            logger.info("[OBSERVER ROUTING] 🔔 OCR results trigger detected → routing to auto_synthesis")
            return "auto_synthesis"
        
        # Check for chain execution (second priority)
        if self._should_chain_to_excel(state):
            logger.info("[CHAIN ROUTING] 🔗 OCR extracted table → routing to execute_excel_task")
            return "execute_excel_task"
        
        # Default to normal aggregation flow
        logger.info("[NORMAL ROUTING] OCR complete → routing to aggregate_results")
        return "aggregate_results"
    
    def _route_after_websearch_execution(self, state: "AgentState") -> Literal["auto_synthesis", "execute_excel_task", "aggregate_results"]:
        """
        Observer pattern routing after WebSearch execution.
        
        Priority routing:
        1. If last_updated_key == 'web_findings' → auto_synthesis (observer trigger)
        2. If CSV file generated → execute_excel_task (chain execution)
        3. Otherwise → aggregate_results (normal flow)
        
        This implements event-driven triggers where web search findings automatically
        trigger synthesis before continuing with normal workflow.
        
        Args:
            state: Current agent state
            
        Returns:
            Route destination node name
        """
        # Check for observer trigger first (highest priority)
        last_updated_key = state.get('last_updated_key', '')
        if last_updated_key == 'web_findings':
            logger.info("[OBSERVER ROUTING] 🔔 Web findings trigger detected → routing to auto_synthesis")
            return "auto_synthesis"
        
        # Check for chain execution (second priority)
        if self._should_chain_to_excel(state):
            logger.info("[CHAIN ROUTING] 🔗 WebSearch created CSV → routing to execute_excel_task")
            return "execute_excel_task"
        
        # Default to normal aggregation flow
        logger.info("[NORMAL ROUTING] WebSearch complete → routing to aggregate_results")
        return "aggregate_results"
