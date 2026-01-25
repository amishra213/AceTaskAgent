"""
Task Manager Agent - LangGraph-based Recursive Task Orchestration System

A generic, production-ready agent that recursively breaks down and executes 
complex tasks using LangGraph for state management and workflow orchestration.

Features:
- Recursive task breakdown with AI analysis
- Parallel execution of independent subtasks
- State persistence and checkpointing
- Error recovery and retry logic
- Progress tracking and visualization
- Human-in-the-loop validation
- Multi-provider LLM support (Anthropic, OpenAI, Google, Local)
- Environment-based configuration

Installation:
pip install langgraph langchain-anthropic langchain-core python-dotenv

Configuration:
    Create a .env file with your LLM provider configuration:
    
    ANTHROPIC_API_KEY=sk-ant-...
    AGENT_LLM_PROVIDER=anthropic
    AGENT_LLM_MODEL=claude-sonnet-4-20250514

Example:
    >>> from task_manager import TaskManagerAgent, AgentConfig, EnvConfig
    >>> 
    >>> # Load configuration from environment
    >>> EnvConfig.load_env_file()
    >>> config = AgentConfig.from_env(prefix="AGENT_")
    >>> 
    >>> agent = TaskManagerAgent(
    ...     objective="Your complex task here",
    ...     config=config,
    ...     metadata={}
    ... )
    >>> final_state = agent.run(thread_id="my-task-001")
    >>> summary = agent.get_results_summary(final_state)
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = [
    'TaskManagerAgent',
    'AgentConfig',
    'LLMConfig',
    'EnvConfig',
    'TaskStatus',
    'Task',
    'AgentState',
]

from task_manager.core import TaskManagerAgent
from task_manager.config import AgentConfig, LLMConfig, EnvConfig
from task_manager.models import TaskStatus, Task, AgentState
