#!/usr/bin/env python
"""
Task Manager Agent - Startup Script

This script starts the Task Manager Agent with Google Gemini configuration
and interactive human review support.
"""

from task_manager import TaskManagerAgent
from task_manager.config import EnvConfig, AgentConfig

def handle_human_review(agent, state):
    """
    Handle human review gate - first try web search, then prompt for human input if needed.
    
    Args:
        agent: TaskManagerAgent instance
        state: Current agent state at human review gate
        
    Returns:
        True to continue execution, False to stop
    """
    print()
    print("=" * 70)
    print("HUMAN REVIEW REQUIRED")
    print("=" * 70)
    
    # Get the current task
    active_task_id = state.get('active_task_id')
    if active_task_id:
        tasks = state.get('tasks', [])
        task = next((t for t in tasks if t['id'] == active_task_id), None)
        
        if task:
            print()
            print(f"Task: {task.get('description', 'Unknown')[:100]}")
            
            result = task.get('result')
            if result:
                if isinstance(result, dict):
                    print()
                    print("Analysis Result:")
                    for key, value in result.items():
                        if key != 'subtasks':  # Don't print full subtasks list yet
                            print(f"  {key}: {str(value)[:80]}")
                        else:
                            print(f"  {key}: {len(value)} subtasks created")
                else:
                    print(f"Analysis: {str(result)[:200]}")
    
    print()
    print("Options:")
    print("  1. Continue with automatic web search (if applicable)")
    print("  2. Provide additional context/guidance for the agent")
    print("  3. Skip this task and continue with next")
    print("  4. Abort execution")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        print()
        print("Proceeding with web search and execution...")
        return True
    
    elif choice == "2":
        print()
        guidance = input("Provide additional context or guidance: ").strip()
        
        if guidance:
            # Add guidance to metadata for the agent to use
            state['metadata']['human_guidance'] = guidance
            state['metadata']['guidance_provided'] = True
            print()
            print("Guidance recorded. Continuing execution...")
            return True
        else:
            return handle_human_review(agent, state)
    
    elif choice == "3":
        print()
        print("Marking current task as completed and continuing with next...")
        # Mark current task as completed
        active_task_id = state.get('active_task_id')
        if active_task_id:
            completed = state.get('completed_task_ids', [])
            if active_task_id not in completed:
                completed.append(active_task_id)
                state['completed_task_ids'] = completed
        return True
    
    elif choice == "4":
        print()
        print("Aborting execution...")
        return False
    
    else:
        print()
        print("Invalid choice. Please enter 1-4.")
        return handle_human_review(agent, state)


def main():
    """Main entry point for the agent."""
    print("=" * 70)
    print("Task Manager Agent - Starting")
    print("=" * 70)
    print()
    
    # Load environment configuration
    print("Loading configuration from .env...")
    EnvConfig.load_env_file()
    
    # Create agent config from environment
    print("Creating agent configuration...")
    config = AgentConfig.from_env(prefix="AGENT_")
    
    print()
    print("Configuration loaded:")
    print(f"  LLM Provider: {config.llm.provider}")
    print(f"  Model: {config.llm.model_name}")
    print(f"  Temperature: {config.llm.temperature}")
    print(f"  Max Iterations: {config.max_iterations}")
    print(f"  Enable Search: {config.enable_search}")
    print(f"  Log Level: {config.log_level}")
    print()
    print("Folder Configuration:")
    print(f"  Input Folder:  {config.folders.input_path}")
    print(f"  Output Folder: {config.folders.output_path}")
    print(f"  Temp Folder:   {config.folders.temp_path}")
    print()
    
    # Show input files if any exist
    input_files = config.folders.get_input_files()
    if input_files:
        print(f"Input files detected ({len(input_files)}):")
        for f in input_files[:5]:  # Show first 5
            print(f"  - {f.name}")
        if len(input_files) > 5:
            print(f"  ... and {len(input_files) - 5} more")
        print()
    
    # Get objective from user
    print("=" * 70)
    print("Enter your objective (what you want the agent to accomplish):")
    print("=" * 70)
    objective = input("\nObjective: ").strip()
    
    if not objective:
        print("No objective provided. Exiting.")
        return
    
    print()
    print("=" * 70)
    print("Starting Task Manager Agent...")
    print("=" * 70)
    print()
    
    try:
        # Create agent
        import uuid
        session_id = f"session-{uuid.uuid4().hex[:8]}"
        print(f"[SESSION] Starting with thread_id: {session_id}")
        
        agent = TaskManagerAgent(
            objective=objective,
            config=config,
            metadata={"started_at": __import__("datetime").datetime.now().isoformat()}
        )
        
        # Run the agent with human review handling
        # The agent will pause at human_review node and we detect this
        final_state = agent.run(thread_id=session_id)
        
        # The agent workflow handles human review internally now
        # Just display the final results
        
        print()
        print("=" * 70)
        print("Task Execution Complete!")
        print("=" * 70)
        
        summary = agent.get_results_summary(final_state)
        
        print()
        print("Results Summary:")
        print(f"  Objective: {summary['objective']}")
        print(f"  Total Tasks: {summary['total_tasks']}")
        print(f"  Completed Tasks: {summary['completed_tasks']}")
        print(f"  Failed Tasks: {summary['failed_tasks']}")
        print(f"  Iterations Used: {summary['iterations_used']}")
        
        print()
        print("=" * 70)
        print("Agent execution finished successfully!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print()
        print("Agent interrupted by user.")
    except Exception as e:
        print()
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
