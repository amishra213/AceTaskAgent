#!/usr/bin/env python
"""
Automated test script for agent with predefined input
"""

from task_manager import TaskManagerAgent
from task_manager.config import EnvConfig, AgentConfig

def main():
    """Run agent with predefined objective."""
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
    
    # Use fixed objective
    objective = "create list of all villages, towns, cities in Karnataka along with the list of their key government officials and representatives, organize the data per district or taluk in a tabular format"
    
    print("=" * 70)
    print("Starting Task Manager Agent...")
    print("=" * 70)
    print()
    
    try:
        # Create agent
        agent = TaskManagerAgent(
            objective=objective,
            config=config,
            metadata={"started_at": __import__("datetime").datetime.now().isoformat()}
        )
        
        # Run the agent
        final_state = agent.run(thread_id="main-task-001")
        
        # Print results
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
