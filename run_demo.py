#!/usr/bin/env python
"""
Task Manager Agent - Demo/Test Execution

This script demonstrates the Task Manager Agent with a sample objective.
"""

from task_manager import TaskManagerAgent
from task_manager.config import EnvConfig, AgentConfig
from datetime import datetime

def main():
    """Main entry point for the agent demo."""
    print("=" * 70)
    print("Task Manager Agent - Demo Execution")
    print("=" * 70)
    print()
    
    # Load environment configuration
    print("Step 1: Loading configuration from .env...")
    EnvConfig.load_env_file()
    print("        Configuration loaded successfully!")
    print()
    
    # Create agent config from environment
    print("Step 2: Creating agent configuration...")
    config = AgentConfig.from_env(prefix="AGENT_")
    
    print("        Configuration:")
    print(f"          - LLM Provider: {config.llm.provider}")
    print(f"          - Model: {config.llm.model_name}")
    print(f"          - Temperature: {config.llm.temperature}")
    print(f"          - Max Iterations: {config.max_iterations}")
    print(f"          - Enable Search: {config.enable_search}")
    print(f"          - Log Level: {config.log_level}")
    print()
    
    # Set a demo objective
    objective = "List the top 5 programming languages in 2026 and their primary use cases"
    
    print("Step 3: Setting objective...")
    print(f"        Objective: {objective}")
    print()
    
    print("=" * 70)
    print("Step 4: Starting Task Manager Agent...")
    print("=" * 70)
    print()
    
    try:
        # Create agent
        agent = TaskManagerAgent(
            objective=objective,
            config=config,
            metadata={
                "demo": True,
                "started_at": datetime.now().isoformat()
            }
        )
        
        print("Agent initialized successfully!")
        print("Running task execution workflow...")
        print()
        
        # Run the agent
        final_state = agent.run(thread_id="demo-task-001")
        
        # Print results
        print()
        print("=" * 70)
        print("Task Execution Complete!")
        print("=" * 70)
        
        summary = agent.get_results_summary(final_state)
        
        print()
        print("Execution Summary:")
        print(f"  Objective: {summary['objective'][:60]}...")
        print(f"  Total Tasks Created: {summary['total_tasks']}")
        print(f"  Completed Tasks: {summary['completed_tasks']}")
        print(f"  Failed Tasks: {summary['failed_tasks']}")
        print(f"  Iterations Used: {summary['iterations_used']}")
        print()
        
        if summary['results']:
            print("Results:")
            for task_id, result in list(summary['results'].items())[:3]:
                print(f"  {task_id}: {str(result)[:50]}...")
        
        print()
        print("=" * 70)
        print("Demo execution completed successfully!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print()
        print("Agent interrupted by user.")
    except Exception as e:
        print()
        print(f"Error during execution: {str(e)}")
        print()
        print("Troubleshooting tips:")
        print("1. Ensure .env file exists with valid API key")
        print("2. Check internet connection for API calls")
        print("3. Verify Google API key is active")
        print()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
