#!/usr/bin/env python
"""
Test script to trigger task breakdown and identify iteration 5 issue.

This tests with an objective that triggers task breakdown specifically
to reproduce the issue shown in the user's logs.
"""

from task_manager import TaskManagerAgent
from task_manager.config import EnvConfig, AgentConfig
from datetime import datetime

def main():
    """Test task breakdown with complex objective."""
    print("=" * 70)
    print("Task Breakdown Iteration Test")
    print("=" * 70)
    print()
    
    # Load environment configuration
    print("Loading configuration...")
    EnvConfig.load_env_file()
    config = AgentConfig.from_env(prefix="AGENT_")
    
    # Use objective that should trigger task breakdown
    # This should create subtasks that mirror the user's hierarchical IDs (1.1.1, 1.1.2, etc)
    objective = """
    Create a comprehensive analysis document about the top 5 latest supply chain trends 
    in the Consumer Packaged Goods (CPG) industry. The document should:
    1. Research and identify the latest trends
    2. Gather comprehensive data for each trend
    3. Create a detailed analysis with examples
    4. Format as a professional DOCX document
    
    The analysis should cover at least 5 key trends with specific examples and industry impacts.
    """
    
    print(f"Objective: {objective[:100]}...")
    print()
    print("Starting agent with enhanced logging...")
    print()
    
    try:
        # Create and run agent
        agent = TaskManagerAgent(
            objective=objective,
            config=config,
            metadata={"test_run": True}
        )
        
        print("Agent initialized, starting execution...")
        print()
        
        result = agent.run(thread_id="breakdown-test-001")
        
        print("\n" + "=" * 70)
        print("EXECUTION COMPLETED")
        print("=" * 70)
        print(f"Final State Keys: {list(result.keys())}")
        print(f"Tasks: {len(result.get('tasks', []))}")
        print(f"Completed: {len(result.get('completed_task_ids', []))}")
        print(f"Failed: {len(result.get('failed_task_ids', []))}")
        print(f"Iterations Used: {result.get('iteration_count', 0)}")
        
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
