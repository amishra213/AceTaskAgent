#!/usr/bin/env python
"""
Test to verify that document creation tasks are properly routed to execute_document_task
instead of being misclassified as research tasks.

This test specifically targets the bug fix for task 1.1.4:
"Create structured document with trends and analyses"

The issue was that the word "trends" was matching research_keywords, causing
the task to be forced as execute_web_search_task instead of being analyzed properly.
"""

import logging
from task_manager.core.agent import TaskManagerAgent
from task_manager.config import AgentConfig, LLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_document_task_routing():
    """Test that document creation tasks at depth 2 are routed correctly."""
    
    print("\n" + "="*80)
    print("TEST: Document Task Routing Fix")
    print("="*80)
    print("\nObjective: Research supply chain trends and create documentation")
    print("Expected: Document tasks should route to execute_document_task, not web_search")
    print("="*80 + "\n")
    
    # Set the objective
    objective = "Research latest supply chain trends in CPG industry and create a professional report with detailed analysis of top 5 trends"
    
    # Create agent with test configuration
    llm_config = LLMConfig(
        provider='deepseek',
        model_name='deepseek-chat',
        temperature=0.2,
    )
    
    config = AgentConfig(
        llm=llm_config,
        max_iterations=50,
        enable_search=True
    )
    
    agent = TaskManagerAgent(objective=objective, config=config)
    
    try:
        final_state = agent.run(thread_id="test-document-routing-001")
        
        # Verify results
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        
        tasks = final_state.get('tasks', [])
        print(f"\nTotal tasks created: {len(tasks)}")
        
        # Find the document tasks
        document_tasks = [t for t in tasks if 'document' in t.get('description', '').lower()]
        print(f"Document tasks found: {len(document_tasks)}")
        
        for task in document_tasks:
            task_id = task.get('id', 'unknown')
            status = task.get('status', 'unknown')
            description = task.get('description', '')[:80]
            result = task.get('result')
            
            # Handle both dict and None result types
            if isinstance(result, dict):
                analysis = result
                action = analysis.get('action', 'unknown')
            else:
                action = 'unknown'
            
            print(f"\n✓ Task {task_id}: {description}...")
            print(f"  Status: {status}")
            print(f"  Analyzed Action: {action}")
            
            # Verify the action is correct
            if action == 'execute_document_task':
                print(f"  ✅ CORRECT: Routed to execute_document_task")
            elif action == 'execute_web_search_task':
                print(f"  ❌ WRONG: Incorrectly routed to execute_web_search_task")
                print(f"     This is the BUG we're trying to fix!")
            else:
                print(f"  ⚠️  UNEXPECTED: Routed to {action}")
        
        # Check for any failed tasks
        failed_tasks = [t for t in tasks if t.get('status') == 'FAILED']
        if failed_tasks:
            print(f"\n⚠️  Warning: {len(failed_tasks)} tasks failed")
            for task in failed_tasks:
                print(f"   - {task['id']}: {task.get('error', 'unknown')}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_document_task_routing()
    exit(0 if success else 1)
