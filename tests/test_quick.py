#!/usr/bin/env python
"""Quick test of agent workflow"""
from task_manager import TaskManagerAgent
from task_manager.config import EnvConfig, AgentConfig

EnvConfig.load_env_file()
config = AgentConfig.from_env(prefix='AGENT_')

objective = 'Collect information about Karnataka administrative divisions - districts, taluks, towns and villages'

print('Starting agent test...')
print('=' * 70)

try:
    agent = TaskManagerAgent(objective=objective, config=config)
    final_state = agent.run(thread_id='test-001')
    
    print()
    print('=' * 70)
    print('Results:')
    print(f'  Total tasks: {len(final_state["tasks"])}')
    print(f'  Completed: {len(final_state["completed_task_ids"])}')
    print(f'  Failed: {len(final_state["failed_task_ids"])}')
    print(f'  Iterations: {final_state["iteration_count"]}')
    print('=' * 70)
    
    # Show task breakdown
    if final_state['tasks']:
        print()
        print('Task Breakdown:')
        for i, task in enumerate(final_state['tasks'][:5], 1):
            print(f"  {i}. {task['description'][:80]}")
        if len(final_state['tasks']) > 5:
            print(f"  ... and {len(final_state['tasks']) - 5} more")
            
except Exception as e:
    print()
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
