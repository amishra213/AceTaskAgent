"""
Example usage of the Task Manager Agent
Karnataka Administrative Data Collection

This example demonstrates how the Task Manager Agent recursively breaks down
and executes a complex data collection objective with parameterized configuration.

Configuration can be:
1. Loaded from environment variables (.env file)
2. Specified explicitly in code
3. Loaded from a dictionary
"""

import json
import os
from task_manager import TaskManagerAgent, AgentConfig
from task_manager.config import EnvConfig


def main():
    """Run the Task Manager Agent example."""
    
    # Load environment variables from .env file (if it exists)
    EnvConfig.load_env_file()
    
    # Define the objective
    OBJECTIVE = """
    Compile a comprehensive list of all villages, towns, and cities in Karnataka 
    (31 districts) along with their key government officials, including counts of 
    municipalities, towns, and villages per district.
    """
    
    # Define metadata (context for the agent)
    METADATA = {
        "districts": [
            'Bagalkot', 'Ballari', 'Belagavi', 'Bengaluru Rural', 'Bengaluru Urban',
            'Bidar', 'Chamarajanagar', 'Chikkaballapur', 'Chikkamagaluru', 'Chitradurga',
            'Dakshina Kannada', 'Davanagere', 'Dharwad', 'Gadag', 'Hassan',
            'Haveri', 'Kalaburagi', 'Kodagu', 'Kolar', 'Koppal',
            'Mandya', 'Mysuru', 'Raichur', 'Ramanagara', 'Shivamogga',
            'Tumakuru', 'Udupi', 'Uttara Kannada', 'Vijayapura', 'Vijayanagara', 'Yadgir'
        ],
        "data_points_needed": [
            "Number of villages",
            "Number of towns",
            "Number of cities",
            "Municipal corporations count",
            "Town panchayats count",
            "Key government officials (DC, ZP CEO, etc.)"
        ]
    }
    
    # Configure agent - Load from environment variables or use defaults
    # Option 1: Load from environment variables (.env file)
    # This allows you to change configuration without modifying code
    if os.getenv("AGENT_LLM_PROVIDER"):
        print("📋 Loading configuration from environment variables...")
        config = AgentConfig.from_env(prefix="AGENT_")
    else:
        # Option 2: Explicit configuration (defaults)
        print("📋 Using default configuration (set env vars to customize)...")
        config = AgentConfig(
            max_iterations=50,
            enable_search=True,
            log_level="INFO"
        )
    
    # Initialize agent
    print("🤖 Initializing Task Manager Agent...")
    agent = TaskManagerAgent(
        objective=OBJECTIVE,
        config=config,
        metadata=METADATA
    )
    
    # Run agent
    print("🚀 Running agent with objective...")
    final_state = agent.run(thread_id="karnataka-data-collection-001")
    
    # Get summary
    summary = agent.get_results_summary(final_state)
    
    # Print results
    print("\n" + "="*60)
    print("✅ FINAL RESULTS SUMMARY")
    print("="*60)
    print(json.dumps(summary, indent=2))
    
    # Save to file
    with open("task_manager_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Results saved to task_manager_results.json")
    
    # Print configuration info
    print("\n" + "="*60)
    print("Configuration Details")
    print("="*60)
    config_dict = config.to_dict(include_secrets=False)
    print(json.dumps(config_dict, indent=2))


if __name__ == "__main__":
    main()
