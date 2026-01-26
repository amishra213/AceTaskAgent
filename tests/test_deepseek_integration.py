"""
Test script to verify DeepSeek AI integration with Task Manager Agent
"""

from task_manager import AgentConfig
from task_manager.config import EnvConfig, LLMConfig

def test_deepseek_basic():
    """Test basic DeepSeek configuration"""
    print("=" * 80)
    print("TEST 1: DeepSeek Configuration from Environment Variables")
    print("=" * 80)
    
    # Load environment variables
    EnvConfig.load_env_file()
    
    # Create config from environment
    config = AgentConfig.from_env(prefix="AGENT_")
    
    print(f"✓ Provider: {config.llm.provider}")
    print(f"✓ Model: {config.llm.model_name}")
    print(f"✓ Base URL: {config.llm.base_url}")
    print(f"✓ API Base URL: {config.llm.api_base_url}")
    print(f"✓ Temperature: {config.llm.temperature}")
    print(f"✓ API Key: {'*' * 20}{config.llm.api_key[-4:] if config.llm.api_key else 'None'}")
    print()
    
    # Verify DeepSeek configuration
    assert config.llm.provider == "deepseek", f"Expected provider 'deepseek', got '{config.llm.provider}'"
    assert config.llm.model_name == "deepseek-chat", f"Expected model 'deepseek-chat', got '{config.llm.model_name}'"
    assert config.llm.api_key is not None, "API key not found"
    assert config.llm.base_url == "https://api.deepseek.com" or config.llm.api_base_url == "https://api.deepseek.com", \
        f"Expected base URL 'https://api.deepseek.com', got base_url='{config.llm.base_url}', api_base_url='{config.llm.api_base_url}'"
    
    print("✅ All configuration checks passed!")
    print()


def test_deepseek_agent_initialization():
    """Test that TaskManagerAgent initializes correctly with DeepSeek"""
    print("=" * 80)
    print("TEST 2: TaskManagerAgent Initialization with DeepSeek")
    print("=" * 80)
    
    from task_manager import TaskManagerAgent
    
    # Load environment variables
    EnvConfig.load_env_file()
    config = AgentConfig.from_env(prefix="AGENT_")
    
    # Initialize agent
    agent = TaskManagerAgent(
        objective="Test DeepSeek integration",
        config=config
    )
    
    print(f"✓ Agent initialized successfully")
    print(f"✓ LLM Provider: {agent.config.llm.provider}")
    print(f"✓ LLM Model: {agent.config.llm.model_name}")
    print(f"✓ LLM Instance Type: {type(agent.llm).__name__}")
    print()
    
    # Verify LLM is ChatOpenAI with correct base_url
    from langchain_openai import ChatOpenAI
    assert isinstance(agent.llm, ChatOpenAI), f"Expected ChatOpenAI instance, got {type(agent.llm)}"
    
    print("✅ Agent initialization successful!")
    print()


def test_deepseek_simple_call():
    """Test a simple LLM call to DeepSeek"""
    print("=" * 80)
    print("TEST 3: Simple DeepSeek LLM Call")
    print("=" * 80)
    
    from task_manager import TaskManagerAgent
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Load environment variables
    EnvConfig.load_env_file()
    config = AgentConfig.from_env(prefix="AGENT_")
    
    # Initialize agent
    agent = TaskManagerAgent(
        objective="Test DeepSeek API call",
        config=config
    )
    
    print("Sending test message to DeepSeek...")
    try:
        response = agent.llm.invoke([
            SystemMessage(content="You are a helpful assistant. Respond with exactly 5 words."),
            HumanMessage(content="Say hello in a friendly way")
        ])
        
        print(f"✓ Response received: {response.content}")
        print()
        print("✅ DeepSeek API call successful!")
        
    except Exception as e:
        print(f"❌ Error calling DeepSeek API: {e}")
        raise
    
    print()


def test_manual_deepseek_config():
    """Test manual DeepSeek configuration (not from env)"""
    print("=" * 80)
    print("TEST 4: Manual DeepSeek Configuration")
    print("=" * 80)
    
    import os
    
    # Get API key from environment
    api_key = os.getenv("LLM_API_KEY")
    
    # Create manual configuration
    config = AgentConfig(
        llm=LLMConfig(
            provider="deepseek",
            model_name="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.2
        ),
        max_iterations=10
    )
    
    print(f"✓ Manual config created")
    print(f"✓ Provider: {config.llm.provider}")
    print(f"✓ Model: {config.llm.model_name}")
    print(f"✓ Base URL: {config.llm.base_url}")
    print()
    
    from task_manager import TaskManagerAgent
    agent = TaskManagerAgent(
        objective="Test manual DeepSeek config",
        config=config
    )
    
    print("✅ Manual configuration successful!")
    print()


if __name__ == "__main__":
    try:
        test_deepseek_basic()
        test_deepseek_agent_initialization()
        test_deepseek_simple_call()
        test_manual_deepseek_config()
        
        print("=" * 80)
        print("🎉 ALL TESTS PASSED! DeepSeek integration is working correctly!")
        print("=" * 80)
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
