"""
Simple DeepSeek AI integration test
"""

from task_manager.config import EnvConfig, LLMConfig, AgentConfig
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr
import os

def test_deepseek():
    """Test DeepSeek configuration and API call"""
    print("=" * 80)
    print("DeepSeek AI Integration Test")
    print("=" * 80)
    print()
    
    # Load environment variables
    EnvConfig.load_env_file()
    
    # Test 1: Configuration from environment
    print("✓ Step 1: Loading configuration from .env file")
    config = AgentConfig.from_env(prefix="AGENT_")
    
    print(f"  - Provider: {config.llm.provider}")
    print(f"  - Model: {config.llm.model_name}")
    print(f"  - Base URL: {config.llm.base_url or config.llm.api_base_url}")
    print(f"  - Temperature: {config.llm.temperature}")
    print(f"  - API Key: ***{config.llm.api_key[-4:] if config.llm.api_key else 'None'}")
    print()
    
    # Test 2: Initialize ChatOpenAI with DeepSeek endpoint
    print("✓ Step 2: Initializing ChatOpenAI with DeepSeek endpoint")
    
    base_url = config.llm.base_url or config.llm.api_base_url or "https://api.deepseek.com/v1"
    llm = ChatOpenAI(
        model=config.llm.model_name,
        api_key=SecretStr(config.llm.api_key) if config.llm.api_key else None,
        base_url=base_url,
        temperature=config.llm.temperature
    )
    
    
    print(f"  - LLM Type: {type(llm).__name__}")
    print(f"  - Model: {llm.model_name}")
    print()
    
    # Test 3: Make a simple API call
    print("✓ Step 3: Making test API call to DeepSeek")
    print("  - Sending: 'Say hello in exactly 5 words'")
    
    try:
        response = llm.invoke([
            SystemMessage(content="You are a helpful assistant. Be concise."),
            HumanMessage(content="Say hello in exactly 5 words")
        ])
        
        print(f"  - Response: {response.content}")
        print()
        print("=" * 80)
        print("✅ SUCCESS! DeepSeek AI integration is working correctly!")
        print("=" * 80)
        
    except Exception as e:
        print(f"  - ❌ Error: {e}")
        print()
        print("=" * 80)
        print("❌ FAILED! Check your API key and network connection")
        print("=" * 80)
        raise


def test_deepseek_direct():
    """Direct test using OpenAI SDK (like user's sample code)"""
    print()
    print("=" * 80)
    print("Direct OpenAI SDK Test (matching user's sample code)")
    print("=" * 80)
    print()
    
    from openai import OpenAI
    
    # Load API key from environment
    api_key = os.getenv('LLM_API_KEY')
    base_url = os.getenv('LLM_API_BASE_URL', 'https://api.deepseek.com')
    
    # Ensure base_url has /v1 at the end
    if not base_url.endswith('/v1'):
        base_url = f"{base_url}/v1"
    
    print(f"  - API Key: ***{api_key[-4:] if api_key else 'None'}")
    print(f"  - Base URL: {base_url}")
    print()
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    print("  - Calling DeepSeek API...")
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )
    
    print(f"  - Response: {response.choices[0].message.content}")
    print()
    print("=" * 80)
    print("✅ SUCCESS! Direct OpenAI SDK call works!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        # Run both tests
        test_deepseek()
        test_deepseek_direct()
        
        print()
        print("🎉 All tests passed! DeepSeek integration is ready to use!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
