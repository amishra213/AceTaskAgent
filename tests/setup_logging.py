#!/usr/bin/env python3
"""
Quick setup script for comprehensive logging and observability

Run this script to:
1. Verify logging configuration
2. Test file logging
3. Test console logging
4. Test Langfuse integration (if enabled)
5. Create sample log files
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from task_manager.config import EnvConfig
from task_manager.utils import ComprehensiveLogger


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_section(text):
    """Print a formatted section."""
    print(f"\n📌 {text}")
    print("-" * 70)


def verify_logging_setup():
    """Verify and display current logging setup."""
    print_header("COMPREHENSIVE LOGGING SETUP VERIFICATION")
    
    # Load configuration
    EnvConfig.load_env_file()
    log_config = EnvConfig.get_logging_config()
    
    print("\n✓ Configuration loaded from .env")
    print_section("Current Logging Configuration")
    
    for key, value in log_config.items():
        if "key" in key.lower() and value:
            value = value[:10] + "..." if len(str(value)) > 10 else value
        print(f"  • {key:30s} : {value}")
    
    return log_config


def test_file_logging(log_config):
    """Test file logging."""
    print_section("Testing File Logging")
    
    log_folder = log_config["log_folder"]
    
    # Initialize logging
    ComprehensiveLogger.initialize(**log_config)
    
    # Get logger
    logger = ComprehensiveLogger.get_logger("setup_test")
    
    # Write test messages
    logger.debug("Debug: Testing file logging")
    logger.info("Info: File logging is working")
    logger.warning("Warning: Test warning message")
    logger.error("Error: Test error message")
    
    # Verify files exist
    log_path = Path(log_folder)
    if log_path.exists():
        log_files = list(log_path.glob("*.log"))
        print(f"✓ Log folder created: {log_folder}")
        print(f"✓ Log files created: {len(log_files)} file(s)")
        
        for log_file in log_files[:5]:  # Show first 5
            file_size = log_file.stat().st_size
            print(f"  • {log_file.name:30s} ({file_size:,} bytes)")
    else:
        print(f"✗ Log folder not found: {log_folder}")


def test_console_logging(log_config):
    """Test console logging."""
    print_section("Testing Console Logging")
    
    if log_config.get("enable_console"):
        print("✓ Console logging is enabled")
        print("  Messages above show console output in action")
    else:
        print("✗ Console logging is disabled")
        print("  Enable it: set AGENT_ENABLE_CONSOLE_LOGGING=true in .env")


def test_structured_logging(log_config):
    """Test structured logging with metadata."""
    print_section("Testing Structured Logging with Metadata")
    
    ComprehensiveLogger.initialize(**log_config)
    logger = ComprehensiveLogger.get_logger("structured_test")
    
    # Log with metadata
    logger.info(
        "Structured logging test",
        extra={
            "user_id": 12345,
            "action": "login",
            "timestamp": "2024-01-26T10:30:00Z",
            "ip_address": "192.168.1.100"
        }
    )
    
    print("✓ Structured logging message written")
    print("  Check log file for JSON metadata")


def test_performance_logging(log_config):
    """Test performance metrics logging."""
    print_section("Testing Performance Metrics Logging")
    
    import time
    
    ComprehensiveLogger.initialize(**log_config)
    logger = ComprehensiveLogger.get_logger("performance_test")
    
    # Simulate operation
    time.sleep(0.1)
    
    logger.log_performance(
        operation="test_operation",
        duration_seconds=0.123,
        success=True,
        metadata={
            "items_processed": 1000,
            "status": "success"
        }
    )
    
    print("✓ Performance metrics logged")
    print("  Check log file for performance data")


def test_langfuse_integration(log_config):
    """Test Langfuse integration."""
    print_section("Testing Langfuse Integration")
    
    if not log_config.get("enable_langfuse"):
        print("⚠️  Langfuse is not enabled")
        print("  To enable:")
        print("  1. Set ENABLE_LANGFUSE=true in .env")
        print("  2. Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        print("  3. Get keys from: https://cloud.langfuse.com/")
        return
    
    ComprehensiveLogger.initialize(**log_config)
    logger = ComprehensiveLogger.get_logger("langfuse_test")
    
    # Try to create a trace
    trace = logger.create_trace(
        "setup_verification",
        metadata={"test": True}
    )
    
    if trace:
        logger.info("Langfuse integration working")
        logger.info("Test event sent to Langfuse")
        print("✓ Langfuse integration is working")
        print("  View traces at: https://cloud.langfuse.com/")
    else:
        print("✗ Langfuse integration failed")
        print("  Check API keys and network connectivity")


def show_next_steps():
    """Show next steps."""
    print_header("NEXT STEPS")
    
    print("""
1. VERIFY CONFIGURATION
   ✓ Check your .env file for logging settings
   ✓ Adjust AGENT_LOG_LEVEL if needed (DEBUG, INFO, WARNING, ERROR)

2. INTEGRATE INTO YOUR APPLICATION
   from task_manager.config import EnvConfig
   from task_manager.utils import ComprehensiveLogger
   
   EnvConfig.load_env_file()
   log_config = EnvConfig.get_logging_config()
   ComprehensiveLogger.initialize(**log_config)
   
   logger = ComprehensiveLogger.get_logger(__name__)
   logger.info("Your message", extra={"metadata": "value"})

3. VIEW LOG FILES
   • Location: ./logs/ (or your configured AGENT_LOG_FOLDER)
   • Each module gets its own log file
   • Logs are rotated when they exceed max size

4. ENABLE LANGFUSE (Optional)
   • Sign up at: https://langfuse.com/
   • Set ENABLE_LANGFUSE=true in .env
   • Add your API keys
   • View traces at: https://cloud.langfuse.com/

5. DOCUMENTATION
   • Comprehensive guide: LOGGING_GUIDE.md
   • Integration examples: LOGGING_INTEGRATION_GUIDE.md
   • Code examples: examples/logging_observability_example.py

6. TROUBLESHOOTING
   • Logs not appearing? Check AGENT_LOG_LEVEL
   • File not created? Verify AGENT_LOG_FOLDER exists
   • Langfuse not working? Check API keys
    """)


def main():
    """Run all tests."""
    try:
        # Verify setup
        log_config = verify_logging_setup()
        
        # Run tests
        test_file_logging(log_config)
        test_console_logging(log_config)
        test_structured_logging(log_config)
        test_performance_logging(log_config)
        test_langfuse_integration(log_config)
        
        # Show next steps
        show_next_steps()
        
        # Flush logs
        ComprehensiveLogger.flush()
        
        print_header("✅ SETUP VERIFICATION COMPLETE")
        print("\nYour logging system is ready to use!")
        print(f"Logs are stored in: {Path(log_config['log_folder']).resolve()}/")
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
