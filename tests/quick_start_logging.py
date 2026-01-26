#!/usr/bin/env python3
"""
Quick Start: TaskManager v2.5 - Logging & Observability

This script demonstrates the easiest way to get started with TaskManager's
comprehensive logging and Langfuse observability features.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from task_manager.config import EnvConfig
from task_manager.utils import ComprehensiveLogger


def main():
    """Quick start demo."""
    
    print("\n" + "="*70)
    print("  TASKMANAGER v2.5 - LOGGING & OBSERVABILITY QUICK START")
    print("="*70)
    
    # Step 1: Load environment
    print("\n1️⃣  Loading environment configuration...")
    EnvConfig.load_env_file()
    log_config = EnvConfig.get_logging_config()
    print("   ✓ Configuration loaded")
    
    # Step 2: Initialize logging
    print("\n2️⃣  Initializing comprehensive logging system...")
    ComprehensiveLogger.initialize(**log_config)
    print(f"   ✓ Logging initialized")
    print(f"   ✓ Log folder: {log_config['log_folder']}")
    print(f"   ✓ Log level: {log_config['log_level']}")
    
    # Step 3: Get logger
    print("\n3️⃣  Creating logger instance...")
    logger = ComprehensiveLogger.get_logger("quick_start")
    print("   ✓ Logger created")
    
    # Step 4: Basic logging
    print("\n4️⃣  Demonstrating basic logging...")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    print("   ✓ Messages logged (check ./logs/quick_start.log)")
    
    # Step 5: Structured logging with metadata
    print("\n5️⃣  Demonstrating structured logging...")
    logger.info("User action", extra={
        "user_id": 12345,
        "action": "data_analysis",
        "timestamp": "2024-01-26T10:30:00Z"
    })
    print("   ✓ Structured message logged with metadata")
    
    # Step 6: Performance logging
    print("\n6️⃣  Demonstrating performance metrics...")
    import time
    time.sleep(0.1)  # Simulate operation
    logger.log_performance(
        operation="sample_operation",
        duration_seconds=0.123,
        success=True,
        metadata={"records": 1000}
    )
    print("   ✓ Performance metrics logged")
    
    # Step 7: Exception logging
    print("\n7️⃣  Demonstrating exception handling...")
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        logger.log_exception("Caught sample exception", exc=e)
    print("   ✓ Exception logged with full traceback")
    
    # Step 8: Langfuse tracing (if enabled)
    print("\n8️⃣  Checking Langfuse integration...")
    if log_config.get("enable_langfuse"):
        trace = logger.create_trace(
            "quick_start_demo",
            metadata={"demo": True}
        )
        if trace:
            logger.info("Langfuse trace created")
            print("   ✓ Langfuse tracing enabled")
            print("   ✓ View traces at: https://cloud.langfuse.com/")
        else:
            print("   ⚠️  Langfuse configured but not connected")
    else:
        print("   ℹ️  Langfuse not enabled (optional)")
        print("   💡  To enable: set ENABLE_LANGFUSE=true in .env")
    
    # Step 9: Flush logs
    print("\n9️⃣  Flushing logs...")
    ComprehensiveLogger.flush()
    print("   ✓ Logs flushed")
    
    # Summary
    print("\n" + "="*70)
    print("  ✅ QUICK START COMPLETED")
    print("="*70)
    
    print("""
📁 LOG FILES:
   • Location: ./logs/ (or your configured AGENT_LOG_FOLDER)
   • Main log: ./logs/quick_start.log
   • Check these files to see all logged messages

📚 NEXT STEPS:
   1. Review LOGGING_GUIDE.md for complete documentation
   2. Check examples/logging_observability_example.py for more patterns
   3. Integrate logging into your TaskManagerAgent
   4. Review LOGGING_INTEGRATION_GUIDE.md for agent integration

🔗 RESOURCES:
   • Documentation: LOGGING_GUIDE.md
   • Integration examples: LOGGING_INTEGRATION_GUIDE.md
   • Code examples: examples/logging_observability_example.py
   • Setup verification: python setup_logging.py
   • Summary: LOGGING_IMPLEMENTATION_SUMMARY.md

🚀 USING IN YOUR CODE:

from task_manager.config import EnvConfig
from task_manager.utils import ComprehensiveLogger

# Initialize
EnvConfig.load_env_file()
ComprehensiveLogger.initialize(**EnvConfig.get_logging_config())

# Get logger
logger = ComprehensiveLogger.get_logger(__name__)

# Log!
logger.info("My message", extra={"key": "value"})
logger.log_performance("operation", 1.23, True)

# Don't forget to flush before exit
ComprehensiveLogger.flush()

💡 PRO TIPS:
   • Use structured logging with metadata for better debugging
   • Log performance metrics for operations you care about
   • Create traces for complex workflows
   • Set log level to DEBUG during development, INFO in production
   • Check ./logs/ folder frequently for issues
    """)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
