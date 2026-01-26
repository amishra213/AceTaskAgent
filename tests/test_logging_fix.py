#!/usr/bin/env python
"""
Test script to verify logging file creation and functionality
"""

import os
import sys
import time

# Test 1: Check if logs folder exists initially
print("=" * 70)
print("Test 1: Initial log folder check")
print("=" * 70)
logs_folder = "./logs"
if os.path.exists(logs_folder):
    print(f"✓ Logs folder already exists: {os.path.abspath(logs_folder)}")
    # List contents
    files = os.listdir(logs_folder)
    print(f"  Current files: {files if files else 'None'}")
else:
    print(f"✗ Logs folder does not exist: {os.path.abspath(logs_folder)}")
print()

# Test 2: Initialize logger and check file creation
print("=" * 70)
print("Test 2: Logger initialization")
print("=" * 70)
from task_manager.utils.logger import get_logger

logger = get_logger(__name__)
print(f"✓ Logger created: {type(logger)}")
print()

# Test 3: Log messages at different levels
print("=" * 70)
print("Test 3: Logging messages")
print("=" * 70)
logger.debug("This is a DEBUG message")
logger.info("This is an INFO message")
logger.warning("This is a WARNING message")
logger.error("This is an ERROR message")
print("✓ Messages logged at all levels")
print()

# Small delay to ensure file writes
time.sleep(0.5)

# Test 4: Check if log files were created
print("=" * 70)
print("Test 4: Log file verification")
print("=" * 70)
if os.path.exists(logs_folder):
    files = os.listdir(logs_folder)
    print(f"✓ Logs folder exists: {os.path.abspath(logs_folder)}")
    print(f"  Files created: {files}")
    
    # Check for __main__.log file
    main_log = os.path.join(logs_folder, "__main__.log")
    if os.path.exists(main_log):
        size = os.path.getsize(main_log)
        print(f"✓ Main log file created: {main_log} ({size} bytes)")
        
        # Show file contents
        with open(main_log, 'r') as f:
            contents = f.read()
            print(f"\n  File contents (first 500 chars):")
            print("  " + "-" * 66)
            for line in contents[:500].split('\n'):
                print(f"  {line}")
            if len(contents) > 500:
                print("  ...")
            print("  " + "-" * 66)
    else:
        print(f"✗ Main log file NOT created: {main_log}")
else:
    print(f"✗ Logs folder does not exist: {os.path.abspath(logs_folder)}")

print()
print("=" * 70)
print("Test completed!")
print("=" * 70)
