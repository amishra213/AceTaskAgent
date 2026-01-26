#!/usr/bin/env python
"""
Test script to verify Unicode logging works on Windows
"""

import os
import sys

print("=" * 70)
print("Unicode Logging Test on Windows")
print("=" * 70)
print(f"Python Version: {sys.version}")
print(f"Console Encoding: {sys.stdout.encoding}")
print()

# Test 1: Import logger
print("Test 1: Import logger")
from task_manager.utils.logger import get_logger

logger = get_logger(__name__)
print("✓ Logger imported successfully")
print()

# Test 2: Log messages with Unicode characters
print("Test 2: Logging messages with Unicode characters")
print()

logger.debug("DEBUG: Testing ✓ checkmark and © copyright")
logger.info("INFO: Testing ✓ checkmark and © copyright")
logger.warning("WARNING: Testing ✓ checkmark and © copyright")
logger.error("ERROR: Testing ✓ checkmark and © copyright")

print()
print("Test 3: Various Unicode characters")
print()

# Test various Unicode symbols
test_messages = [
    "Testing checkmark: ✓",
    "Testing cross mark: ✗",
    "Testing bullet: •",
    "Testing arrows: → ← ↑ ↓",
    "Testing stars: ★ ☆",
    "Testing hearts: ♥ ♦ ♣ ♠",
    "Testing copyright: © ® ™",
    "Testing mathematical: ± × ÷ ≈ ≠ ≤ ≥",
]

for msg in test_messages:
    logger.info(f"[UNICODE TEST] {msg}")

print()
print("=" * 70)
print("✓ All Unicode tests completed successfully!")
print("=" * 70)
