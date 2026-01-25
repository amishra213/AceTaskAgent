"""
Unit tests for the Task Manager Agent

This module contains tests for the core components of the Task Manager Agent.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime

from task_manager.models import Task, TaskStatus, AgentState
from task_manager.config import AgentConfig, LLMConfig
from task_manager.utils import PromptBuilder


class TestTaskModel(unittest.TestCase):
    """Tests for Task model."""
    
    def test_task_creation(self):
        """Test creating a task."""
        task = Task(
            id="1",
            description="Test task",
            status=TaskStatus.PENDING,
            parent_id=None,
            depth=0,
            context="Test context",
            result=None,
            error=None,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.assertEqual(task['id'], "1")
        self.assertEqual(task['status'], TaskStatus.PENDING)
        self.assertIsNone(task['parent_id'])


class TestAgentConfig(unittest.TestCase):
    """Tests for AgentConfig."""
    
    def test_valid_config(self):
        """Test creating valid configuration."""
        config = AgentConfig(
            max_iterations=50,
            llm=LLMConfig(
                provider="anthropic",
                model_name="claude-opus-4-20250805",
                api_key="sk-ant-test-key",
                temperature=0.5
            ),
            enable_search=True
        )
        
        self.assertEqual(config.max_iterations, 50)
        self.assertEqual(config.llm.temperature, 0.5)
        self.assertTrue(config.enable_search)
    
    def test_invalid_temperature(self):
        """Test invalid temperature raises error."""
        with self.assertRaises(ValueError):
            LLMConfig(
                provider="anthropic",
                model_name="claude-opus-4-20250805",
                api_key="sk-ant-test-key",
                temperature=3.0
            )
    
    def test_invalid_max_iterations(self):
        """Test invalid max_iterations raises error."""
        with self.assertRaises(ValueError):
            AgentConfig(max_iterations=0)
    
    def test_invalid_log_level(self):
        """Test invalid log_level raises error."""
        with self.assertRaises(ValueError):
            AgentConfig(log_level="INVALID")


class TestPromptBuilder(unittest.TestCase):
    """Tests for PromptBuilder utility."""
    
    def test_analysis_prompt_building(self):
        """Test building analysis prompt."""
        task = Task(
            id="1",
            description="Test task",
            status=TaskStatus.PENDING,
            parent_id=None,
            depth=0,
            context="Test context",
            result=None,
            error=None,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        prompt = PromptBuilder.build_analysis_prompt(
            objective="Test objective",
            task=task,
            metadata={}
        )
        
        self.assertIn("Test objective", prompt)
        self.assertIn("Test task", prompt)
        self.assertIn("breakdown", prompt)
        self.assertIn("execute", prompt)
    
    def test_search_prompt_building(self):
        """Test building search prompt."""
        prompt = PromptBuilder.build_search_prompt(
            search_query="Test query",
            task_description="Search for data",
            objective="Main goal"
        )
        
        self.assertIn("Test query", prompt)
        self.assertIn("Search for data", prompt)
        self.assertIn("Main goal", prompt)


if __name__ == '__main__':
    unittest.main()
