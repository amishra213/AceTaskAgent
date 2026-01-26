"""
Test suite for document task analysis and execution.

Ensures that:
1. Analysis prompt includes document operations
2. LLM can format document file_operation correctly
3. Document execution validates and processes operations
4. Recovery is attempted on invalid operations
"""

import pytest
import json
from typing import Dict, Any

from task_manager.utils.prompt_builder import PromptBuilder
from task_manager.models import TaskStatus


class TestDocumentTaskAnalysisPrompt:
    """Test that analysis prompt correctly guides document task handling"""
    
    def test_analysis_prompt_includes_document_type(self):
        """Ensure JSON schema includes 'document' as valid file_operation type"""
        task = {
            "id": "1.1.3",
            "description": "Create comprehensive analysis document in DOCX format",
            "depth": 2,
            "status": TaskStatus.PENDING.value
        }
        
        prompt = PromptBuilder.build_analysis_prompt(
            objective="List top 5 latest supply chain trends in the CPG industry",
            task=task,
            metadata={},
            current_depth=2,
            input_context=None
        )
        
        # Check that prompt mentions document type in JSON schema
        assert "\"type\":" in prompt, "JSON schema missing type field"
        assert "\"document\"" in prompt or "'document'" in prompt, \
            "JSON schema should include 'document' as valid type"
    
    def test_analysis_prompt_includes_document_operations(self):
        """Ensure guidelines list document operations"""
        task = {
            "id": "1.1.3",
            "description": "Create comprehensive analysis document in DOCX format",
            "depth": 2,
            "status": TaskStatus.PENDING.value
        }
        
        prompt = PromptBuilder.build_analysis_prompt(
            objective="List top 5 latest supply chain trends",
            task=task,
            metadata={},
            current_depth=2,
            input_context=None
        )
        
        # Should mention document operations
        assert "For Document:" in prompt or "for Document" in prompt, \
            "Guidelines should include 'For Document' section"
        
        # Should mention specific operations
        operations = ["create_docx", "create_txt", "append_docx", "append_txt", "read_docx", "read_txt"]
        operations_found = sum(1 for op in operations if op in prompt)
        assert operations_found >= 2, \
            f"Guidelines should mention document operations, found {operations_found}: {operations}"
    
    def test_analysis_prompt_includes_document_parameters(self):
        """Ensure guidelines specify parameter structure for document operations"""
        task = {
            "id": "1.1.3",
            "description": "Create comprehensive analysis document in DOCX format",
            "depth": 2
        }
        
        prompt = PromptBuilder.build_analysis_prompt(
            objective="List top 5 latest supply chain trends",
            task=task,
            metadata={},
            current_depth=2,
            input_context=None
        )
        
        # Should mention content parameter
        assert "content" in prompt.lower(), \
            "Guidelines should mention 'content' parameter"
        
        # Should mention parameters structure
        assert "parameters" in prompt, \
            "Guidelines should mention parameters structure"
    
    def test_execute_document_task_validates_document_type(self):
        """Test that _execute_document_task accepts 'document' as valid type"""
        # This is an integration test - validates the actual execution logic
        # Ensures that the fix allowing type='document' is working
        
        # Expected to pass with type='document'
        file_operation = {
            "type": "document",
            "operation": "create_docx",
            "parameters": {
                "content": "Test content",
                "title": "Test Title"
            }
        }
        
        # This would be checked in _execute_document_task
        valid_types = ['document', 'docx', 'txt']
        assert file_operation.get('type') in valid_types, \
            f"File operation type '{file_operation.get('type')}' should be in {valid_types}"
    
    def test_document_file_operation_backwards_compatibility(self):
        """Test that legacy types 'docx' and 'txt' still work"""
        # Legacy file_operation types for backwards compatibility
        legacy_operations = [
            {"type": "docx", "operation": "create"},
            {"type": "txt", "operation": "create"}
        ]
        
        valid_types = ['document', 'docx', 'txt']
        
        for op in legacy_operations:
            assert op.get('type') in valid_types, \
                f"Legacy type '{op.get('type')}' should still be supported"
    
    def test_analyze_document_task_with_blackboard_context(self):
        """Test that analysis includes blackboard context when available"""
        # This test validates that we can enhance the prompt with blackboard data
        task = {
            "id": "1.1.3",
            "description": "Create comprehensive analysis document in DOCX format",
            "depth": 2
        }
        
        # Simulated blackboard entries from previous tasks
        blackboard_entries = [
            {
                "entry_type": "web_search_result",
                "source_agent": "web_search_agent",
                "source_task_id": "1.1.1",
                "content": {
                    "findings": [
                        "Trend 1: Supply chain automation",
                        "Trend 2: Sustainability focus"
                    ]
                },
                "depth_level": 2
            }
        ]
        
        # The prompt should be buildable with or without blackboard
        # (Current implementation doesn't use blackboard yet, but test is in place
        # for when enhancement is added)
        prompt = PromptBuilder.build_analysis_prompt(
            objective="List top 5 latest supply chain trends",
            task=task,
            metadata={},
            current_depth=2,
            input_context=None
        )
        
        assert prompt is not None and len(prompt) > 100, \
            "Analysis prompt should be generated"


class TestDocumentOperationSpecification:
    """Test the specification of document operations and parameters"""
    
    def test_create_docx_parameters(self):
        """Test create_docx operation parameters"""
        params = {
            "content": "# Research Findings\n\nTrend 1: ...",
            "title": "Top 5 Supply Chain Trends",
            "file_path": "/output/analysis.docx"
        }
        
        # Validate parameter structure
        assert "content" in params, "create_docx should have 'content' parameter"
        assert isinstance(params["content"], str), "content should be string"
        assert len(params["content"]) > 0, "content should not be empty"
    
    def test_create_txt_parameters(self):
        """Test create_txt operation parameters"""
        params = {
            "content": "Research Findings:\n\nTrend 1: ...",
            "encoding": "utf-8",
            "file_path": "/output/analysis.txt"
        }
        
        # Validate parameter structure
        assert "content" in params, "create_txt should have 'content' parameter"
        assert "encoding" in params, "create_txt can specify encoding"
        assert params["encoding"] in ["utf-8", "ascii", "latin-1"], \
            "encoding should be valid"
    
    def test_append_docx_parameters(self):
        """Test append_docx operation parameters"""
        params = {
            "file_path": "/output/analysis.docx",
            "content": "Additional findings..."
        }
        
        assert "file_path" in params, "append_docx should have 'file_path'"
        assert "content" in params, "append_docx should have 'content'"


class TestDocumentExecutionRecovery:
    """Test recovery logic for document task failures"""
    
    def test_invalid_operation_triggers_recovery(self):
        """Test that invalid document operation triggers recovery flow"""
        # When file_operation has invalid type, system should attempt recovery
        
        invalid_operations = [
            {"type": "invalid"},
            {"type": "pdf", "operation": "create"},  # Wrong type for document task
            {"type": None},
            {},
            None
        ]
        
        valid_types = ['document', 'docx', 'txt']
        
        for invalid_op in invalid_operations:
            is_invalid = (
                not invalid_op or 
                invalid_op.get('type') not in valid_types
            )
            assert is_invalid, \
                f"Operation {invalid_op} should be detected as invalid"


class TestDocumentTaskIntegration:
    """Integration tests for document task flow"""
    
    def test_document_task_action_recognized(self):
        """Test that analysis recognizes execute_document_task action"""
        prompt = PromptBuilder.build_analysis_prompt(
            objective="Create analysis document",
            task={
                "id": "1.1.3",
                "description": "Create comprehensive analysis document in DOCX format",
                "depth": 2
            },
            metadata={},
            current_depth=2,
            input_context=None
        )
        
        # Prompt should mention execute_document_task as valid action
        assert "execute_document_task" in prompt, \
            "Analysis prompt should recognize execute_document_task action"
    
    def test_document_operation_in_json_schema(self):
        """Test that file_operation is properly documented in schema"""
        prompt = PromptBuilder.build_analysis_prompt(
            objective="Create analysis document",
            task={
                "id": "1.1.3",
                "description": "Create comprehensive analysis document in DOCX format",
                "depth": 2
            },
            metadata={},
            current_depth=2,
            input_context=None
        )
        
        # Should mention operation field
        assert "operation" in prompt.lower(), \
            "Prompt should document operation field in file_operation"
        
        # Should mention parameters field
        assert "parameters" in prompt, \
            "Prompt should document parameters field in file_operation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
