#!/usr/bin/env python3
"""
Test suite for execution tracer and diagnostic enhancements.

Validates that:
1. State snapshots are captured correctly
2. Event transactions are tracked
3. Data audits work properly
4. Routing decisions are recorded
5. Node execution times are measured
6. Diagnostic reports can be generated
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any

from task_manager.utils.execution_tracer import ExecutionTracer, StateSnapshot, EventTransaction, DataTransactionAudit


class TestExecutionTracer:
    """Test the ExecutionTracer functionality."""
    
    def setup_method(self):
        """Initialize tracer before each test."""
        self.tracer = ExecutionTracer(enable_detailed_logging=True, max_history=100)
    
    def test_tracer_initialization(self):
        """Test that tracer initializes correctly."""
        assert self.tracer is not None
        assert self.tracer.enable_detailed_logging is True
        assert self.tracer.max_history == 100
        assert len(self.tracer.state_snapshots) == 0
        assert len(self.tracer.event_transactions) == 0
        assert len(self.tracer.data_audits) == 0
    
    def test_state_snapshot_recording(self):
        """Test recording state snapshots."""
        # Create a mock state
        state = {
            "objective": "Test objective",
            "tasks": [
                {"id": "1", "status": "COMPLETED", "description": "Task 1"},
                {"id": "2", "status": "PENDING", "description": "Task 2"},
            ],
            "completed_task_ids": ["1"],
            "failed_task_ids": [],
            "active_task_id": "2",
            "iteration_count": 5,
            "blackboard": [
                {"source_agent": "test_agent", "content": "Finding 1"}
            ]
        }
        
        # Record snapshot
        self.tracer.record_state_snapshot("test_node", state, phase="entry", notes="Testing")
        
        # Verify snapshot was recorded
        assert len(self.tracer.state_snapshots) == 1
        snapshot = self.tracer.state_snapshots[0]
        
        assert snapshot.node_name == "test_node"
        assert snapshot.phase == "entry"
        assert snapshot.notes == "Testing"
        assert snapshot.tasks_count == 2
        assert snapshot.completed_tasks == 1
        assert snapshot.pending_tasks == 1
        assert snapshot.failed_tasks == 0
        assert snapshot.active_task_id == "2"
        assert snapshot.iteration_count == 5
        assert snapshot.blackboard_size == 1
        assert len(snapshot.state_hash) == 16  # SHA256 first 16 chars
        assert snapshot.state_size_bytes > 0
    
    def test_multiple_state_snapshots(self):
        """Test recording multiple snapshots."""
        state1 = {"tasks": [{"id": "1", "status": "PENDING"}], "completed_task_ids": []}
        state2 = {"tasks": [{"id": "1", "status": "COMPLETED"}], "completed_task_ids": ["1"]}
        
        self.tracer.record_state_snapshot("node1", state1, phase="entry")
        self.tracer.record_state_snapshot("node1", state2, phase="exit")
        
        assert len(self.tracer.state_snapshots) == 2
        
        # Hashes should be different
        hash1 = self.tracer.state_snapshots[0].state_hash
        hash2 = self.tracer.state_snapshots[1].state_hash
        assert hash1 != hash2
    
    def test_state_snapshot_history_limit(self):
        """Test that history is trimmed when exceeding max."""
        # Set small history limit
        tracer = ExecutionTracer(enable_detailed_logging=True, max_history=5)
        
        state = {"tasks": [], "completed_task_ids": []}
        
        # Record more than limit
        for i in range(10):
            tracer.record_state_snapshot(f"node_{i}", state, phase="entry")
        
        # Should only keep last 5
        assert len(tracer.state_snapshots) == 5
    
    def test_event_transaction_recording(self):
        """Test recording event transactions."""
        payload = {
            "task_id": "1.2.3",
            "data": "Test data"
        }
        
        event_id = self.tracer.record_event_transaction(
            event_type="task_completed",
            source_agent="web_search_agent",
            target_agent="synthesis_node",
            payload=payload,
            operation="publish",
            status="initiated"
        )
        
        assert event_id.startswith("evt_")
        assert len(self.tracer.event_transactions) == 1
        
        txn = self.tracer.event_transactions[0]
        assert txn.event_type == "task_completed"
        assert txn.source_agent == "web_search_agent"
        assert txn.target_agent == "synthesis_node"
        assert txn.operation == "publish"
        assert txn.status == "initiated"
        assert len(txn.payload_hash) == 16
        assert txn.payload_size_bytes > 0
    
    def test_data_transaction_audit(self):
        """Test recording data transactions."""
        data = {
            "research_findings": {
                "trend_1": "Supply chain diversification",
                "trend_2": "Digital transformation"
            },
            "source": "web_search"
        }
        
        self.tracer.record_data_transaction(
            from_agent="web_search_agent",
            to_agent="document_agent",
            data_key="research_findings",
            data=data,
            operation_type="output"
        )
        
        assert len(self.tracer.data_audits) == 1
        
        audit = self.tracer.data_audits[0]
        assert audit.from_agent == "web_search_agent"
        assert audit.to_agent == "document_agent"
        assert audit.data_key == "research_findings"
        assert audit.operation_type == "output"
        assert audit.validation_status == "valid"
        assert len(audit.data_hash) == 16
        assert audit.data_size_bytes > 0
    
    def test_data_validation_states(self):
        """Test data validation for different data states."""
        # Valid data
        self.tracer.record_data_transaction(
            "agent1", "agent2", "valid_data", {"key": "value"}, "input"
        )
        assert self.tracer.data_audits[-1].validation_status == "valid"
        
        # Missing data (None)
        self.tracer.record_data_transaction(
            "agent1", "agent2", "missing_data", None, "input"
        )
        assert self.tracer.data_audits[-1].validation_status == "missing"
        
        # Empty dict
        self.tracer.record_data_transaction(
            "agent1", "agent2", "empty_data", {}, "input"
        )
        assert self.tracer.data_audits[-1].validation_status == "invalid"
        
        # Empty string
        self.tracer.record_data_transaction(
            "agent1", "agent2", "empty_string", "   ", "input"
        )
        assert self.tracer.data_audits[-1].validation_status == "invalid"
    
    def test_routing_decision_recording(self):
        """Test recording routing decisions."""
        analysis_data = {
            "action": "execute_document_task",
            "reasoning": "Document formatting required",
            "confidence": 0.95
        }
        
        self.tracer.record_routing_decision(
            current_node="analyze_task",
            active_task_id="1.1.4",
            decision="execute_document_task",
            reasoning="Task requires document formatting",
            analysis_data=analysis_data
        )
        
        assert len(self.tracer.routing_decisions) == 1
        
        decision = self.tracer.routing_decisions[0]
        assert decision["current_node"] == "analyze_task"
        assert decision["active_task_id"] == "1.1.4"
        assert decision["decision"] == "execute_document_task"
        assert decision["reasoning"] == "Task requires document formatting"
        assert decision["analysis_data"] == analysis_data
    
    def test_node_execution_timing(self):
        """Test recording node execution times."""
        self.tracer.record_node_execution_time("execute_document_task", 125.45)
        self.tracer.record_node_execution_time("execute_document_task", 130.20)
        self.tracer.record_node_execution_time("select_task", 45.67)
        
        assert "execute_document_task" in self.tracer.node_execution_times
        assert "select_task" in self.tracer.node_execution_times
        
        doc_times = self.tracer.node_execution_times["execute_document_task"]
        assert len(doc_times) == 2
        assert 125.45 in doc_times
        assert 130.20 in doc_times
    
    def test_timing_statistics(self):
        """Test that timing report calculates statistics."""
        times = [100.0, 120.0, 110.0, 130.0, 105.0]
        for t in times:
            self.tracer.record_node_execution_time("test_node", t)
        
        report = self.tracer.get_timing_report()
        
        # Report should contain node name and timing info
        assert "test_node" in report
        assert "Avg:" in report
        assert "Min:" in report
        assert "Max:" in report
        assert "Calls:" in report
    
    def test_state_transition_report(self):
        """Test generating state transition report."""
        state = {"tasks": [], "completed_task_ids": []}
        
        self.tracer.record_state_snapshot("node1", state, phase="entry")
        self.tracer.record_state_snapshot("node1", state, phase="exit")
        self.tracer.record_state_snapshot("node2", state, phase="entry")
        
        report = self.tracer.get_state_transition_report()
        
        assert "STATE TRANSITION REPORT" in report
        assert "node1:entry" in report
        assert "node1:exit" in report  # Using colon without space
        assert "node2:entry" in report
    
    def test_event_transaction_report(self):
        """Test generating event transaction report."""
        self.tracer.record_event_transaction(
            "task_completed",
            "web_search_agent",
            "synthesis_node",
            {"task_id": "1"},
            "publish"
        )
        
        report = self.tracer.get_event_transaction_report()
        
        assert "EVENT TRANSACTION REPORT" in report
        assert "publish" in report.lower()  # Check lowercase version
        assert "web_search_agent" in report
        assert "synthesis_node" in report
        assert "task_completed" in report
    
    def test_routing_decision_report(self):
        """Test generating routing decision report."""
        self.tracer.record_routing_decision(
            "analyze_task",
            "1.1.4",
            "execute_document_task",
            "Document formatting required"
        )
        
        report = self.tracer.get_routing_decision_report()
        
        assert "ROUTING DECISION REPORT" in report
        assert "analyze_task" in report
        assert "1.1.4" in report
        assert "execute_document_task" in report
    
    def test_full_diagnostic_report(self):
        """Test generating full diagnostic report."""
        # Add some data
        state = {"tasks": [{"id": "1"}], "completed_task_ids": []}
        self.tracer.record_state_snapshot("node1", state, phase="entry")
        self.tracer.record_event_transaction(
            "test_event",
            "agent1",
            "agent2",
            {"data": "test"}
        )
        self.tracer.record_routing_decision(
            "node1",
            "task1",
            "node2",
            "Test routing"
        )
        self.tracer.record_node_execution_time("node1", 100.0)
        
        report = self.tracer.get_full_diagnostic_report()
        
        assert "COMPREHENSIVE EXECUTION DIAGNOSTIC REPORT" in report
        assert "STATE TRANSITION REPORT" in report
        assert "EVENT TRANSACTION REPORT" in report
        assert "ROUTING DECISION REPORT" in report
        assert "NODE EXECUTION TIMING REPORT" in report
        assert f"State snapshots: {len(self.tracer.state_snapshots)}" in report
        assert f"Event transactions: {len(self.tracer.event_transactions)}" in report
        assert f"Routing decisions: {len(self.tracer.routing_decisions)}" in report
    
    def test_hash_consistency(self):
        """Test that identical states produce identical hashes."""
        state = {
            "task": "Format document",
            "status": "PENDING",
            "priority": 1
        }
        
        # Record same state twice
        self.tracer.record_state_snapshot("node", state, phase="entry")
        hash1 = self.tracer.state_snapshots[0].state_hash
        
        self.tracer.record_state_snapshot("node", state, phase="entry")
        hash2 = self.tracer.state_snapshots[1].state_hash
        
        # Same state should produce same hash
        assert hash1 == hash2
    
    def test_hash_different_for_different_states(self):
        """Test that different states produce different hashes."""
        state1 = {"task": "Task 1", "status": "PENDING"}
        state2 = {"task": "Task 1", "status": "COMPLETED"}
        
        self.tracer.record_state_snapshot("node", state1, phase="entry")
        hash1 = self.tracer.state_snapshots[0].state_hash
        
        self.tracer.record_state_snapshot("node", state2, phase="entry")
        hash2 = self.tracer.state_snapshots[1].state_hash
        
        # Different states should produce different hashes
        assert hash1 != hash2
    
    def test_data_serialization_to_dict(self):
        """Test that all data types can be serialized to dict."""
        state = {"tasks": [], "completed_task_ids": []}
        self.tracer.record_state_snapshot("node", state, phase="entry")
        
        snapshot = self.tracer.state_snapshots[0]
        snapshot_dict = snapshot.to_dict()
        
        # Should be serializable to JSON
        json_str = json.dumps(snapshot_dict)
        assert len(json_str) > 0


class TestExecutionTracerIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_full_workflow_trace(self):
        """Test tracing a complete workflow execution."""
        tracer = ExecutionTracer(enable_detailed_logging=True)
        
        # Simulate workflow
        initial_state = {
            "objective": "Test objective",
            "tasks": [
                {"id": "1", "status": "PENDING"},
                {"id": "2", "status": "PENDING"},
            ],
            "completed_task_ids": [],
            "failed_task_ids": [],
            "active_task_id": "1",
            "iteration_count": 0,
            "blackboard": []
        }
        
        # Node 1: Initialize
        state1 = {**initial_state, "iteration_count": 1}
        tracer.record_state_snapshot("initialize", initial_state, phase="entry")
        tracer.record_state_snapshot("initialize", state1, phase="exit")
        
        # Node 2: Select task
        tracer.record_state_snapshot("select_task", state1, phase="entry")
        state2 = {**state1}
        tracer.record_state_snapshot("select_task", state2, phase="exit")
        
        # Node 3: Analyze task
        tracer.record_state_snapshot("analyze_task", state2, phase="entry")
        tracer.record_routing_decision(
            "analyze_task", "1", "execute_document_task",
            "Analysis determined document execution needed"
        )
        state3 = {**state2}
        tracer.record_state_snapshot("analyze_task", state3, phase="exit")
        
        # Node 4: Execute
        tracer.record_state_snapshot("execute_document_task", state3, phase="entry")
        tracer.record_data_transaction(
            "workflow", "document_agent",
            "task_input", {"description": "Format document"},
            "input"
        )
        state4 = {
            **state3,
            "completed_task_ids": ["1"],
            "active_task_id": "2",
            "blackboard": [{"content": "Document formatted"}]
        }
        tracer.record_data_transaction(
            "document_agent", "workflow",
            "formatted_document", {"file": "output.docx"},
            "output"
        )
        tracer.record_node_execution_time("execute_document_task", 245.67)
        tracer.record_state_snapshot("execute_document_task", state4, phase="exit")
        
        # Verify comprehensive trace
        report = tracer.get_full_diagnostic_report()
        
        assert len(tracer.state_snapshots) == 8  # 4 nodes × 2 (entry/exit)
        assert len(tracer.routing_decisions) == 1
        assert len(tracer.data_audits) == 2
        assert len(tracer.node_execution_times) == 1
        
        # Verify reports can be generated
        assert "STATE TRANSITION REPORT" in report
        assert "ROUTING DECISION REPORT" in report
        assert len(report) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
