"""
Test suite for Graph-of-Thought planning implementation.

Tests graph-based task dependencies, non-linear execution flows,
and dependency resolution logic in MasterPlanner.
"""

import unittest
from unittest.mock import MagicMock
from task_manager.core.master_planner import MasterPlanner, PlanStatus
from task_manager.models import PlanNode, BlackboardEntry, AgentState


class TestGraphOfThoughtPlanning(unittest.TestCase):
    """Test Graph-of-Thought planning with cross-branch dependencies."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.planner = MasterPlanner(llm=None)
    
    
    def test_dependency_task_ids_field(self):
        """Test that PlanNode supports dependency_task_ids field."""
        node = PlanNode(
            task_id="task_1",
            parent_id="task_0",
            depth=1,
            description="Test task with dependencies",
            status=PlanStatus.PENDING.value,
            priority=1,
            dependency_task_ids=["task_2", "task_3"],  # Depends on multiple tasks
            estimated_effort="medium"
        )
        
        self.assertEqual(node['task_id'], "task_1")
        self.assertEqual(node.get('dependency_task_ids'), ["task_2", "task_3"])
        self.assertEqual(node['priority'], 1)
    
    
    def test_get_ready_tasks_no_dependencies(self):
        """Test task selection when tasks have no dependencies."""
        plan = [
            PlanNode(
                task_id="plan_0",
                parent_id=None,
                depth=0,
                description="Root",
                status=PlanStatus.READY.value,
                priority=1,
                dependency_task_ids=[],
                estimated_effort="medium"
            ),
            PlanNode(
                task_id="plan_1",
                parent_id="plan_0",
                depth=1,
                description="Task 1",
                status=PlanStatus.PENDING.value,
                priority=2,
                dependency_task_ids=[],  # No dependencies
                estimated_effort="low"
            ),
            PlanNode(
                task_id="plan_2",
                parent_id="plan_0",
                depth=1,
                description="Task 2",
                status=PlanStatus.PENDING.value,
                priority=3,
                dependency_task_ids=[],  # No dependencies
                estimated_effort="low"
            )
        ]
        
        # Mark plan_0 as completed so plan_1 and plan_2 become ready
        plan[0]['status'] = PlanStatus.COMPLETED.value
        
        ready = self.planner.get_ready_tasks(plan)
        
        # Should have 2 ready tasks
        self.assertEqual(len(ready), 2)
        ready_ids = [t['task_id'] for t in ready]
        self.assertIn('plan_1', ready_ids)
        self.assertIn('plan_2', ready_ids)
    
    
    def test_get_ready_tasks_with_single_dependency(self):
        """Test task selection with single task dependency."""
        plan = [
            PlanNode(
                task_id="plan_0",
                parent_id=None,
                depth=0,
                description="Root",
                status=PlanStatus.COMPLETED.value,
                priority=1,
                dependency_task_ids=[],
                estimated_effort="medium"
            ),
            PlanNode(
                task_id="plan_1",
                parent_id="plan_0",
                depth=1,
                description="Search task",
                status=PlanStatus.PENDING.value,
                priority=1,
                dependency_task_ids=[],
                estimated_effort="low"
            ),
            PlanNode(
                task_id="plan_2",
                parent_id="plan_0",
                depth=1,
                description="Analysis task (depends on search)",
                status=PlanStatus.PENDING.value,
                priority=2,
                dependency_task_ids=["plan_1"],  # Depends on plan_1
                estimated_effort="medium"
            )
        ]
        
        ready = self.planner.get_ready_tasks(plan)
        
        # Only plan_1 should be ready (plan_2 depends on it)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0]['task_id'], 'plan_1')
        
        # Now mark plan_1 as completed
        plan[1]['status'] = PlanStatus.COMPLETED.value
        ready = self.planner.get_ready_tasks(plan)
        
        # Now plan_2 should be ready
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0]['task_id'], 'plan_2')
    
    
    def test_get_ready_tasks_with_multiple_dependencies(self):
        """Test task selection with multiple task dependencies (Graph-of-Thought)."""
        plan = [
            PlanNode(
                task_id="plan_0",
                parent_id=None,
                depth=0,
                description="Root",
                status=PlanStatus.COMPLETED.value,
                priority=1,
                dependency_task_ids=[],
                estimated_effort="medium"
            ),
            PlanNode(
                task_id="plan_1",
                parent_id="plan_0",
                depth=1,
                description="Search task",
                status=PlanStatus.PENDING.value,
                priority=1,
                dependency_task_ids=[],
                estimated_effort="low"
            ),
            PlanNode(
                task_id="plan_2",
                parent_id="plan_0",
                depth=1,
                description="Extract from PDF",
                status=PlanStatus.PENDING.value,
                priority=2,
                dependency_task_ids=[],
                estimated_effort="medium"
            ),
            PlanNode(
                task_id="plan_3",
                parent_id="plan_0",
                depth=1,
                description="Synthesize findings",
                status=PlanStatus.PENDING.value,
                priority=3,
                dependency_task_ids=["plan_1", "plan_2"],  # Depends on BOTH plan_1 and plan_2
                estimated_effort="high"
            )
        ]
        
        ready = self.planner.get_ready_tasks(plan)
        
        # plan_1 and plan_2 should be ready, but not plan_3
        self.assertEqual(len(ready), 2)
        ready_ids = [t['task_id'] for t in ready]
        self.assertIn('plan_1', ready_ids)
        self.assertIn('plan_2', ready_ids)
        self.assertNotIn('plan_3', ready_ids)
        
        # Mark plan_1 as completed (plan_3 still needs plan_2)
        plan[1]['status'] = PlanStatus.COMPLETED.value
        ready = self.planner.get_ready_tasks(plan)
        
        # Still only plan_2 should be ready
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0]['task_id'], 'plan_2')
        
        # Mark plan_2 as completed (now plan_3 can run)
        plan[2]['status'] = PlanStatus.COMPLETED.value
        ready = self.planner.get_ready_tasks(plan)
        
        # Now plan_3 should be ready
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0]['task_id'], 'plan_3')
    
    
    def test_parallel_execution_with_independent_tasks(self):
        """Test that independent tasks can execute in parallel."""
        plan = [
            PlanNode(
                task_id="plan_0",
                parent_id=None,
                depth=0,
                description="Root",
                status=PlanStatus.COMPLETED.value,
                priority=1,
                dependency_task_ids=[],
                estimated_effort="medium"
            ),
            PlanNode(
                task_id="plan_1",
                parent_id="plan_0",
                depth=1,
                description="Search for evidence",
                status=PlanStatus.PENDING.value,
                priority=1,
                dependency_task_ids=[],  # No dependencies - can run now
                estimated_effort="low"
            ),
            PlanNode(
                task_id="plan_2",
                parent_id="plan_0",
                depth=1,
                description="Extract PDF data",
                status=PlanStatus.PENDING.value,
                priority=2,
                dependency_task_ids=[],  # No dependencies - can run in parallel
                estimated_effort="medium"
            ),
            PlanNode(
                task_id="plan_3",
                parent_id="plan_0",
                depth=1,
                description="Analyze spreadsheet",
                status=PlanStatus.PENDING.value,
                priority=3,
                dependency_task_ids=[],  # No dependencies - can run in parallel
                estimated_effort="medium"
            )
        ]
        
        ready = self.planner.get_ready_tasks(plan)
        
        # All three tasks should be ready for parallel execution
        self.assertEqual(len(ready), 3)
        ready_ids = [t['task_id'] for t in ready]
        self.assertEqual(set(ready_ids), {'plan_1', 'plan_2', 'plan_3'})
    
    
    def test_diamond_dependency_pattern(self):
        """Test diamond dependency pattern: A->C, B->C, C->D."""
        plan = [
            PlanNode(
                task_id="plan_0",
                parent_id=None,
                depth=0,
                description="Root",
                status=PlanStatus.COMPLETED.value,
                priority=1,
                dependency_task_ids=[],
                estimated_effort="medium"
            ),
            PlanNode(
                task_id="plan_A",
                parent_id="plan_0",
                depth=1,
                description="Search A",
                status=PlanStatus.PENDING.value,
                priority=1,
                dependency_task_ids=[],
                estimated_effort="low"
            ),
            PlanNode(
                task_id="plan_B",
                parent_id="plan_0",
                depth=1,
                description="Search B",
                status=PlanStatus.PENDING.value,
                priority=2,
                dependency_task_ids=[],
                estimated_effort="low"
            ),
            PlanNode(
                task_id="plan_C",
                parent_id="plan_0",
                depth=1,
                description="Combine A and B",
                status=PlanStatus.PENDING.value,
                priority=3,
                dependency_task_ids=["plan_A", "plan_B"],  # Diamond: depends on both
                estimated_effort="medium"
            ),
            PlanNode(
                task_id="plan_D",
                parent_id="plan_0",
                depth=1,
                description="Final synthesis",
                status=PlanStatus.PENDING.value,
                priority=4,
                dependency_task_ids=["plan_C"],
                estimated_effort="high"
            )
        ]
        
        # Stage 1: A and B ready
        ready = self.planner.get_ready_tasks(plan)
        ready_ids = [t['task_id'] for t in ready]
        self.assertEqual(set(ready_ids), {'plan_A', 'plan_B'})
        
        # Stage 2: Mark A and B complete, C ready but D not yet
        plan[1]['status'] = PlanStatus.COMPLETED.value
        plan[2]['status'] = PlanStatus.COMPLETED.value
        ready = self.planner.get_ready_tasks(plan)
        ready_ids = [t['task_id'] for t in ready]
        self.assertEqual(ready_ids, ['plan_C'])  # Only C is ready
        
        # Stage 3: Mark C complete, D ready
        plan[3]['status'] = PlanStatus.COMPLETED.value
        ready = self.planner.get_ready_tasks(plan)
        ready_ids = [t['task_id'] for t in ready]
        self.assertEqual(ready_ids, ['plan_D'])  # D is ready
    
    
    def test_heuristic_plan_with_graph_dependencies(self):
        """Test heuristic planning creates correct dependency structure."""
        objective = "Search for data, extract from PDF, and summarize findings"
        
        plan = self.planner.create_initial_plan(objective, {})
        
        # Should have root + sub-tasks
        self.assertGreater(len(plan), 1)
        
        # Root should have no dependencies
        root = [p for p in plan if p['task_id'] == 'plan_0'][0]
        self.assertEqual(root.get('dependency_task_ids', []), [])
        
        # Find synthesis task if it exists (should depend on others)
        synthesis_tasks = [p for p in plan if 'synthesize' in p['description'].lower()]
        if synthesis_tasks:
            syn_task = synthesis_tasks[0]
            # Should have dependencies on other tasks
            self.assertGreater(len(syn_task.get('dependency_task_ids', [])), 0)
    
    
    def test_dependency_blocking(self):
        """Test that tasks with incomplete dependencies are blocked."""
        plan = [
            PlanNode(
                task_id="plan_0",
                parent_id=None,
                depth=0,
                description="Root",
                status=PlanStatus.COMPLETED.value,
                priority=1,
                dependency_task_ids=[],
                estimated_effort="medium"
            ),
            PlanNode(
                task_id="plan_1",
                parent_id="plan_0",
                depth=1,
                description="Task 1",
                status=PlanStatus.PENDING.value,
                priority=1,
                dependency_task_ids=[],
                estimated_effort="low"
            ),
            PlanNode(
                task_id="plan_2",
                parent_id="plan_0",
                depth=1,
                description="Task 2 (blocked)",
                status=PlanStatus.PENDING.value,
                priority=2,
                dependency_task_ids=["plan_1"],
                estimated_effort="medium"
            ),
            PlanNode(
                task_id="plan_3",
                parent_id="plan_0",
                depth=1,
                description="Task 3 (blocked by 1 and 2)",
                status=PlanStatus.PENDING.value,
                priority=3,
                dependency_task_ids=["plan_1", "plan_2"],
                estimated_effort="high"
            )
        ]
        
        ready = self.planner.get_ready_tasks(plan)
        ready_ids = [t['task_id'] for t in ready]
        
        # Only plan_1 should be ready
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready_ids[0], 'plan_1')
        
        # Other tasks should be blocked
        blocked = [t for t in plan if t['task_id'] not in ready_ids + ['plan_0']]
        self.assertEqual(len(blocked), 2)


class TestGraphOfThoughtIntegration(unittest.TestCase):
    """Integration tests for graph-of-thought with state updates."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.planner = MasterPlanner(llm=None)
    
    
    def test_complex_workflow_scenario(self):
        """Test complex real-world scenario with multiple dependency chains."""
        # Scenario: Analyze company data
        # 1. Search web for company info (parallel with PDF extraction)
        # 2. Extract from PDF documents (parallel with search)
        # 3. Process Excel spreadsheet (depends on nothing)
        # 4. Validate data across all sources (depends on 1, 2, 3)
        # 5. Generate report (depends on 4)
        
        plan = [
            PlanNode(
                task_id="root",
                parent_id=None,
                depth=0,
                description="Analyze company data",
                status=PlanStatus.COMPLETED.value,
                priority=1,
                dependency_task_ids=[],
                estimated_effort="high"
            ),
            PlanNode(
                task_id="search",
                parent_id="root",
                depth=1,
                description="Search web for company info",
                status=PlanStatus.PENDING.value,
                priority=1,
                dependency_task_ids=[],
                estimated_effort="low"
            ),
            PlanNode(
                task_id="pdf",
                parent_id="root",
                depth=1,
                description="Extract from PDF documents",
                status=PlanStatus.PENDING.value,
                priority=2,
                dependency_task_ids=[],
                estimated_effort="medium"
            ),
            PlanNode(
                task_id="excel",
                parent_id="root",
                depth=1,
                description="Process Excel spreadsheet",
                status=PlanStatus.PENDING.value,
                priority=3,
                dependency_task_ids=[],
                estimated_effort="medium"
            ),
            PlanNode(
                task_id="validate",
                parent_id="root",
                depth=1,
                description="Validate data across all sources",
                status=PlanStatus.PENDING.value,
                priority=4,
                dependency_task_ids=["search", "pdf", "excel"],  # Depends on all three
                estimated_effort="high"
            ),
            PlanNode(
                task_id="report",
                parent_id="root",
                depth=1,
                description="Generate final report",
                status=PlanStatus.PENDING.value,
                priority=5,
                dependency_task_ids=["validate"],
                estimated_effort="medium"
            )
        ]
        
        # Phase 1: Parallel execution
        ready = self.planner.get_ready_tasks(plan)
        ready_ids = {t['task_id'] for t in ready}
        self.assertEqual(ready_ids, {'search', 'pdf', 'excel'})
        
        # Phase 2: Complete search
        plan[1]['status'] = PlanStatus.COMPLETED.value
        ready = self.planner.get_ready_tasks(plan)
        ready_ids = {t['task_id'] for t in ready}
        self.assertEqual(ready_ids, {'pdf', 'excel'})  # Validate still blocked
        
        # Phase 3: Complete PDF
        plan[2]['status'] = PlanStatus.COMPLETED.value
        ready = self.planner.get_ready_tasks(plan)
        ready_ids = {t['task_id'] for t in ready}
        self.assertEqual(ready_ids, {'excel'})  # Validate still blocked
        
        # Phase 4: Complete Excel
        plan[3]['status'] = PlanStatus.COMPLETED.value
        ready = self.planner.get_ready_tasks(plan)
        ready_ids = {t['task_id'] for t in ready}
        self.assertEqual(ready_ids, {'validate'})  # Now validate is ready
        
        # Phase 5: Complete validate
        plan[4]['status'] = PlanStatus.COMPLETED.value
        ready = self.planner.get_ready_tasks(plan)
        ready_ids = {t['task_id'] for t in ready}
        self.assertEqual(ready_ids, {'report'})  # Report is ready


if __name__ == '__main__':
    unittest.main()
