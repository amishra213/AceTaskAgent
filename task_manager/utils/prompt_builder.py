"""
Prompt builder module - Constructs prompts for LLM interactions
"""

import json
from typing import Dict, Any, Optional, Union

try:
    from task_manager.models import Task
except ImportError:
    Task = Any  # type: ignore


class PromptBuilder:
    """
    Utility class for building structured prompts for the Task Manager Agent.
    
    This centralizes all prompt construction logic, making it easier to
    maintain and test LLM interactions.
    """
    
    @staticmethod
    def build_analysis_prompt(
        objective: str,
        task: Union[Dict[str, Any], Any],
        metadata: Dict[str, Any],
        current_depth: int = 0,
        depth_limit: int = 5,
        input_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build analysis prompt for task breakdown decision.
        
        Args:
            objective: Main objective of the agent
            task: Current task being analyzed (as dict or Task TypedDict)
            metadata: Additional context
            current_depth: Current depth in task hierarchy
            depth_limit: Maximum allowed depth
            input_context: Context from user-provided input files
            
        Returns:
            Formatted prompt string
        """
        # Ensure task is converted to dict (handles both Task TypedDict and regular dicts)
        if hasattr(task, '__dict__'):
            task_dict = task.__dict__
        else:
            task_dict = dict(task) if isinstance(task, dict) else task
        
        # Add depth-awareness to guidelines
        depth_guidance = ""
        if current_depth >= 2:
            depth_guidance = f"""
IMPORTANT DEPTH CONSTRAINT:
- Current depth: {current_depth} (approaching limit of {depth_limit})
- At depth 2+, STRONGLY PREFER "execute" or specific agent actions (web_search_task, etc.)
- Only use "breakdown" if absolutely necessary for complex multi-step operations
- For research/search tasks, use "web_search_task" directly
- For data analysis, use "code_interpreter_task" directly
"""
        
        # Build input context section if available (from DataExtractionAgent)
        input_context_section = ""
        if input_context and input_context.get('success'):
            extractions = input_context.get('extractions', [])
            if extractions:
                input_context_section = f"""
USER-PROVIDED INPUT DATA (Relevance-Filtered):
Summary: {input_context.get('summary', 'No summary available')}

Extracted Data ({len(extractions)} files, {input_context.get('total_content_size', 0)} chars):
"""
                for ext in extractions[:5]:  # Limit to top 5 most relevant
                    file_name = ext.get('file_name', 'Unknown')
                    file_type = ext.get('file_type', 'unknown')
                    relevance = ext.get('relevance_score', 0)
                    key_points = ext.get('key_points', [])
                    
                    input_context_section += f"""
--- {file_name} ({file_type}, relevance: {relevance:.2f}) ---
"""
                    if key_points:
                        input_context_section += "Key Points:\n"
                        for point in key_points[:3]:
                            input_context_section += f"  • {point[:150]}\n"
                    
                    # Include truncated content preview
                    content = ext.get('extracted_content', '')
                    if content and len(content) > 100:
                        preview = content[:500] + "..." if len(content) > 500 else content
                        input_context_section += f"Preview:\n{preview}\n"
                
                if len(extractions) > 5:
                    input_context_section += f"\n... and {len(extractions) - 5} more files available\n"
                
                input_context_section += """
IMPORTANT: Use data from these input files when relevant to the task.
- Use "pdf_task" for deeper PDF analysis
- Use "excel_task" for Excel operations  
- Use "ocr_task" for image text extraction
- Use "data_extraction_task" for more detailed extraction from specific files
"""
        
        prompt = f"""
You are analyzing a task as part of a recursive task management system.

MAIN OBJECTIVE: {objective}
{input_context_section}
CURRENT TASK:
ID: {task_dict.get('id', 'unknown')}
Description: {task_dict.get('description', 'no description')}
Depth: {task_dict.get('depth', 0)}
Context: {task_dict.get('context', 'no context')}

METADATA: {json.dumps(metadata, indent=2)}
{depth_guidance}

Analyze this task and respond with ONLY a JSON object:
{{
  "action": "breakdown" or "execute" or "skip" or "pdf_task" or "excel_task" or "ocr_task" or "web_search_task" or "code_interpreter_task" or "data_extraction_task",
  "reasoning": "brief explanation",
  "subtasks": ["subtask1", "subtask2", ...] or null,
  "search_query": "query for web search" or null,
  "file_operation": {{"type": "pdf" or "excel" or "ocr" or "web_search" or "code_interpreter" or "data_extraction", "operation": "operation_name", "parameters": {{...}}}} or null,
  "estimated_complexity": "low/medium/high",
  "requires_human_review": false
}}

Guidelines:
- Use "breakdown" if task is complex and needs splitting into 2-5 subtasks
- Use "execute" if task is specific enough to be completed now (e.g., search for data)
- Use "skip" if task is already implicit in other tasks or not needed
- Use "pdf_task" if task involves PDF file operations (read, create, merge, extract)
- Use "excel_task" if task involves Excel file operations (read, create, write, format)
- Use "ocr_task" if task involves OCR or image extraction (extract text from images, extract images from PDFs/docs)
- Use "web_search_task" if task involves web search or content retrieval (search, scrape, fetch, summarize)
- Use "code_interpreter_task" if task involves data analysis, code generation, or computational tasks (analyze data, generate code, create visualizations)
- Use "data_extraction_task" if task requires extracting specific data from input files or searching within input folder contents
- Provide search_query only if action is "execute" and requires web search
- Provide file_operation only if action is "pdf_task", "excel_task", "ocr_task", "web_search_task", "code_interpreter_task", or "data_extraction_task"
  - For PDF: operations are "read", "create", "merge", "extract_pages"
  - For Excel: operations are "read", "create", "write", "format"
  - For OCR: operations are "ocr_image", "extract_images_from_pdf", "batch_ocr", "process_screenshot"
  - For Web Search: operations are "search", "scrape", "fetch", "summarize", "smart_scrape", "capture_screenshot", "handle_pagination"
  - For Data Extraction: operations are "extract_relevant", "search_content", "get_file_preview", "summarize_folder"
- Be concise and specific in subtasks
- Set requires_human_review to false (human review happens later if needed)
"""
        return prompt.strip()
    
    
    @staticmethod
    def build_search_prompt(
        search_query: str,
        task_description: str,
        objective: str
    ) -> str:
        """
        Build prompt for executing a search task.
        
        Args:
            search_query: Query to search for
            task_description: Description of the current task
            objective: Main objective for context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Search for: {search_query}

Task context: {task_description}
Main objective: {objective}

Extract and summarize the key data points relevant to this task.
Provide structured information that can be aggregated later.
"""
        return prompt.strip()
    
    
    @staticmethod
    def build_aggregation_prompt(
        task_results: list,
        objective: str,
        output_format: Optional[str] = None
    ) -> str:
        """
        Build prompt for aggregating task results.
        
        Args:
            task_results: List of task results to aggregate
            objective: Main objective for context
            output_format: Desired format for aggregated results
            
        Returns:
            Formatted prompt string
        """
        format_instruction = f"Format the output as: {output_format}" if output_format else ""
        
        prompt = f"""
Aggregate the following task results into a cohesive summary.

Main objective: {objective}

Results to aggregate:
{json.dumps(task_results, indent=2)}

{format_instruction}

Provide a well-structured summary that consolidates all findings.
"""
        return prompt.strip()
    
    
    @staticmethod
    def build_synthesis_prompt(
        objective: str,
        blackboard_entries: list,
        objective_context: Dict[str, Any],
        input_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for synthesizing multi-level research findings.
        
        This prompt instructs the LLM to:
        1. Analyze all blackboard entries across hierarchy levels
        2. Compare WebSearch findings against PDF/Excel data
        3. Identify and flag numerical contradictions
        4. Synthesize into a research brief format
        5. Flag if human review is needed for conflicts
        
        Args:
            objective: Main research objective
            blackboard_entries: All blackboard findings from all agents and levels
            objective_context: Additional context about the research task
            input_context: Context from user-provided input files (from DataExtractionAgent)
            
        Returns:
            Formatted synthesis prompt for LLM
        """
        # Build input context section if available (from DataExtractionAgent)
        input_files_section = ""
        if input_context and input_context.get('success'):
            extractions = input_context.get('extractions', [])
            if extractions:
                input_files_section = f"""
USER-PROVIDED INPUT DATA:
{input_context.get('summary', 'Input files processed')}

Key data from input files:
"""
                for ext in extractions[:5]:
                    file_name = ext.get('file_name', 'Unknown')
                    key_points = ext.get('key_points', [])
                    if key_points:
                        input_files_section += f"From {file_name}:\n"
                        for point in key_points[:2]:
                            input_files_section += f"  • {point[:100]}\n"
                
                input_files_section += """
IMPORTANT: Data from input files should be treated as PRIMARY sources. 
Cross-reference web findings against input file data.
"""
        
        prompt = f"""
You are reviewing a comprehensive research analysis conducted across multiple agents and hierarchical levels.

RESEARCH OBJECTIVE:
{objective}
{input_files_section}
CONTEXT:
{json.dumps(objective_context, indent=2)}

BLACKBOARD FINDINGS (All Findings Across All Levels):
{json.dumps(blackboard_entries, indent=2)}

Your task is to:

1. ANALYZE SOURCE DATA
   - Identify all WebSearch findings (from web_search_agent)
   - Identify all PDF data (from pdf_agent)
   - Identify all Excel data (from excel_agent)
   - Identify all OCR/image findings (from ocr_agent)
   - Note the hierarchical level (depth) of each finding

2. CROSS-REFERENCE FINDINGS
   - Compare numerical values across sources
   - Look for contradictions in percentages, dates, counts, amounts
   - Note if same metric has different values from different sources
   - Consider context: are contradictions due to different time periods or scopes?

3. FLAG CONTRADICTIONS
   - For each contradiction, note:
     * The metric/data point in question
     * Values from each source
     * Sources (agents/levels) reporting conflicting data
     * Severity: CRITICAL, HIGH, MEDIUM, LOW
   - CRITICAL: Core objective data conflicts significantly
   - HIGH: Important metrics differ by >10%
   - MEDIUM: Supporting data with minor discrepancies
   - LOW: Conflicting metadata or formatting

4. SYNTHESIZE INTO RESEARCH BRIEF
   - Format as professional research summary
   - Structure:
     a) Executive Summary (2-3 sentences)
     b) Key Findings (bullet points, synthesized across sources)
     c) Data Sources (which agents/levels contributed)
     d) Contradictions & Conflicts (if any)
     e) Confidence Assessment (based on agreement across sources)
     f) Recommendations (for handling conflicts)

5. DETERMINE IF HUMAN REVIEW NEEDED
   - Set requires_human_review = true if:
     * Any CRITICAL contradictions found
     * Core metrics differ significantly between sources
     * Conflicting data about same fact from different levels
     * Unclear which source is more reliable
   - Set requires_human_review = false if:
     * All data is consistent across sources
     * Minor contradictions are clearly due to scope/time differences
     * High confidence in synthesized findings

RESPOND WITH ONLY A JSON OBJECT:
{{
  "executive_summary": "2-3 sentence summary of findings",
  "key_findings": [
    "Finding 1 (synthesized from sources)",
    "Finding 2 with data from [agents]",
    "..."
  ],
  "data_sources": {{
    "web_search": ["summary of what was searched"],
    "pdf": ["documents analyzed"],
    "excel": ["data processed"],
    "ocr": ["images analyzed"]
  }},
  "contradictions": [
    {{
      "metric": "metric name or data point",
      "sources": {{"source_agent1": "value1", "source_agent2": "value2"}},
      "severity": "CRITICAL|HIGH|MEDIUM|LOW",
      "explanation": "Why this is a contradiction and what might explain it"
    }},
    ...
  ],
  "confidence_level": "HIGH|MEDIUM|LOW",
  "confidence_explanation": "Why we can/cannot trust these findings",
  "synthesis": "Full research brief formatted as professional summary",
  "requires_human_review": true or false,
  "human_review_reason": "Why human review is needed (if required)" or null
}}

Guidelines:
- Be thorough in cross-referencing data
- Distinguish between real conflicts and differences due to scope/methodology
- Prioritize accuracy over completion
- Flag uncertainty clearly
- Recommend human review if there's any doubt about data reliability
- Consider the hierarchical context: findings at different depths may have different scopes
"""
        return prompt.strip()
