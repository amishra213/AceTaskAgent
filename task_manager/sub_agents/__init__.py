"""
Sub-agents for specialized file operations and code execution.

This module provides dedicated sub-agents for handling:
- PDF file operations (read, create, extract, merge)
- Excel file operations (read, create, format, aggregate)
- OCR and image extraction (text extraction from images, image extraction from documents)
- Web search and content retrieval (search, scrape, fetch, summarize)
- Code interpretation and execution (generate and execute Python code for data analysis)
- Data extraction (intelligent extraction of relevant data from input/temp files)
- Problem solving (LLM-based error analysis, solution generation, human input interpretation)
"""

from .pdf_agent import PDFAgent
from .excel_agent import ExcelAgent
from .ocr_image_agent import OCRImageAgent
from .web_search_agent import WebSearchAgent
from .code_interpreter_agent import CodeInterpreterAgent
from .data_extraction_agent import DataExtractionAgent
from .problem_solver_agent import ProblemSolverAgent

__all__ = [
    "PDFAgent",
    "ExcelAgent",
    "OCRImageAgent",
    "WebSearchAgent",
    "CodeInterpreterAgent",
    "DataExtractionAgent",
    "ProblemSolverAgent",
]
