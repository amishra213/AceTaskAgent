"""
PDF Sub-Agent for handling PDF file operations.

Capabilities:
- Read PDF files and extract text/data
- Create PDF documents with formatted content
- Merge multiple PDF files
- Extract specific pages from PDFs
- Add metadata and annotations

Migration Status:
- Week 8 Day 1: ✅ COMPLETED - Legacy support removed, standardized-only interface
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import json
import time

from task_manager.utils.logger import get_logger

# Import standardized schemas and utilities (Week 1-2 implementation)
from task_manager.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    create_system_event,
    create_error_response
)
from task_manager.utils import (
    auto_convert_response,
    validate_agent_execution_response,
    exception_to_error_response
)
from task_manager.core.event_bus import get_event_bus

logger = get_logger(__name__)


class PDFAgent:
    """
    Sub-agent for PDF file operations.
    
    This agent handles all PDF-related tasks:
    - Reading and extracting data from PDFs
    - Creating new PDF documents
    - Merging PDFs
    - Page extraction
    - Metadata management
    """
    
    def __init__(self):
        """Initialize PDF Agent with dual-format support."""
        self.agent_name = "pdf_agent"
        self.supported_operations = [
            "read",
            "create",
            "merge",
            "extract_pages",
            "add_metadata",
            "extract_text"
        ]
        
        # Initialize event bus for event-driven workflows
        self.event_bus = get_event_bus()
        
        logger.info("PDF Agent initialized with dual-format support")
        self._check_dependencies()
    
    
    def _check_dependencies(self):
        """Check if required PDF libraries are installed."""
        try:
            import PyPDF2  # noqa: F401
            self.pdf_lib = "PyPDF2"
            logger.debug("Using PyPDF2 for PDF operations")
        except ImportError:
            try:
                import pypdf  # noqa: F401
                self.pdf_lib = "pypdf"
                logger.debug("Using pypdf for PDF operations")
            except ImportError:
                logger.warning(
                    "No PDF library found. Install with: "
                    "pip install PyPDF2 (or pip install pypdf)"
                )
                self.pdf_lib = None
    
    
    def read_pdf(
        self,
        file_path: str,
        extract_text: bool = True,
        extract_tables: bool = False,
        page_range: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """
        Read and extract data from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            extract_text: Whether to extract text content
            extract_tables: Whether to attempt table extraction
            page_range: Tuple of (start_page, end_page) to extract
        
        Returns:
            Dictionary with extracted data
        """
        try:
            if not self.pdf_lib:
                return {
                    "success": False,
                    "error": "PDF library not installed",
                    "file": file_path
                }
            
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "file": str(file_path_obj)
                }
            
            logger.info(f"Reading PDF: {file_path}")
            
            if self.pdf_lib == "PyPDF2":
                import PyPDF2  # noqa: F401
                pdf_reader = PyPDF2.PdfReader(str(file_path))
                num_pages = len(pdf_reader.pages)
                
                # Determine page range
                start = page_range[0] if page_range else 0
                end = page_range[1] if page_range else num_pages
                start = max(0, start)
                end = min(num_pages, end)
                
                # Extract text
                text_content = []
                metadata = {
                    "file": str(file_path),
                    "total_pages": num_pages,
                    "extracted_pages": f"{start+1}-{end}",
                    "extraction_time": datetime.now().isoformat()
                }
                
                if extract_text:
                    for page_num in range(start, end):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        text_content.append({
                            "page": page_num + 1,
                            "content": text,
                            "length": len(text) if text else 0
                        })
                
                logger.info(f"Extracted {len(text_content)} pages from PDF")
                
                return {
                    "success": True,
                    "file": str(file_path),
                    "metadata": metadata,
                    "text_content": text_content,
                    "page_count": num_pages,
                    "extracted_data": "\n\n".join([
                        f"--- Page {p['page']} ---\n{p['content']}"
                        for p in text_content
                    ])
                }
            
            elif self.pdf_lib == "pypdf":
                import pypdf  # noqa: F401
                pdf_reader = pypdf.PdfReader(str(file_path))
                num_pages = len(pdf_reader.pages)
                
                # Similar extraction logic for pypdf
                start = page_range[0] if page_range else 0
                end = page_range[1] if page_range else num_pages
                start = max(0, start)
                end = min(num_pages, end)
                
                text_content = []
                metadata = {
                    "file": str(file_path),
                    "total_pages": num_pages,
                    "extracted_pages": f"{start+1}-{end}",
                    "extraction_time": datetime.now().isoformat()
                }
                
                if extract_text:
                    for page_num in range(start, end):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        text_content.append({
                            "page": page_num + 1,
                            "content": text,
                            "length": len(text) if text else 0
                        })
                
                logger.info(f"Extracted {len(text_content)} pages from PDF")
                
                return {
                    "success": True,
                    "file": str(file_path),
                    "metadata": metadata,
                    "text_content": text_content,
                    "page_count": num_pages,
                    "extracted_data": "\n\n".join([
                        f"--- Page {p['page']} ---\n{p['content']}"
                        for p in text_content
                    ])
                }
            
            return {
                "success": False,
                "error": "No PDF library available",
                "file": str(file_path)
            }
        
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(file_path)
            }
    
    
    def create_pdf(
        self,
        output_path: str,
        content: List[Dict[str, Any]],
        title: Optional[str] = None,
        author: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a PDF document from content.
        
        Args:
            output_path: Path where PDF will be saved
            content: List of content items, each with 'type' and 'data'
                - type: 'text', 'heading', 'table', etc.
                - data: content string or structure
            title: PDF title/subject
            author: PDF author
        
        Returns:
            Dictionary with creation result
        """
        try:
            if not self.pdf_lib:
                return {
                    "success": False,
                    "error": "PDF library not installed",
                    "output_path": output_path
                }
            
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating PDF: {output_path}")
            
            # Use reportlab for better PDF creation
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.lib import colors
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                
                # Create PDF document
                doc = SimpleDocTemplate(
                    str(output_path),
                    pagesize=letter,
                    title=title or "Generated PDF",
                    author=author or "Task Manager Agent"
                )
                
                story = []
                styles = getSampleStyleSheet()
                
                # Add title if provided
                if title:
                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Heading1'],
                        fontSize=24,
                        textColor=colors.HexColor('#1f4788'),
                        spaceAfter=30,
                        alignment=1  # Center alignment
                    )
                    story.append(Paragraph(title, title_style))
                    story.append(Spacer(1, 0.3*inch))
                
                # Process content
                for item in content:
                    item_type = item.get('type', 'text').lower()
                    data = item.get('data', '')
                    
                    if item_type == 'heading':
                        story.append(Paragraph(str(data), styles['Heading2']))
                        story.append(Spacer(1, 0.2*inch))
                    
                    elif item_type == 'text':
                        story.append(Paragraph(str(data), styles['Normal']))
                        story.append(Spacer(1, 0.1*inch))
                    
                    elif item_type == 'table':
                        if isinstance(data, list) and len(data) > 0:
                            table = Table(data)
                            table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, 0), 12),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black)
                            ]))
                            story.append(table)
                            story.append(Spacer(1, 0.2*inch))
                    
                    elif item_type == 'pagebreak':
                        story.append(PageBreak())
                
                # Build PDF
                doc.build(story)
                
                logger.info(f"PDF created successfully: {output_path}")
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "file_size": output_path_obj.stat().st_size,
                    "items_added": len(content),
                    "created_at": datetime.now().isoformat()
                }
            
            except ImportError:
                logger.warning("reportlab not installed. Using basic PDF creation with PyPDF2.")
                return {
                    "success": False,
                    "error": "reportlab required for PDF creation. Install with: pip install reportlab",
                    "output_path": str(output_path)
                }
        
        except Exception as e:
            logger.error(f"Error creating PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "output_path": str(output_path)
            }
    
    
    def merge_pdfs(
        self,
        input_files: List[str],
        output_path: str
    ) -> Dict[str, Any]:
        """
        Merge multiple PDF files into one.
        
        Args:
            input_files: List of PDF file paths to merge
            output_path: Path for the output merged PDF
        
        Returns:
            Dictionary with merge result
        """
        try:
            if not self.pdf_lib:
                return {
                    "success": False,
                    "error": "PDF library not installed",
                    "output_path": output_path
                }
            
            logger.info(f"Merging {len(input_files)} PDFs")
            
            if self.pdf_lib == "PyPDF2":
                import PyPDF2  # noqa: F401
                
                pdf_merger = PyPDF2.PdfMerger()
                
                for file_path in input_files:
                    file_path_obj = Path(file_path)
                    if not file_path_obj.exists():
                        logger.warning(f"File not found, skipping: {file_path}")
                        continue
                    
                    pdf_merger.append(str(file_path_obj))
                
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                
                pdf_merger.write(str(output_path_obj))
                pdf_merger.close()
                
                logger.info(f"PDFs merged successfully: {output_path}")
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "merged_files": len(input_files),
                    "file_size": output_path_obj.stat().st_size,
                    "created_at": datetime.now().isoformat()
                }
            
            elif self.pdf_lib == "pypdf":
                import pypdf  # noqa: F401
                
                pdf_merger = pypdf.PdfWriter()
                
                for file_path in input_files:
                    file_path_obj = Path(file_path)
                    if not file_path_obj.exists():
                        logger.warning(f"File not found, skipping: {file_path}")
                        continue
                    
                    # Read and merge pages
                    reader = pypdf.PdfReader(str(file_path_obj))
                    for page in reader.pages:
                        pdf_merger.add_page(page)
                
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                
                pdf_merger.write(str(output_path_obj))
                pdf_merger.close()
                
                logger.info(f"PDFs merged successfully: {output_path}")
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "merged_files": len(input_files),
                    "file_size": output_path_obj.stat().st_size,
                    "created_at": datetime.now().isoformat()
                }
            
            return {
                "success": False,
                "error": "No PDF library available",
                "output_path": output_path
            }
        
        except Exception as e:
            logger.error(f"Error merging PDFs: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "output_path": str(output_path)
            }
    
    
    def extract_pages(
        self,
        input_path: str,
        page_numbers: List[int],
        output_path: str
    ) -> Dict[str, Any]:
        """
        Extract specific pages from a PDF.
        
        Args:
            input_path: Source PDF file
            page_numbers: List of page numbers to extract (1-indexed)
            output_path: Path for the output PDF
        
        Returns:
            Dictionary with extraction result
        """
        try:
            if not self.pdf_lib:
                return {
                    "success": False,
                    "error": "PDF library not installed",
                    "output_path": output_path
                }
            
            input_path_obj = Path(input_path)
            if not input_path_obj.exists():
                return {
                    "success": False,
                    "error": f"Input file not found: {input_path}",
                    "output_path": output_path
                }
            
            logger.info(f"Extracting pages {page_numbers} from {input_path}")
            
            if self.pdf_lib == "PyPDF2":
                import PyPDF2  # noqa: F401
                
                pdf_reader = PyPDF2.PdfReader(str(input_path_obj))
                pdf_writer = PyPDF2.PdfWriter()
                
                # Convert 1-indexed to 0-indexed
                for page_num in page_numbers:
                    if 1 <= page_num <= len(pdf_reader.pages):
                        pdf_writer.add_page(pdf_reader.pages[page_num - 1])
                
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path_obj, 'wb') as output_file:
                    pdf_writer.write(output_file)
                
                logger.info(f"Pages extracted successfully: {output_path}")
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "pages_extracted": len(page_numbers),
                    "file_size": output_path_obj.stat().st_size,
                    "created_at": datetime.now().isoformat()
                }
            
            elif self.pdf_lib == "pypdf":
                import pypdf  # noqa: F401
                
                pdf_reader = pypdf.PdfReader(str(input_path_obj))
                pdf_writer = pypdf.PdfWriter()
                
                for page_num in page_numbers:
                    if 1 <= page_num <= len(pdf_reader.pages):
                        pdf_writer.add_page(pdf_reader.pages[page_num - 1])
                
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path_obj, 'wb') as output_file:
                    pdf_writer.write(output_file)
                
                logger.info(f"Pages extracted successfully: {output_path}")
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "pages_extracted": len(page_numbers),
                    "file_size": output_path_obj.stat().st_size,
                    "created_at": datetime.now().isoformat()
                }
            
            return {
                "success": False,
                "error": "No PDF library available",
                "output_path": str(output_path)
            }
        
        except Exception as e:
            logger.error(f"Error extracting pages: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "output_path": str(output_path)
            }
    
    
    def execute_task(self, request: AgentExecutionRequest) -> AgentExecutionResponse:
        """
        Execute a PDF operation using standardized interface.
        
        Args:
            request: AgentExecutionRequest with operation and parameters
        
        Returns:
            AgentExecutionResponse: Standardized response
        """
        start_time = time.time()
        
        try:
            # Extract request parameters
            task_id = request.get("task_id", f"pdf_{int(time.time())}")
            operation = request.get("operation", "unknown")
            parameters = request.get("parameters", {})
            
            logger.info(f"[{self.agent_name}] Executing operation: {operation} (task_id={task_id})")
            
            # Execute the operation using existing methods
            if operation == "read":
                result = self.read_pdf(
                    file_path=parameters.get('file_path', ''),
                    extract_text=parameters.get('extract_text', True),
                    extract_tables=parameters.get('extract_tables', False),
                    page_range=parameters.get('page_range')
                )
            
            elif operation == "create":
                result = self.create_pdf(
                    output_path=parameters.get('output_path', ''),
                    content=parameters.get('content', []),
                    title=parameters.get('title'),
                    author=parameters.get('author')
                )
            
            elif operation == "merge":
                result = self.merge_pdfs(
                    input_files=parameters.get('input_files', []),
                    output_path=parameters.get('output_path', '')
                )
            
            elif operation == "extract_pages":
                result = self.extract_pages(
                    input_path=parameters.get('input_path', ''),
                    page_numbers=parameters.get('page_numbers', []),
                    output_path=parameters.get('output_path', '')
                )
            
            else:
                result = {
                    "success": False,
                    "error": f"Unknown operation: {operation}",
                    "supported_operations": self.supported_operations
                }
            
            # Convert legacy result to standardized response
            standard_response = self._convert_to_standard_response(
                result,
                operation,
                task_id,
                start_time
            )
            
            # Publish completion event for event-driven workflows
            self._publish_completion_event(task_id, operation, standard_response)
            
            return standard_response
        
        except Exception as e:
            logger.error(f"[{self.agent_name}] Error executing task: {e}", exc_info=True)
            
            # Create standardized error response
            error = exception_to_error_response(
                e,
                source=self.agent_name,
                task_id=request.get("task_id", "unknown")
            )
            
            error_response: AgentExecutionResponse = {
                "status": "failure",
                "success": False,
                "result": {},
                "artifacts": [],
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "timestamp": datetime.now().isoformat(),
                "agent_name": self.agent_name,
                "operation": request.get("operation", "unknown"),
                "blackboard_entries": [],
                "warnings": []
            }
            # Add error field separately to handle TypedDict
            error_response["error"] = error  # type: ignore
            
            return error_response
    
    def _convert_to_standard_response(
        self,
        legacy_result: Dict[str, Any],
        operation: str,
        task_id: str,
        start_time: float
    ) -> AgentExecutionResponse:
        """Convert legacy result dict to standardized AgentExecutionResponse."""
        success = legacy_result.get("success", False)
        
        # Extract artifacts from result
        artifacts = []
        output_file = legacy_result.get("file") or legacy_result.get("output_path")
        if output_file and success:
            file_path = Path(output_file)
            if file_path.exists():
                artifacts.append({
                    "type": "pdf",
                    "path": str(output_file),
                    "size_bytes": file_path.stat().st_size,
                    "description": f"PDF output from {operation} operation"
                })
        
        # Create blackboard entries for sharing data
        blackboard_entries = []
        if success and "text_content" in legacy_result:
            blackboard_entries.append({
                "key": f"pdf_text_{task_id}",
                "value": legacy_result.get("extracted_data", ""),
                "scope": "workflow",
                "ttl_seconds": 3600
            })
        
        # Build standardized response
        response: AgentExecutionResponse = {
            "status": "success" if success else "failure",
            "success": success,
            "result": {
                k: v for k, v in legacy_result.items()
                if k not in ["success", "file", "output_path"]
            },
            "artifacts": artifacts,
            "execution_time_ms": int((time.time() - start_time) * 1000),
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.agent_name,
            "operation": operation,
            "blackboard_entries": blackboard_entries,
            "warnings": []
        }
        
        # Add error field if present (handle TypedDict)
        if not success and "error" in legacy_result:
            response["error"] = create_error_response(  # type: ignore
                error_code="PDF_001",
                error_type="execution_error",
                message=legacy_result.get("error", "Unknown error"),
                source=self.agent_name
            )
        
        return response
    
    def _publish_completion_event(
        self,
        task_id: str,
        operation: str,
        response: AgentExecutionResponse
    ):
        """Publish task completion event for event-driven workflows."""
        try:
            event = create_system_event(
                event_type="pdf_extraction_completed" if operation == "read" else "pdf_operation_completed",
                event_category="task_lifecycle",
                source_agent=self.agent_name,
                payload={
                    "task_id": task_id,
                    "operation": operation,
                    "success": response["success"],
                    "artifacts": response["artifacts"],
                    "blackboard_keys": [entry["key"] for entry in response["blackboard_entries"]]
                }
            )
            self.event_bus.publish(event)
            logger.debug(f"Published completion event for task {task_id}")
        except Exception as e:
            logger.warning(f"Failed to publish completion event: {e}")
