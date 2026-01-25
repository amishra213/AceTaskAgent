"""
Excel Sub-Agent for handling Excel file operations.

Capabilities:
- Read Excel files and extract data
- Create Excel workbooks with formatted sheets
- Write data to specific cells/ranges
- Perform data aggregation and formatting
- Create charts and pivot tables
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import json

from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


class ExcelAgent:
    """
    Sub-agent for Excel file operations.
    
    This agent handles all Excel-related tasks:
    - Reading and extracting data from Excel files
    - Creating new Excel workbooks
    - Writing data to sheets
    - Data aggregation and formatting
    - Chart creation
    """
    
    def __init__(self):
        """Initialize Excel Agent."""
        self.supported_operations = [
            "read",
            "create",
            "write",
            "append",
            "format",
            "aggregate",
            "create_chart"
        ]
        logger.info("Excel Agent initialized")
        self._check_dependencies()
    
    
    def _check_dependencies(self):
        """Check if required Excel libraries are installed."""
        try:
            import openpyxl
            self.excel_lib = "openpyxl"
            logger.debug("Using openpyxl for Excel operations")
        except ImportError:
            try:
                import pandas as pd
                self.excel_lib = "pandas"
                logger.debug("Using pandas for Excel operations")
            except ImportError:
                logger.warning(
                    "No Excel library found. Install with: "
                    "pip install openpyxl (or pip install pandas)"
                )
                self.excel_lib = None
    
    
    def read_excel(
        self,
        file_path: str,
        sheet_name: Union[str, int, None] = None,
        header: int = 0,
        index_col: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Read and extract data from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name or index of sheet to read (None = all sheets)
            header: Row number to use as column names
            index_col: Column number to use as index
        
        Returns:
            Dictionary with extracted data
        """
        try:
            if not self.excel_lib:
                return {
                    "success": False,
                    "error": "Excel library not installed",
                    "file": file_path
                }
            
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "file": str(file_path_obj)
                }
            
            logger.info(f"Reading Excel file: {file_path}")
            
            if self.excel_lib == "pandas":
                import pandas as pd  # noqa: F401
                
                try:
                    # Read Excel file
                    if sheet_name is None:
                        # Read all sheets
                        excel_file = pd.read_excel(
                            str(file_path_obj),
                            sheet_name=None,
                            header=header,
                            index_col=index_col
                        )
                        
                        sheets_data = {}
                        for sheet_name_key, df in excel_file.items():
                            sheets_data[sheet_name_key] = {
                                "rows": len(df),
                                "columns": list(df.columns),
                                "data": df.to_dict('records'),
                                "summary": {
                                    "shape": df.shape,
                                    "dtypes": df.dtypes.to_dict(),
                                    "missing_values": df.isnull().sum().to_dict()
                                }
                            }
                        
                        logger.info(f"Read {len(sheets_data)} sheets from Excel file")
                        
                        return {
                            "success": True,
                            "file": str(file_path),
                            "sheets": list(sheets_data.keys()),
                            "data": sheets_data,
                            "total_sheets": len(sheets_data),
                            "read_at": datetime.now().isoformat()
                        }
                    else:
                        # Read specific sheet
                        df = pd.read_excel(
                            str(file_path_obj),
                            sheet_name=sheet_name,
                            header=header,
                            index_col=index_col
                        )
                        
                        logger.info(f"Read sheet '{sheet_name}' with {len(df)} rows")
                        
                        return {
                            "success": True,
                            "file": str(file_path),
                            "sheet": str(sheet_name),
                            "rows": len(df),
                            "columns": list(df.columns),
                            "data": df.to_dict('records'),
                            "summary": {
                                "shape": df.shape,
                                "dtypes": df.dtypes.to_dict(),
                                "missing_values": df.isnull().sum().to_dict()
                            },
                            "read_at": datetime.now().isoformat()
                        }
                
                except Exception as e:
                    logger.error(f"Error reading with pandas: {str(e)}")
                    raise
            
            elif self.excel_lib == "openpyxl":
                from openpyxl import load_workbook  # noqa: F401
                
                workbook = load_workbook(str(file_path_obj))
                
                if sheet_name is None:
                    # Read all sheets
                    sheets_data = {}
                    
                    for sheet_name_key in workbook.sheetnames:
                        ws = workbook[sheet_name_key]
                        data = []
                        
                        for row in ws.iter_rows(values_only=True):
                            data.append(row)
                        
                        sheets_data[sheet_name_key] = {
                            "rows": len(data),
                            "columns": data[0] if data else [],
                            "data": [dict(zip(data[0], row)) for row in data[1:]] if len(data) > 1 else [],
                            "raw_data": data
                        }
                    
                    logger.info(f"Read {len(sheets_data)} sheets from Excel file")
                    
                    return {
                        "success": True,
                        "file": str(file_path),
                        "sheets": list(sheets_data.keys()),
                        "data": sheets_data,
                        "total_sheets": len(sheets_data),
                        "read_at": datetime.now().isoformat()
                    }
                else:
                    # Read specific sheet
                    if isinstance(sheet_name, int):
                        ws = workbook.worksheets[sheet_name]
                    else:
                        ws = workbook[sheet_name]
                    
                    data = []
                    for row in ws.iter_rows(values_only=True):
                        data.append(row)
                    
                    logger.info(f"Read sheet with {len(data)} rows")
                    
                    return {
                        "success": True,
                        "file": str(file_path),
                        "sheet": str(sheet_name),
                        "rows": len(data),
                        "columns": data[0] if data else [],
                        "data": [dict(zip(data[0], row)) for row in data[1:]] if len(data) > 1 else [],
                        "raw_data": data,
                        "read_at": datetime.now().isoformat()
                    }
            
            return {
                "success": False,
                "error": "No Excel library available",
                "file": str(file_path)
            }
        
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(file_path)
            }
    
    
    def create_excel(
        self,
        output_path: str,
        sheets: Dict[str, List[List[Any]]],
        title: Optional[str] = None,
        auto_format: bool = True
    ) -> Dict[str, Any]:
        """
        Create an Excel workbook with multiple sheets.
        
        Args:
            output_path: Path where Excel file will be saved
            sheets: Dictionary of sheet_name -> data (list of rows)
            title: Workbook title
            auto_format: Whether to auto-format the sheets
        
        Returns:
            Dictionary with creation result
        """
        try:
            if not self.excel_lib:
                return {
                    "success": False,
                    "error": "Excel library not installed",
                    "output_path": output_path
                }
            
            # Validate output path
            if not output_path or output_path == '.':
                return {
                    "success": False,
                    "error": "Invalid or empty output path provided",
                    "output_path": output_path
                }
            
            output_path_obj = Path(output_path)
            
            # Create parent directory if it doesn't exist (but only if parent is not current dir)
            if output_path_obj.parent != Path('.'):
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating Excel file: {output_path}")
            
            if self.excel_lib == "pandas":
                import pandas as pd  # noqa: F401
                
                with pd.ExcelWriter(str(output_path_obj), engine='openpyxl') as writer:
                    for sheet_name, data in sheets.items():
                        if isinstance(data, list) and len(data) > 0:
                            # Assume first row is headers
                            headers = data[0]
                            rows = data[1:]
                            
                            df = pd.DataFrame(rows, columns=headers)
                            df.to_excel(writer, sheet_name=str(sheet_name), index=False)
                
                logger.info(f"Excel file created successfully: {output_path}")
                
                return {
                    "success": True,
                    "output_path": str(output_path_obj),
                    "sheets_created": len(sheets),
                    "file_size": output_path_obj.stat().st_size,
                    "created_at": datetime.now().isoformat()
                }
            
            elif self.excel_lib == "openpyxl":
                from openpyxl import Workbook  # noqa: F401
                from openpyxl.styles import Font, PatternFill, Alignment  # noqa: F401
                
                workbook = Workbook()
                
                # Remove default sheet if adding new ones
                if len(sheets) > 0:
                    if workbook.active:
                        workbook.remove(workbook.active)
                
                for sheet_name, data in sheets.items():
                    ws = workbook.create_sheet(title=str(sheet_name)[:31])  # Excel limits to 31 chars
                    
                    # Write data
                    for row_idx, row_data in enumerate(data, 1):
                        for col_idx, cell_value in enumerate(row_data, 1):
                            cell = ws.cell(row=row_idx, column=col_idx, value=cell_value)
                            
                            # Format header row if auto_format is True
                            if auto_format and row_idx == 1:
                                cell.font = Font(bold=True)
                                cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                                cell.font = Font(bold=True, color="FFFFFF")
                                cell.alignment = Alignment(horizontal="center")
                    
                    # Auto-adjust column widths
                    if auto_format:
                        for column in ws.columns:
                            max_length = 0
                            col_cell = column[0]
                            # Handle MergedCell objects by using get_column_letter
                            try:
                                # Try to get column_letter directly
                                column_letter = col_cell.column_letter  # type: ignore
                            except (AttributeError, ValueError):
                                # Fallback: use get_column_letter from openpyxl.utils
                                try:
                                    from openpyxl.utils import get_column_letter  # noqa: F401
                                    col = col_cell.column
                                    if col is not None:
                                        column_letter = get_column_letter(col)
                                    else:
                                        continue
                                except (AttributeError, ValueError):
                                    continue
                            
                            for cell in column:
                                if cell.value:
                                    max_length = max(max_length, len(str(cell.value)))
                            adjusted_width = min(max_length + 2, 50)
                            ws.column_dimensions[column_letter].width = adjusted_width
                
                workbook.save(str(output_path_obj))
                
                logger.info(f"Excel file created successfully: {output_path}")
                
                return {
                    "success": True,
                    "output_path": str(output_path_obj),
                    "sheets_created": len(sheets),
                    "file_size": output_path_obj.stat().st_size,
                    "created_at": datetime.now().isoformat()
                }
            
            return {
                "success": False,
                "error": "No Excel library available",
                "output_path": str(output_path)
            }
        
        except Exception as e:
            logger.error(f"Error creating Excel file: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "output_path": str(output_path)
            }
    
    
    def write_data(
        self,
        file_path: str,
        sheet_name: str,
        data: List[List[Any]],
        start_row: int = 1,
        start_col: int = 1,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Write data to an existing Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name to write to
            data: Data to write (list of rows)
            start_row: Starting row (1-indexed)
            start_col: Starting column (1-indexed)
            overwrite: Whether to overwrite existing data
        
        Returns:
            Dictionary with write result
        """
        try:
            if not self.excel_lib:
                return {
                    "success": False,
                    "error": "Excel library not installed",
                    "file": file_path
                }
            
            file_path_obj = Path(file_path)
            
            if self.excel_lib == "openpyxl":
                from openpyxl import load_workbook  # noqa: F401
                from openpyxl import Workbook  # noqa: F401
                
                if not file_path_obj.exists():
                    logger.warning(f"File does not exist, creating new: {file_path}")
                    workbook = Workbook()
                    ws = workbook.active
                    if ws:
                        ws.title = sheet_name
                else:
                    workbook = load_workbook(str(file_path_obj))
                    if sheet_name not in workbook.sheetnames:
                        ws = workbook.create_sheet(title=sheet_name)
                    else:
                        ws = workbook[sheet_name]
                
                # Write data
                if ws:
                    for row_idx, row_data in enumerate(data, start=start_row):
                        for col_idx, cell_value in enumerate(row_data, start=start_col):
                            ws.cell(row=row_idx, column=col_idx, value=cell_value)
                
                workbook.save(str(file_path_obj))
                
                logger.info(f"Data written successfully: {file_path}")
                
                return {
                    "success": True,
                    "file": str(file_path_obj),
                    "sheet": sheet_name,
                    "rows_written": len(data),
                    "columns_written": len(data[0]) if data else 0,
                    "written_at": datetime.now().isoformat()
                }
            
            elif self.excel_lib == "pandas":
                import pandas as pd  # noqa: F401
                
                if not file_path_obj.exists():
                    logger.warning(f"File does not exist, creating new: {file_path}")
                    df = pd.DataFrame(data)
                    with pd.ExcelWriter(str(file_path_obj), engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                else:
                    # Read existing file and append
                    with pd.ExcelWriter(str(file_path_obj), engine='openpyxl', mode='a') as writer:
                        df = pd.DataFrame(data)
                        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                
                logger.info(f"Data written successfully: {file_path}")
                
                return {
                    "success": True,
                    "file": str(file_path_obj),
                    "sheet": sheet_name,
                    "rows_written": len(data),
                    "columns_written": len(data[0]) if data else 0,
                    "written_at": datetime.now().isoformat()
                }
            
            return {
                "success": False,
                "error": "No valid Excel library",
                "file": str(file_path)
            }
        
        except Exception as e:
            logger.error(f"Error writing data: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(file_path)
            }
    
    
    def format_sheet(
        self,
        file_path: str,
        sheet_name: str,
        header_style: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply formatting to an Excel sheet.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name to format
            header_style: Header formatting options
        
        Returns:
            Dictionary with format result
        """
        try:
            if self.excel_lib != "openpyxl":
                return {
                    "success": False,
                    "error": "Formatting requires openpyxl library",
                    "file": file_path
                }
            
            from openpyxl import load_workbook  # noqa: F401
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side  # noqa: F401
            
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "file": str(file_path_obj)
                }
            
            workbook = load_workbook(str(file_path_obj))
            if sheet_name not in workbook.sheetnames:
                return {
                    "success": False,
                    "error": f"Sheet not found: {sheet_name}",
                    "file": str(file_path_obj)
                }
            
            ws = workbook[sheet_name]
            
            # Apply header formatting
            if header_style is None:
                header_style = {
                    "background_color": "366092",
                    "font_color": "FFFFFF",
                    "bold": True
                }
            
            for cell in ws[1]:  # First row
                if header_style.get("background_color"):
                    cell.fill = PatternFill(
                        start_color=header_style["background_color"],
                        end_color=header_style["background_color"],
                        fill_type="solid"
                    )
                if header_style.get("font_color"):
                    cell.font = Font(
                        bold=header_style.get("bold", False),
                        color=header_style["font_color"]
                    )
                cell.alignment = Alignment(horizontal="center", vertical="center")
            
            # Auto-adjust column widths
            for column in ws.columns:
                max_length = 0
                col_cell = column[0]
                # Handle MergedCell objects by using get_column_letter
                try:
                    # Try to get column_letter directly
                    column_letter = col_cell.column_letter  # type: ignore
                except (AttributeError, ValueError):
                    # Fallback: use get_column_letter from openpyxl.utils
                    try:
                        from openpyxl.utils import get_column_letter  # noqa: F401
                        col = col_cell.column
                        if col is not None:
                            column_letter = get_column_letter(col)
                        else:
                            continue
                    except (AttributeError, ValueError):
                        continue
                
                for cell in column:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column_letter].width = adjusted_width
            
            workbook.save(str(file_path_obj))
            
            logger.info(f"Sheet formatted successfully: {sheet_name}")
            
            return {
                "success": True,
                "file": str(file_path_obj),
                "sheet": sheet_name,
                "formatted_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error formatting sheet: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(file_path)
            }
    
    
    def execute_task(
        self,
        operation: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an Excel operation based on operation type.
        
        Args:
            operation: Type of operation (read, create, write, format, etc.)
            parameters: Operation parameters
        
        Returns:
            Result dictionary
        """
        logger.info(f"Executing Excel operation: {operation}")
        
        if operation == "read":
            return self.read_excel(
                file_path=parameters.get('file_path', ''),
                sheet_name=parameters.get('sheet_name'),
                header=parameters.get('header', 0),
                index_col=parameters.get('index_col')
            )
        
        elif operation == "create":
            # Support both 'output_path' and 'file_path' parameter names for flexibility
            output_path = parameters.get('output_path') or parameters.get('file_path', '')
            sheets = parameters.get('sheets', {})
            
            # If sheets is empty but headers are provided, create a sheet with headers
            if not sheets and parameters.get('headers'):
                sheet_name = parameters.get('sheet_name', 'Sheet1')
                headers = parameters.get('headers', [])
                sheets = {sheet_name: [headers]}  # Single row with headers
            
            return self.create_excel(
                output_path=output_path,
                sheets=sheets,
                title=parameters.get('title'),
                auto_format=parameters.get('auto_format', True)
            )
        
        elif operation == "write":
            return self.write_data(
                file_path=parameters.get('file_path', ''),
                sheet_name=parameters.get('sheet_name', ''),
                data=parameters.get('data', []),
                start_row=parameters.get('start_row', 1),
                start_col=parameters.get('start_col', 1),
                overwrite=parameters.get('overwrite', False)
            )
        
        elif operation == "format":
            return self.format_sheet(
                file_path=parameters.get('file_path', ''),
                sheet_name=parameters.get('sheet_name', ''),
                header_style=parameters.get('header_style')
            )
        
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}",
                "supported_operations": self.supported_operations
            }
