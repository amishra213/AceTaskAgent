"""
Input Context Manager - Scans input folder and builds context for agent processing.

This module provides:
1. File discovery and cataloging from input folder
2. Content extraction and summarization for different file types
3. Context building for agent state and prompts
4. File metadata tracking for easy reference
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
import hashlib


@dataclass
class FileInfo:
    """Information about a single input file."""
    path: Path
    name: str
    extension: str
    size_bytes: int
    modified_time: datetime
    file_type: str  # pdf, excel, image, text, other
    content_hash: str
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": str(self.path),
            "name": self.name,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "modified_time": self.modified_time.isoformat(),
            "file_type": self.file_type,
            "content_hash": self.content_hash,
            "summary": self.summary,
            "metadata": self.metadata
        }


class InputContext:
    """
    Manages input folder scanning, file cataloging, and context building.
    
    This class:
    1. Scans the input folder for user-provided files
    2. Categorizes files by type (PDF, Excel, images, text)
    3. Extracts basic metadata and optionally content summaries
    4. Builds a context object that can be passed to all agents
    """
    
    # File type mappings
    FILE_TYPE_MAPPINGS = {
        'pdf': ['.pdf'],
        'excel': ['.xlsx', '.xls', '.xlsm', '.csv'],
        'image': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'],
        'text': ['.txt', '.md', '.rst', '.json', '.xml', '.yaml', '.yml'],
        'document': ['.doc', '.docx', '.rtf', '.odt'],
        'data': ['.json', '.xml', '.csv', '.parquet'],
    }
    
    def __init__(self, input_folder: Union[str, Path], auto_scan: bool = True):
        """
        Initialize the InputContext manager.
        
        Args:
            input_folder: Path to the input folder
            auto_scan: Whether to automatically scan on initialization
        """
        self.input_folder = Path(input_folder).resolve()
        self.files: Dict[str, FileInfo] = {}  # path -> FileInfo
        self.files_by_type: Dict[str, List[FileInfo]] = {}
        self.scan_time: Optional[datetime] = None
        self._context_cache: Optional[Dict] = None
        
        # Ensure folder exists
        self.input_folder.mkdir(parents=True, exist_ok=True)
        
        if auto_scan:
            self.scan()
    
    def scan(self) -> Dict[str, List[FileInfo]]:
        """
        Scan the input folder and catalog all files.
        
        Returns:
            Dictionary mapping file types to list of FileInfo objects
        """
        self.files.clear()
        self.files_by_type.clear()
        self._context_cache = None
        
        # Initialize type buckets
        for file_type in self.FILE_TYPE_MAPPINGS.keys():
            self.files_by_type[file_type] = []
        self.files_by_type['other'] = []
        
        # Scan recursively
        for root, dirs, files in os.walk(self.input_folder):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in files:
                if filename.startswith('.'):
                    continue
                    
                file_path = Path(root) / filename
                file_info = self._create_file_info(file_path)
                
                self.files[str(file_path)] = file_info
                self.files_by_type[file_info.file_type].append(file_info)
        
        self.scan_time = datetime.now()
        return self.files_by_type
    
    def _create_file_info(self, file_path: Path) -> FileInfo:
        """Create FileInfo for a single file."""
        stat = file_path.stat()
        extension = file_path.suffix.lower()
        
        # Determine file type
        file_type = 'other'
        for ftype, extensions in self.FILE_TYPE_MAPPINGS.items():
            if extension in extensions:
                file_type = ftype
                break
        
        # Calculate content hash (first 8KB for performance)
        content_hash = self._calculate_hash(file_path)
        
        return FileInfo(
            path=file_path,
            name=file_path.name,
            extension=extension,
            size_bytes=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            file_type=file_type,
            content_hash=content_hash,
            summary=None,
            metadata={}
        )
    
    def _calculate_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Calculate MD5 hash of file (first chunk only for large files)."""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(chunk_size)
                hasher.update(chunk)
            return hasher.hexdigest()[:16]
        except Exception:
            return "unknown"
    
    def get_files(self, file_type: Optional[str] = None) -> List[FileInfo]:
        """
        Get files, optionally filtered by type.
        
        Args:
            file_type: Optional file type filter (pdf, excel, image, text, other)
            
        Returns:
            List of FileInfo objects
        """
        if file_type:
            return self.files_by_type.get(file_type, [])
        return list(self.files.values())
    
    def get_file_by_name(self, name: str) -> Optional[FileInfo]:
        """Get file info by filename."""
        for file_info in self.files.values():
            if file_info.name == name:
                return file_info
        return None
    
    def get_file_by_path(self, path: Union[str, Path]) -> Optional[FileInfo]:
        """Get file info by path."""
        return self.files.get(str(Path(path).resolve()))
    
    def build_context(self, include_summaries: bool = True) -> Dict[str, Any]:
        """
        Build a context dictionary for agent consumption.
        
        This context can be added to agent metadata and passed to all sub-agents.
        
        Args:
            include_summaries: Whether to include file summaries
            
        Returns:
            Context dictionary with input folder information
        """
        if self._context_cache:
            return self._context_cache
        
        context = {
            "input_folder": str(self.input_folder),
            "scan_time": self.scan_time.isoformat() if self.scan_time else None,
            "total_files": len(self.files),
            "files_by_type": {
                ftype: len(files) for ftype, files in self.files_by_type.items()
            },
            "file_list": [],
            "available_data": {
                "pdfs": [],
                "excel_files": [],
                "images": [],
                "text_files": [],
                "other_files": []
            }
        }
        
        # Build detailed file list
        for file_info in self.files.values():
            file_entry = {
                "name": file_info.name,
                "path": str(file_info.path),
                "type": file_info.file_type,
                "size_kb": round(file_info.size_bytes / 1024, 2),
                "modified": file_info.modified_time.strftime("%Y-%m-%d %H:%M")
            }
            
            if include_summaries and file_info.summary:
                file_entry["summary"] = file_info.summary
            
            context["file_list"].append(file_entry)
            
            # Also add to type-specific lists
            if file_info.file_type == 'pdf':
                context["available_data"]["pdfs"].append(file_info.name)
            elif file_info.file_type == 'excel':
                context["available_data"]["excel_files"].append(file_info.name)
            elif file_info.file_type == 'image':
                context["available_data"]["images"].append(file_info.name)
            elif file_info.file_type == 'text':
                context["available_data"]["text_files"].append(file_info.name)
            else:
                context["available_data"]["other_files"].append(file_info.name)
        
        # Build a human-readable summary
        context["summary"] = self._build_summary_text()
        
        self._context_cache = context
        return context
    
    def _build_summary_text(self) -> str:
        """Build a human-readable summary of input files."""
        if not self.files:
            return "No input files found in the input folder."
        
        parts = [f"Found {len(self.files)} file(s) in input folder:"]
        
        for file_type, files in self.files_by_type.items():
            if files:
                parts.append(f"  - {file_type.upper()}: {len(files)} file(s)")
                for f in files[:3]:  # Show first 3
                    parts.append(f"      • {f.name} ({round(f.size_bytes/1024, 1)} KB)")
                if len(files) > 3:
                    parts.append(f"      ... and {len(files) - 3} more")
        
        return "\n".join(parts)
    
    def get_context_for_prompt(self) -> str:
        """
        Get a formatted context string for inclusion in LLM prompts.
        
        Returns:
            Formatted string describing available input files
        """
        if not self.files:
            return "No user-provided input files available."
        
        lines = [
            "USER-PROVIDED INPUT DATA:",
            f"Location: {self.input_folder}",
            f"Files Available: {len(self.files)}",
            ""
        ]
        
        for file_type, files in self.files_by_type.items():
            if files:
                lines.append(f"{file_type.upper()} FILES:")
                for f in files:
                    size_str = f"{round(f.size_bytes/1024, 1)} KB"
                    lines.append(f"  - {f.name} ({size_str})")
                    if f.summary:
                        lines.append(f"    Summary: {f.summary[:100]}...")
                lines.append("")
        
        lines.append("Use these files as data sources when relevant to the task.")
        
        return "\n".join(lines)
    
    def set_file_summary(self, file_path: Union[str, Path], summary: str) -> bool:
        """
        Set a summary for a file (after content extraction).
        
        Args:
            file_path: Path to the file
            summary: Summary text
            
        Returns:
            True if file was found and updated
        """
        path_key = str(Path(file_path).resolve())
        if path_key in self.files:
            self.files[path_key].summary = summary
            self._context_cache = None  # Invalidate cache
            return True
        return False
    
    def set_file_metadata(self, file_path: Union[str, Path], metadata: Dict[str, Any]) -> bool:
        """
        Set metadata for a file.
        
        Args:
            file_path: Path to the file
            metadata: Metadata dictionary
            
        Returns:
            True if file was found and updated
        """
        path_key = str(Path(file_path).resolve())
        if path_key in self.files:
            self.files[path_key].metadata.update(metadata)
            self._context_cache = None
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "input_folder": str(self.input_folder),
            "scan_time": self.scan_time.isoformat() if self.scan_time else None,
            "files": {k: v.to_dict() for k, v in self.files.items()}
        }
    
    def save_catalog(self, output_path: Union[str, Path]) -> None:
        """Save file catalog to JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __bool__(self) -> bool:
        return len(self.files) > 0
