"""
Temp Data Manager - Manages temporary data storage for agent operations.

This module provides:
1. Structured temp folder organization for tasks, caches, checkpoints
2. Easy save/load for intermediate results
3. Search and retrieval capabilities
4. Automatic cleanup policies
5. Persistence for long-running tasks
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib

from .exceptions import InvalidParameterError


@dataclass
class TempDataEntry:
    """Metadata for a temp data entry."""
    key: str
    category: str  # tasks, cache, checkpoints, results, blackboard
    created_at: datetime
    updated_at: datetime
    size_bytes: int
    data_type: str  # json, text, binary
    task_id: Optional[str] = None
    description: Optional[str] = None
    ttl_hours: Optional[int] = None  # Time to live in hours
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "category": self.category,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "size_bytes": self.size_bytes,
            "data_type": self.data_type,
            "task_id": self.task_id,
            "description": self.description,
            "ttl_hours": self.ttl_hours
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TempDataEntry":
        return cls(
            key=data["key"],
            category=data["category"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            size_bytes=data["size_bytes"],
            data_type=data["data_type"],
            task_id=data.get("task_id"),
            description=data.get("description"),
            ttl_hours=data.get("ttl_hours")
        )


class TempDataManager:
    """
    Manages temporary data storage with organized structure.
    
    Folder Structure:
    temp_folder/
    ├── index.json              # Index of all stored data
    ├── tasks/                  # Task-specific data
    │   ├── task_1/
    │   │   ├── subtasks.json
    │   │   ├── results.json
    │   │   └── ...
    │   └── task_1.2/
    ├── cache/                  # Cached data (web searches, API responses)
    │   ├── web_search_*.json
    │   └── llm_response_*.json
    ├── checkpoints/            # Workflow checkpoints for recovery
    │   ├── checkpoint_001.json
    │   └── ...
    ├── blackboard/             # Blackboard entries for agent communication
    │   └── entries.json
    ├── results/                # Intermediate results before final output
    │   └── ...
    └── sessions/               # Session-specific data
        └── session_*.json
    """
    
    CATEGORIES = ['tasks', 'cache', 'checkpoints', 'blackboard', 'results', 'sessions']
    
    def __init__(self, temp_folder: Union[str, Path], session_id: Optional[str] = None):
        """
        Initialize the TempDataManager.
        
        Args:
            temp_folder: Path to the temp folder
            session_id: Optional session identifier for grouping data
        """
        self.temp_folder = Path(temp_folder).resolve()
        self.session_id = session_id or self._generate_session_id()
        self.index_path = self.temp_folder / "index.json"
        self._index: Dict[str, TempDataEntry] = {}
        
        # Initialize folder structure
        self._init_folders()
        self._load_index()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    def _init_folders(self) -> None:
        """Initialize the folder structure."""
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        
        for category in self.CATEGORIES:
            (self.temp_folder / category).mkdir(exist_ok=True)
    
    def _load_index(self) -> None:
        """Load the data index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._index = {
                        k: TempDataEntry.from_dict(v) 
                        for k, v in data.get('entries', {}).items()
                    }
            except Exception:
                self._index = {}
    
    def _save_index(self) -> None:
        """Save the data index to disk."""
        data = {
            "session_id": self.session_id,
            "updated_at": datetime.now().isoformat(),
            "entries": {k: v.to_dict() for k, v in self._index.items()}
        }
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _get_path(self, category: str, key: str, extension: str = ".json") -> Path:
        """Get the file path for a data entry."""
        # Sanitize key for filesystem
        safe_key = "".join(c if c.isalnum() or c in '-_.' else '_' for c in key)
        return self.temp_folder / category / f"{safe_key}{extension}"
    
    def _get_key(self, category: str, name: str) -> str:
        """Generate a unique key for a data entry."""
        return f"{category}:{name}"
    
    # ========================================================================
    # SAVE OPERATIONS
    # ========================================================================
    
    def save(
        self,
        category: str,
        name: str,
        data: Any,
        task_id: Optional[str] = None,
        description: Optional[str] = None,
        ttl_hours: Optional[int] = None
    ) -> str:
        """
        Save data to temp storage.
        
        Args:
            category: Category (tasks, cache, checkpoints, blackboard, results, sessions)
            name: Unique name within category
            data: Data to save (will be JSON serialized)
            task_id: Optional associated task ID
            description: Optional description
            ttl_hours: Optional time-to-live in hours
            
        Returns:
            The key for retrieval
        """
        if category not in self.CATEGORIES:
            raise InvalidParameterError(
                parameter_name="category",
                message=f"Invalid category: {category}. Must be one of {self.CATEGORIES}"
            )
        
        key = self._get_key(category, name)
        file_path = self._get_path(category, name)
        
        # Serialize and save
        now = datetime.now()
        json_data = json.dumps(data, indent=2, default=str)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_data)
        
        # Update index
        entry = TempDataEntry(
            key=key,
            category=category,
            created_at=self._index.get(key, TempDataEntry(key, category, now, now, 0, "json")).created_at,
            updated_at=now,
            size_bytes=len(json_data.encode('utf-8')),
            data_type="json",
            task_id=task_id,
            description=description,
            ttl_hours=ttl_hours
        )
        self._index[key] = entry
        self._save_index()
        
        return key
    
    def save_task_data(
        self,
        task_id: str,
        name: str,
        data: Any,
        description: Optional[str] = None
    ) -> str:
        """
        Save task-specific data.
        
        Creates a subfolder for the task if needed.
        
        Args:
            task_id: The task ID
            name: Data name (e.g., "subtasks", "results", "analysis")
            data: Data to save
            description: Optional description
            
        Returns:
            The key for retrieval
        """
        # Create task subfolder
        task_folder = self.temp_folder / "tasks" / task_id.replace('.', '_')
        task_folder.mkdir(parents=True, exist_ok=True)
        
        # Save data
        file_path = task_folder / f"{name}.json"
        json_data = json.dumps(data, indent=2, default=str)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_data)
        
        key = f"tasks:{task_id}:{name}"
        now = datetime.now()
        
        entry = TempDataEntry(
            key=key,
            category="tasks",
            created_at=self._index.get(key, TempDataEntry(key, "tasks", now, now, 0, "json")).created_at,
            updated_at=now,
            size_bytes=len(json_data.encode('utf-8')),
            data_type="json",
            task_id=task_id,
            description=description
        )
        self._index[key] = entry
        self._save_index()
        
        return key
    
    def save_cache(
        self,
        name: str,
        data: Any,
        ttl_hours: int = 24
    ) -> str:
        """
        Save cached data with TTL.
        
        Args:
            name: Cache key name
            data: Data to cache
            ttl_hours: Time to live in hours (default 24)
            
        Returns:
            The key for retrieval
        """
        return self.save(
            category="cache",
            name=name,
            data=data,
            ttl_hours=ttl_hours,
            description=f"Cache entry with {ttl_hours}h TTL"
        )
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        Save a workflow checkpoint for recovery.
        
        Args:
            state: The agent state to checkpoint
            checkpoint_name: Optional name (auto-generated if not provided)
            
        Returns:
            The key for retrieval
        """
        if not checkpoint_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{timestamp}"
        
        return self.save(
            category="checkpoints",
            name=checkpoint_name,
            data=state,
            description=f"Workflow checkpoint at iteration {state.get('iteration_count', 'unknown')}"
        )
    
    def save_blackboard(self, entries: List[Dict[str, Any]]) -> str:
        """
        Save blackboard entries.
        
        Args:
            entries: List of blackboard entries
            
        Returns:
            The key for retrieval
        """
        return self.save(
            category="blackboard",
            name="entries",
            data=entries,
            description=f"{len(entries)} blackboard entries"
        )
    
    # ========================================================================
    # LOAD OPERATIONS
    # ========================================================================
    
    def load(self, category: str, name: str) -> Optional[Any]:
        """
        Load data from temp storage.
        
        Args:
            category: Category
            name: Name within category
            
        Returns:
            Loaded data or None if not found
        """
        file_path = self._get_path(category, name)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def load_by_key(self, key: str) -> Optional[Any]:
        """
        Load data by key.
        
        Args:
            key: The key returned from save operations
            
        Returns:
            Loaded data or None if not found
        """
        if key not in self._index:
            return None
        
        entry = self._index[key]
        
        # Parse key to get category and name
        parts = key.split(":", 1)
        if len(parts) != 2:
            return None
        
        return self.load(parts[0], parts[1])
    
    def load_task_data(self, task_id: str, name: str) -> Optional[Any]:
        """
        Load task-specific data.
        
        Args:
            task_id: The task ID
            name: Data name
            
        Returns:
            Loaded data or None if not found
        """
        task_folder = self.temp_folder / "tasks" / task_id.replace('.', '_')
        file_path = task_folder / f"{name}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Checkpoint state or None if no checkpoints exist
        """
        checkpoint_entries = [
            entry for entry in self._index.values()
            if entry.category == "checkpoints"
        ]
        
        if not checkpoint_entries:
            return None
        
        # Sort by updated_at descending
        latest = max(checkpoint_entries, key=lambda e: e.updated_at)
        return self.load_by_key(latest.key)
    
    def load_blackboard(self) -> List[Dict[str, Any]]:
        """
        Load blackboard entries.
        
        Returns:
            List of blackboard entries or empty list
        """
        return self.load("blackboard", "entries") or []
    
    # ========================================================================
    # SEARCH AND QUERY
    # ========================================================================
    
    def search(
        self,
        category: Optional[str] = None,
        task_id: Optional[str] = None,
        keyword: Optional[str] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None
    ) -> List[TempDataEntry]:
        """
        Search for data entries.
        
        Args:
            category: Filter by category
            task_id: Filter by task ID
            keyword: Search in key and description
            after: Filter entries updated after this time
            before: Filter entries updated before this time
            
        Returns:
            List of matching TempDataEntry objects
        """
        results = []
        
        for entry in self._index.values():
            # Apply filters
            if category and entry.category != category:
                continue
            if task_id and entry.task_id != task_id:
                continue
            if keyword:
                keyword_lower = keyword.lower()
                if keyword_lower not in entry.key.lower():
                    if not entry.description or keyword_lower not in entry.description.lower():
                        continue
            if after and entry.updated_at < after:
                continue
            if before and entry.updated_at > before:
                continue
            
            results.append(entry)
        
        # Sort by updated_at descending
        results.sort(key=lambda e: e.updated_at, reverse=True)
        return results
    
    def get_task_entries(self, task_id: str) -> List[TempDataEntry]:
        """Get all entries for a specific task."""
        return self.search(task_id=task_id)
    
    def list_all(self) -> Dict[str, List[TempDataEntry]]:
        """
        List all entries grouped by category.
        
        Returns:
            Dictionary mapping category to list of entries
        """
        result = {category: [] for category in self.CATEGORIES}
        
        for entry in self._index.values():
            result[entry.category].append(entry)
        
        return result
    
    # ========================================================================
    # CLEANUP OPERATIONS
    # ========================================================================
    
    def cleanup_expired(self) -> int:
        """
        Remove entries that have exceeded their TTL.
        
        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self._index.items():
            if entry.ttl_hours:
                expiry = entry.created_at + timedelta(hours=entry.ttl_hours)
                if now > expiry:
                    expired_keys.append(key)
        
        for key in expired_keys:
            self.delete_by_key(key)
        
        return len(expired_keys)
    
    def delete_by_key(self, key: str) -> bool:
        """
        Delete a data entry by key.
        
        Args:
            key: The key to delete
            
        Returns:
            True if deleted, False if not found
        """
        if key not in self._index:
            return False
        
        entry = self._index[key]
        
        # Parse key to get file path
        parts = key.split(":")
        if len(parts) >= 2:
            file_path = self._get_path(parts[0], ":".join(parts[1:]))
            if file_path.exists():
                file_path.unlink()
        
        del self._index[key]
        self._save_index()
        return True
    
    def clear_category(self, category: str) -> int:
        """
        Clear all entries in a category.
        
        Args:
            category: Category to clear
            
        Returns:
            Number of entries removed
        """
        if category not in self.CATEGORIES:
            raise InvalidParameterError(
                parameter_name="category",
                message=f"Invalid category: {category}"
            )
        
        keys_to_delete = [
            key for key, entry in self._index.items()
            if entry.category == category
        ]
        
        for key in keys_to_delete:
            self.delete_by_key(key)
        
        # Also clear the folder
        category_folder = self.temp_folder / category
        if category_folder.exists():
            shutil.rmtree(category_folder)
            category_folder.mkdir()
        
        return len(keys_to_delete)
    
    def clear_all(self) -> None:
        """Clear all temp data."""
        self._index.clear()
        
        for category in self.CATEGORIES:
            category_folder = self.temp_folder / category
            if category_folder.exists():
                shutil.rmtree(category_folder)
                category_folder.mkdir()
        
        self._save_index()
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        stats = {
            "session_id": self.session_id,
            "total_entries": len(self._index),
            "total_size_bytes": sum(e.size_bytes for e in self._index.values()),
            "by_category": {
                category: {
                    "count": 0,
                    "size_bytes": 0
                }
                for category in self.CATEGORIES
            },
            "oldest_entry": None,
            "newest_entry": None
        }
        
        oldest_dt: Optional[datetime] = None
        newest_dt: Optional[datetime] = None
        
        for entry in self._index.values():
            stats["by_category"][entry.category]["count"] += 1
            stats["by_category"][entry.category]["size_bytes"] += entry.size_bytes
            
            if oldest_dt is None or entry.created_at < oldest_dt:
                oldest_dt = entry.created_at
            if newest_dt is None or entry.updated_at > newest_dt:
                newest_dt = entry.updated_at
        
        stats["oldest_entry"] = oldest_dt.isoformat() if oldest_dt else None
        stats["newest_entry"] = newest_dt.isoformat() if newest_dt else None
        
        return stats
    
    def __len__(self) -> int:
        return len(self._index)
    
    def __contains__(self, key: str) -> bool:
        return key in self._index
