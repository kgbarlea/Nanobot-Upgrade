"""Hybrid Memory System for Nanobot: Combines Chronicle (Log), Archive (Semantic), and Persona (Episodic) layers."""

import sqlite3
import json
import os
from datetime import datetime
from typing import Any, List, Dict, Optional
from pathlib import Path

# For the Archive (Semantic) layer, we use ChromaDB (Local Persistent Mode)
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None


class MemoryManager:
    def __init__(self, storage_dir: str = "nanobot_hybrid_memory"):
        """
        Initialize the Hybrid Memory System.
        
        Args:
            storage_dir: Directory to store all memory components
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # 1. Chronicle (Log Layer) - SQLite for state-change logging
        self.chronicle_path = self.storage_dir / "chronicle.db"
        self._init_chronicle()

        # 2. Archive (Semantic Layer) - ChromaDB for vector search
        if CHROMADB_AVAILABLE:
            self.archive_client = chromadb.PersistentClient(path=str(self.storage_dir / "archive_index"))
            self.archive_collection = self.archive_client.get_or_create_collection(name="session_archive")
        else:
            self.archive_client = None
            self.archive_collection = None
            print("Warning: ChromaDB not available. Archive layer disabled.")

        # 3. Persona (Episodic Layer) - JSON for user preferences and learned truths
        self.persona_path = self.storage_dir / "persona.json"
        self._init_persona()

    def _init_chronicle(self):
        """Initialize the SQL table for state-change logging."""
        with sqlite3.connect(self.chronicle_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    session_id TEXT,
                    user_id TEXT,
                    event_type TEXT,
                    state_blob TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON state_logs(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id ON state_logs(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON state_logs(user_id)
            """)
            conn.commit()

    def _init_persona(self):
        """Initialize the JSON file for user preferences."""
        if not self.persona_path.exists():
            with open(self.persona_path, "w") as f:
                json.dump({"preferences": {}, "learned_truths": {}}, f)

    # --- WRITE METHODS ---

    def log_state_change(self, session_id: str, user_id: str, event_type: str, state: Dict[str, Any]):
        """
        Chronicle: Log every state change for auditability and historical lookup.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            event_type: Type of event (e.g., 'goal_update', 'task_completion')
            state: The state blob to store
        """
        timestamp = datetime.now().isoformat()
        state_json = json.dumps(state)
        with sqlite3.connect(self.chronicle_path) as conn:
            conn.execute(
                """INSERT INTO state_logs 
                   (timestamp, session_id, user_id, event_type, state_blob) 
                   VALUES (?, ?, ?, ?, ?)""",
                (timestamp, session_id, user_id, event_type, state_json)
            )

    def archive_session(self, session_id: str, summary: str, metadata: Dict[str, Any]):
        """
        Archive: Store session summary in the Vector DB for semantic search.
        
        Args:
            session_id: Unique session identifier
            summary: Text summary of the session
            metadata: Additional metadata to store with the summary
        """
        if not CHROMADB_AVAILABLE or not self.archive_collection:
            return
            
        # Add to ChromaDB collection
        self.archive_collection.add(
            documents=[summary],
            metadatas=[metadata],
            ids=[session_id]
        )

    def update_persona(self, user_id: str, key: str, value: Any, is_preference: bool = True):
        """
        Persona: Update long-term user preferences or learned truths.
        
        Args:
            user_id: User identifier
            key: The preference/truth key
            value: The value to store
            is_preference: Whether this is a preference (True) or learned truth (False)
        """
        with open(self.persona_path, "r") as f:
            persona = json.load(f)
        
        if is_preference:
            persona["preferences"][key] = value
        else:
            persona["learned_truths"][key] = value
            
        with open(self.persona_path, "w") as f:
            json.dump(persona, f, indent=4)

    # --- READ METHODS ---

    def get_history_by_date(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Chronicle: Lookup history between two dates.
        
        Args:
            start_date: Start date in ISO format (inclusive)
            end_date: End date in ISO format (inclusive)
            
        Returns:
            List of state change records
        """
        with sqlite3.connect(self.chronicle_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT * FROM state_logs 
                   WHERE timestamp BETWEEN ? AND ? 
                   ORDER BY timestamp ASC""",
                (start_date, end_date)
            )
            return [dict(row) for row in cursor.fetchall()]

    def search_knowledge(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Archive: Semantic search for relevant history.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            
        Returns:
            List of matching documents with metadata
        """
        if not CHROMADB_AVAILABLE or not self.archive_collection:
            return []
            
        results = self.archive_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Return the documents and metadata
        return [{"doc": doc, "meta": meta} for doc, meta in zip(results['documents'][0], results['metadatas'][0])]

    def get_user_preference(self, user_id: str, key: str) -> Optional[Any]:
        """
        Persona: Retrieve a specific user preference or learned truth.
        
        Args:
            user_id: User identifier
            key: The preference/truth key
            
        Returns:
            The value if found, None otherwise
        """
        with open(self.persona_path, "r") as f:
            persona = json.load(f)
        # Check preferences first, then learned truths
        return persona["preferences"].get(key) or persona["learned_truths"].get(key)

    def get_all_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Persona: Get all preferences and learned truths for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing preferences and learned truths
        """
        with open(self.persona_path, "r") as f:
            persona = json.load(f)
        return {
            "preferences": persona.get("preferences", {}),
            "learned_truths": persona.get("learned_truths", {})
        }


# Global instance for easy access
_memory_manager_instance = None

def get_memory_manager(storage_dir: str = "nanobot_hybrid_memory") -> MemoryManager:
    """
    Get or create the global MemoryManager instance.
    
    Args:
        storage_dir: Directory to store memory components
        
    Returns:
        MemoryManager instance
    """
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager(storage_dir)
    return _memory_manager_instance