"""
Persistence layer for NeuroMem-Agents
Handles saving and loading memory data to/from persistent storage
"""

import json
import pickle
import sqlite3
from typing import Dict, List, Optional

from .neuromorphic_memory import MemoryNode, MemoryType


class MemoryDatabase:
    """Persistent storage for memory data using SQLite"""

    def __init__(self, db_path: str = "neuromem.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        self._configure_connection()
        self.init_database()

    def _configure_connection(self):
        self.conn.execute("PRAGMA temp_store = MEMORY")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        try:
            self.conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:
            # WAL is an optimization, not a correctness requirement.
            pass

    def init_database(self):
        """Initialize the database schema"""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                activation_level REAL DEFAULT 0.0,
                connectivity_strength REAL DEFAULT 1.0,
                importance_score REAL DEFAULT 1.0,
                decay_rate REAL DEFAULT 0.01,
                tags TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS connections (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_id, target_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS access_frequencies (
                node_id TEXT PRIMARY KEY,
                frequency INTEGER DEFAULT 1,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS working_memory_buffer (
                node_id TEXT PRIMARY KEY,
                position INTEGER NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                node_id TEXT PRIMARY KEY,
                consolidation_reason TEXT,
                consolidated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    def save_memory_node(self, node: MemoryNode):
        """Save a memory node to the database"""
        embedding_blob = pickle.dumps(node.embedding)
        tags_json = json.dumps(node.tags)

        self.conn.execute("""
            INSERT OR REPLACE INTO memory_nodes
            (id, content, embedding, memory_type, timestamp,
             activation_level, connectivity_strength, importance_score,
             decay_rate, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node.id,
            node.content,
            embedding_blob,
            node.memory_type.value,
            node.timestamp,
            node.activation_level,
            node.connectivity_strength,
            node.importance_score,
            node.decay_rate,
            tags_json,
        ))
        self.conn.commit()

    def load_memory_node(self, node_id: str) -> Optional[MemoryNode]:
        """Load a memory node from the database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, content, embedding, memory_type, timestamp,
                   activation_level, connectivity_strength, importance_score,
                   decay_rate, tags
            FROM memory_nodes WHERE id = ?
        """, (node_id,))
        row = cursor.fetchone()

        if not row:
            return None

        embedding = pickle.loads(row[2])
        tags = json.loads(row[9])

        return MemoryNode(
            id=row[0],
            content=row[1],
            embedding=embedding,
            memory_type=MemoryType(row[3]),
            timestamp=row[4],
            activation_level=row[5],
            connectivity_strength=row[6],
            importance_score=row[7],
            decay_rate=row[8],
            tags=tags,
        )

    def load_all_memory_nodes(self) -> List[MemoryNode]:
        """Load all memory nodes from the database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, content, embedding, memory_type, timestamp,
                   activation_level, connectivity_strength, importance_score,
                   decay_rate, tags
            FROM memory_nodes
        """)

        nodes = []
        for row in cursor.fetchall():
            embedding = pickle.loads(row[2])
            tags = json.loads(row[9])
            nodes.append(
                MemoryNode(
                    id=row[0],
                    content=row[1],
                    embedding=embedding,
                    memory_type=MemoryType(row[3]),
                    timestamp=row[4],
                    activation_level=row[5],
                    connectivity_strength=row[6],
                    importance_score=row[7],
                    decay_rate=row[8],
                    tags=tags,
                )
            )
        return nodes

    def save_connection(self, source_id: str, target_id: str, weight: float):
        """Save a connection between memory nodes"""
        self.conn.execute("""
            INSERT OR REPLACE INTO connections
            (source_id, target_id, weight)
            VALUES (?, ?, ?)
        """, (source_id, target_id, weight))
        self.conn.commit()

    def save_connections_batch(self, connections: List[tuple]):
        """Save multiple connections in one transaction."""
        if not connections:
            return

        self.conn.executemany("""
            INSERT OR REPLACE INTO connections
            (source_id, target_id, weight)
            VALUES (?, ?, ?)
        """, connections)
        self.conn.commit()

    def load_connections(self, node_id: str) -> List[tuple]:
        """Load all connections for a given node"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT target_id, weight FROM connections WHERE source_id = ?
        """, (node_id,))
        return cursor.fetchall()

    def load_all_connections(self) -> Dict[str, List[tuple]]:
        """Load the full connection graph."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_id, target_id, weight FROM connections
        """)

        graph: Dict[str, List[tuple]] = {}
        for source_id, target_id, weight in cursor.fetchall():
            graph.setdefault(source_id, []).append((target_id, weight))
        return graph

    def save_access_frequency(self, node_id: str, frequency: int):
        """Save access frequency for a node"""
        self.conn.execute("""
            INSERT OR REPLACE INTO access_frequencies
            (node_id, frequency)
            VALUES (?, ?)
        """, (node_id, frequency))
        self.conn.commit()

    def save_access_frequencies(self, frequencies: List[tuple]):
        """Save multiple access-frequency updates in one transaction."""
        if not frequencies:
            return

        self.conn.executemany("""
            INSERT OR REPLACE INTO access_frequencies
            (node_id, frequency)
            VALUES (?, ?)
        """, frequencies)
        self.conn.commit()

    def load_access_frequency(self, node_id: str) -> Optional[int]:
        """Load access frequency for a node"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT frequency FROM access_frequencies WHERE node_id = ?
        """, (node_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def load_all_access_frequencies(self) -> Dict[str, int]:
        """Load access frequencies for all nodes."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT node_id, frequency FROM access_frequencies
        """)
        return {node_id: frequency for node_id, frequency in cursor.fetchall()}

    def save_to_working_memory_buffer(self, node_id: str, position: int):
        """Save a node to the working memory buffer"""
        self.conn.execute("""
            INSERT OR REPLACE INTO working_memory_buffer
            (node_id, position)
            VALUES (?, ?)
        """, (node_id, position))
        self.conn.commit()

    def load_working_memory_buffer(self) -> List[str]:
        """Load the working memory buffer"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT node_id FROM working_memory_buffer ORDER BY position
        """)
        return [row[0] for row in cursor.fetchall()]

    def save_to_long_term_memory(self, node_id: str, reason: str = ""):
        """Save a node to long-term memory"""
        self.conn.execute("""
            INSERT OR REPLACE INTO long_term_memory
            (node_id, consolidation_reason)
            VALUES (?, ?)
        """, (node_id, reason))
        self.conn.commit()

    def load_long_term_memory(self) -> List[str]:
        """Load all long-term memory node IDs"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT node_id FROM long_term_memory")
        return [row[0] for row in cursor.fetchall()]

    def checkpoint(self):
        """Flush WAL pages before measurement or shutdown."""
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.conn.commit()
        except sqlite3.DatabaseError:
            pass

    def clear_database(self):
        """Clear all data from the database"""
        self.conn.execute("DELETE FROM memory_nodes")
        self.conn.execute("DELETE FROM connections")
        self.conn.execute("DELETE FROM access_frequencies")
        self.conn.execute("DELETE FROM working_memory_buffer")
        self.conn.execute("DELETE FROM long_term_memory")
        self.conn.commit()

    def close(self):
        """Close the shared SQLite connection."""
        if getattr(self, "conn", None) is None:
            return
        try:
            self.checkpoint()
        finally:
            self.conn.close()
            self.conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def attach_persistence_methods(cls):
    """Decorator to add persistence methods to MemoryManager"""

    def save_to_db(self, db_path: Optional[str] = None):
        """Save the entire memory system to database"""
        if db_path is None:
            db = getattr(self, "db", None)
            if db is None:
                db = MemoryDatabase()
                self.db = db
        else:
            existing_db = getattr(self, "db", None)
            if existing_db is not None and getattr(existing_db, "db_path", None) != db_path:
                close = getattr(existing_db, "close", None)
                if callable(close):
                    close()
            self.db = MemoryDatabase(db_path)
            db = self.db

        for node in self.memory_nodes.values():
            db.save_memory_node(node)

        save_connection_batch = getattr(db, "save_connections_batch", None)
        if callable(save_connection_batch):
            connection_rows = []
            for source_id, connections in self.connections.items():
                for connection_tuple in connections:
                    target_id, weight = connection_tuple[:2]
                    connection_rows.append((source_id, target_id, weight))
            save_connection_batch(connection_rows)
        else:
            for source_id, connections in self.connections.items():
                for connection_tuple in connections:
                    target_id, weight = connection_tuple[:2]
                    db.save_connection(source_id, target_id, weight)

        save_access_batch = getattr(db, "save_access_frequencies", None)
        if callable(save_access_batch):
            save_access_batch(list(self.access_frequency.items()))
        else:
            for node_id, frequency in self.access_frequency.items():
                db.save_access_frequency(node_id, frequency)

        for position, node_id in enumerate(self.working_memory_buffer):
            db.save_to_working_memory_buffer(node_id, position)

        for node_id in self.long_term_memory.keys():
            db.save_to_long_term_memory(node_id, "consolidated")

    def load_from_db(self, db_path: Optional[str] = None):
        """Load the entire memory system from database"""
        if db_path is None:
            db = getattr(self, "db", None)
            if db is None:
                db = MemoryDatabase()
                self.db = db
        else:
            existing_db = getattr(self, "db", None)
            if existing_db is not None and getattr(existing_db, "db_path", None) != db_path:
                close = getattr(existing_db, "close", None)
                if callable(close):
                    close()
            self.db = MemoryDatabase(db_path)
            db = self.db

        self.memory_nodes = {node.id: node for node in db.load_all_memory_nodes()}
        self.long_term_memory = {}

        load_all_connections = getattr(db, "load_all_connections", None)
        if callable(load_all_connections):
            self.connections = load_all_connections()
        else:
            self.connections = {}
            for node_id in self.memory_nodes.keys():
                self.connections[node_id] = db.load_connections(node_id)
        for node_id in self.memory_nodes.keys():
            self.connections.setdefault(node_id, [])

        load_all_access = getattr(db, "load_all_access_frequencies", None)
        if callable(load_all_access):
            self.access_frequency = load_all_access()
        else:
            self.access_frequency = {}
            for node_id in self.memory_nodes.keys():
                freq = db.load_access_frequency(node_id)
                if freq is not None:
                    self.access_frequency[node_id] = freq
        for node_id in self.memory_nodes.keys():
            self.access_frequency.setdefault(node_id, 1)

        self.working_memory_buffer = db.load_working_memory_buffer()

        for node_id in db.load_long_term_memory():
            if node_id in self.memory_nodes:
                self.long_term_memory[node_id] = self.memory_nodes[node_id]

        rebuild_indices = getattr(self, "_rebuild_indices", None)
        if callable(rebuild_indices):
            rebuild_indices()

    cls.save_to_db = save_to_db
    cls.load_from_db = load_from_db

    return cls


__all__ = ["MemoryDatabase", "attach_persistence_methods"]
