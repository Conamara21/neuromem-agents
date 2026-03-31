"""
Persistence layer for NeuroMem-Agents
Handles saving and loading memory data to/from persistent storage
"""

import json
import pickle
import sqlite3
import os
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from datetime import datetime
import hashlib
from .neuromorphic_memory import MemoryNode, MemoryType


class MemoryDatabase:
    """Persistent storage for memory data using SQLite"""
    
    def __init__(self, db_path: str = "neuromem.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for memory nodes
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
        
        # Table for connections between nodes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS connections (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_id, target_id)
            )
        """)
        
        # Table for access frequencies
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS access_frequencies (
                node_id TEXT PRIMARY KEY,
                frequency INTEGER DEFAULT 1,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table for working memory buffer (temporary storage)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS working_memory_buffer (
                node_id TEXT PRIMARY KEY,
                position INTEGER NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table for long-term memory (consolidated memories)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                node_id TEXT PRIMARY KEY,
                consolidation_reason TEXT,
                consolidated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_memory_node(self, node: MemoryNode):
        """Save a memory node to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize embedding and tags
        embedding_blob = pickle.dumps(node.embedding)
        tags_json = json.dumps(node.tags)
        
        cursor.execute("""
            INSERT OR REPLACE INTO memory_nodes 
            (id, content, embedding, memory_type, timestamp, 
             activation_level, connectivity_strength, importance_score, 
             decay_rate, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node.id, node.content, embedding_blob, node.memory_type.value,
            node.timestamp, node.activation_level, node.connectivity_strength,
            node.importance_score, node.decay_rate, tags_json
        ))
        
        conn.commit()
        conn.close()
    
    def load_memory_node(self, node_id: str) -> Optional[MemoryNode]:
        """Load a memory node from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, embedding, memory_type, timestamp,
                   activation_level, connectivity_strength, importance_score,
                   decay_rate, tags
            FROM memory_nodes WHERE id = ?
        """, (node_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Deserialize embedding and tags
        embedding = pickle.loads(row[2])
        tags = json.loads(row[9])
        
        return MemoryNode(
            id=row[0], content=row[1], embedding=embedding,
            memory_type=MemoryType(row[3]), timestamp=row[4],
            activation_level=row[5], connectivity_strength=row[6],
            importance_score=row[7], decay_rate=row[8], tags=tags
        )
    
    def load_all_memory_nodes(self) -> List[MemoryNode]:
        """Load all memory nodes from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, embedding, memory_type, timestamp,
                   activation_level, connectivity_strength, importance_score,
                   decay_rate, tags
            FROM memory_nodes
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        nodes = []
        for row in rows:
            embedding = pickle.loads(row[2])
            tags = json.loads(row[9])
            
            node = MemoryNode(
                id=row[0], content=row[1], embedding=embedding,
                memory_type=MemoryType(row[3]), timestamp=row[4],
                activation_level=row[5], connectivity_strength=row[6],
                importance_score=row[7], decay_rate=row[8], tags=tags
            )
            nodes.append(node)
        
        return nodes
    
    def save_connection(self, source_id: str, target_id: str, weight: float):
        """Save a connection between memory nodes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO connections
            (source_id, target_id, weight)
            VALUES (?, ?, ?)
        """, (source_id, target_id, weight))
        
        conn.commit()
        conn.close()
    
    def load_connections(self, node_id: str) -> List[tuple]:
        """Load all connections for a given node"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT target_id, weight FROM connections WHERE source_id = ?
        """, (node_id,))
        
        connections = cursor.fetchall()
        conn.close()
        
        return connections
    
    def save_access_frequency(self, node_id: str, frequency: int):
        """Save access frequency for a node"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO access_frequencies
            (node_id, frequency)
            VALUES (?, ?)
        """, (node_id, frequency))
        
        conn.commit()
        conn.close()
    
    def load_access_frequency(self, node_id: str) -> Optional[int]:
        """Load access frequency for a node"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT frequency FROM access_frequencies WHERE node_id = ?
        """, (node_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def save_to_working_memory_buffer(self, node_id: str, position: int):
        """Save a node to the working memory buffer"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO working_memory_buffer
            (node_id, position)
            VALUES (?, ?)
        """, (node_id, position))
        
        conn.commit()
        conn.close()
    
    def load_working_memory_buffer(self) -> List[str]:
        """Load the working memory buffer"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT node_id FROM working_memory_buffer ORDER BY position
        """)
        
        nodes = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return nodes
    
    def save_to_long_term_memory(self, node_id: str, reason: str = ""):
        """Save a node to long-term memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO long_term_memory
            (node_id, consolidation_reason)
            VALUES (?, ?)
        """, (node_id, reason))
        
        conn.commit()
        conn.close()
    
    def load_long_term_memory(self) -> List[str]:
        """Load all long-term memory node IDs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT node_id FROM long_term_memory")
        
        nodes = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return nodes
    
    def clear_database(self):
        """Clear all data from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM memory_nodes")
        cursor.execute("DELETE FROM connections")
        cursor.execute("DELETE FROM access_frequencies")
        cursor.execute("DELETE FROM working_memory_buffer")
        cursor.execute("DELETE FROM long_term_memory")
        
        conn.commit()
        conn.close()


def attach_persistence_methods(cls):
    """Decorator to add persistence methods to MemoryManager"""
    def save_to_db(self, db_path: Optional[str] = None):
        """Save the entire memory system to database"""
        if db_path is None:
            db = getattr(self, "db", MemoryDatabase())
        else:
            self.db = MemoryDatabase(db_path)
            db = self.db
        
        # Save all memory nodes
        for node_id, node in self.memory_nodes.items():
            db.save_memory_node(node)
        
        # Save all connections
        for source_id, connections in self.connections.items():
            for connection_tuple in connections:
                if len(connection_tuple) == 3:  # (target_id, weight, timestamp)
                    target_id, weight, _ = connection_tuple
                else:  # (target_id, weight) - backward compatibility
                    target_id, weight = connection_tuple
                db.save_connection(source_id, target_id, weight)
        
        # Save access frequencies
        for node_id, freq in self.access_frequency.items():
            db.save_access_frequency(node_id, freq)
        
        # Save working memory buffer
        for pos, node_id in enumerate(self.working_memory_buffer):
            db.save_to_working_memory_buffer(node_id, pos)
        
        # Save long-term memory
        for node_id in self.long_term_memory.keys():
            db.save_to_long_term_memory(node_id, "consolidated")
    
    def load_from_db(self, db_path: Optional[str] = None):
        """Load the entire memory system from database"""
        if db_path is None:
            db = getattr(self, "db", MemoryDatabase())
        else:
            self.db = MemoryDatabase(db_path)
            db = self.db
        
        # Load all memory nodes
        self.memory_nodes = {node.id: node for node in db.load_all_memory_nodes()}
        
        # Load all connections
        # First, rebuild the connections dict structure
        self.connections = {}
        for node_id in self.memory_nodes.keys():
            connections = db.load_connections(node_id)
            self.connections[node_id] = connections
        
        # Load access frequencies
        for node_id in self.memory_nodes.keys():
            freq = db.load_access_frequency(node_id)
            if freq is not None:
                self.access_frequency[node_id] = freq
        
        # Load working memory buffer
        self.working_memory_buffer = db.load_working_memory_buffer()
        
        # Load long-term memory
        for node_id in db.load_long_term_memory():
            if node_id in self.memory_nodes:
                self.long_term_memory[node_id] = self.memory_nodes[node_id]
    
    cls.save_to_db = save_to_db
    cls.load_from_db = load_from_db
    
    return cls


# Export the database class and decorator
__all__ = ['MemoryDatabase', 'attach_persistence_methods']
