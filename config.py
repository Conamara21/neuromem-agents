"""
Configuration settings for NeuroMem-Agents
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryConfig:
    """Configuration for memory systems"""
    capacity: int = 10000
    working_memory_size: int = 100
    consolidation_threshold: int = 3
    decay_threshold: float = 0.1
    threshold: float = 0.5
    embedding_dim: int = 128


@dataclass 
class ExperimentConfig:
    """Configuration for experiments"""
    default_iterations: int = 10
    top_k_default: int = 5


@dataclass
class APIConfig:
    """API and integration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


# Load configuration from environment variables
def get_config():
    """Load configuration from environment variables"""
    return {
        "memory": MemoryConfig(
            capacity=int(os.getenv("MEM_CAPACITY", "10000")),
            working_memory_size=int(os.getenv("WORKING_MEM_SIZE", "100")),
            consolidation_threshold=int(os.getenv("CONSOLIDATION_THRESHOLD", "3")),
            decay_threshold=float(os.getenv("DECAY_THRESHOLD", "0.1")),
            threshold=float(os.getenv("SPIKE_THRESHOLD", "0.5")),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "128"))
        ),
        "experiment": ExperimentConfig(
            default_iterations=int(os.getenv("EXP_ITERATIONS", "10")),
            top_k_default=int(os.getenv("TOP_K_DEFAULT", "5"))
        ),
        "api": APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
    }


CONFIG = get_config()