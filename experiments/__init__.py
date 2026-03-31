"""
Backward-compatible imports for the legacy ``experiments`` package path.
"""

from neuromem.experiments import __all__ as __all__
from neuromem.experiments import __version__ as __version__
from neuromem.experiments import *  # noqa: F401,F403
