"""
Backward-compatible imports for the legacy ``core`` package path.
"""

from neuromem.core import __all__ as __all__
from neuromem.core import __version__ as __version__
from neuromem.core import *  # noqa: F401,F403
