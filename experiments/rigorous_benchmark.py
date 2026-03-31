"""
Backward-compatible wrapper for ``experiments.rigorous_benchmark``.
"""

from neuromem.experiments.rigorous_benchmark import *  # noqa: F401,F403


if __name__ == "__main__":
    from neuromem.experiments.rigorous_benchmark import main

    main()
