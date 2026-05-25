"""
netfd
=====

Active fault diagnosis for networked LTI systems.

Sub-packages
------------
systems    Control-theoretic modeling (nodes, networks, synthesis, ν-gap)
diagnosis  Single-shot diagnosis algorithms (probe design + classifier)
sequential Sequential AFD environment + PPO trainer
viz        Plotting utilities
io         YAML config loading + results I/O
"""

__version__ = "0.1.0"

from netfd import systems, diagnosis, sequential, viz, io

__all__ = ["systems", "diagnosis", "sequential", "viz", "io", "__version__"]
