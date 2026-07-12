"""
SNA Framework Engine Package
=============================
Core engine modules for Social Network Analysis with Closeness Centrality
on Dynamic Barabási-Albert Graphs.

Modules:
    graph_generator  - BA graph creation and batch update utilities
    full_recompute   - Full closeness centrality recomputation (baseline)
    icc              - Incremental Closeness Centrality (approximation)
    lba              - Landmark-Based Approximation of closeness centrality
"""

from engine.graph_generator import GraphGenerator
from engine.full_recompute import FullRecompute
from engine.icc import IncrementalCloseness
from engine.lba import LandmarkApproximation

__all__ = [
    "GraphGenerator",
    "FullRecompute",
    "IncrementalCloseness",
    "LandmarkApproximation",
]
