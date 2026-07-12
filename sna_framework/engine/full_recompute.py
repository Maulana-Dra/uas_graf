"""
full_recompute.py
=================
Full (baseline) closeness-centrality recomputation.

NOTE: As of 2026-07-11, this module uses igraph (C backend)
instead of pure NetworkX for closeness centrality computation,
for performance reasons on resource-constrained hardware
(Intel i3-8130U, 2 cores). This is a deliberate methodological
choice documented in BAB II metodologi. ICC (engine/icc.py)
remains pure Python/NetworkX, as it is the primary object of
study in this research — only the Full Recompute BASELINE uses
the faster backend, to make large-N benchmarking feasible.

Classes:
    FullRecompute  – computes closeness centrality for all nodes from scratch
                     using igraph's C-level implementation.
"""

from __future__ import annotations

import time
from typing import Any

import networkx as nx


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------

def _nx_to_igraph(G: nx.Graph):
    """Convert a NetworkX graph to an igraph Graph, preserving node identity.

    Parameters
    ----------
    G : nx.Graph
        The NetworkX graph to convert.

    Returns
    -------
    tuple[ig.Graph, list]
        (ig_graph, node_list) where node_list[i] is the original NetworkX
        node id corresponding to igraph vertex i.
    """
    import igraph as ig
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    ig_graph = ig.Graph(n=len(node_list), edges=edges)
    return ig_graph, node_list


# ---------------------------------------------------------------------------
# FullRecompute class
# ---------------------------------------------------------------------------

class FullRecompute:
    """Compute closeness centrality for every node using igraph (C backend).

    This is the **baseline** method; it recomputes from scratch on each call
    and is used to benchmark speed and accuracy against ICC and LBA.

    The igraph C-level implementation is used instead of pure NetworkX
    for performance on resource-constrained hardware, making large-N
    (N=50,000 / N=100,000) benchmarking feasible. The return dict structure
    is identical to the original NetworkX-based version — callers need not
    change.
    """

    def compute(self, G: nx.Graph) -> dict[str, Any]:
        """Compute closeness centrality for all nodes in *G* from scratch.

        Uses igraph's ``closeness()`` (C backend) and measures wall-clock
        elapsed time with ``time.perf_counter``. The graph is converted from
        NetworkX to igraph internally; callers always pass/receive NetworkX
        objects.

        Parameters
        ----------
        G : nx.Graph
            The graph on which to compute centrality.

        Returns
        -------
        dict
            ``{
                "centrality": {node_id: float, ...},
                "top5":       [(node_id, score), ...],  # 5 highest, desc.
                "elapsed_ms": float
            }``
        """
        t0: float = time.perf_counter()

        ig_graph, node_list = _nx_to_igraph(G)

        # igraph's closeness() returns a list aligned with vertex indices,
        # in the same order as node_list.
        closeness_values = ig_graph.closeness()

        # Map back to original NetworkX node ids.
        centrality: dict[int, float] = {}
        for i, node in enumerate(node_list):
            val = closeness_values[i]
            # igraph returns NaN for isolated/unreachable nodes;
            # convert to 0.0 for consistency with the existing pipeline.
            centrality[node] = 0.0 if (val is None or val != val) else val

        elapsed_ms: float = (time.perf_counter() - t0) * 1_000.0

        top5: list[tuple[int, float]] = sorted(
            centrality.items(), key=lambda kv: kv[1], reverse=True
        )[:5]

        print(
            f"[FullRecompute] Elapsed: {elapsed_ms:.3f} ms | "
            f"Top1: node {top5[0][0]} score={top5[0][1]:.6f}"
        )
        return {
            "centrality": centrality,
            "top5": top5,
            "elapsed_ms": elapsed_ms,
        }
