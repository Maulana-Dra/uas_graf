"""
icc.py
======
Incremental Closeness Centrality (ICC) approximation.

NOTE: As of 2026-07-11, the affected-node closeness recomputation step
uses igraph's batched distance computation (C backend) instead of a
per-node NetworkX BFS loop, for performance reasons on
resource-constrained hardware. This was approved by the course instructor
(Dr. Dian Puspita Hapsari) as an acceleration layer only — the 2-hop
neighbourhood detection logic and the incremental update algorithm itself
remain unchanged from the original design.

The ICC heuristic avoids full graph recomputation after small edge changes
by re-evaluating only nodes within a 2-hop neighbourhood of changed edges.
This is an intentional *approximation*: after many batches, values may drift
from full recompute — that accuracy-vs-speed tradeoff is a core research
question (H1 and H3 hypotheses).

Classes:
    IncrementalCloseness  – incremental updater for closeness centrality.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import networkx as nx


# ---------------------------------------------------------------------------
# Module-level igraph conversion helper
# ---------------------------------------------------------------------------

def _nx_to_igraph(G: nx.Graph):
    """Convert a NetworkX graph to an igraph Graph, preserving node identity.

    Parameters
    ----------
    G : nx.Graph
        The NetworkX graph to convert.

    Returns
    -------
    tuple[ig.Graph, list, dict]
        (ig_graph, node_list, node_to_idx) where:
        - node_list[i]   is the original NetworkX node id for igraph vertex i
        - node_to_idx[v] is the igraph vertex index for NetworkX node v
    """
    import igraph as ig
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    ig_graph = ig.Graph(n=len(node_list), edges=edges)
    return ig_graph, node_list, node_to_idx


# ---------------------------------------------------------------------------
# IncrementalCloseness class
# ---------------------------------------------------------------------------

class IncrementalCloseness:
    """Maintain and incrementally update closeness centrality estimates.

    On construction a full BFS-based closeness computation is performed
    once (via NetworkX).  Subsequent :meth:`update` calls recompute only
    the *affected* neighbourhood (≤ 2 hops from any changed edge endpoint),
    leaving all other nodes' scores unchanged.

    The per-affected-node BFS recomputation in :meth:`update` is accelerated
    using igraph's batched C-level ``distances()`` call, while the 2-hop
    neighbourhood detection and all algorithm logic remain NetworkX-based
    and unchanged from the original design.

    Parameters
    ----------
    G : nx.Graph
        The initial graph.  A copy is stored internally; the caller's *G*
        is not mutated by ICC.
    """

    def __init__(self, G: nx.Graph) -> None:
        """Initialise ICC with a full closeness computation (igraph backend).

        Uses :class:`engine.full_recompute.FullRecompute` (igraph C-backend)
        for the one-time initial closeness computation, which is mathematically
        identical to ``nx.closeness_centrality`` but orders of magnitude faster
        (e.g. ~2-3 min vs ~41 min at N=50,000).

        The import is done locally to avoid a circular import at module level
        (both modules are in the same ``engine`` package).

        Parameters
        ----------
        G : nx.Graph
            Base graph.  ICC keeps its own internal copy.
        """
        self.G: nx.Graph = G.copy()
        print("[ICC] Computing initial full closeness centrality (igraph) …")
        t0 = time.perf_counter()
        # Local import to avoid circular dependency at module level.
        from engine.full_recompute import FullRecompute
        init_result = FullRecompute().compute(self.G)
        self.centrality: dict[int, float] = init_result["centrality"]
        elapsed = (time.perf_counter() - t0) * 1_000.0
        print(f"[ICC] Initial full computation done in {elapsed:.3f} ms.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bfs_closeness(G: nx.Graph, source: int) -> float:
        """Compute closeness centrality for *source* via iterative BFS.

        Uses an iterative (queue-based) BFS to avoid recursion-depth issues
        on large graphs.

        Parameters
        ----------
        G : nx.Graph
            The current graph.
        source : int
            Node for which to compute closeness.

        Returns
        -------
        float
            Normalised closeness centrality (0.0 if node is isolated or
            the graph is disconnected from *source*).
        """
        n: int = G.number_of_nodes()
        if n <= 1:
            return 0.0

        # Iterative BFS
        visited: dict[int, int] = {source: 0}
        queue: deque[int] = deque([source])
        total_dist: int = 0

        while queue:
            node = queue.popleft()
            d_node = visited[node]
            for neighbour in G.neighbors(node):
                if neighbour not in visited:
                    visited[neighbour] = d_node + 1
                    total_dist += visited[neighbour]
                    queue.append(neighbour)

        reachable: int = len(visited)  # includes source itself
        if reachable <= 1 or total_dist == 0:
            return 0.0

        # NetworkX normalisation: (reachable-1)^2 / ((n-1) * sum_of_dist)
        # This accounts for disconnected components correctly.
        return (reachable - 1) ** 2 / ((n - 1) * total_dist)

    @staticmethod
    def _two_hop_neighbourhood(G: nx.Graph, endpoints: set[int]) -> set[int]:
        """Return all nodes within 2 hops of any node in *endpoints*.

        Uses ``nx.single_source_shortest_path_length`` with ``cutoff=2``
        for each endpoint, then merges the results.

        Parameters
        ----------
        G : nx.Graph
            Graph to traverse.
        endpoints : set[int]
            Seed nodes (typically u and v from changed edges).

        Returns
        -------
        set[int]
            Union of all nodes reachable within 2 hops from any endpoint.
        """
        affected: set[int] = set()
        for ep in endpoints:
            if ep in G:
                lengths = nx.single_source_shortest_path_length(
                    G, ep, cutoff=2
                )
                affected.update(lengths.keys())
        return affected

    # Maximum number of affected nodes to process per igraph.distances() call.
    # Keeps peak memory at roughly CHUNK_SIZE × N × 8 bytes.
    # At CHUNK_SIZE=200, N=100,000: 200 × 100,000 × 8 = 160 MB per chunk — safe.
    _IGRAPH_CHUNK_SIZE: int = 200

    def _recompute_affected_closeness(
        self, affected_nodes: set[int]
    ) -> dict[int, float]:
        """Recompute closeness for *affected_nodes* using igraph batch BFS.

        Converts ``self.G`` to igraph once, then calls igraph's C-level
        ``distances()`` for affected nodes in chunks of ``_IGRAPH_CHUNK_SIZE``.
        This is significantly faster than looping over Python-level NetworkX BFS
        calls, while keeping peak memory bounded regardless of how many nodes are
        affected (calling distances() on all nodes at once would allocate
        n_affected × N floats, which can exceed available RAM at large N and
        high churn rates — e.g. 30,000 × 50,000 × 8 bytes ≈ 12 GB).

        The closeness formula is **identical** to :meth:`_bfs_closeness`:
        ``(reachable-1)^2 / ((n-1) * sum_of_dist)``
        which correctly handles disconnected components (same as NetworkX).

        Parameters
        ----------
        affected_nodes : set[int]
            NetworkX node IDs for which to recompute closeness.

        Returns
        -------
        dict[int, float]
            Mapping from node ID → updated closeness score.
        """
        ig_graph, node_list, node_to_idx = _nx_to_igraph(self.G)
        n: int = self.G.number_of_nodes()

        affected_list = list(affected_nodes)
        affected_indices = [node_to_idx[node] for node in affected_list]

        results: dict[int, float] = {}
        chunk_size = self._IGRAPH_CHUNK_SIZE

        # Process in chunks to keep memory bounded.
        # Each igraph.distances() call allocates chunk_size × N distances.
        for chunk_start in range(0, len(affected_list), chunk_size):
            chunk_nodes = affected_list[chunk_start: chunk_start + chunk_size]
            chunk_indices = affected_indices[chunk_start: chunk_start + chunk_size]

            # igraph C-level BFS for this chunk.
            # distances_matrix[i] = distances from chunk_nodes[i] to all nodes.
            # igraph uses float('inf') for unreachable nodes.
            distances_matrix = ig_graph.distances(
                source=chunk_indices, target=None
            )

            for i, node in enumerate(chunk_nodes):
                dists = distances_matrix[i]
                # Collect finite, non-zero distances (exclude source itself
                # at d=0 and unreachable nodes at d=inf).
                reachable_dists = [
                    d for d in dists if d != float("inf") and d > 0
                ]
                total_dist = sum(reachable_dists)
                reachable = len(reachable_dists) + 1  # +1 for source itself

                if total_dist > 0 and reachable > 1:
                    # Same formula as _bfs_closeness / NetworkX:
                    # (reachable-1)^2 / ((n-1) * sum_of_dist)
                    results[node] = (reachable - 1) ** 2 / ((n - 1) * total_dist)
                else:
                    results[node] = 0.0

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, batch: dict[str, list[tuple[int, int]]]
    ) -> dict[str, Any]:
        """Apply *batch* and incrementally re-estimate closeness centrality.

        Steps
        -----
        1. Apply added/removed edges to ``self.G``.
        2. Identify endpoints of all changed edges.
        3. Expand to 2-hop neighbourhood → *affected* nodes.
        4. Recompute closeness only for affected nodes (igraph batch BFS).
        5. Leave all other nodes' scores unchanged.
        6. Return updated centrality dict, top-5, count of affected nodes,
           and elapsed time.

        Parameters
        ----------
        batch : dict
            ``{"added": [(u, v), ...], "removed": [(u, v), ...]}``
            as produced by :meth:`GraphGenerator.generate_batch_updates`.

        Returns
        -------
        dict
            ``{
                "centrality":  {node_id: float, ...},
                "top5":        [(node_id, score), ...],
                "n_affected":  int,
                "elapsed_ms":  float
            }``
        """
        t_start: float = time.perf_counter()

        added: list[tuple[int, int]] = batch.get("added", [])
        removed: list[tuple[int, int]] = batch.get("removed", [])

        # -- Step 1: apply edge changes ------------------------------------
        for u, v in added:
            self.G.add_edge(u, v)
        for u, v in removed:
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)

        # -- Step 2: collect changed endpoints -----------------------------
        endpoints: set[int] = set()
        for u, v in added:
            endpoints.update([u, v])
        for u, v in removed:
            endpoints.update([u, v])

        # -- Step 3: 2-hop neighbourhood (unchanged — NetworkX) -----------
        affected: set[int] = self._two_hop_neighbourhood(self.G, endpoints)

        # -- Step 4: recompute closeness for affected nodes (igraph batch) -
        affected_centrality = self._recompute_affected_closeness(affected)
        self.centrality.update(affected_centrality)

        elapsed_ms: float = (time.perf_counter() - t_start) * 1_000.0

        top5: list[tuple[int, float]] = sorted(
            self.centrality.items(), key=lambda kv: kv[1], reverse=True
        )[:5]

        print(
            f"[ICC] Update done in {elapsed_ms:.3f} ms | "
            f"Affected nodes: {len(affected)} | "
            f"Top1: node {top5[0][0]} score={top5[0][1]:.6f}"
        )
        return {
            "centrality": self.centrality,
            "top5": top5,
            "n_affected": len(affected),
            "elapsed_ms": elapsed_ms,
        }
