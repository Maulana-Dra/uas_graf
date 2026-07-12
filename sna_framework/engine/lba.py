"""
lba.py
======
Landmark-Based Approximation (LBA) of closeness centrality.

NOTE: As of 2026-07-12, LBA uses igraph (C backend) for 
chunked BFS distance computation instead of pure NetworkX, AND 
stores landmark distances as a numpy int32 array instead of a 
nested Python dict. This was necessary because at N=100,000 with 
5% landmarks (5,000 landmarks), the dict-based storage required 
~25GB RAM (exceeding available 20GB system memory) due to Python 
dict overhead, while a numpy array requires only ~2GB for the 
same data. This backend/storage change was approved by the course 
instructor as an acceleration layer — the LBA algorithm design 
(5% random landmark sampling, distance-based closeness 
approximation formula) is unchanged from the original.
"""

from __future__ import annotations

import time
import random
from typing import Any
import numpy as np
import networkx as nx


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


class LandmarkApproximation:
    """Approximate closeness centrality using landmark BFS distances.

    On construction, landmark nodes are selected and BFS is run from each
    landmark once.  :meth:`compute_approximation` then derives per-node
    closeness estimates in O(N × K) time where K = number of landmarks.

    Parameters
    ----------
    G : nx.Graph
        The graph to analyse.  LBA keeps a *reference* (not a copy) because
        it is designed as a read-only, one-shot estimator.
    landmark_fraction : float, optional
        Fraction of nodes to use as landmarks (default 0.05 → 5 %).
        A minimum of 10 landmarks is always enforced regardless of *N*.
    seed : int or None, optional
        RNG seed for landmark selection.
    """

    MIN_LANDMARKS: int = 10  # hard minimum regardless of graph size

    def __init__(
        self,
        G: nx.Graph,
        landmark_fraction: float = 0.05,
        seed: int | None = None,
    ) -> None:
        """Initialise LBA: select landmarks and precompute their 
        distances to all nodes using igraph's batched, chunked 
        distance computation. Distances are stored as a compact 
        numpy int32 array (n_landmarks x n_nodes), NOT a Python dict, 
        to keep memory usage feasible at N=100,000+ scale.
        """
        self.G: nx.Graph = G  # reference only — do not modify
        self.n: int = G.number_of_nodes()

        if seed is not None:
            random.seed(seed)

        all_nodes: list[int] = list(G.nodes())

        # Enforce minimum landmark count
        n_landmarks: int = max(
            self.MIN_LANDMARKS,
            int(landmark_fraction * self.n),
        )
        n_landmarks = min(n_landmarks, self.n)  # can't exceed node count

        ig_graph, node_list, node_to_idx = _nx_to_igraph(G)
        self._node_list = node_list  # preserve node order for later lookup
        self._node_to_idx = node_to_idx

        # Select landmarks (by original NetworkX node id)
        self.landmarks: list[int] = random.sample(node_list, n_landmarks)
        landmark_indices = [node_to_idx[l] for l in self.landmarks]

        print(
            f"[LBA] Selected {len(self.landmarks)} landmarks "
            f"({landmark_fraction * 100:.1f}% of N={self.n}). "
            f"Precomputing BFS distances using igraph ..."
        )

        # Precompute BFS from every landmark chunk-by-chunk
        t0 = time.perf_counter()
        
        # Preallocate numpy array (n_landmarks x self.n) with -1 (unreachable)
        self.landmark_distances = np.full(
            (n_landmarks, self.n), -1, dtype=np.int32
        )
        UNREACHABLE_SENTINEL = -1
        CHUNK_SIZE = 200

        for chunk_start in range(0, n_landmarks, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_landmarks)
            chunk_indices = landmark_indices[chunk_start:chunk_end]

            # igraph C-level BFS chunk distances
            dist_chunk = ig_graph.distances(source=chunk_indices, target=None)

            # Fast vectorized conversion to numpy array and sentinel replacement
            chunk_arr = np.array(dist_chunk, dtype=np.float32)
            chunk_arr[np.isinf(chunk_arr)] = UNREACHABLE_SENTINEL
            self.landmark_distances[chunk_start:chunk_end, :] = chunk_arr.astype(np.int32)

        elapsed = (time.perf_counter() - t0) * 1_000.0
        print(f"[LBA] BFS precomputation done in {elapsed:.3f} ms.")
        
        array_size_mb = self.landmark_distances.nbytes / 1024 / 1024
        print(f"  LBA landmark_distances array: {n_landmarks} x {self.n} "
              f"int32 = {array_size_mb:.1f} MB")

        # Store landmark index mapping for compute_approximation()
        self._landmark_indices = landmark_indices
        self.n_landmarks = n_landmarks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_approximation(self) -> dict[str, Any]:
        """Estimate closeness centrality for all nodes via landmark distances.
        Entirely vectorized using numpy.

        Formula for node *v*
        --------------------
        Let *L(v)* be the set of landmarks that can reach *v* (finite dist).

            If |L(v)| == 0:
                score(v) = 0.0
            Else:
                avg_dist = mean( dist(l, v) for l in L(v) )
                score(v) = (n - 1) / (avg_dist * n / |L(v)|)

        This scales the average landmark distance to approximate the full
        closeness formula, correcting for the sample fraction of landmarks.

        Returns
        -------
        dict
            ``{
                "centrality":  {node_id: float, ...},
                "top5":        [(node_id, score), ...],
                "n_landmarks": int,
                "elapsed_ms":  float
            }``
        """
        t_start: float = time.perf_counter()

        UNREACHABLE_SENTINEL = -1
        dist_matrix = self.landmark_distances  # shape: (n_landmarks, n_nodes)
        
        # Boolean mask of reachable node-landmark pairs
        reachable_mask = (dist_matrix != UNREACHABLE_SENTINEL)
        
        # Count how many landmarks can reach each node
        reachable_count = reachable_mask.sum(axis=0)  # shape: (n_nodes,)

        # Sum of distances to landmarks that can reach the node
        masked_dist = np.where(reachable_mask, dist_matrix, 0)
        sum_dist = masked_dist.sum(axis=0)  # shape: (n_nodes,)

        # Calculate average distance safely
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_landmark_dist = np.where(
                reachable_count > 0,
                sum_dist / np.maximum(reachable_count, 1),
                0.0
            )
            # Scale calculation
            # score = (n - 1) / (avg_landmark_dist * n / reachable_count)
            denom = avg_landmark_dist * self.n / np.maximum(reachable_count, 1)
            scores = np.where(
                (reachable_count > 0) & (denom > 0),
                (self.n - 1) / np.maximum(denom, 1e-10),
                0.0
            )

        # Map back to original NetworkX node IDs
        centrality = {
            self._node_list[i]: float(scores[i])
            for i in range(self.n)
        }

        elapsed_ms: float = (time.perf_counter() - t_start) * 1_000.0

        top5: list[tuple[int, float]] = sorted(
            centrality.items(), key=lambda kv: kv[1], reverse=True
        )[:5]

        print(
            f"[LBA] Approximation done in {elapsed_ms:.3f} ms | "
            f"Landmarks used: {self.n_landmarks} | "
            f"Top1: node {top5[0][0]} score={top5[0][1]:.6f}"
        )
        return {
            "centrality": centrality,
            "top5": top5,
            "n_landmarks": self.n_landmarks,
            "elapsed_ms": elapsed_ms,
        }

    def get_landmark_ids(self) -> list[int]:
        """Return the list of selected landmark node IDs.

        Returns
        -------
        list[int]
            Landmark node identifiers, in selection order.
        """
        return list(self.landmarks)
