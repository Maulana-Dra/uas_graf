"""
graph_generator.py
==================
Barabási-Albert graph generation and dynamic batch-update utilities.

Classes:
    GraphGenerator  – creates BA graphs and produces edge-change batches.
"""

from __future__ import annotations

import random
from typing import Any

import networkx as nx


class GraphGenerator:
    """Generate Barabási-Albert graphs and produce safe batch edge updates.

    All random operations are reproducible via an optional *seed* parameter.
    No global state is maintained; every method operates on the arguments
    passed in or on the *G* object supplied by the caller.
    """

    # ------------------------------------------------------------------
    # Graph creation
    # ------------------------------------------------------------------

    def generate_ba_graph(
        self,
        N: int,
        m: int = 2,
        seed: int | None = None,
    ) -> nx.Graph:
        """Generate a Barabási-Albert preferential-attachment graph.

        Parameters
        ----------
        N : int
            Number of nodes (must be > m).
        m : int, optional
            Number of edges to attach from each new node (default 2).
            With m=2 the expected average degree ≈ 4.
        seed : int or None, optional
            RNG seed for reproducibility.

        Returns
        -------
        nx.Graph
            The generated undirected BA graph.
        """
        if N <= m:
            raise ValueError(
                f"N ({N}) must be strictly greater than m ({m})."
            )

        G: nx.Graph = nx.barabasi_albert_graph(n=N, m=m, seed=seed)

        avg_degree: float = (
            sum(d for _, d in G.degree()) / G.number_of_nodes()
        )
        print(
            f"[GraphGenerator] BA graph generated: "
            f"N={N}, m={m}, avg_degree={avg_degree:.4f}, "
            f"edges={G.number_of_edges()}"
        )
        return G

    # ------------------------------------------------------------------
    # Batch-update planning
    # ------------------------------------------------------------------

    def generate_batch_updates(
        self,
        G: nx.Graph,
        churn_rate: float = 0.05,
        add_fraction: float = 0.80,
        seed: int | None = None,
    ) -> dict[str, list[tuple[int, int]]]:
        """Plan a batch of edge additions and removals (does NOT apply them).

        *churn_rate* × |E| edges are targeted for change per batch.
        Of those, *add_fraction* are additions and the rest are removals.
        Removals that would disconnect the graph or create an isolated node
        are silently skipped.

        Parameters
        ----------
        G : nx.Graph
            The current graph (read-only in this method).
        churn_rate : float, optional
            Fraction of current edge count to modify (default 0.05).
        add_fraction : float, optional
            Fraction of changes that are additions (default 0.80).
        seed : int or None, optional
            RNG seed for this batch.

        Returns
        -------
        dict
            ``{"added": [(u, v), ...], "removed": [(u, v), ...]}``
        """
        rng = random.Random(seed)

        n_changes: int = max(1, int(G.number_of_edges() * churn_rate))
        n_add: int = max(0, int(n_changes * add_fraction))
        n_remove: int = max(0, n_changes - n_add)

        nodes: list[int] = list(G.nodes())

        # ---- Edge additions ------------------------------------------------
        added: list[tuple[int, int]] = []
        existing_edges: set[frozenset[int]] = {
            frozenset(e) for e in G.edges()
        }

        attempts: int = 0
        max_attempts: int = n_add * 20  # guard against dense graphs
        while len(added) < n_add and attempts < max_attempts:
            u, v = rng.sample(nodes, 2)
            if frozenset({u, v}) not in existing_edges:
                added.append((u, v))
                existing_edges.add(frozenset({u, v}))
            attempts += 1

        # ---- Edge removals -------------------------------------------------
        removed: list[tuple[int, int]] = []
        candidate_edges: list[tuple[int, int]] = list(G.edges())
        rng.shuffle(candidate_edges)

        for edge in candidate_edges:
            if len(removed) >= n_remove:
                break
            u, v = edge
            # Safety: never remove an edge that would isolate a node
            if G.degree(u) <= 1 or G.degree(v) <= 1:
                continue
            # Safety: skip removal if it would disconnect the graph
            G.remove_edge(u, v)
            connected: bool = nx.is_connected(G)
            G.add_edge(u, v)  # restore immediately
            if connected:
                removed.append((u, v))

        print(
            f"[GraphGenerator] Batch plan: "
            f"+{len(added)} edges / -{len(removed)} edges "
            f"(target +{n_add}/-{n_remove})"
        )
        return {"added": added, "removed": removed}

    # ------------------------------------------------------------------
    # Batch application
    # ------------------------------------------------------------------

    def apply_batch(
        self,
        G: nx.Graph,
        batch: dict[str, list[tuple[int, int]]],
    ) -> nx.Graph:
        """Apply a pre-planned batch of edge changes to *G* in-place.

        Parameters
        ----------
        G : nx.Graph
            Graph to modify in-place.
        batch : dict
            Output of :meth:`generate_batch_updates` with keys
            ``"added"`` and ``"removed"``.

        Returns
        -------
        nx.Graph
            The same *G* object (modified in-place).
        """
        for u, v in batch.get("added", []):
            if not G.has_edge(u, v):
                G.add_edge(u, v)

        for u, v in batch.get("removed", []):
            if G.has_edge(u, v):
                G.remove_edge(u, v)

        print(
            f"[GraphGenerator] Batch applied: "
            f"now {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges, "
            f"connected={nx.is_connected(G)}"
        )
        return G

    # ------------------------------------------------------------------
    # Graph statistics
    # ------------------------------------------------------------------

    def get_graph_stats(self, G: nx.Graph) -> dict[str, Any]:
        """Return a summary statistics dict for graph *G*.

        Parameters
        ----------
        G : nx.Graph
            Graph to inspect.

        Returns
        -------
        dict
            Keys: ``n_nodes``, ``n_edges``, ``avg_degree``,
            ``is_connected``, ``density``.
        """
        n: int = G.number_of_nodes()
        e: int = G.number_of_edges()
        avg_deg: float = (2 * e / n) if n > 0 else 0.0
        return {
            "n_nodes": n,
            "n_edges": e,
            "avg_degree": round(avg_deg, 4),
            "is_connected": nx.is_connected(G),
            "density": round(nx.density(G), 6),
        }
