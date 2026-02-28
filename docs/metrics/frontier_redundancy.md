# Frontier Redundancy Clusters

This document defines the overlap-based redundancy metrics for frontier points, exposing which expansion cells offer truly distinct opportunities versus those that are functionally identical.

## Overview
As the board fills up, players often have multiple frontier points that allow placements in the exact same region using the exact same set of legal moves. Evaluating these redundant points individually inflates the perceived value of a position and wastes search time. 

The `frontier_clusters` metric identifies groups of frontier points that share a high degree of overlap in the moves they support.

## Metrics Exposed

The telemetry payload inside `get_state()` provides a `frontier_clusters` dictionary:

```json
"frontier_clusters": {
  "cluster_id": { "row,col": int, ... },
  "cluster_sizes": [ int, int, ... ],
  "num_clusters": int
}
```

*   **`cluster_id`**: A mapping from a frontier point string (`"row,col"`) to its integer cluster ID.
    *   If a point supports *zero* moves (Utility = 0), it is assigned a cluster ID of `-1`.
*   **`cluster_sizes`**: An array where the index corresponds to the `cluster_id` and the value is the total number of frontier points belonging to that cluster.
*   **`num_clusters`**: The total count of distinct clusters found (excluding the `-1` zero-support points).

## Computation Method

1.  **Support Sets**: For each frontier point $f$, we collect the set of all legal move indices $S_f$ that utilize $f$.
2.  **Overlap Calculation**: For any pair of frontier points $(f_1, f_2)$, we calculate their overlap rating using the formula:
    $$Overlap(f_1, f_2) = \frac{|S_1 \cap S_2|}{\min(|S_1|, |S_2|)}$$
3.  **Adjacency & Clustering**: If the $Overlap \ge 0.35$, the two frontier points are considered adjacent in a redundancy graph. We then find all connected components (clusters) using Depth-First Search (DFS).
4.  **Guardrails**: To prevent $O(F^2)$ scaling issues in the early game when the frontier is massive, overlap is only computed for the top 60 frontier points ranked by their Utility score. Points outside the top 60 are placed into their own isolated clusters of size 1.
