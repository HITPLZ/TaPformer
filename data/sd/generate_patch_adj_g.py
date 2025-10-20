import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from collections import defaultdict
import json
import math

# Configure logging
tmp = logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# === Step 1: Load Data ===
df = pd.read_csv("sd_meta.csv")
coords = df[['Lng', 'Lat']].values
adj_matrix = np.load('sd_rn_adj.npy')
N = len(coords)

# === Step 2: Compute graph distances ===
dist_matrix = 1.0 - adj_matrix
np.fill_diagonal(dist_matrix, 0.0)
graph_distances = dijkstra(csgraph=csr_matrix(dist_matrix), directed=False)

# === Step 3: Graph-distance CCVT ===
def ccvt_graph_medoid(graph_distances: np.ndarray, points: np.ndarray,
                      num_centers: int, max_iter: int = 50) -> (np.ndarray, np.ndarray):
    rng = np.random.default_rng(seed=0)
    centers_idx = rng.choice(len(points), size=num_centers, replace=False)

    for it in range(max_iter):
        dists = graph_distances[:, centers_idx]  # shape (N, k)
        labels = np.argmin(dists, axis=1)

        converged = True
        new_centers_idx = np.empty_like(centers_idx)

        for j in range(num_centers):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                new_centers_idx[j] = centers_idx[j]
                continue

            sub_dists = graph_distances[np.ix_(members, members)]

            sum_dists = sub_dists.sum(axis=1)

            best = members[np.argmin(sum_dists)]
            new_centers_idx[j] = best

            if best != centers_idx[j]:
                converged = False

        centers_idx = new_centers_idx
        if converged:
            logger.info(f"CCVT converged at iteration {it}")
            break

    centers = points[centers_idx]
    return centers, centers_idx

# === Step 4: Capacity-constrained assignment ===
def assign_with_capacity(graph_distances: np.ndarray, centers_idx: np.ndarray,
                         max_capacity: int) -> (dict, np.ndarray, bool):

    dists = graph_distances[:, centers_idx]  # shape (N, k)
    order = np.argsort(dists, axis=1)

    patch_nodes = defaultdict(list)
    assigned = np.full(len(graph_distances), -1, dtype=int)

    # Greedy
    for node in range(len(graph_distances)):
        for j in order[node]:
            if len(patch_nodes[j]) < max_capacity:
                patch_nodes[j].append(node)
                assigned[node] = j
                break

    unassigned = np.where(assigned < 0)[0]
    for node in unassigned:
        jmin = min(patch_nodes.items(), key=lambda x: len(x[1]))[0]
        patch_nodes[jmin].append(node)
        assigned[node] = jmin

    violated = any(len(v) > max_capacity for v in patch_nodes.values())
    return patch_nodes, assigned, violated

def extract_patch_adjacencies(adj_matrix: np.ndarray, patch_nodes: dict) -> dict:
    return {pid: adj_matrix[np.ix_(nodes, nodes)] for pid, nodes in patch_nodes.items()}

if __name__ == '__main__':
    max_capacity = 30
    total_nodes = len(coords)
    start_patches = math.ceil(total_nodes / max_capacity)
    max_tries = 50

    for k in range(start_patches, start_patches + max_tries):
        logger.info(f"Trying {k} patches")
        centers, centers_idx = ccvt_graph_medoid(graph_distances, coords, k)
        patch_nodes, assigned, violated = assign_with_capacity(graph_distances, centers_idx, max_capacity)
        max_size = max(len(v) for v in patch_nodes.values())
        logger.info(f"Max patch size: {max_size}")

        if not violated:
            logger.info(f"Capacity constraint satisfied with {k} patches")
            break
    else:
        logger.error("Unable to satisfy capacity constraint within tries")
        raise RuntimeError("Cannot satisfy capacity")

    with open(f"patch_to_nodes_{max_capacity}_dj_g.json", 'w') as f:
        json.dump({int(k): v for k, v in patch_nodes.items()}, f, indent=2)
    patch_adjs = extract_patch_adjacencies(adj_matrix, patch_nodes)
    with open(f"patch_adj_matrices_{max_capacity}_dj_g.json", 'w') as f:
        json.dump({int(k): m.tolist() for k, m in patch_adjs.items()}, f, indent=2)

    logger.info("Done")
