import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import Polygon
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from collections import defaultdict
import math

# === Step 1: Load Data ===
df = pd.read_csv("ca_meta.csv")
coords = df[['Lng', 'Lat']].values
adj_matrix = np.load('ca_rn_adj.npy')
N = len(coords)

# === Step 2: Compute graph distances from adjacency matrix ===
# Convert adjacency weights to distances
dist_matrix = 1.0 - adj_matrix
np.fill_diagonal(dist_matrix, 0.0)
graph_distances = dijkstra(csgraph=csr_matrix(dist_matrix), directed=False)

def ccvt_relaxation(points, num_centers, bbox, max_iter=50, tol=1e-4):
    rng = np.random.default_rng(seed=0)
    centers = points[rng.choice(len(points), size=num_centers, replace=False)]
    centers_idx = np.array(rng.choice(len(points), size=num_centers, replace=False))
    for iteration in range(max_iter):
        dists_to_centers = graph_distances[:, centers_idx]
        labels = np.argmin(dists_to_centers, axis=1)
        patch_points = defaultdict(list)
        for i, lbl in enumerate(labels):
            patch_points[lbl].append(points[i])
        new_centers = []
        converged = True
        for j in range(num_centers):
            pts = patch_points.get(j, [])
            if not pts:
                new_centers.append(centers[j])
                continue
            centroid = np.mean(pts, axis=0)
            if np.linalg.norm(centroid - centers[j]) > tol:
                converged = False
            new_centers.append(centroid)
        centers = np.array(new_centers)

        centers_idx = np.argmin(graph_distances[:, centers_idx][:, :], axis=0)
        if converged:
            print(f"CCVT converged in iteration {iteration}")
            break
    return centers, centers_idx

def assign_with_capacity(points, centers_idx, max_capacity):
    # precompute distances[i, j] = graph_distances[i, centers_idx[j]]
    dists = graph_distances[:, centers_idx]
    indices = np.argsort(dists, axis=1)
    patch_nodes = defaultdict(list)
    assigned = np.full(len(points), -1)
    for i in range(len(points)):
        for j_idx in indices[i]:
            if len(patch_nodes[j_idx]) < max_capacity:
                patch_nodes[j_idx].append(i)
                assigned[i] = j_idx
                break
    unassigned = np.where(assigned == -1)[0]
    for i in unassigned:
        smallest = min(patch_nodes.items(), key=lambda x: len(x[1]))[0]
        patch_nodes[smallest].append(i)
        assigned[i] = smallest
    violated = any(len(v) > max_capacity for v in patch_nodes.values())
    return patch_nodes, assigned, violated

# === Step 6: Main loop to find suitable patch count ===
max_capacity = 120
start_patches = math.ceil(8600 / max_capacity)
max_try = 50

for num_patches in range(start_patches, start_patches + max_try):
    print(f"\n Trying {num_patches} patches")
    centers, centers_idx = ccvt_relaxation(coords, num_patches)
    patch_nodes, assigned, violated = assign_with_capacity(coords, centers_idx, max_capacity)
    max_size = max(len(v) for v in patch_nodes.values())
    print(f"Max patch size: {max_size}")
    if not violated:
        print(f"Capacity constraint satisfied with {num_patches} patches")
        break
else:
    print("Reached limit without satisfying capacity")
    raise RuntimeError("Cannot satisfy capacity")

# === Step 7: Extract patch adjacency matrices ===
def extract_patch_adjacencies(adj_matrix, patch_nodes):
    patch_adjs = {}
    for pid, nodes in patch_nodes.items():
        sub = adj_matrix[np.ix_(nodes, nodes)]
        patch_adjs[pid] = sub
    return patch_adjs

patch_adj_matrices = extract_patch_adjacencies(adj_matrix, patch_nodes)

# === Step 8: Save results ===
with open(f"patch_to_nodes_{max_capacity}_dj.json", "w") as f:
    json.dump({str(k): v for k, v in patch_nodes.items()}, f, indent=2)
with open(f"patch_adj_matrices_{max_capacity}_dj.json", "w") as f:
    json.dump({str(k): mat.tolist() for k, mat in patch_adj_matrices.items()}, f, indent=2)
