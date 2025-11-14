"""
TSP Heuristic Utilities (tsp_utils.py) - V2

Refactored for modular benchmarking. Each estimator is now a separate,
timed function. Loads a pre-trained single-output ML model.
This version is TSP-specific and works with the JSON data formats
from the benchmark generators.
"""

import os
import time
import math
from math import inf
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd
import json
import subprocess
import re
from pathlib import Path
import joblib # Added for loading the ML model
from collections import deque
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.sparse.csgraph import connected_components
import joblib
import pandas as pd
# --- Configuration ---
# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
LKH_EXECUTABLE_PATH = "C:\\LKH\\LKH-3.exe"

# Path for GART 1.0 Model (your original)
GART_MODEL_PATH = SCRIPT_DIR / "GART_1.0" / "alpha_predictor_model.joblib"

# Path for Linear Model
LINEAR_MODEL_DIR = SCRIPT_DIR / "linear_model"
LINEAR_MODEL_PATH = LINEAR_MODEL_DIR / "linear_alpha_model.joblib"
LINEAR_MODEL_FEATURES_PATH = LINEAR_MODEL_DIR / "selected_features_stats.csv"

# Path for Boosted Linear Model
BOOSTED_MODEL_DIR = SCRIPT_DIR / "boosted_linear_model"
BOOSTED_MODEL_PATH = BOOSTED_MODEL_DIR / "boosted_linear_alpha_model.joblib"
BOOSTED_MODEL_FEATURES_PATH = BOOSTED_MODEL_DIR / "selected_features_stats.csv"

# Constants for feature generation
MAX_D = 5 # From feature_creator.py
RANDOM_STATE = 42 # From feature_creator.py

# ====================================================================
# JSON DATA PARSING
# ====================================================================

def parse_tsp_instance(json_path):
    """
    Parses a TSP instance .json file into a standard dictionary
    and converts coordinates to a NumPy array.
    """
    with open(json_path, 'r') as f:
        instance_data = json.load(f)
    
    instance_data['coordinates'] = np.array(instance_data['coordinates'])
    return instance_data

def parse_tsp_solution(json_path):
    """
    Parses a TSP solution .sol.json file into a standard dictionary.
    """
    with open(json_path, 'r') as f:
        solution_data = json.load(f)
    return solution_data

# ====================================================================
# CORE TSP SOLVER (LKH)
# ====================================================================

def _save_lkh_par_tsp(par_path, tsp_path, tour_path, time_limit_s=None):
    """Writes a simple LKH parameter file for TSP solving."""
    with open(par_path, "w") as f:
        f.write(f"PROBLEM_FILE = {tsp_path}\nTOUR_FILE = {tour_path}\n")
        f.write("MTSP_MIN_SIZE = 0\n") 
        if time_limit_s:
            f.write(f"TIME_LIMIT = {time_limit_s}\n")
        f.write("RUNS = 1\n")
        f.write("MAX_TRIALS = 1000\n")

def _compute_distance_tsp(p1, p2):
    """Computes Euclidean distance, rounded to nearest integer."""
    return int(math.sqrt(np.sum((p1 - p2)**2)) + 0.5)

def _save_as_tsplib_tsp(file_path, coords, tsp_name):
    """Saves coordinates as a TSPLIB file (Full Matrix)."""
    n = len(coords)
    with open(file_path, "w") as f:
        f.write(f"NAME : {tsp_name}\nTYPE : TSP\nCOMMENT : TSP Utility Solver\nDIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
        for i in range(n):
            row_distances = [_compute_distance_tsp(coords[i], coords[j]) for j in range(n)]
            f.write(" ".join(map(str, row_distances)) + "\n")
        f.write("EOF\n")

def solve_tsp_lkh(coordinates, instance_name, lkh_exe_path, scratch_dir, time_limit_s=None):
    """
    Solves a TSP defined by 'coordinates' using the LKH-3 executable.
    
    Args:
        coordinates (np.array): The array of [x, y] coordinates.
        instance_name (str): A name for the instance (used in temp files).
        lkh_exe_path (str): Path to the LKH-3.exe.
        scratch_dir (str or Path): Directory to write temp .tsp, .par, .tour files.
        time_limit_s (int, optional): Time limit for LKH.

    Returns:
        (int, list, float): A tuple of (tour_length, tour_nodes, time_taken)
    """
    solver_name = f"{instance_name.split('.')[0]}_{int(time.time())}"
    
    tsp_path = str(Path(scratch_dir) / f"{solver_name}.tsp")
    par_path = str(Path(scratch_dir) / f"{solver_name}.par")
    tour_path = str(Path(scratch_dir) / f"{solver_name}.tour")
    
    start_time = time.perf_counter()
    
    try:
        _save_as_tsplib_tsp(tsp_path, coordinates, instance_name)
        _save_lkh_par_tsp(par_path, tsp_path, tour_path, time_limit_s)
        
        lkh_cmd = [lkh_exe_path, par_path]
        
        subprocess.run(lkh_cmd, capture_output=True, text=True, check=True, timeout=300)
        
        with open(tour_path, 'r') as f:
            tour_content = f.read()
            
        tour_match = re.search(r"TOUR_SECTION\s*([\s\d-]*?)\s*EOF", tour_content, re.DOTALL)
        if not tour_match:
            raise ValueError(f"LKH ran but could not parse tour from output for {instance_name}")
        
        tour_nodes = [int(n) for n in tour_match.group(1).strip().split() if int(n) != -1]
        
        if not tour_nodes:
            raise ValueError(f"LKH ran but tour was empty for {instance_name}")
            
        tour_length = calculate_tour_cost(coordinates, tour_nodes)
        
        total_time = time.perf_counter() - start_time
        return tour_length, tour_nodes, total_time
        
    finally:
        for f_path in [tsp_path, par_path, tour_path]:
            if os.path.exists(f_path):
                os.remove(f_path)

# ====================================================================
# TSP COST CALCULATION
# ====================================================================

def calculate_tour_cost(coordinates, tour_nodes):
    """
    Calculates the integer cost of a TSP tour.
    
    Args:
        coordinates (np.array): The array of [x, y] coordinates.
        tour_nodes (list): A list of 1-based node indices in tour order.
        
    Returns:
        int: The total cost of the tour, calculated using integer math.
    """
    tour_length = 0
    for i in range(len(tour_nodes)):
        p1 = coordinates[tour_nodes[i]-1]
        p2 = coordinates[tour_nodes[(i + 1) % len(tour_nodes)]-1]
        tour_length += _compute_distance_tsp(p1, p2)
    return tour_length

# ====================================================================
# TIMED TSP ESTIMATOR FUNCTIONS
# ====================================================================

def estimate_tsp_cavdar(nodes_coords, a0=2.791, a1=0.2669):
    """
    Estimates TSP cost using Çavdar & Sokol (2015) formula.
    Returns: (estimated_cost, time_taken)
    """
    start_time = time.perf_counter()
    
    n = len(nodes_coords)
    if n <= 1: return 0.0, 0.0
    
    coords = np.asarray(nodes_coords, dtype=float)
    hull = ConvexHull(coords)
    area = hull.volume
    mu = coords.mean(axis=0)
    stdev = coords.std(axis=0)
    abs_dev = np.abs(coords - mu)
    c_bar = abs_dev.mean(axis=0)
    cstdev = np.sqrt(np.mean((abs_dev - c_bar)**2, axis=0))
    term1 = a0 * math.sqrt(n * cstdev[0] * cstdev[1])
    term2 = a1 * math.sqrt(n * stdev[0] * stdev[1] * area / (c_bar[0] * c_bar[1]))
    estimated_cost = term1 + term2
    
    if n < 1000:
        corr = 0.9325 * math.exp(0.00005298 * n) - 0.2972 * math.exp(-0.01452 * n)
        if corr > 0:
            estimated_cost = estimated_cost / corr
            
    total_time = time.perf_counter() - start_time
    return estimated_cost, total_time

def estimate_tsp_held_karp(nodes_coords):
    """
    Estimates TSP cost using Held-Karp (exact for small n).
    Returns: (estimated_cost, time_taken)
    """
    start_time = time.perf_counter()
    
    n = len(nodes_coords)
    if n <= 1: return 0.0, 0.0
    if n == 2:
        dist = np.linalg.norm(nodes_coords[0] - nodes_coords[1]) * 2
        return dist, time.perf_counter() - start_time
    if n == 3:
        cost = (np.linalg.norm(nodes_coords[0] - nodes_coords[1]) +
                np.linalg.norm(nodes_coords[1] - nodes_coords[2]) +
                np.linalg.norm(nodes_coords[2] - nodes_coords[0]))
        return cost, time.perf_counter() - start_time
        
    dist = [[0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = nodes_coords[i]
        for j in range(i+1, n):
            d = ((xi - nodes_coords[j][0])**2 + (yi - nodes_coords[j][1])**2)**0.5
            dist[i][j] = dist[j][i] = d
            
    dp = [[inf]*n for _ in range(1<<n)]
    dp[1][0] = 0.0
    
    for r in range(2, n+1):
        for subset in combinations(range(1, n), r-1):
            mask = 1
            for bit in subset:
                mask |= 1<<bit
            for last in subset:
                prev_mask = mask ^ (1<<last)
                best = inf
                rem = prev_mask
                while rem:
                    bit = rem & -rem
                    i = bit.bit_length() - 1
                    rem ^= bit
                    cand = dp[prev_mask][i] + dist[i][last]
                    if cand < best:
                        best = cand
                dp[mask][last] = best
                
    full = (1<<n) - 1
    ans = inf
    for last in range(1, n):
        cand = dp[full][last] + dist[last][0]
        if cand < ans:
            ans = cand
            
    total_time = time.perf_counter() - start_time
    return ans, total_time

def estimate_tsp_vinel(nodes_coords, b=0.768):
    """
    Estimates TSP cost using Vinel & Silva (2018) BHH-based formula.
    Returns: (estimated_cost, time_taken)
    """
    start_time = time.perf_counter()
    
    n = len(nodes_coords)
    if n <= 1: return 0.0, 0.0
    
    coords = np.asarray(nodes_coords, dtype=float)
    hull = ConvexHull(coords)
    area = hull.volume
    estimated_cost = b * math.sqrt(n * area)
    
    total_time = time.perf_counter() - start_time
    return estimated_cost, total_time

def get_mst_length(nodes_coords):
    """
    Calculates the MST length for a set of coordinates.
    Returns: (mst_length, time_taken)
    """
    start_time = time.perf_counter()
    
    num_nodes = len(nodes_coords)
    if num_nodes <= 1: return 0.0, 0.0
    
    dist_matrix = np.linalg.norm(nodes_coords[:, np.newaxis, :] - nodes_coords[np.newaxis, :, :], axis=2)
    mst = minimum_spanning_tree(dist_matrix)
    mst_length = mst.sum()
    
    total_time = time.perf_counter() - start_time
    return mst_length, total_time

def estimate_tsp_ml_alpha(nodes_coords, ml_model):
    """
    Estimates TSP cost using a pre-trained single-output alpha regressor.
    Returns: (estimated_cost, time_taken)
    """
    start_time = time.perf_counter()
    
    n = len(nodes_coords)
    if n <= 1: return 0.0, 0.0
    
    features_dict, mst_length = calculate_features_and_mst_length(nodes_coords)
    if mst_length == 0:
        return 0.0, time.perf_counter() - start_time

    # --- Single-Output Regressor Logic ---
    # Use .feature_name_in_ for modern scikit-learn
    feature_cols = ml_model.feature_name_in_
    feature_df = pd.DataFrame([features_dict])[feature_cols]
    predicted_alpha = ml_model.predict(feature_df)[0]
    estimated_cost = predicted_alpha * mst_length
    # --- End Single-Output Logic ---

    final_cost = max(mst_length, estimated_cost) # Bound by MST
    
    total_time = time.perf_counter() - start_time
    return final_cost, total_time

# ====================================================================
# ML FEATURE ENGINEERING
# ====================================================================

def calculate_features_and_mst_length(coords_list):
    """
    Optimized function to calculate the full, expanded feature set and MST length.
    Assumes depot is the first coordinate in the list.
    """
    coords = np.array(coords_list)
    features = {'n': len(coords)}
    
    dist_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    
    try:
        hull = ConvexHull(coords)
        features['convex_hull_area'] = hull.volume
        features['convex_hull_perimeter'] = hull.area
        features['hull_vertex_count'] = len(hull.vertices)
        features['hull_ratio'] = features['hull_vertex_count'] / features['n']
    except Exception:
        features.update({'convex_hull_area': 0, 'convex_hull_perimeter': 0, 'hull_vertex_count': 0, 'hull_ratio': 0})
    
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    width, height = max_c[0] - min_c[0], max_c[1] - min_c[1]
    features['bounding_box_area'] = width * height
    
    np.fill_diagonal(dist_matrix, np.inf)
    one_nn_dists = np.min(dist_matrix, axis=1)
    features['one_nn_dist_mean'] = one_nn_dists.mean()
    features['one_nn_dist_std'] = one_nn_dists.std()
    
    try:
        pca = PCA(n_components=2).fit(coords)
        eigenvalues = pca.explained_variance_
        features['pca_eigenvalue_ratio'] = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else 0
    except Exception:
        features['pca_eigenvalue_ratio'] = 1.0
        
    np.fill_diagonal(dist_matrix, 0)
    mst = minimum_spanning_tree(dist_matrix)
    mst_length = mst.sum()
    degrees = np.count_nonzero(mst.toarray() + mst.toarray().T, axis=1)
    features['mst_degree_mean'] = degrees.mean()
    features['mst_degree_max'] = degrees.max()
    features['mst_degree_std'] = degrees.std()
    features['mst_leaf_nodes_fraction'] = np.sum(degrees == 1) / features['n']

    features['coord_std_dev_x'] = coords[:, 0].std()
    features['coord_std_dev_y'] = coords[:, 1].std()
    depot_coord, customer_coords = coords[0], coords[1:]
    if len(customer_coords) > 0:
        dists_from_depot = np.linalg.norm(customer_coords - depot_coord, axis=1)
        features['avg_dist_from_depot'] = dists_from_depot.mean()
        features['max_dist_from_depot'] = dists_from_depot.max()
    else:
        features['avg_dist_from_depot'] = 0.0
        features['max_dist_from_depot'] = 0.0
        
    return features, mst_length

# ====================================================================
# VISUALIZATION
# ====================================================================

def plot_tsp_solution(instance_data, solution_data, output_path):
    """
    Generates and saves a plot of the TSP solution, showing LKH and
    Concorde tours if they exist.
    
    Args:
        instance_data (dict): Loaded data from parse_tsp_instance().
        solution_data (dict): Loaded data from parse_tsp_solution().
        output_path (str or Path): The file path to save the .png image.
    """
    
    coords = instance_data['coordinates']
    instance_name = instance_data['instance_name']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(coords[0, 0], coords[0, 1], c='red', marker='s', s=80, label='Depot (Node 1)')
    if len(coords) > 1:
        ax.scatter(coords[1:, 0], coords[1:, 1], c='blue', label='Customers', s=20, alpha=0.7)
    
    if solution_data.get('lkh_tour'):
        tour = solution_data['lkh_tour']
        tour_coords = np.array([coords[i-1] for i in tour] + [coords[tour[0]-1]])
        ax.plot(tour_coords[:, 0], tour_coords[:, 1], c='red', ls='--', lw=0.8, 
                label=f"LKH Tour (Cost: {solution_data['lkh_length']})")

    if solution_data.get('concorde_tour'):
        tour = solution_data['concorde_tour']
        tour_coords = np.array([coords[i-1] for i in tour] + [coords[tour[0]-1]])
        ax.plot(tour_coords[:, 0], tour_coords[:, 1], c='blue', ls='-', lw=0.8, 
                label=f"Concorde Tour (Cost: {solution_data['concorde_length']})")

    ax.set_title(f"Instance: {instance_name}", fontsize=10)
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

# ====================================================================
# ML FEATURE ENGINEERING (On-Demand)
# ====================================================================

# --- Internal Helper: _compute_tree_diameter (from feature_creator.py) ---
def _compute_tree_diameter(mst_adj, n):
    """Compute the weighted diameter of the tree using two BFS runs."""
    def farthest(start_node):
        distances = np.full(n, -1.0)
        distances[start_node] = 0.0
        queue = deque([start_node])
        farthest_node, max_dist = start_node, 0.0
        
        while queue:
            u = queue.popleft()
            if distances[u] > max_dist:
                max_dist = distances[u]
                farthest_node = u
            
            for v, weight in mst_adj[u]:
                if distances[v] < 0:
                    distances[v] = distances[u] + weight
                    queue.append(v)
        
        final_farthest_node = np.argmax(distances)
        return final_farthest_node, distances[final_farthest_node]

    if n < 2: return 0.0
    node1, _ = farthest(0)
    _, diameter = farthest(node1)
    return diameter

# --- Internal Helper: _compute_cluster_features (from feature_creator.py) ---
def _compute_cluster_features(coords, dist_matrix, mst_csr, n):
    """
    Computes advanced cluster features by finding an optimal K.
    """
    K_MIN = 2
    K_MAX = max(K_MIN, int(np.ceil(np.log(n))))
    MIN_SILHOUETTE_SCORE = 0.4
    
    k_feature_names = [
        'k_num_clusters', 'k_silhouette_score', 'k_alignment_error', 'k_size_ratio',
        'k_total_intra_mst_cost', 'k_inter_centroid_mst_cost', 'k_cost_ratio',
        'k_centroid_dist_mean', 'k_centroid_dist_std'
    ]
    default_output = {key: np.nan for key in k_feature_names}

    if n < 4 or K_MAX < K_MIN:
        return default_output

    best_k = -1; min_alignment_error = np.inf
    best_mst_labels = None; best_kmeans_centroids = None
    mst_data = mst_csr.data
    
    for k in range(K_MIN, K_MAX + 1):
        num_cuts = k - 1
        cut_indices = np.argpartition(mst_data, -num_cuts)[-num_cuts:]
        
        temp_csr = mst_csr.copy()
        temp_csr.data[cut_indices] = 0
        temp_csr.eliminate_zeros()
        
        n_components, mst_labels = connected_components(
            csgraph=temp_csr, directed=False, return_labels=True
        )
        
        if n_components != k:
            continue
            
        mst_centroids = np.array([coords[mst_labels == i].mean(axis=0) for i in range(k)])
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=10, random_state=RANDOM_STATE).fit(coords)
        kmeans_centroids = kmeans.cluster_centers_
        
        alignment_dist_matrix = cdist(mst_centroids, kmeans_centroids)
        row_ind, col_ind = linear_sum_assignment(alignment_dist_matrix)
        current_alignment_error = alignment_dist_matrix[row_ind, col_ind].sum()
        
        if current_alignment_error < min_alignment_error:
            min_alignment_error = current_alignment_error
            best_k = k
            best_mst_labels = mst_labels
            best_kmeans_centroids = kmeans_centroids[col_ind] 

    if best_k == -1: return default_output

    # Per your convention, this will fail hard if a cluster has 1 member
    current_silhouette_score = silhouette_score(coords, best_mst_labels, metric='euclidean')
        
    if current_silhouette_score < MIN_SILHOUETTE_SCORE:
        return default_output

    output = {}
    output['k_num_clusters'] = best_k
    output['k_silhouette_score'] = current_silhouette_score
    output['k_alignment_error'] = min_alignment_error
    
    cluster_sizes = np.bincount(best_mst_labels)
    cluster_sizes = cluster_sizes[cluster_sizes > 0]
    output['k_size_ratio'] = np.max(cluster_sizes) / np.min(cluster_sizes)
    
    total_intra_mst = 0
    for i in range(best_k):
        cluster_indices = np.where(best_mst_labels == i)[0]
        if len(cluster_indices) > 1:
            intra_dist_matrix = dist_matrix[cluster_indices, :][:, cluster_indices]
            total_intra_mst += minimum_spanning_tree(intra_dist_matrix).sum()
            
    output['k_total_intra_mst_cost'] = total_intra_mst

    if best_k > 1:
        centroid_dist_matrix = cdist(best_kmeans_centroids, best_kmeans_centroids)
        output['k_inter_centroid_mst_cost'] = minimum_spanning_tree(centroid_dist_matrix).sum()
        
        upper_tri_dists = centroid_dist_matrix[np.triu_indices(best_k, k=1)]
        if len(upper_tri_dists) > 0:
            output['k_centroid_dist_mean'] = np.mean(upper_tri_dists)
            output['k_centroid_dist_std'] = np.std(upper_tri_dists)
        else:
            output.update({'k_centroid_dist_mean': 0.0, 'k_centroid_dist_std': 0.0})
    else:
        output.update({'k_inter_centroid_mst_cost': 0.0, 'k_centroid_dist_mean': 0.0, 'k_centroid_dist_std': 0.0})

    if total_intra_mst > 1e-9:
        output['k_cost_ratio'] = output['k_inter_centroid_mst_cost'] / total_intra_mst
    else:
        output['k_cost_ratio'] = np.nan

    return output


# --- The "Smart" Feature Calculator ---
def _calculate_minimal_features(nodes_coords, n, d, required_features):
    """
    Calculates *only* the features present in the required_features set.
    Returns: (features_dict, mst_length)
    """
    features = {'n_customers': n, 'dimension': d}
    coords = nodes_coords
    
    # --- Pre-computation ---
    # These are almost always needed, so we compute them upfront
    if n > 1:
        dist_matrix = cdist(coords, coords, 'euclidean')
        mst_csr = minimum_spanning_tree(dist_matrix)
        mst_edges = mst_csr.data
        mst_total_length = np.sum(mst_edges)
    else:
        dist_matrix = np.array([[]])
        mst_csr = None
        mst_edges = np.array([])
        mst_total_length = 0.0

    # --- Bounding Box & Density Features ---
    if any(f in required_features for f in ['bounding_hypervolume', 'node_density', 'aspect_ratio']):
        dim_ranges = np.ptp(coords, axis=0)
        dim_ranges[dim_ranges < 1e-9] = 1e-9 
        features['bounding_hypervolume'] = np.prod(dim_ranges)
        features['node_density'] = n / features['bounding_hypervolume']
        features['aspect_ratio'] = np.max(dim_ranges) / np.min(dim_ranges)

    # --- Coordinate & Centroid Statistics ---
    if any(f.startswith('mean_dim_') or f.startswith('std_dim_') or f.startswith('centroid_dist_') for f in required_features):
        per_dim_mean = np.mean(coords, axis=0)
        per_dim_std = np.std(coords, axis=0)
        for i in range(MAX_D):
            if f'mean_dim_{i}' in required_features:
                features[f'mean_dim_{i}'] = per_dim_mean[i] if i < d else np.nan
            if f'std_dim_{i}' in required_features:
                features[f'std_dim_{i}'] = per_dim_std[i] if i < d else np.nan
        
        centroid = per_dim_mean
        centroid_dists = np.linalg.norm(coords - centroid, axis=1)
        if n > 1:
            if 'centroid_dist_min' in required_features: features['centroid_dist_min'] = np.min(centroid_dists)
            if 'centroid_dist_mean' in required_features: features['centroid_dist_mean'] = np.mean(centroid_dists)
            if 'centroid_dist_std' in required_features: features['centroid_dist_std'] = np.std(centroid_dists)
            if 'centroid_dist_max' in required_features: features['centroid_dist_max'] = np.max(centroid_dists)
            if 'centroid_dist_iqr' in required_features: features['centroid_dist_iqr'] = np.subtract(*np.percentile(centroid_dists, [75, 25]))
    
    # --- Pairwise Distance Statistics ---
    if any(f.startswith('pairwise_') for f in required_features) and n > 1:
        upper_tri = dist_matrix[np.triu_indices(n, k=1)]
        if 'pairwise_min' in required_features: features['pairwise_min'] = np.min(upper_tri)
        if 'pairwise_mean' in required_features: features['pairwise_mean'] = np.mean(upper_tri)
        if 'pairwise_std' in required_features: features['pairwise_std'] = np.std(upper_tri)
        if 'pairwise_max' in required_features: features['pairwise_max'] = np.max(upper_tri)
        if 'pairwise_skew' in required_features: features['pairwise_skew'] = stats.skew(upper_tri)
        if 'pairwise_kurtosis' in required_features: features['pairwise_kurtosis'] = stats.kurtosis(upper_tri)
        
        if any(f.startswith('pairwise_q') or f == 'pairwise_iqr' for f in required_features):
            p_percs = np.percentile(upper_tri, [10, 25, 50, 75, 90])
            if 'pairwise_q10' in required_features: features['pairwise_q10'] = p_percs[0]
            if 'pairwise_q25' in required_features: features['pairwise_q25'] = p_percs[1]
            if 'pairwise_q50' in required_features: features['pairwise_q50'] = p_percs[2]
            if 'pairwise_q75' in required_features: features['pairwise_q75'] = p_percs[3]
            if 'pairwise_q90' in required_features: features['pairwise_q90'] = p_percs[4]
            if 'pairwise_iqr' in required_features: features['pairwise_iqr'] = p_percs[3] - p_percs[1]

    # --- Nearest Neighbor Statistics ---
    if any(f.startswith('nn_') or f.startswith('avg_') for f in required_features) and n > 1:
        np.fill_diagonal(dist_matrix, np.inf) # Must be done after pairwise
        nn_dists = np.min(dist_matrix, axis=1) # 1-NN
        
        if 'nn_min' in required_features: features['nn_min'] = np.min(nn_dists)
        if 'nn_mean' in required_features: features['nn_mean'] = np.mean(nn_dists)
        if 'nn_std' in required_features: features['nn_std'] = np.std(nn_dists)
        if 'nn_max' in required_features: features['nn_max'] = np.max(nn_dists)
        if 'nn_iqr' in required_features: features['nn_iqr'] = np.subtract(*np.percentile(nn_dists, [75, 25]))
        if 'nn_std_mean_ratio' in required_features: features['nn_std_mean_ratio'] = features.get('nn_std', 0) / features.get('nn_mean', 1e-9)

        if n > 3 and ('avg_3nn_dist' in required_features or 'nn_ratio_2_3' in required_features):
            partitioned = np.partition(dist_matrix, 4, axis=1)[:, 1:4] 
            features['avg_3nn_dist'] = np.mean(partitioned) 
            d_2nn, d_3nn = partitioned[:, 1], partitioned[:, 2]
            valid_ratio_indices = d_3nn > 1e-9
            if np.sum(valid_ratio_indices) > 0:
                features['nn_ratio_2_3'] = np.mean(d_2nn[valid_ratio_indices] / d_3nn[valid_ratio_indices])

        if n > 1 and 'avg_ln_n_nn_dist' in required_features:
             k_nn = max(1, int(np.round(np.log(n))))
             if n > k_nn:
                features['avg_ln_n_nn_dist'] = np.mean(np.partition(dist_matrix, k_nn, axis=1)[:, k_nn])
        
        np.fill_diagonal(dist_matrix, 0.0) # Restore for cluster/MST

    # --- MST Features ---
    if any(f.startswith('mst_') for f in required_features) and n > 1:
        features['mst_total_length'] = mst_total_length
        mst_edge_mean = np.mean(mst_edges)
        mst_edge_std = np.std(mst_edges)

        if 'mst_edge_min' in required_features: features['mst_edge_min'] = np.min(mst_edges)
        if 'mst_edge_mean' in required_features: features['mst_edge_mean'] = mst_edge_mean
        if 'mst_edge_std' in required_features: features['mst_edge_std'] = mst_edge_std
        if 'mst_edge_max' in required_features: features['mst_edge_max'] = np.max(mst_edges)
        
        if mst_edge_std > 1e-9:
            if 'mst_edge_skew' in required_features: features['mst_edge_skew'] = stats.skew(mst_edges)
            if 'mst_edge_kurtosis' in required_features: features['mst_edge_kurtosis'] = stats.kurtosis(mst_edges)
        
        if 'mst_edge_std_ratio' in required_features: features['mst_edge_std_ratio'] = mst_edge_std / (mst_edge_mean + 1e-9)
        if 'mst_max_mean_ratio' in required_features: features['mst_max_mean_ratio'] = features.get('mst_edge_max', 0) / (mst_edge_mean + 1e-9)

        if any(f.startswith('mst_degree_') or f == 'mst_diameter' or f == 'mst_high_degree_count' for f in required_features):
            mst_adj = [[] for _ in range(n)]
            rows, cols = mst_csr.nonzero()
            degrees = np.zeros(n, dtype=int)
            for i in range(len(rows)):
                u, v, dist = rows[i], cols[i], mst_edges[i]
                mst_adj[u].append((v, dist))
                mst_adj[v].append((u, dist))
                degrees[u] += 1
                degrees[v] += 1
            
            mean_deg, std_deg = np.mean(degrees), np.std(degrees)
            if 'mst_degree_min' in required_features: features['mst_degree_min'] = np.min(degrees)
            if 'mst_degree_mean' in required_features: features['mst_degree_mean'] = mean_deg
            if 'mst_degree_std' in required_features: features['mst_degree_std'] = std_deg
            if 'mst_degree_max' in required_features: features['mst_degree_max'] = np.max(degrees)
            if 'mst_diameter' in required_features: features['mst_diameter'] = _compute_tree_diameter(mst_adj, n)
            if 'mst_high_degree_count' in required_features: features['mst_high_degree_count'] = np.sum(degrees > mean_deg + std_deg)
        
        if 'large_edge_count' in required_features:
            features['large_edge_count'] = np.sum(mst_edges > mst_edge_mean + mst_edge_std)
    
    # --- Cluster Features ---
    if any(f.startswith('k_') for f in required_features) and n > 1:
        cluster_features = _compute_cluster_features(coords, dist_matrix, mst_csr, n)
        # Only add the ones that were actually requested
        for k in cluster_features:
            if k in required_features:
                features[k] = cluster_features[k]

    # --- PCA Features ---
    if any(f.startswith('pca_') for f in required_features):
        pca = PCA()
        pca.fit(coords)
        evr, ev = pca.explained_variance_ratio_, pca.explained_variance_
        num_comp = len(evr)

        if 'pca_eigenvalue_ratio' in required_features:
            features['pca_eigenvalue_ratio'] = ev[0] / ev[-1] if len(ev) > 0 and ev[-1] > 1e-9 else np.nan
        
        if 'pca_cum_evr_k2' in required_features:
            features['pca_cum_evr_k2'] = (evr[0] + evr[1]) if num_comp >= 2 else (evr[0] if num_comp == 1 else np.nan)
        
        for i in range(MAX_D):
            if f'pca_evr_dim_{i}' in required_features:
                features[f'pca_evr_dim_{i}'] = evr[i] if i < num_comp else np.nan
            if f'pca_ev_dim_{i}' in required_features:
                features[f'pca_ev_dim_{i}'] = ev[i] if i < num_comp else np.nan
    
    # Fill any remaining requested features with NaN (handles n=1 case)
    for f in required_features:
        if f not in features:
            features[f] = np.nan
            
    return features, mst_total_length

# --- Internal Helper: _create_boosted_features (from Boosted_Linear_Model.py) ---
def _create_boosted_features(features_df, required_features_set):
    """
    Engineers new log and interaction features IN-PLACE on a DataFrame.
    Only computes features that are in the required_features_set.
    """
    df = features_df # Works on a copy
    
    # List of all *possible* log features
    log_features_list = [
        'n_customers', 'grid_size', 'bounding_hypervolume', 'node_density',
        'aspect_ratio', 'centroid_dist_min', 'centroid_dist_mean',
        'centroid_dist_std', 'centroid_dist_max', 'centroid_dist_iqr',
        'pairwise_min', 'pairwise_mean', 'pairwise_std', 'pairwise_max',
        'pairwise_q10', 'pairwise_q25', 'pairwise_q50', 'pairwise_q75',
        'pairwise_q90', 'pairwise_iqr', 'nn_min', 'nn_mean', 'nn_std',
        'nn_max', 'nn_iqr', 'avg_3nn_dist', 'avg_ln_n_nn_dist',
        'mst_edge_min', 'mst_edge_mean', 'mst_edge_std', 'mst_edge_max',
        'mst_diameter', 'large_edge_count', 'mst_high_degree_count',
        'k_size_ratio', 'k_total_intra_mst_cost', 'k_inter_centroid_mst_cost',
        'k_cost_ratio', 'k_centroid_dist_mean', 'k_centroid_dist_std'
    ]
    
    original_cols_to_drop = []
    
    for col in log_features_list:
        log_col_name = f'log_{col}'
        if log_col_name in required_features_set:
            col_data = df[col].fillna(0) # Impute NaNs with 0 for log
            df[log_col_name] = np.log1p(col_data)
            original_cols_to_drop.append(col)

    if 'int_cost_x_silhouette' in required_features_set:
        col1_imputed = df['k_cost_ratio'].fillna(0)
        col2_imputed = df['k_silhouette_score'].fillna(0)
        df['int_cost_x_silhouette'] = col1_imputed * col2_imputed
    
    # Drop original columns *if* they are not also in the final feature list
    for col in original_cols_to_drop:
        if col not in required_features_set:
            df = df.drop(columns=[col])
    
    return df
# ====================================================================
# NEW TIMED ML-TSP ESTIMATOR FUNCTIONS (V3)
# ====================================================================

def load_model_and_features(model_path, features_csv_path):
    """
    Public helper to load a model and its feature list from CSV.
    Returns: (model, feature_list_set)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(features_csv_path):
        raise FileNotFoundError(f"Feature list not found: {features_csv_path}")
            
    model = joblib.load(model_path)
    features_df = pd.read_csv(features_csv_path)
    feature_list_set = set(features_df['feature'].tolist())
    
    return model, feature_list_set

def estimate_tsp_from_model(nodes_coords, n, d, ml_model, feature_list_set, use_boosted_features=False):
    """
    Generic timed function to estimate TSP cost from a loaded model
    using on-demand feature calculation.
    
    Args:
        nodes_coords (np.array): Array of [x, y] coordinates.
        n (int): Number of nodes.
        d (int): Number of dimensions.
        ml_model (Pipeline): The loaded scikit-learn model.
        feature_list_set (set): The set of feature names this model needs.
        use_boosted_features (bool): Whether to run the boosted feature engineering.
        
    Returns: (estimated_cost, time_taken)
    """
    start_time = time.perf_counter()
    
    if n <= 1: return 0.0, 0.0
    
    # 1. Generate *only* the features we need
    features_dict, mst_length = _calculate_minimal_features(nodes_coords, n, d, feature_list_set)
    if mst_length == 0:
        return 0.0, time.perf_counter() - start_time
    
    features_df = pd.DataFrame([features_dict])
    
    # 2. (Optional) Engineer boosted features
    if use_boosted_features:
        features_df = _create_boosted_features(features_df.copy(), feature_list_set)
    
    # 3. Filter to the final feature list
    # The list *from the model* is the source of truth
    final_feature_list = ml_model.feature_name_in_
    
    # Ensure all required features are columns, add as NaN if not
    # (e.g., if a boosted feature wasn't created, it becomes NaN)
    for col in final_feature_list:
        if col not in features_df.columns:
            features_df[col] = np.nan
    
    X_predict = features_df[final_feature_list]

    # 4. Predict (model is a pre-fit Pipeline)
    predicted_alpha = ml_model.predict(X_predict)[0]
    estimated_cost = predicted_alpha * mst_length

    final_cost = max(mst_length, estimated_cost) # Bound by MST
    
    total_time = time.perf_counter() - start_time
    return final_cost, total_time