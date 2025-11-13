import numpy as np
import os
import json
from collections import deque
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from concurrent.futures import ThreadPoolExecutor
import math

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCES_DIR = os.path.join(ROOT_DIR, "instances")
SOLUTIONS_DIR = os.path.join(ROOT_DIR, "solutions")
MAX_D = 5
RANDOM_STATE = 42 # For reproducible splits
# ---

def _compute_cluster_features(coords, dist_matrix, mst_csr, n):
    """
    Computes advanced cluster features by finding an optimal K that balances
    MST-based topological cuts with K-Means geometric centers.
    """
    
    # Define constants for clustering
    K_MIN = 2
    # Define search range for K. We use ln(n) as a fast, reasonable upper bound.
    K_MAX = max(K_MIN, int(np.ceil(np.log(n))))
    MIN_SILHOUETTE_SCORE = 0.4 # Quality gate for accepting a cluster
    
    # Initialize the output dictionary with NaNs
    k_feature_names = [
        'k_num_clusters', 'k_silhouette_score', 'k_alignment_error', 'k_size_ratio',
        'k_total_intra_mst_cost', 'k_inter_centroid_mst_cost', 'k_cost_ratio',
        'k_centroid_dist_mean', 'k_centroid_dist_std'
    ]
    default_output = {key: np.nan for key in k_feature_names}

    # Clustering is not meaningful for very few points
    if n < 4 or K_MAX < K_MIN:
        return default_output

    best_k = -1
    min_alignment_error = np.inf
    best_mst_labels = None
    best_kmeans_centroids = None
    
    mst_data = mst_csr.data
    
    # --- Loop to Find Optimal K ---
    for k in range(K_MIN, K_MAX + 1):
        num_cuts = k - 1
        
        # 1. Method 1: Get MST-Topological Clusters
        
        # Find the indices of the k-1 longest edges in the MST
        cut_indices = np.argpartition(mst_data, -num_cuts)[-num_cuts:]
        
        # Create a copy of the MST and "cut" these edges
        temp_csr = mst_csr.copy()
        temp_csr.data[cut_indices] = 0
        temp_csr.eliminate_zeros()
        
        # Get the resulting connected components (clusters)
        n_components, mst_labels = connected_components(
            csgraph=temp_csr, directed=False, return_labels=True
        )
        
        # If this cut didn't produce exactly k components, it's not a
        # stable cut. Skip this k.
        if n_components != k:
            continue
            
        # Get the geometric centers of these topological clusters
        mst_centroids = np.array([coords[mst_labels == i].mean(axis=0) for i in range(k)])
        
        # 2. Method 2: Get K-Means-Geometric Clusters
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=10, random_state=RANDOM_STATE).fit(coords)
        kmeans_centroids = kmeans.cluster_centers_
        
        # 3. Compare and Score (The "Balancing" Step)
        
        # Find the minimum-cost matching between the two sets of centroids
        alignment_dist_matrix = cdist(mst_centroids, kmeans_centroids)
        row_ind, col_ind = linear_sum_assignment(alignment_dist_matrix)
        current_alignment_error = alignment_dist_matrix[row_ind, col_ind].sum()
        
        if current_alignment_error < min_alignment_error:
            min_alignment_error = current_alignment_error
            best_k = k
            best_mst_labels = mst_labels
            # Reorder K-Means centroids to match MST label order for consistency
            best_kmeans_centroids = kmeans_centroids[col_ind] 

    # If no stable k was found, exit
    if best_k == -1:
        return default_output

    # --- Quality Gate: Check if the found clustering is "good" ---
    try:
        current_silhouette_score = silhouette_score(coords, best_mst_labels, metric='euclidean')
    except ValueError:
        # Happens if a cluster has only 1 member
        return default_output
        
    if current_silhouette_score < MIN_SILHOUETTE_SCORE:
        # It's a "blob," not a real cluster. Reject it.
        return default_output

    # --- Compute Final Features (Success Case) ---
    output = {}
    output['k_num_clusters'] = best_k
    output['k_silhouette_score'] = current_silhouette_score
    output['k_alignment_error'] = min_alignment_error
    
    cluster_sizes = np.bincount(best_mst_labels)
    cluster_sizes = cluster_sizes[cluster_sizes > 0] # Handle non-sequential labels if any
    output['k_size_ratio'] = np.max(cluster_sizes) / np.min(cluster_sizes)
    
    # Calculate total Intra-cluster MST cost
    total_intra_mst = 0
    for i in range(best_k):
        cluster_indices = np.where(best_mst_labels == i)[0]
        if len(cluster_indices) > 1:
            # Slice the main distance matrix for this cluster
            intra_dist_matrix = dist_matrix[cluster_indices, :][:, cluster_indices]
            intra_mst_csr = minimum_spanning_tree(intra_dist_matrix)
            total_intra_mst += intra_mst_csr.sum()
            
    output['k_total_intra_mst_cost'] = total_intra_mst

    # Calculate Inter-cluster Centroid MST cost
    if best_k > 1:
        centroid_dist_matrix = cdist(best_kmeans_centroids, best_kmeans_centroids)
        inter_centroid_mst_csr = minimum_spanning_tree(centroid_dist_matrix)
        output['k_inter_centroid_mst_cost'] = inter_centroid_mst_csr.sum()
        
        # Get distances for mean/std stats
        upper_tri_dists = centroid_dist_matrix[np.triu_indices(best_k, k=1)]
        if len(upper_tri_dists) > 0:
            output['k_centroid_dist_mean'] = np.mean(upper_tri_dists)
            output['k_centroid_dist_std'] = np.std(upper_tri_dists)
        else:
            output['k_centroid_dist_mean'] = 0.0
            output['k_centroid_dist_std'] = 0.0
    else:
        output['k_inter_centroid_mst_cost'] = 0.0
        output['k_centroid_dist_mean'] = 0.0
        output['k_centroid_dist_std'] = 0.0

    # The "Alpha-Driver" Feature
    if total_intra_mst > 1e-9:
        output['k_cost_ratio'] = output['k_inter_centroid_mst_cost'] / total_intra_mst
    else:
        # Avoid divide-by-zero if intra-cost is 0 (e.g., all points in one spot)
        output['k_cost_ratio'] = np.nan

    return output


def compute_tree_diameter(mst_adj, n):
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
        
        # Final check in case the last node popped was the farthest
        final_farthest_node = np.argmax(distances)
        return final_farthest_node, distances[final_farthest_node]

    if n < 2: return 0.0
    node1, _ = farthest(0)
    _, diameter = farthest(node1)
    return diameter

def compute_features_for_instance(inst_data, sol_data):
    """Compute all features for a single loaded instance."""
    coords = np.array(inst_data['coordinates'])
    n = inst_data['n_customers']
    d = inst_data['dimension']
    
    features = {
        'instance_name': inst_data['instance_name'], 'n_customers': n,
        'dimension': d, 'grid_size': inst_data['grid_size'],
        'distribution_type': inst_data['distribution_type'],
    }

    # Target & Solver Info
    features['optimal_cost'] = sol_data['optimal_cost']
    features['optimal_solver'] = sol_data['optimal_solver']
    features['solve_time_s'] = sol_data['concorde_time_s'] if sol_data['optimal_solver'] == 'concorde' else sol_data['lkh_time_s']

    # Bounding Box & Density Features
    dim_ranges = np.ptp(coords, axis=0)
    # Add a small epsilon for cases where all points are on a line/plane
    dim_ranges[dim_ranges < 1e-9] = 1e-9 
    features['bounding_hypervolume'] = np.prod(dim_ranges)
    features['node_density'] = n / features['bounding_hypervolume']
    features['aspect_ratio'] = np.max(dim_ranges) / np.min(dim_ranges)

    # Coordinate & Centroid Statistics
    per_dim_mean = np.mean(coords, axis=0)
    per_dim_std = np.std(coords, axis=0)
    for i in range(MAX_D):
        features[f'mean_dim_{i}'] = per_dim_mean[i] if i < d else np.nan
        features[f'std_dim_{i}'] = per_dim_std[i] if i < d else np.nan

    centroid = per_dim_mean
    centroid_dists = np.linalg.norm(coords - centroid, axis=1)
    if n > 1:
        features.update({
            'centroid_dist_min': np.min(centroid_dists), 'centroid_dist_mean': np.mean(centroid_dists),
            'centroid_dist_std': np.std(centroid_dists), 'centroid_dist_max': np.max(centroid_dists),
            'centroid_dist_iqr': np.subtract(*np.percentile(centroid_dists, [75, 25]))
        })
    else:
        for key in ['centroid_dist_min', 'centroid_dist_mean', 'centroid_dist_std', 'centroid_dist_max', 'centroid_dist_iqr']:
            features[key] = np.nan
        

    # Pairwise, NN, and MST features
    if n > 1:
        dist_matrix = cdist(coords, coords, 'euclidean')
        
        # Pairwise Distance Statistics
        upper_tri = dist_matrix[np.triu_indices(n, k=1)]
        p_percs = np.percentile(upper_tri, [10, 25, 50, 75, 90])
        features.update({
            'pairwise_min': np.min(upper_tri), 'pairwise_mean': np.mean(upper_tri),
            'pairwise_std': np.std(upper_tri), 'pairwise_max': np.max(upper_tri),
            'pairwise_skew': stats.skew(upper_tri), 'pairwise_kurtosis': stats.kurtosis(upper_tri),
            'pairwise_q10': p_percs[0], 'pairwise_q25': p_percs[1], 'pairwise_q50': p_percs[2],
            'pairwise_q75': p_percs[3], 'pairwise_q90': p_percs[4], 'pairwise_iqr': p_percs[3] - p_percs[1],
        })
        
        # Nearest Neighbor Statistics
        np.fill_diagonal(dist_matrix, np.inf)
        nn_dists = np.min(dist_matrix, axis=1) # The 1-NN distances
        
        mean_nn = np.mean(nn_dists)
        std_nn = np.std(nn_dists)
        features.update({
            'nn_min': np.min(nn_dists), 'nn_mean': mean_nn, 'nn_std': std_nn,
            'nn_max': np.max(nn_dists), 'nn_iqr': np.subtract(*np.percentile(nn_dists, [75, 25])),
        })
        
        if mean_nn > 1e-9:
            features['nn_std_mean_ratio'] = std_nn / mean_nn
        else:
            features['nn_std_mean_ratio'] = np.nan

        # N-NN Features
        if n > 3:
            partitioned = np.partition(dist_matrix, 4, axis=1)[:, 1:4] 
            d_2nn = partitioned[:, 1]
            d_3nn = partitioned[:, 2]
            
            features['avg_3nn_dist'] = np.mean(partitioned) 

            # 2NN/3NN Ratio
            valid_ratio_indices = d_3nn > 1e-9
            if np.sum(valid_ratio_indices) > 0:
                nn_ratios = d_2nn[valid_ratio_indices] / d_3nn[valid_ratio_indices]
                features['nn_ratio_2_3'] = np.mean(nn_ratios)
            else:
                features['nn_ratio_2_3'] = np.nan
                
        else:
            features['avg_3nn_dist'] = np.nan
            features['nn_ratio_2_3'] = np.nan
            
        # ln(n) NN distance
        k_nn = max(1, int(np.round(np.log(n))))
        if n > k_nn:
            k_nn_dists = np.partition(dist_matrix, k_nn, axis=1)[:, k_nn]
            features['avg_ln_n_nn_dist'] = np.mean(k_nn_dists)
        else:
            features['avg_ln_n_nn_dist'] = np.nan


        # MST Features
        mst_csr = minimum_spanning_tree(dist_matrix)
        mst_edges = mst_csr.data
        
        mst_edge_mean = np.mean(mst_edges)
        mst_edge_std = np.std(mst_edges)

        features.update({
            'mst_total_length': np.sum(mst_edges), 'mst_edge_min': np.min(mst_edges),
            'mst_edge_mean': mst_edge_mean,
            'mst_edge_std': mst_edge_std,
            'mst_edge_max': np.max(mst_edges),
        })
        
        if mst_edge_std > 1e-9:
            features['mst_edge_skew'] = stats.skew(mst_edges)
            features['mst_edge_kurtosis'] = stats.kurtosis(mst_edges)
        else:
            features['mst_edge_skew'] = 0.0
            features['mst_edge_kurtosis'] = 0.0
        
        # --- NEW: Call Cluster Feature Function ---
        # This function is complex and contains all the new logic
        cluster_features = _compute_cluster_features(coords, dist_matrix, mst_csr, n)
        features.update(cluster_features)
        # ---
        
        # MST Topology & Cluster Proxies
        mst_adj = [[] for _ in range(n)]
        rows, cols = mst_csr.nonzero()
        degrees = np.zeros(n, dtype=int)
        for i in range(len(rows)):
            u, v, dist = rows[i], cols[i], mst_edges[i]
            mst_adj[u].append((v, dist))
            mst_adj[v].append((u, dist))
            degrees[u] += 1
            degrees[v] += 1
        
        mean_deg = np.mean(degrees)
        std_deg = np.std(degrees)
        features.update({
            'mst_degree_min': np.min(degrees), 'mst_degree_mean': mean_deg,
            'mst_degree_std': std_deg, 'mst_degree_max': np.max(degrees),
            'mst_diameter': compute_tree_diameter(mst_adj, n),
            'large_edge_count': np.sum(mst_edges > mst_edge_mean + mst_edge_std),
        })
        
        features['mst_high_degree_count'] = np.sum(degrees > mean_deg + std_deg)
        
        if mst_edge_mean > 1e-9:
            features['mst_edge_std_ratio'] = mst_edge_std / mst_edge_mean
            features['mst_max_mean_ratio'] = features['mst_edge_max'] / mst_edge_mean
        else:
            features['mst_edge_std_ratio'] = np.nan
            features['mst_max_mean_ratio'] = np.nan
    else:
        # Default all pairwise/NN/MST/cluster features to NaN for n=1
        nan_keys = [
            'pairwise_min', 'pairwise_mean', 'pairwise_std', 'pairwise_max', 'pairwise_skew', 'pairwise_kurtosis', 
            'pairwise_q10', 'pairwise_q25', 'pairwise_q50', 'pairwise_q75', 'pairwise_q90', 'pairwise_iqr',
            'nn_min', 'nn_mean', 'nn_std', 'nn_max', 'nn_iqr', 'nn_std_mean_ratio', 'avg_3nn_dist', 
            'nn_ratio_2_3', 'avg_ln_n_nn_dist',
            'mst_total_length', 'mst_edge_min', 'mst_edge_mean', 'mst_edge_std', 'mst_edge_max', 'mst_edge_skew', 
            'mst_edge_kurtosis', 'mst_degree_min', 'mst_degree_mean', 'mst_degree_std', 'mst_degree_max', 
            'mst_diameter', 'large_edge_count', 'mst_high_degree_count', 'mst_edge_std_ratio', 'mst_max_mean_ratio',
            # Add new cluster keys
            'k_num_clusters', 'k_silhouette_score', 'k_alignment_error', 'k_size_ratio',
            'k_total_intra_mst_cost', 'k_inter_centroid_mst_cost', 'k_cost_ratio',
            'k_centroid_dist_mean', 'k_centroid_dist_std'
        ]
        for key in nan_keys:
            features[key] = np.nan
            

    # PCA Features
    if n >= 1 and d >= 1:
        pca = PCA()
        pca.fit(coords)
        evr = pca.explained_variance_ratio_
        ev = pca.explained_variance_
        if len(ev) > 0 and ev[-1] > 1e-9:
            features['pca_eigenvalue_ratio'] = ev[0] / ev[-1]
        else:
            features['pca_eigenvalue_ratio'] = np.nan
        num_comp = len(evr)
        
        if num_comp >= 2:
            features['pca_cum_evr_k2'] = evr[0] + evr[1]
        elif num_comp == 1:
            features['pca_cum_evr_k2'] = evr[0]
        else:
            features['pca_cum_evr_k2'] = np.nan

    else:
        evr = []
        ev = []
        num_comp = 0
        features['pca_eigenvalue_ratio'] = np.nan
        features['pca_cum_evr_k2'] = np.nan
        
    for i in range(MAX_D):
        features[f'pca_evr_dim_{i}'] = evr[i] if i < num_comp else np.nan
        features[f'pca_ev_dim_{i}'] = ev[i] if i < num_comp else np.nan
            
    return features

def process_file(filename):
    """Worker function to load a file pair and compute its features."""
    try:
        inst_path = os.path.join(INSTANCES_DIR, filename)
        with open(inst_path, 'r') as f:
            inst_data = json.load(f)

        sol_filename = filename.replace('.json', '.sol.json')
        sol_path = os.path.join(SOLUTIONS_DIR, sol_filename)
        with open(sol_path, 'r') as f:
            sol_data = json.load(f)

        return compute_features_for_instance(inst_data, sol_data)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def create_stratified_split(df):
    """Adds a 'split' column to the DataFrame with stratified assignments."""
    stratum_cols = ['dimension', 'n_customers', 'grid_size', 'distribution_type']
    
    # Shuffle within each stratum for random assignment
    df = df.groupby(stratum_cols, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=RANDOM_STATE))
    
    # Assign splits based on cumulative percentage within each group
    df['group_frac'] = df.groupby(stratum_cols).cumcount() / df.groupby(stratum_cols)['instance_name'].transform('count')
    
    conditions = [
        df['group_frac'] < 0.05,                      # 5% for Test
        df['group_frac'] < (0.05 + 0.19),             # 19% for Validation
    ]
    choices = ['test', 'val']
    df['split'] = np.select(conditions, choices, default='train') # Remaining 76% is Train
    
    return df.drop(columns=['group_frac'])

if __name__ == '__main__':
    all_files = [f for f in os.listdir(INSTANCES_DIR) if f.endswith('.json')]
    all_features = []

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, all_files), total=len(all_files), desc="Computing Features"))

    all_features = [res for res in results if res is not None]

    if not all_features:
        print("No valid instance/solution pairs found to process.")
    else:
        df = pd.DataFrame(all_features)
        
        print("\nStratifying dataset into train, validation, and test sets...")
        df = create_stratified_split(df)
        
        output_path = os.path.join(ROOT_DIR, 'tsp_features.csv')
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Successfully generated and saved features to {output_path}")
        print("\nDataset Split Summary:")
        print(df['split'].value_counts(normalize=True).map("{:.2%}".format))