"""
NN Estimator v7.2: A "run-all" script that implements a "Two-Pass"
feature creation process to generate SOTA features and trains a 
"Two-Tower" (Varol-style) PyTorch model on them.

This version fixes:
1. [FATAL] A NameError for 'silhouette_score' that caused all workers to fail.
2. [Warning] A KMeans convergence warning for duplicate points.
3. [Warning] A divide-by-zero warning for node_density/aspect_ratio.
"""

import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import optuna
from tqdm import tqdm
import random
import json
import math
from concurrent.futures import ThreadPoolExecutor

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy import stats
from scipy.optimize import linear_sum_assignment
from collections import deque
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score # <-- FIX 1: Added import

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, 'tsp_features_v2.csv') # Caches V2 features
INSTANCES_DIR = os.path.join(ROOT_DIR, "instances")
SOLUTIONS_DIR = os.path.join(ROOT_DIR, "solutions")

MODEL_DIR = os.path.join(ROOT_DIR, 'nn_model_pytorch_v7_twopass')
MODEL_OUTPUT_FILE = os.path.join(MODEL_DIR, 'nn_alpha_model.pt')
CONT_PREPROCESSOR_FILE = os.path.join(MODEL_DIR, 'nn_cont_preprocessor.joblib')
CAT_PREPROCESSOR_FILE = os.path.join(MODEL_DIR, 'nn_cat_preprocessor.joblib')

RANDOM_STATE = 42
OPTUNA_N_TRIALS = 50
EARLY_STOPPING_PATIENCE = 20
N_TRIAL_EPOCHS = 20
N_FINAL_EPOCHS = 200
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
CLUSTER_THRESHOLD = 0.4
MAX_D = 5

# --- 1. SETUP ---

def set_random_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. V2 FEATURE CREATION (PASS 1 & 2) ---

def compute_tree_diameter(mst_adj, n):
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

def _compute_cluster_features(coords, dist_matrix, mst_csr, n):
    K_MIN = 2
    K_MAX = max(K_MIN, int(np.ceil(np.log(n))))
    
    # FIX 2: Get the number of unique points
    n_unique_points = len(np.unique(coords, axis=0))
    
    if n < 4 or K_MAX < K_MIN or n_unique_points < K_MIN:
        return { 'k_num_clusters': np.nan, 'k_silhouette_score': np.nan, 
                 'k_alignment_error': np.nan, 'best_mst_labels': None, 
                 'best_kmeans_centroids': None }

    best_k = -1
    min_alignment_error = np.inf
    best_mst_labels = None
    best_kmeans_centroids = None
    mst_data = mst_csr.data
    
    # Only search up to the number of unique points
    for k in range(K_MIN, min(K_MAX, n_unique_points) + 1):
        num_cuts = k - 1
        if num_cuts >= len(mst_data): continue
            
        cut_indices = np.argpartition(mst_data, -num_cuts)[-num_cuts:]
        temp_csr = mst_csr.copy()
        temp_csr.data[cut_indices] = 0
        temp_csr.eliminate_zeros()
        n_components, mst_labels = connected_components(csgraph=temp_csr, directed=False, return_labels=True)
        
        if n_components != k: continue
            
        mst_centroids = np.array([coords[mst_labels == i].mean(axis=0) for i in range(k)])
        
        # We already know k <= n_unique_points, so this is safe
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

    if best_k == -1:
        return { 'k_num_clusters': np.nan, 'k_silhouette_score': np.nan, 
                 'k_alignment_error': np.nan, 'best_mst_labels': None, 
                 'best_kmeans_centroids': None }

    try:
        # FIX 1: Use silhouette_score (from sklearn) not stats.silhouette_score
        current_silhouette_score = silhouette_score(coords, best_mst_labels, metric='euclidean')
    except ValueError:
        current_silhouette_score = np.nan
        
    if current_silhouette_score < CLUSTER_THRESHOLD:
        return { 'k_num_clusters': best_k, 'k_silhouette_score': current_silhouette_score, 
                 'k_alignment_error': min_alignment_error, 'best_mst_labels': None, 
                 'best_kmeans_centroids': None }

    return { 'k_num_clusters': best_k, 'k_silhouette_score': current_silhouette_score, 
             'k_alignment_error': min_alignment_error, 'best_mst_labels': best_mst_labels, 
             'best_kmeans_centroids': best_kmeans_centroids }

def _compute_sub_cluster_features(sub_coords, sub_dist_matrix):
    sub_n = len(sub_coords)
    sub_feats = {'sub_pca_ratio': np.nan, 'sub_nn_mean': np.nan, 'sub_density': np.nan}
    
    if sub_n < 2:
        return sub_feats
    
    # Sub-PCA
    if sub_n >= 2 and sub_coords.shape[1] > 1:
        try:
            sub_pca = PCA(n_components=2)
            sub_pca.fit(sub_coords)
            ev = sub_pca.explained_variance_
            if len(ev) > 1 and ev[-1] > 1e-9:
                sub_feats['sub_pca_ratio'] = ev[0] / ev[-1]
        except ValueError:
            pass

    # Sub-NN Mean
    np.fill_diagonal(sub_dist_matrix, np.inf)
    sub_feats['sub_nn_mean'] = np.mean(np.min(sub_dist_matrix, axis=1))
    
    # Sub-Density
    dim_ranges = np.ptp(sub_coords, axis=0)
    dim_ranges[dim_ranges < 1e-9] = 1e-9
    sub_hypervolume = np.prod(dim_ranges)
    
    # FIX 3: Robust check for zero hypervolume
    if sub_hypervolume > 1e-9:
        sub_feats['sub_density'] = sub_n / sub_hypervolume
    else:
        sub_feats['sub_density'] = np.inf # Will be handled by preprocessor
        
    return sub_feats

def compute_features_for_instance_v2(inst_data, sol_data):
    coords = np.array(inst_data['coordinates'])
    n = inst_data['n_customers']
    d = inst_data['dimension']
    
    features = {
        'instance_name': inst_data['instance_name'], 'n_customers': n,
        'dimension': d, 'grid_size': inst_data['grid_size'],
        'optimal_cost': sol_data['optimal_cost']
    }

    # --- Pass 1: Global Features ---
    dim_ranges = np.ptp(coords, axis=0)
    dim_ranges[dim_ranges < 1e-9] = 1e-9 
    features['bounding_hypervolume'] = np.prod(dim_ranges)
    
    # FIX 3: Robust check for zero hypervolume
    if features['bounding_hypervolume'] > 1e-9:
        features['node_density'] = n / features['bounding_hypervolume']
        features['aspect_ratio'] = np.max(dim_ranges) / np.min(dim_ranges)
    else:
        features['node_density'] = np.inf # Will be handled by preprocessor
        features['aspect_ratio'] = 1.0 # Min and Max are equal

    per_dim_mean = np.mean(coords, axis=0)
    per_dim_std = np.std(coords, axis=0)
    for i in range(MAX_D):
        features[f'mean_dim_{i}'] = per_dim_mean[i] if i < d else np.nan
        features[f'std_dim_{i}'] = per_dim_std[i] if i < d else np.nan

    if n > 1:
        centroid_dists = np.linalg.norm(coords - per_dim_mean, axis=1)
        features.update({
            'centroid_dist_mean': np.mean(centroid_dists),
            'centroid_dist_std': np.std(centroid_dists),
        })
        
        dist_matrix = cdist(coords, coords, 'euclidean')
        upper_tri = dist_matrix[np.triu_indices(n, k=1)]
        features.update({
            'pairwise_mean': np.mean(upper_tri),
            'pairwise_std': np.std(upper_tri),
            'pairwise_skew': stats.skew(upper_tri),
        })
        
        np.fill_diagonal(dist_matrix, np.inf)
        nn_dists = np.min(dist_matrix, axis=1)
        features.update({
            'nn_mean': np.mean(nn_dists), 'nn_std': np.std(nn_dists),
        })

        mst_csr = minimum_spanning_tree(dist_matrix)
        mst_edges = mst_csr.data
        mst_edge_mean = np.mean(mst_edges)
        mst_edge_std = np.std(mst_edges)
        features['mst_total_length'] = np.sum(mst_edges)
        features['mst_edge_mean'] = mst_edge_mean
        features['mst_edge_std'] = mst_edge_std
        if mst_edge_std > 1e-9:
            features['mst_edge_skew'] = stats.skew(mst_edges)
        else:
            features['mst_edge_skew'] = 0.0
            
        # --- Pass 1: Cluster Features ---
        cluster_info = _compute_cluster_features(coords, dist_matrix, mst_csr, n)
        features.update({
            'k_num_clusters': cluster_info['k_num_clusters'],
            'k_silhouette_score': cluster_info['k_silhouette_score'],
            'k_alignment_error': cluster_info['k_alignment_error']
        })
        
        # --- Pass 2: Sub-Problem Features ---
        if cluster_info['best_mst_labels'] is not None:
            best_k = cluster_info['k_num_clusters']
            best_labels = cluster_info['best_mst_labels']
            best_centroids = cluster_info['best_kmeans_centroids']
            
            cluster_sizes = np.bincount(best_labels)
            features['k_size_ratio'] = np.max(cluster_sizes) / np.min(cluster_sizes)
            
            sub_pca_ratios = []
            sub_nn_means = []
            sub_densities = []
            
            total_intra_mst = 0
            for i in range(int(best_k)):
                cluster_indices = np.where(best_labels == i)[0]
                if len(cluster_indices) < 2: continue
                    
                sub_coords = coords[cluster_indices]
                sub_dist_matrix = dist_matrix[cluster_indices, :][:, cluster_indices]
                
                sub_feats = _compute_sub_cluster_features(sub_coords, sub_dist_matrix)
                sub_pca_ratios.append(sub_feats['sub_pca_ratio'])
                sub_nn_means.append(sub_feats['sub_nn_mean'])
                sub_densities.append(sub_feats['sub_density'])
                
                sub_mst_csr = minimum_spanning_tree(sub_dist_matrix)
                total_intra_mst += sub_mst_csr.sum()
            
            features['k_total_intra_mst_cost'] = total_intra_mst
            
            with np.warnings.catch_warnings():
                np.warnings.filterwarnings('ignore', r'Mean of empty slice')
                np.warnings.filterwarnings('ignore', r'Degrees of freedom <= 0')
                features['k_pca_ratio_mean'] = np.nanmean(sub_pca_ratios)
                features['k_pca_ratio_std'] = np.nanstd(sub_pca_ratios)
                features['k_nn_mean_mean'] = np.nanmean(sub_nn_means)
                features['k_nn_mean_std'] = np.nanstd(sub_nn_means)
                features['k_density_mean'] = np.nanmean(sub_densities)
                features['k_density_std'] = np.nanstd(sub_densities)

            centroid_dist_matrix = cdist(best_centroids, best_centroids)
            inter_centroid_mst_csr = minimum_spanning_tree(centroid_dist_matrix)
            features['k_inter_centroid_mst_cost'] = inter_centroid_mst_csr.sum()
            
            if total_intra_mst > 1e-9:
                features['k_cost_ratio'] = features['k_inter_centroid_mst_cost'] / total_intra_mst
            else:
                features['k_cost_ratio'] = np.nan
        else:
            sub_feat_names = ['k_size_ratio', 'k_total_intra_mst_cost', 
                              'k_pca_ratio_mean', 'k_pca_ratio_std',
                              'k_nn_mean_mean', 'k_nn_mean_std', 'k_density_mean',
                              'k_density_std', 'k_inter_centroid_mst_cost', 'k_cost_ratio']
            for name in sub_feat_names:
                features[name] = np.nan

    if n >= 2 and d >= 1:
        try:
            pca = PCA()
            pca.fit(coords)
            evr = pca.explained_variance_ratio_
            if len(evr) > 1 and evr[-1] > 1e-9:
                features['pca_eigenvalue_ratio'] = evr[0] / evr[-1]
            for i in range(MAX_D):
                features[f'pca_evr_dim_{i}'] = evr[i] if i < len(evr) else np.nan
        except ValueError:
            pass
            
    return features

def process_file_v2(instance_filename):
    try:
        inst_path = os.path.join(INSTANCES_DIR, instance_filename)
        with open(inst_path, 'r') as f:
            inst_data = json.load(f)

        sol_filename = instance_filename.replace('.json', '.sol.json')
        sol_path = os.path.join(SOLUTIONS_DIR, sol_filename)
        with open(sol_path, 'r') as f:
            sol_data = json.load(f)

        return compute_features_for_instance_v2(inst_data, sol_data)
    except Exception:
        # import traceback
        # print(f"Error processing {instance_filename}:\n{traceback.format_exc()}")
        return None

def load_or_generate_features_v2():
    if os.path.exists(DATA_FILE):
        print(f"Loading cached V2 features from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
        return df
        
    print(f"V2 feature file not found. Generating from raw data...")
    print(f"Scanning {INSTANCES_DIR} and {SOLUTIONS_DIR}...")
    
    if not (os.path.exists(INSTANCES_DIR) and os.path.exists(SOLUTIONS_DIR)):
        print(f"Error: INSTANCES_DIR ('{INSTANCES_DIR}') or SOLUTIONS_DIR ('{SOLUTIONS_DIR}') not found.")
        print("Please check your configuration.")
        return None

    all_files = [f for f in os.listdir(INSTANCES_DIR) if f.endswith('.json')]
    all_features = []

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file_v2, all_files), total=len(all_files), desc="Computing V2 Features"))

    all_features = [res for res in results if res is not None]

    if not all_features:
        print("CRITICAL ERROR: No valid instance/solution pairs found to process.")
        return None
        
    df = pd.DataFrame(all_features)
    
    print("Creating stratified data splits...")
    stratum_cols = ['dimension', 'n_customers', 'grid_size']
    
    df['distribution_type'] = df['instance_name'].apply(lambda x: x.split('_')[-2])
    
    # FIX 4: Add include_groups=False to silence the warning
    df = df.groupby(stratum_cols, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=RANDOM_STATE), include_groups=False)
    
    df['group_frac'] = df.groupby(stratum_cols).cumcount() / df.groupby(stratum_cols)['instance_name'].transform('count')
    conditions = [
        df['group_frac'] < 0.05,
        df['group_frac'] < (0.05 + 0.19),
    ]
    choices = ['test', 'val']
    df['split'] = np.select(conditions, choices, default='train')
    df = df.drop(columns=['group_frac', 'distribution_type'])
    
    print(f"Saving new V2 features to {DATA_FILE} for future runs...")
    df.to_csv(DATA_FILE, index=False)
    
    return df

# --- 3. PYTORCH "TWO-TOWER" MODEL (v7 ARCH) ---

class TabularDataset(Dataset):
    def __init__(self, X_cont, X_cat, y):
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cont[idx], self.X_cat[idx], self.y[idx]

def build_preprocessors(X_train_df):
    print("Building SOTA 'Two-Tower' preprocessing pipelines for V2 features...")
    
    categorical_features_list = ['dimension', 'grid_size']
    
    continuous_features_list = [
        col for col in X_train_df.columns if col not in categorical_features_list
    ]
    print(f"  Found {len(continuous_features_list)} continuous features.")
    print(f"  Found {len(categorical_features_list)} categorical features.")

    continuous_pipe = Pipeline([
        # FIX 5: Add keep_empty_feature=True to silence warning
        ('imputer', SimpleImputer(strategy='constant', fill_value=0, add_indicator=True, keep_empty_features=True)),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    cont_preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', continuous_pipe, continuous_features_list)
        ],
        remainder='drop'
    )
    
    cat_preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', categorical_pipe, categorical_features_list)
        ],
        remainder='drop'
    )
    
    cont_preprocessor.fit(X_train_df)
    cat_preprocessor.fit(X_train_df)
    
    n_cont_features = cont_preprocessor.transform(X_train_df).shape[1]
    n_cat_features = cat_preprocessor.transform(X_train_df).shape[1]
    
    print(f"  Continuous pipeline output features: {n_cont_features}")
    print(f"  Categorical pipeline output features: {n_cat_features}")
    
    return (
        cont_preprocessor, cat_preprocessor, 
        n_cont_features, n_cat_features,
        continuous_features_list, categorical_features_list
    )

def _build_mlp_tower(in_features, n_units_list, activation, dropout_rate):
    layers = []
    for n_units in n_units_list:
        layers.append(nn.Linear(in_features, n_units))
        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU())
        else:
            layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        in_features = n_units
    return nn.Sequential(*layers)

class TabularNet(nn.Module):
    def __init__(self, n_cont_features, n_cat_features, params):
        super(TabularNet, self).__init__()
        
        self.cont_tower = _build_mlp_tower(
            in_features=n_cont_features,
            n_units_list=params['n_units_cont_list'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate']
        )
        
        self.cat_tower = _build_mlp_tower(
            in_features=n_cat_features,
            n_units_list=params['n_units_cat_list'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate']
        )
        
        head_input_size = params['n_units_cont_list'][-1] + params['n_units_cat_list'][-1]
        
        self.head = _build_mlp_tower(
            in_features=head_input_size,
            n_units_list=params['n_units_head_list'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate']
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(params['n_units_head_list'][-1], 1),
            nn.Sigmoid()
        )

    def forward(self, x_cont, x_cat):
        cont_embedding = self.cont_tower(x_cont)
        cat_embedding = self.cat_tower(x_cat)
        combined = torch.cat([cont_embedding, cat_embedding], dim=1)
        head_output = self.head(combined)
        return self.output_layer(head_output)

# --- 4. TRAINING & VALIDATION LOOPS ---

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for x_cont, x_cat, targets in loader:
        x_cont, x_cat, targets = x_cont.to(device), x_cat.to(device), targets.to(device)
        outputs = model(x_cont, x_cat)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate_model(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_cont, x_cat, targets in loader:
            x_cont, x_cat, targets = x_cont.to(device), x_cat.to(device), targets.to(device)
            outputs = model(x_cont, x_cat)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(loader)

# --- 5. OPTUNA HYPERPARAMETER TUNING ---

def objective(trial, X_train_cont_tf, X_train_cat_tf, y_train, X_val_cont_tf, X_val_cat_tf, y_val, n_cont_features, n_cat_features):
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    n_layers_cont = trial.suggest_int('n_layers_cont', 2, 5)
    n_units_cont_list = []
    for i in range(n_layers_cont):
        n_units = trial.suggest_int(f'n_units_cont_{i}', 64, 512, log=True)
        n_units_cont_list.append(n_units)

    n_layers_cat = trial.suggest_int('n_layers_cat', 1, 3)
    n_units_cat_list = []
    for i in range(n_layers_cat):
        n_units = trial.suggest_int(f'n_units_cat_{i}', 16, 128, log=True)
        n_units_cat_list.append(n_units)

    n_layers_head = trial.suggest_int('n_layers_head', 1, 3)
    n_units_head_list = []
    for i in range(n_layers_head):
        n_units = trial.suggest_int(f'n_units_head_{i}', 32, 256, log=True)
        n_units_head_list.append(n_units)

    params = {
        'n_layers_cont': n_layers_cont,
        'n_units_cont_list': n_units_cont_list,
        'n_layers_cat': n_layers_cat,
        'n_units_cat_list': n_units_cat_list,
        'n_layers_head': n_layers_head,
        'n_units_head_list': n_units_head_list,
        'activation': activation,
        'dropout_rate': dropout_rate,
    }
    
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])

    train_dataset = TabularDataset(X_train_cont_tf, X_train_cat_tf, y_train)
    val_dataset = TabularDataset(X_val_cont_tf, X_val_cat_tf, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TabularNet(n_cont_features, n_cat_features, params).to(device)
    criterion = nn.MSELoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
    
    best_val_loss = np.inf
    for epoch in range(N_TRIAL_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate_model(model, val_loader, criterion)
        
        if np.isnan(val_loss) or np.isinf(val_loss):
            return np.inf
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_val_loss

# --- 6. MAIN EXECUTION ---

if __name__ == "__main__":
    set_random_seed(RANDOM_STATE)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("--- 1. Loading/Generating V2 Features ---")
    df = load_or_generate_features_v2()
    
    if df is None:
        print("Halting script due to feature generation error.")
    else:
        print(f"V2 features loaded. Shape: {df.shape}")
        
        print("Creating data splits and scaled target...")
        
        mst_divisor = df['mst_total_length'].replace(0, 1e-9)
        df['alpha'] = df['optimal_cost'] / mst_divisor
        df['alpha_scaled'] = df['alpha'].clip(1.0, 2.0) - 1.0
        
        y = df['alpha_scaled']
        
        features_to_drop = [
            'instance_name', 'optimal_cost', 'mst_total_length', 
            'alpha', 'alpha_scaled', 'split'
        ]
        existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
        X = df.drop(columns=existing_cols_to_drop)
        
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        train_mask = (df['split'] == 'train')
        val_mask = (df['split'] == 'val')
        test_mask = (df['split'] == 'test')
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        X_train_full = pd.concat([X_train, X_val])
        y_train_full = pd.concat([y_train, y_val])
        
        print(f"Data splits created: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")

        (
            cont_preprocessor, cat_preprocessor, 
            n_cont_features, n_cat_features,
            cont_features_list, cat_features_list
        ) = build_preprocessors(X_train)
        
        print(f"Saving preprocessors to {MODEL_DIR}...")
        joblib.dump(cont_preprocessor, CONT_PREPROCESSOR_FILE)
        joblib.dump(cat_preprocessor, CAT_PREPROCESSOR_FILE)
        
        print("Transforming data splits...")
        X_train_cont_tf = cont_preprocessor.transform(X_train)
        X_train_cat_tf = cat_preprocessor.transform(X_train)
        
        X_val_cont_tf = cont_preprocessor.transform(X_val)
        X_val_cat_tf = cat_preprocessor.transform(X_val)
        
        X_test_cont_tf = cont_preprocessor.transform(X_test)
        X_test_cat_tf = cat_preprocessor.transform(X_test)
        
        X_train_full_cont_tf = cont_preprocessor.transform(X_train_full)
        X_train_full_cat_tf = cat_preprocessor.transform(X_train_full)
        
        print(f"Data transformed.")
        
        print(f"\n--- 2. Running Optuna Hyperparameter Tuning ({OPTUNA_N_TRIALS} trials) ---")
        objective_func = lambda trial: objective(
            trial, 
            X_train_cont_tf, X_train_cat_tf, y_train,
            X_val_cont_tf, X_val_cat_tf, y_val,
            n_cont_features, n_cat_features
        )
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_func, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
        
        if study.best_value == np.inf or np.isnan(study.best_value):
            print("\n\nCRITICAL ERROR: Optuna tuning failed. All trials resulted in 'inf' or 'nan'.")
        else:
            print(f"\nOptuna tuning complete. Best Validation MSE: {study.best_value:.6f}")
            print("Best parameters found:")
            print(study.best_params)

            print("\n--- 3. Training Final Model ---")
            best_params = study.best_params
            
            final_model_params = {
                'n_layers_cont': best_params['n_layers_cont'],
                'n_units_cont_list': [best_params[f'n_units_cont_{i}'] for i in range(best_params['n_layers_cont'])],
                'n_layers_cat': best_params['n_layers_cat'],
                'n_units_cat_list': [best_params[f'n_units_cat_{i}'] for i in range(best_params['n_layers_cat'])],
                'n_layers_head': best_params['n_layers_head'],
                'n_units_head_list': [best_params[f'n_units_head_{i}'] for i in range(best_params['n_layers_head'])],
                'activation': best_params['activation'],
                'dropout_rate': best_params['dropout_rate'],
            }
            
            train_full_dataset = TabularDataset(X_train_full_cont_tf, X_train_full_cat_tf, y_train_full)
            val_size = int(len(train_full_dataset) * VALIDATION_SPLIT)
            train_size = len(train_full_dataset) - val_size
            train_data, val_data = random_split(train_full_dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_data, batch_size=best_params["batch_size"], shuffle=True)
            val_loader = DataLoader(val_data, batch_size=best_params["batch_size"], shuffle=False)

            final_model = TabularNet(n_cont_features, n_cat_features, final_model_params).to(device)
            
            criterion = nn.MSELoss()
            optimizer = getattr(optim, best_params["optimizer"])(
                final_model.parameters(), lr=best_params["learning_rate"]
            )

            best_val_loss = np.inf
            epochs_no_improve = 0
            best_model_state = final_model.state_dict()

            for epoch in range(N_FINAL_EPOCHS):
                # --- FIX 2: Use final_model, not model ---
                train_loss = train_one_epoch(final_model, train_loader, criterion, optimizer)
                val_loss = validate_model(final_model, val_loader, criterion)
                print(f"Epoch {epoch+1}/{N_FINAL_EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = final_model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                        print(f"Early stopping triggered after {epoch+1} epochs.")
                        break
            
            print("Final model training complete.")
            
            print("\n--- 4. Evaluating Final Model on Test Set ---")
            final_model.load_state_dict(best_model_state)
            
            test_dataset = TabularDataset(X_test_cont_tf, X_test_cat_tf, y_test)
            test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)
            
            final_model.eval()
            all_preds_scaled = []
            all_test_scaled = []
            with torch.no_grad():
                for x_cont, x_cat, targets in test_loader:
                    x_cont, x_cat, targets = x_cont.to(device), x_cat.to(device), targets.to(device)
                    outputs = final_model(x_cont, x_cat)
                    all_preds_scaled.append(outputs.cpu().numpy())
                    all_test_scaled.append(targets.cpu().numpy())
            
            y_pred_scaled = np.concatenate(all_preds_scaled).flatten()
            y_test_scaled = np.concatenate(all_test_scaled).flatten()

            y_pred_final = 1.0 + y_pred_scaled
            y_test_final = 1.0 + y_test_scaled
            
            y_pred_clipped = y_pred_final.clip(1.0, 2.0)

            test_rmse = np.sqrt(mean_squared_error(y_test_final, y_pred_clipped))
            test_mae = mean_absolute_error(y_test_final, y_pred_clipped)
            test_r2 = r2_score(y_test_final, y_pred_clipped)

            print("\n--- Final PyTorch 'Two-Tower' NN Model Test Results (v7.2) ---")
            print(f"  Final Test RMSE: {test_rmse:.4f}")
            print(f"  Final Test MAE : {test_mae:.4f}")
            print(f"  Final Test R^2   : {test_r2:.4f}")

            print(f"\n--- 5. Saving Model ---")
            checkpoint = {
                "model_state_dict": best_model_state,
                "model_class": "TabularNet",
                "model_params": {
                    "n_cont_features": n_cont_features,
                    "n_cat_features": n_cat_features,
                    **final_model_params
                },
                "best_val_loss": best_val_loss,
                "optuna_best_params": best_params,
                "cont_features_list": cont_features_list,
                "cat_features_list": cat_features_list
            }
            torch.save(checkpoint, MODEL_OUTPUT_FILE)
            print(f"Model checkpoint saved to {MODEL_OUTPUT_FILE}")
            
            print("\n✅ Process complete. Preprocessors (v7.2) and Model (v7.2) are saved.")