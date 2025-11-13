import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, 'tsp_features.csv')

# --- New Model Directory ---
MODEL_DIR = os.path.join(ROOT_DIR, 'piecewise_linear_model')
ROUTER_MODEL_FILE = os.path.join(MODEL_DIR, 'router_model.joblib')
EXPERT_BLOB_FILE = os.path.join(MODEL_DIR, 'expert_blob_model.joblib')
EXPERT_CLUSTER_FILE = os.path.join(MODEL_DIR, 'expert_cluster_model.joblib')
EXPERT_BLOB_FEATURES_FILE = os.path.join(MODEL_DIR, 'expert_blob_features.csv')
EXPERT_CLUSTER_FEATURES_FILE = os.path.join(MODEL_DIR, 'expert_cluster_features.csv')

RANDOM_STATE = 42
SIGNIFICANCE_ALPHA = 0.05 # p-value threshold for feature selection
CLUSTER_THRESHOLD = 0.4    # Silhouette score threshold to define a "cluster"

def create_boosted_features(df):
    """
    Engineers new log and interaction features to boost the linear model.
    """
    # This is a lighter-touch function for the script
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
    
    log_features_exist = [col for col in log_features_list if col in df.columns]
    original_cols_to_drop = []
    
    for col in log_features_exist:
        # Impute NaNs with 0 for log
        col_data = df[col].fillna(0)
        
        # --- FIX 1: Clip data at 0 to prevent log1p(x < -1) error ---
        col_data_clipped = col_data.clip(lower=0)
        
        df[f'log_{col}'] = np.log1p(col_data_clipped)
        original_cols_to_drop.append(col)

    col1_imputed = df['k_cost_ratio'].fillna(0)
    col2_imputed = df['k_silhouette_score'].fillna(0)
    df['int_cost_x_silhouette'] = col1_imputed * col2_imputed
    
    return df, original_cols_to_drop


def load_and_preprocess(data_path, cluster_threshold):
    """
    Loads data, engineers features, creates alpha target, creates
    the binary cluster feature, and splits the data.
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)

    # --- 1. Engineer Boosted Features ---
    # We do this before anything else
    df, original_cols_to_drop = create_boosted_features(df)

    # --- 2. Create the Target Variable: alpha ---
    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = df['optimal_cost'] / mst_divisor
    df['alpha'] = df['alpha'].clip(1.0, 2.0)
    
    # --- 3. Create the Binary "Router" Feature ---
    # We fillna(False) because a NaN silhouette score means no cluster was found.
    df['is_clustered'] = (df['k_silhouette_score'] > cluster_threshold).fillna(False)
    print(f"Created 'is_clustered' feature. Split: (True: {df['is_clustered'].sum()} / False: {len(df) - df['is_clustered'].sum()})")

    # --- 4. Define Features (X) and Target (y) ---
    y = df['alpha']
    
    features_to_drop = [
        'instance_name', 'optimal_cost', 'optimal_solver', 'solve_time_s',
        'mst_total_length', 'alpha', 'split', 'distribution_type'
    ]
    features_to_drop.extend(original_cols_to_drop)
    
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- 5. Split Data based on pre-defined 'split' column ---
    train_mask = (df['split'] == 'train')
    val_mask = (df['split'] == 'val')
    test_mask = (df['split'] == 'test')

    X_train_full = X[train_mask | val_mask]
    y_train_full = y[train_mask | val_mask]
    
    X_test_full = X[test_mask]
    y_test_full = y[test_mask]

    return X_train_full, y_train_full, X_test_full, y_test_full

def select_features(X_train, y_train, model_name):
    """
    Uses an F-test to select all statistically significant features.
    """
    print(f"  Starting feature selection for '{model_name}' model...")
    
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    # This should now work, as X_train has no all-NaN columns
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    f_scores, p_values = f_regression(X_train_scaled, y_train)
    
    feature_stats = pd.DataFrame({
        'feature': X_train.columns,
        'p_value': p_values,
        'f_score': f_scores
    })
    
    feature_stats = feature_stats.dropna(subset=['p_value'])
    significant_features_df = feature_stats[feature_stats['p_value'] < SIGNIFICANCE_ALPHA]
    significant_features_df = significant_features_df.sort_values(by='p_value', ascending=True)
    
    print(f"  Found {len(significant_features_df)} significant features for '{model_name}'.")
    return significant_features_df

def train_single_expert(X_train, y_train, model_name):
    """
    Selects features and trains a single expert linear model pipeline.
    """
    
    # --- FIX 2: Drop all-NaN columns *before* feature selection ---
    # This stops the imputer from dropping columns, which fixes the length mismatch
    X_train_clean = X_train.dropna(axis=1, how='all')
    
    # 1. Select features for this expert
    features_df = select_features(X_train_clean, y_train, model_name)
    selected_features_list = features_df['feature'].tolist()
    
    # Filter data to only selected features
    X_train_selected = X_train_clean[selected_features_list]
    
    # 2. Define and train the pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    # 3. Benchmark the expert
    print(f"  Benchmarking '{model_name}' model...")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores_neg_mse = cross_val_score(
        pipeline, X_train_selected, y_train, cv=cv, scoring='neg_mean_squared_error'
    )
    cv_rmse_scores = np.sqrt(-cv_scores_neg_mse)
    print(f"  CV Avg. RMSE for '{model_name}': {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std():.4f})")
    
    # 4. Final training on all its data
    pipeline.fit(X_train_selected, y_train)
    
    return pipeline, features_df


if __name__ == '__main__':
    # Ensure the script is run from the main directory
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found.")
        print(f"Please run this script from the main project directory (c:/TSP_ND_ML_Project/).")
    else:
        # Create the output directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        print("--- 1. Loading and Preprocessing Data ---")
        X_train_full, y_train_full, X_test_full, y_test_full = load_and_preprocess(
            DATA_FILE, CLUSTER_THRESHOLD
        )
        
        print("\n--- 2. Training 'Router' Model ---")
        # The router is trained *only* on the non-continuous 'is_clustered' feature
        X_router_train = X_train_full[['is_clustered']]
        y_router_train = X_train_full['is_clustered']
        
        router_model = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
        router_model.fit(X_router_train, y_router_train)
        
        print("Router model trained. Saving...")
        joblib.dump(router_model, ROUTER_MODEL_FILE)

        print("\n--- 3. Training 'Expert' Models ---")
        
        # Split the training data for the two experts
        blob_mask = (X_train_full['is_clustered'] == False)
        cluster_mask = (X_train_full['is_clustered'] == True)
        
        # We must drop the 'is_clustered' feature from the expert's training data
        X_train_blobs = X_train_full[blob_mask].drop(columns=['is_clustered'])
        y_train_blobs = y_train_full[blob_mask]
        
        X_train_clusters = X_train_full[cluster_mask].drop(columns=['is_clustered'])
        y_train_clusters = y_train_full[cluster_mask]
        
        print(f"Training 'Blob' expert on {len(X_train_blobs)} instances...")
        expert_blob_model, expert_blob_features_df = train_single_expert(
            X_train_blobs, y_train_blobs, "Blob"
        )
        
        print(f"\nTraining 'Cluster' expert on {len(X_train_clusters)} instances...")
        expert_cluster_model, expert_cluster_features_df = train_single_expert(
            X_train_clusters, y_train_clusters, "Cluster"
        )
        
        print("\nSaving expert models and feature lists...")
        joblib.dump(expert_blob_model, EXPERT_BLOB_FILE)
        joblib.dump(expert_cluster_model, EXPERT_CLUSTER_FILE)
        expert_blob_features_df.to_csv(EXPERT_BLOB_FEATURES_FILE, index=False)
        expert_cluster_features_df.to_csv(EXPERT_CLUSTER_FEATURES_FILE, index=False)

        print("\n--- 4. Evaluating Full Piecewise Model on Test Set ---")
        
        # 1. Use Router to split the test set
        X_router_test = X_test_full[['is_clustered']]
        router_predictions = router_model.predict(X_router_test) # Boolean mask
        
        test_blob_mask = (router_predictions == False)
        test_cluster_mask = (router_predictions == True)
        
        # 2. Prepare test sets for each expert
        # We must drop 'is_clustered' and filter to the *exact* features
        # that each expert was trained on.
        
        blob_features = expert_blob_features_df['feature'].tolist()
        X_test_blobs = X_test_full[test_blob_mask].drop(columns=['is_clustered'])
        X_test_blobs_selected = X_test_blobs[blob_features]
        
        cluster_features = expert_cluster_features_df['feature'].tolist()
        X_test_clusters = X_test_full[test_cluster_mask].drop(columns=['is_clustered'])
        X_test_clusters_selected = X_test_clusters[cluster_features]

        # 3. Get predictions from each expert
        y_pred = np.zeros(len(y_test_full))
        
        if len(X_test_blobs_selected) > 0:
            y_pred[test_blob_mask] = expert_blob_model.predict(X_test_blobs_selected)
            
        if len(X_test_clusters_selected) > 0:
            y_pred[test_cluster_mask] = expert_cluster_model.predict(X_test_clusters_selected)
        
        # 4. Clip final predictions and report results
        y_pred_clipped = y_pred.clip(1.0, 2.0)
        
        test_rmse = np.sqrt(mean_squared_error(y_test_full, y_pred_clipped))
        test_mae = mean_absolute_error(y_test_full, y_pred_clipped)
        test_r2 = r2_score(y_test_full, y_pred_clipped)

        print("\n--- Final Piecewise Model Test Results ---")
        print(f"  Final Test RMSE: {test_rmse:.4f}")
        print(f"  Final Test MAE : {test_mae:.4f}")
        print(f"  Final Test R^2   : {test_r2:.4f}")
        
        print("\n✅ Process complete.")