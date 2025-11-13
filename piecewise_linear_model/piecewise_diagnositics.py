import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- CONFIGURATION ---
# The script lives inside 'piecewise_linear_model'
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODEL_DIR)

DATA_FILE = os.path.join(ROOT_DIR, 'tsp_features.csv')
PLOT_DIR = os.path.join(MODEL_DIR, 'plots')

# Model files to load
ROUTER_MODEL_FILE = os.path.join(MODEL_DIR, 'router_model.joblib')
EXPERT_BLOB_FILE = os.path.join(MODEL_DIR, 'expert_blob_model.joblib')
EXPERT_CLUSTER_FILE = os.path.join(MODEL_DIR, 'expert_cluster_model.joblib')
EXPERT_BLOB_FEATURES_FILE = os.path.join(MODEL_DIR, 'expert_blob_features.csv')
EXPERT_CLUSTER_FEATURES_FILE = os.path.join(MODEL_DIR, 'expert_cluster_features.csv')

# This MUST match the setting from your training script
CLUSTER_THRESHOLD = 0.4

def create_boosted_features(df):
    """
    Engineers new log and interaction features to boost the linear model.
    This function MUST match the one in the training script.
    """
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
        col_data = df[col].fillna(0)
        col_data_clipped = col_data.clip(lower=0)
        df[f'log_{col}'] = np.log1p(col_data_clipped)
        original_cols_to_drop.append(col)

    col1_imputed = df['k_cost_ratio'].fillna(0)
    col2_imputed = df['k_silhouette_score'].fillna(0)
    df['int_cost_x_silhouette'] = col1_imputed * col2_imputed
    
    return df, original_cols_to_drop


def load_data_and_predict(data_path, cluster_threshold):
    """
    Loads all data, model artifacts, and runs the full piecewise
    prediction pipeline on the test set.
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)

    # --- 1. Engineer Boosted Features ---
    df, original_cols_to_drop = create_boosted_features(df)

    # --- 2. Create the Target Variable: alpha ---
    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = df['optimal_cost'] / mst_divisor
    df['alpha'] = df['alpha'].clip(1.0, 2.0)
    
    # --- 3. Create the Binary "Router" Feature ---
    df['is_clustered'] = (df['k_silhouette_score'] > cluster_threshold).fillna(False)

    # --- 4. Define Features (X) and Target (y) ---
    y = df['alpha']
    features_to_drop = [
        'instance_name', 'optimal_cost', 'optimal_solver', 'solve_time_s',
        'mst_total_length', 'alpha', 'split', 'distribution_type'
    ]
    features_to_drop.extend(original_cols_to_drop)
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
    X_full = df.drop(columns=existing_cols_to_drop)
    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- 5. Split Data to get Test Set ---
    test_mask = (df['split'] == 'test')
    X_test_full = X_full[test_mask]
    y_test_full = y[test_mask]
    
    print("Test data loaded. Loading models...")

    # --- 6. Load All Model Artifacts ---
    router_model = joblib.load(ROUTER_MODEL_FILE)
    expert_blob_model = joblib.load(EXPERT_BLOB_FILE)
    expert_cluster_model = joblib.load(EXPERT_CLUSTER_FILE)
    
    blob_features = pd.read_csv(EXPERT_BLOB_FEATURES_FILE)['feature'].tolist()
    cluster_features = pd.read_csv(EXPERT_CLUSTER_FEATURES_FILE)['feature'].tolist()
    
    print("All models and feature lists loaded.")

    # --- 7. Run the Piecewise Prediction ---
    print("Running router to split test set...")
    X_router_test = X_test_full[['is_clustered']]
    router_predictions = router_model.predict(X_router_test) # Boolean mask
    
    test_blob_mask = (router_predictions == False)
    test_cluster_mask = (router_predictions == True)
    
    # Prepare test sets for each expert
    X_test_blobs = X_test_full[test_blob_mask].drop(columns=['is_clustered'])
    X_test_blobs_selected = X_test_blobs[blob_features]
    
    X_test_clusters = X_test_full[test_cluster_mask].drop(columns=['is_clustered'])
    X_test_clusters_selected = X_test_clusters[cluster_features]

    # Get predictions from each expert
    y_pred = np.zeros(len(y_test_full))
    
    if len(X_test_blobs_selected) > 0:
        y_pred[test_blob_mask] = expert_blob_model.predict(X_test_blobs_selected)
        
    if len(X_test_clusters_selected) > 0:
        y_pred[test_cluster_mask] = expert_cluster_model.predict(X_test_clusters_selected)
    
    # Clip final predictions
    y_pred_clipped = y_pred.clip(1.0, 2.0)
    
    return y_test_full, y_pred_clipped

def plot_predicted_vs_actual(y_test, y_pred, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    plt.plot([1.0, 2.0], [1.0, 2.0], 'r--', label='Ideal (y=x)')
    plt.title('Piecewise Model: Predicted vs. Actual $\\alpha$')
    plt.xlabel('Actual $\\alpha$')
    plt.ylabel('Predicted $\\alpha$')
    plt.xlim(1.0, 2.0)
    plt.ylim(1.0, 2.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '1_piecewise_predicted_vs_actual.png'))
    plt.close()

def plot_residuals_vs_predicted(y_pred, residuals, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, s=10)
    plt.axhline(y=0, color='r', linestyle='--', label='y=0')
    plt.title('Piecewise Model: Residuals vs. Predicted $\\alpha$')
    plt.xlabel('Predicted $\\alpha$')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '2_piecewise_residuals_vs_predicted.png'))
    plt.close()

def plot_residual_histogram(residuals, save_path):
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, density=True, alpha=0.7, label='Residuals')
    
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Normal Fit ($\\mu$={mu:.3f}, $\\sigma$={std:.3f})')
    
    plt.title('Piecewise Model: Histogram of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '3_piecewise_residual_histogram.png'))
    plt.close()

def plot_qq_plot(residuals, save_path):
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Piecewise Model: Q-Q Plot of Residuals')
    plt.xlabel('Theoretical Quantiles (Normal)')
    plt.ylabel('Sample Quantiles (Residuals)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '4_piecewise_qq_plot.png'))
    plt.close()

if __name__ == '__main__':
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    print("--- 1. Loading Models and Evaluating Test Set ---")
    try:
        y_test, y_pred = load_data_and_predict(DATA_FILE, CLUSTER_THRESHOLD)
        
        residuals = y_test - y_pred
        
        print(f"\n--- 2. Generating Plots (Saving to {PLOT_DIR}) ---")
        
        print("  Plotting 1/4: Predicted vs. Actual...")
        plot_predicted_vs_actual(y_test, y_pred, PLOT_DIR)
        
        print("  Plotting 2/4: Residuals vs. Predicted...")
        plot_residuals_vs_predicted(y_pred, residuals, PLOT_DIR)
        
        print("  Plotting 3/4: Residual Histogram...")
        plot_residual_histogram(residuals, PLOT_DIR)
        
        print("  Plotting 4/4: Q-Q Plot...")
        plot_qq_plot(residuals, PLOT_DIR)
        
        print("\n✅ All plots saved successfully.")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find a required file: {e.fileName}")
        print("Please ensure you have run the training script first and all files are in the correct folder.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")