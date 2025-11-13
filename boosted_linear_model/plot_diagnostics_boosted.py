import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- CONFIGURATION ---
# The script lives inside 'boosted_linear_model'
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODEL_DIR)

DATA_FILE = os.path.join(ROOT_DIR, 'tsp_features.csv')
MODEL_FILE = os.path.join(MODEL_DIR, 'boosted_linear_alpha_model.joblib')
FEATURE_STATS_FILE = os.path.join(MODEL_DIR, 'selected_features_stats.csv')
PLOT_DIR = os.path.join(MODEL_DIR, 'plots')

def create_boosted_features(df):
    """
    Engineers new log and interaction features to boost the linear model.
    This function MUST match the one in the training script.
    """
    # This is a lighter version for inference, only prints once.
    
    # 1. Define features for log transformation
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
        df[f'log_{col}'] = np.log1p(col_data)
        original_cols_to_drop.append(col)

    # 2. Define interaction features
    col1_imputed = df['k_cost_ratio'].fillna(0)
    col2_imputed = df['k_silhouette_score'].fillna(0)
    df['int_cost_x_silhouette'] = col1_imputed * col2_imputed
    
    return df, original_cols_to_drop


def load_data_and_model(data_path, model_path, feature_path):
    """
    Loads data, model, and filters test data to the selected features.
    """
    print(f"Loading model from {model_path}")
    pipeline = joblib.load(model_path)
    
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Loading selected features from {feature_path}")
    selected_features_df = pd.read_csv(feature_path)
    selected_features = selected_features_df['feature'].tolist()
    print(f"Found {len(selected_features)} features used by the model.")

    # --- CRITICAL: Re-engineer the boosted features ---
    print("Engineering 'boosted' features for the test set...")
    df, original_cols_to_drop = create_boosted_features(df)
    
    # --- Create the Target Variable: alpha ---
    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = df['optimal_cost'] / mst_divisor
    df['alpha'] = df['alpha'].clip(1.0, 2.0)

    # --- Define Features (X) and Target (y) ---
    y = df['alpha']
    
    features_to_drop = [
        'instance_name', 'optimal_cost', 'optimal_solver', 'solve_time_s',
        'mst_total_length', 'alpha', 'split', 'distribution_type'
    ]
    features_to_drop.extend(original_cols_to_drop)
    
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
    X_full = df.drop(columns=existing_cols_to_drop)

    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- Split Data based on pre-defined 'split' column ---
    test_mask = (df['split'] == 'test')
    X_test_full = X_full[test_mask]
    y_test = y[test_mask]

    # --- Filter test set to only the features read from the file ---
    X_test_selected = X_test_full[selected_features]
    
    return pipeline, X_test_selected, y_test

def plot_predicted_vs_actual(y_test, y_pred, save_path):
    """Plots a scatter plot of predicted vs. actual values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    plt.plot([1.0, 2.0], [1.0, 2.0], 'r--', label='Ideal (y=x)')
    plt.title('Boosted Model: Predicted vs. Actual $\\alpha$')
    plt.xlabel('Actual $\\alpha$')
    plt.ylabel('Predicted $\\alpha$')
    plt.xlim(1.0, 2.0)
    plt.ylim(1.0, 2.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '1_boosted_predicted_vs_actual.png'))
    plt.close()

def plot_residuals_vs_predicted(y_pred, residuals, save_path):
    """Plots a scatter plot of residuals vs. predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, s=10)
    plt.axhline(y=0, color='r', linestyle='--', label='y=0')
    plt.title('Boosted Model: Residuals vs. Predicted $\\alpha$')
    plt.xlabel('Predicted $\\alpha$')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '2_boosted_residuals_vs_predicted.png'))
    plt.close()

def plot_residual_histogram(residuals, save_path):
    """Plots a histogram of the residuals to check for normality."""
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, density=True, alpha=0.7, label='Residuals')
    
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Normal Fit ($\\mu$={mu:.3f}, $\\sigma$={std:.3f})')
    
    plt.title('Boosted Model: Histogram of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '3_boosted_residual_histogram.png'))
    plt.close()

def plot_qq_plot(residuals, save_path):
    """Plots a Q-Q plot to check for normality of residuals."""
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Boosted Model: Q-Q Plot of Residuals')
    plt.xlabel('Theoretical Quantiles (Normal)')
    plt.ylabel('Sample Quantiles (Residuals)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '4_boosted_qq_plot.png'))
    plt.close()

if __name__ == '__main__':
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    print("--- 1. Loading Model and Test Data ---")
    # We must provide all file paths
    model, X_test, y_test = load_data_and_model(
        DATA_FILE, MODEL_FILE, FEATURE_STATS_FILE
    )
    
    if model:
        print("\n--- 2. Generating Predictions ---")
        y_pred = model.predict(X_test)
        y_pred_clipped = y_pred.clip(1.0, 2.0)
        
        residuals = y_test - y_pred_clipped
        
        print(f"\n--- 3. Generating Plots (Saving to {PLOT_DIR}) ---")
        
        print("  Plotting 1/4: Predicted vs. Actual...")
        plot_predicted_vs_actual(y_test, y_pred_clipped, PLOT_DIR)
        
        print("  Plotting 2/4: Residuals vs. Predicted...")
        plot_residuals_vs_predicted(y_pred_clipped, residuals, PLOT_DIR)
        
        print("  Plotting 3/4: Residual Histogram...")
        plot_residual_histogram(residuals, PLOT_DIR)
        
        print("  Plotting 4/4: Q-Q Plot...")
        plot_qq_plot(residuals, PLOT_DIR)
        
        print("\n✅ All plots saved successfully.")
    else:
        print("\n❌ Script terminated due to loading errors.")