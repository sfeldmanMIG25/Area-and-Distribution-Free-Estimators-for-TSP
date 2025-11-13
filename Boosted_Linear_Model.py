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

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, 'tsp_features.csv')

# --- New Model Directory ---
MODEL_DIR = os.path.join(ROOT_DIR, 'boosted_linear_model')
MODEL_OUTPUT_FILE = os.path.join(MODEL_DIR, 'boosted_linear_alpha_model.joblib')
FEATURE_STATS_FILE = os.path.join(MODEL_DIR, 'selected_features_stats.csv')

RANDOM_STATE = 42
SIGNIFICANCE_ALPHA = 0.05 # p-value threshold for feature selection

def create_boosted_features(df):
    """
    Engineers new log and interaction features to boost the linear model.
    """
    print("Engineering new 'boosted' features (log transforms and interactions)...")
    
    # 1. Define features for log transformation (non-negative, skewed)
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
    
    # Filter list to only columns that actually exist
    log_features_exist = [col for col in log_features_list if col in df.columns]
    
    original_cols_to_drop = []
    
    for col in log_features_exist:
        # Use np.log1p (log(1+x)) to handle zero values gracefully
        # We must impute NaNs first before logging
        col_data = df[col].fillna(0) # Impute NaNs with 0 for log
        df[f'log_{col}'] = np.log1p(col_data)
        original_cols_to_drop.append(col)

    # 2. Define interaction features
    # We impute with 0, so if cluster info is NaN, the interaction is 0.
    col1_imputed = df['k_cost_ratio'].fillna(0)
    col2_imputed = df['k_silhouette_score'].fillna(0)
    df['int_cost_x_silhouette'] = col1_imputed * col2_imputed
    
    # We will keep k_cost_ratio and k_silhouette_score in the feature set,
    # as they are still handled by the main imputer and might be
    # independently predictive.
    
    # Return the modified DataFrame and the list of original columns to drop
    return df, original_cols_to_drop


def load_and_preprocess(data_path):
    """
    Loads data, engineers features, creates alpha target, and splits.
    """
    df = pd.read_csv(data_path)

    # --- 1. Engineer Boosted Features ---
    df, original_cols_to_drop = create_boosted_features(df)

    # --- 2. Create the Target Variable: alpha ---
    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = df['optimal_cost'] / mst_divisor
    df['alpha'] = df['alpha'].clip(1.0, 2.0)

    # --- 3. Define Features (X) and Target (y) ---
    y = df['alpha']
    
    # Base features to drop
    features_to_drop = [
        'instance_name', 'optimal_cost', 'optimal_solver', 'solve_time_s',
        'mst_total_length', 'alpha', 'split', 'distribution_type'
    ]
    
    # Add the original, now-transformed columns to the drop list
    features_to_drop.extend(original_cols_to_drop)
    
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- 4. Split Data based on pre-defined 'split' column ---
    train_mask = (df['split'] == 'train')
    val_mask = (df['split'] == 'val')
    test_mask = (df['split'] == 'test')

    X_train_full = X[train_mask | val_mask]
    y_train_full = y[train_mask | val_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]

    return X_train_full, y_train_full, X_test, y_test

def select_features(X_train, y_train):
    """
    Uses an F-test to select all statistically significant features
    and returns them as a DataFrame.
    """
    print(f"\nStarting feature selection with p-value < {SIGNIFICANCE_ALPHA}...")
    
    # We must run the F-test on imputed and scaled data
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    # Run the F-test, which returns F-scores and p-values
    # Note: f_regression can handle NaNs in p-values if a feature is constant
    f_scores, p_values = f_regression(X_train_scaled, y_train)
    
    feature_stats = pd.DataFrame({
        'feature': X_train.columns,
        'p_value': p_values,
        'f_score': f_scores
    })
    
    # Filter for significant features (and drop any NaNs from p-values)
    feature_stats = feature_stats.dropna(subset=['p_value'])
    significant_features_df = feature_stats[feature_stats['p_value'] < SIGNIFICANCE_ALPHA]
    significant_features_df = significant_features_df.sort_values(by='p_value', ascending=True)
    
    print(f"\nFound {len(significant_features_df)} significant features:")
    for _, row in significant_features_df.iterrows():
        print(f"  - {row['feature']} (p-value: {row['p_value']:.2e})")
        
    return significant_features_df

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Builds the final pipeline, benchmarks with cross-val, evaluates on test set,
    and saves the model.
    """
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    print("\nBenchmarking with 5-fold Cross-Validation on training data...")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    cv_scores_neg_mse = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'
    )
    
    cv_rmse_scores = np.sqrt(-cv_scores_neg_mse)
    cv_rmse_mean = cv_rmse_scores.mean()
    cv_rmse_std = cv_rmse_scores.std()
    
    print(f"  CV Avg. RMSE: {cv_rmse_mean:.4f} (+/- {cv_rmse_std:.4f})")

    print("Training final model on selected features...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model on held-out test set...")
    y_pred = pipeline.predict(X_test)
    y_pred_clipped = y_pred.clip(1.0, 2.0)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
    test_mae = mean_absolute_error(y_test, y_pred_clipped)
    test_r2 = r2_score(y_test, y_pred_clipped)
    
    print("\n--- Test Set Results ---")
    print(f"  Final Test RMSE: {test_rmse:.4f}")
    print(f"  Final Test MAE : {test_mae:.4f}")
    print(f"  Final Test R^2   : {test_r2:.4f}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\nSaving model pipeline to {MODEL_OUTPUT_FILE}...")
    joblib.dump(pipeline, MODEL_OUTPUT_FILE)

    return pipeline

if __name__ == '__main__':
    # --- THIS BLOCK IS THE FIX ---
    # Check for the DATA_FILE using its full path variable
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found.")
        print(f"Please ensure 'tsp_features.csv' is in the root directory: {ROOT_DIR}")
    else:
        print("--- 1. Loading and Preprocessing Data ---")
        X_train_full, y_train_full, X_test_full, y_test = load_and_preprocess(DATA_FILE)
        
        print(f"Loaded {len(X_train_full)} training instances and {len(X_test_full)} test instances.")

        print("\n--- 2. Performing Feature Selection ---")
        significant_features_df = select_features(X_train_full, y_train_full)
        
        selected_features_list = significant_features_df['feature'].tolist()

        # Save the feature stats to the new model directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"\nSaving feature stats to {FEATURE_STATS_FILE}...")
        significant_features_df.to_csv(FEATURE_STATS_FILE, index=False)
        
        # Filter data to only use selected features
        X_train = X_train_full[selected_features_list]
        X_test = X_test_full[selected_features_list]

        print("\n--- 3. Training and Benchmarking Model ---")
        final_model = train_and_evaluate(X_train, y_train_full, X_test, y_test)
        
        print("\n✅ Process complete.")