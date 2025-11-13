import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# The script lives inside 'lgbm_model'
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODEL_DIR)

DATA_FILE = os.path.join(ROOT_DIR, 'tsp_features.csv')
MODEL_FILE = os.path.join(MODEL_DIR, 'lgbm_alpha_model.joblib')
PLOT_DIR = os.path.join(MODEL_DIR, 'plots')
IMPORTANCE_PLOT_FILE = os.path.join(PLOT_DIR, '5_lgbm_feature_importance.png')

# --- Number of top features to show in the plot ---
TOP_N_FEATURES = 30

def load_data_and_model(data_path, model_path):
    """
    Loads the model and the feature names from the test data.
    """
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    print(f"Loading data from {data_path} to get feature names...")
    df = pd.read_csv(data_path, nrows=0) # Read 0 rows, just get columns

    # --- Define Features (X) ---
    features_to_drop = [
        'instance_name', 'optimal_cost', 'optimal_solver', 'solve_time_s',
        'mst_total_length', 'alpha', 'split', 'distribution_type'
    ]
    
    # Get all column names from CSV
    all_cols = df.columns.tolist()
    
    # Filter out the ones we need to drop
    feature_names = [
        col for col in all_cols if col not in features_to_drop
    ]
    
    return model, feature_names

def plot_importance(model, feature_names, save_path):
    """
    Generates and saves the feature importance plot.
    """
    print(f"Generating feature importance plot for top {TOP_N_FEATURES} features...")
    
    # Set the feature names in the loaded model
    # (joblib-saved LGBM models sometimes forget them)
    model.booster_.feature_name_ = feature_names
    
    fig, ax = plt.subplots(figsize=(10, 12))
    lgb.plot_importance(
        model,
        ax=ax,
        max_num_features=TOP_N_FEATURES,
        importance_type='split', # 'split' = times used, 'gain' = total impact
        title=f'LightGBM Feature Importance (Top {TOP_N_FEATURES})'
    )
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Importance plot saved to {save_path}")

if __name__ == '__main__':
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    print("--- 1. Loading Model and Feature Names ---")
    try:
        # Note: We don't need the full 'boosted' feature creation logic
        # because the model was saved *after* it was trained on them.
        # We just need the list of column names from the *original* CSV
        # as LightGBM handles all feature transformations internally.
        
        # ... Wait, the LightGBM model *wasn't* trained on boosted features.
        # The boosted features were for the linear model.
        # Let's re-verify the lgbm training script.
        
        # Ah, correct. The LGBM script did NOT create boosted features.
        # It used the raw features (and handled categoricals).
        # This is much simpler.
        
        model, feature_names = load_data_and_model(DATA_FILE, MODEL_FILE)
        
        print(f"Model loaded. Found {len(feature_names)} features.")

        # --- 2. Generating Plot ---
        plot_importance(model, feature_names, IMPORTANCE_PLOT_FILE)
        
        print("\n✅ Process complete.")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find a required file: {e}")
        print("Please ensure you have run the training script and all files are in the correct folder.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")