import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- CONFIGURATION ---
# The script lives inside 'lgbm_model'
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODEL_DIR)

DATA_FILE = os.path.join(ROOT_DIR, 'tsp_features.csv')
MODEL_FILE = os.path.join(MODEL_DIR, 'lgbm_alpha_model.joblib')
PLOT_DIR = os.path.join(MODEL_DIR, 'plots')


def load_data_and_predict(data_path, model_path):
    """
    Loads data, model, and runs prediction on the test set.
    """
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # --- 1. Create the Target Variable: alpha ---
    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = df['optimal_cost'] / mst_divisor
    df['alpha'] = df['alpha'].clip(1.0, 2.0)

    # --- 2. Define Features (X) and Target (y) ---
    y = df['alpha']
    
    features_to_drop = [
        'instance_name', 'optimal_cost', 'optimal_solver', 'solve_time_s',
        'mst_total_length', 'alpha', 'split', 'distribution_type'
    ]
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
    X_full = df.drop(columns=existing_cols_to_drop)

    # --- 3. Set Categorical Features ---
    # This MUST match the training script so the model handles them correctly
    categorical_features = ['dimension', 'grid_size']
    for col in categorical_features:
        if col in X_full.columns:
            X_full[col] = X_full[col].astype('category')

    # --- 4. Split Data to get Test Set ---
    test_mask = (df['split'] == 'test')
    X_test = X_full[test_mask]
    y_test = y[test_mask]
    
    print("Test data loaded. Generating predictions...")

    # --- 5. Run Prediction ---
    y_pred = model.predict(X_test)
    y_pred_clipped = y_pred.clip(1.0, 2.0)
    
    return y_test, y_pred_clipped

def plot_predicted_vs_actual(y_test, y_pred, save_path):
    """Plots a scatter plot of predicted vs. actual values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    plt.plot([1.0, 2.0], [1.0, 2.0], 'r--', label='Ideal (y=x)')
    plt.title('LightGBM Model: Predicted vs. Actual $\\alpha$')
    plt.xlabel('Actual $\\alpha$')
    plt.ylabel('Predicted $\\alpha$')
    plt.xlim(1.0, 2.0)
    plt.ylim(1.0, 2.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '1_lgbm_predicted_vs_actual.png'))
    plt.close()

def plot_residuals_vs_predicted(y_pred, residuals, save_path):
    """Plots a scatter plot of residuals vs. predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, s=10)
    plt.axhline(y=0, color='r', linestyle='--', label='y=0')
    plt.title('LightGBM Model: Residuals vs. Predicted $\\alpha$')
    plt.xlabel('Predicted $\\alpha$')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '2_lgbm_residuals_vs_predicted.png'))
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
    
    plt.title('LightGBM Model: Histogram of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '3_lgbm_residual_histogram.png'))
    plt.close()

def plot_qq_plot(residuals, save_path):
    """Plots a Q-Q plot to check for normality of residuals."""
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('LightGBM Model: Q-Q Plot of Residuals')
    plt.xlabel('Theoretical Quantiles (Normal)')
    plt.ylabel('Sample Quantiles (Residuals)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '4_lgbm_qq_plot.png'))
    plt.close()

if __name__ == '__main__':
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    print("--- 1. Loading Model and Evaluating Test Set ---")
    try:
        y_test, y_pred = load_data_and_predict(DATA_FILE, MODEL_FILE)
        
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
        print(f"\n❌ ERROR: Could not find a required file: {e}")
        print("Please ensure you have run the training script and all files are in the correct folder.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")