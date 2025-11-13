import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, 'tsp_features.csv')

MODEL_DIR = os.path.join(ROOT_DIR, 'lgbm_model')
MODEL_OUTPUT_FILE = os.path.join(MODEL_DIR, 'lgbm_alpha_model.joblib')

RANDOM_STATE = 42
OPTUNA_N_TRIALS = 50 # Number of tuning trials
EARLY_STOPPING_ROUNDS = 100 # Stop if val_score doesn't improve for 100 rounds

def load_and_preprocess(data_path):
    """
    Loads data, creates alpha target, and splits into train/val/test.
    No scaling or imputation is needed for LightGBM.
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)

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
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)

    # --- Identify Categorical Features ---
    # LightGBM can get a performance boost by knowing which features are
    # categorical (like 'dimension') vs. continuous (like 'pairwise_mean')
    categorical_features = ['dimension', 'grid_size']
    # Convert them to 'category' dtype for LightGBM
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype('category')
            
    # --- Split Data based on pre-defined 'split' column ---
    train_mask = (df['split'] == 'train')
    val_mask = (df['split'] == 'val')
    test_mask = (df['split'] == 'test')
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Combine train and val for final model training
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    return X_train, y_train, X_val, y_val, X_test, y_test, X_train_full, y_train_full, categorical_features


def optuna_objective(trial, X_train, y_train, X_val, y_val, categorical_features):
    """
    The objective function for Optuna to minimize (RMSE).
    """
    # Define a "lightweight" search space focused on simple, fast models
    params = {
        'objective': 'regression_l2', # Standard RMSE
        'metric': 'rmse',
        'n_estimators': 2000, # We set a high number and let early_stopping find the best
        'boosting_type': 'gbdt',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1, # Suppress logging
        
        # --- Parameters to Tune ---
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8), # Simple trees
        'num_leaves': trial.suggest_int('num_leaves', 10, 30), # num_leaves < 2^max_depth
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
    }

    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        categorical_feature=categorical_features
    )
    
    # Get predictions on the *best* iteration
    y_pred = model.predict(X_val, num_iteration=model.best_iteration_)
    y_pred_clipped = y_pred.clip(1.0, 2.0)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_clipped))
    return rmse


if __name__ == '__main__':
    # Ensure the script is run from the main directory
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found.")
        print(f"Please run this script from the main project directory (c:/TSP_ND_ML_Project/).")
    else:
        # Create the output directory
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        print("--- 1. Loading and Preprocessing Data ---")
        (
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            X_train_full, y_train_full,
            categorical_features
        ) = load_and_preprocess(DATA_FILE)
        
        print(f"Loaded {len(X_train)} train, {len(X_val)} val, {len(X_test)} test instances.")
        print(f"Categorical features identified: {categorical_features}")

        print(f"\n--- 2. Running Optuna Hyperparameter Tuning ({OPTUNA_N_TRIALS} trials) ---")
        # We need to pass the datasets to the objective function
        objective_func = lambda trial: optuna_objective(
            trial, X_train, y_train, X_val, y_val, categorical_features
        )
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_func, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
        
        print(f"\nOptuna tuning complete. Best RMSE: {study.best_value:.6f}")
        print("Best parameters found:")
        print(study.best_params)

        print("\n--- 3. Training Final Model ---")
        
        # 1. Get the best hyperparameters from the study
        best_params = study.best_params
        
        # 2. Find the optimal n_estimators using these params
        print("Finding optimal number of trees using early stopping...")
        temp_model = lgb.LGBMRegressor(
            **best_params,
            n_estimators=5000, # High number to allow early stopping
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        temp_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            categorical_feature=categorical_features
        )
        
        n_best_trees = temp_model.best_iteration_
        print(f"Optimal number of trees found: {n_best_trees}")
        
        # 3. Train the *actual* final model on ALL training data
        print("Training final model on combined (train + val) dataset...")
        final_params = best_params.copy()
        final_params['n_estimators'] = n_best_trees
        
        final_model = lgb.LGBMRegressor(**final_params, random_state=RANDOM_STATE, n_jobs=-1)
        
        final_model.fit(
            X_train_full,
            y_train_full,
            categorical_feature=categorical_features
        )
        
        print("Final model trained.")
        
        print("\n--- 4. Evaluating Final Model on Test Set ---")
        y_pred = final_model.predict(X_test)
        y_pred_clipped = y_pred.clip(1.0, 2.0)

        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
        test_mae = mean_absolute_error(y_test, y_pred_clipped)
        test_r2 = r2_score(y_test, y_pred_clipped)

        print("\n--- Final LightGBM Model Test Results ---")
        print(f"  Final Test RMSE: {test_rmse:.4f}")
        print(f"  Final Test MAE : {test_mae:.4f}")
        print(f"  Final Test R^2   : {test_r2:.4f}")

        print(f"\n--- 5. Saving Model ---")
        print(f"Saving model to {MODEL_OUTPUT_FILE}...")
        joblib.dump(final_model, MODEL_OUTPUT_FILE)
        
        print("\n✅ Process complete.")