# File: run_benchmark.py
# Purpose: Run all trained alpha-estimators on the 2D benchmark dataset,
# save detailed results, and calculate summary statistics.

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time
from tqdm import tqdm
from glob import glob
import warnings
from sklearn.metrics import r2_score, mean_absolute_error

# --- Add script directory to path to find tsp_utils ---
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

# --- Import all required functions from tsp_utils ---
try:
    from tsp_utils import (
        parse_tsp_instance, parse_tsp_solution, get_mst_length,
        # V1 Feature Models
        load_gart_model, estimate_tsp_ml_alpha,
        load_model_and_features, estimate_tsp_from_model,
        load_lgbm_model, estimate_tsp_lgbm,
        load_piecewise_models, estimate_tsp_piecewise,
        load_pytorch_nn_v4, estimate_tsp_pytorch_nn_v4,
        load_pytorch_nn_v6, estimate_tsp_pytorch_nn_v6,
        # V2 Feature Model
        load_pytorch_nn_v7, estimate_tsp_pytorch_nn_v7,
        # Paths
        SCRIPT_DIR, GART_MODEL_PATH,
        LINEAR_MODEL_PATH, LINEAR_MODEL_FEATURES_PATH,
        BOOSTED_MODEL_PATH, BOOSTED_MODEL_FEATURES_PATH,
        PIECEWISE_ROUTER_MODEL_FILE, PIECEWISE_EXPERT_BLOB_FILE,
        PIECEWISE_EXPERT_BLOB_FEATURES_FILE, PIECEWISE_EXPERT_CLUSTER_FILE,
        PIECEWISE_EXPERT_CLUSTER_FEATURES_FILE,
        LGBM_MODEL_FILE, V1_FEATURES_FILE,
        NN_V4_MODEL_FILE, NN_V4_PREPROCESSOR_FILE,
        NN_V6_MODEL_FILE, NN_V6_CONT_PREPROCESSOR_FILE, NN_V6_CAT_PREPROCESSOR_FILE,
        NN_V7_MODEL_FILE, NN_V7_CONT_PREPROCESSOR_FILE, NN_V7_CAT_PREPROCESSOR_FILE
    )
except ImportError as e:
    print(f"FATAL: Could not import from tsp_utils.py. Ensure it is in the same directory.")
    print(f"Error: {e}")
    sys.exit(1)

# --- Configuration ---
ROOT_DIR = SCRIPT_DIR
RESULTS_DIR = ROOT_DIR / "Generalized_TSP_Analysis"
INSTANCES_DIR = RESULTS_DIR / "instances"
SOLUTIONS_DIR = RESULTS_DIR / "solutions"
BENCHMARK_RESULTS_FILE = RESULTS_DIR / "benchmark_results.csv"

# Suppress warnings from scikit-learn/KMeans
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_all_models():
    """
    Loads all models and their assets (preprocessors, feature lists)
    into a single dictionary. This is done once.
    """
    print("Loading all models...")
    assets = {}
    
    # 1. GART (V1-Simple)
    assets['gart'] = load_gart_model(GART_MODEL_PATH)
    
    # 2. Linear (V1-VIF)
    assets['lin_model'], assets['lin_features'] = load_model_and_features(
        LINEAR_MODEL_PATH, LINEAR_MODEL_FEATURES_PATH
    )
    
    # 3. Boosted (V1-Eng)
    assets['boost_model'], assets['boost_features'] = load_model_and_features(
        BOOSTED_MODEL_PATH, BOOSTED_MODEL_FEATURES_PATH
    )
    
    # 4. LGBM (V1-All)
    assets['lgbm_model'], assets['lgbm_features'] = load_lgbm_model(
        LGBM_MODEL_FILE, V1_FEATURES_FILE
    )
    
    # 5. Piecewise (V1-Eng)
    router, blob_m, blob_f, clust_m, clust_f = load_piecewise_models(
        PIECEWISE_ROUTER_MODEL_FILE, PIECEWISE_EXPERT_BLOB_FILE,
        PIECEWISE_EXPERT_BLOB_FEATURES_FILE, PIECEWISE_EXPERT_CLUSTER_FILE,
        PIECEWISE_EXPERT_CLUSTER_FEATURES_FILE
    )
    assets['piecewise'] = {
        "router_model": router,
        "blob_model": blob_m, "blob_features_set": blob_f,
        "cluster_model": clust_m, "cluster_features_set": clust_f
    }
    
    # 6. NN v4 (V1-All)
    model, prep, device = load_pytorch_nn_v4(NN_V4_MODEL_FILE, NN_V4_PREPROCESSOR_FILE)
    assets['nn_v4'] = {"model": model, "preprocessor": prep, "device": device}

    # 7. NN v6 (V1-All)
    model, cont_prep, cat_prep, device = load_pytorch_nn_v6(
        NN_V6_MODEL_FILE, NN_V6_CONT_PREPROCESSOR_FILE, NN_V6_CAT_PREPROCESSOR_FILE
    )
    assets['nn_v6'] = {
        "model": model, "cont_prep": cont_prep, 
        "cat_prep": cat_prep, "device": device
    }
    
    # 8. NN v7 (V2-Features) - (User Req: Commented out)
    # model, cont_prep, cat_prep, device = load_pytorch_nn_v7(
    #     NN_V7_MODEL_FILE, NN_V7_CONT_PREPROCESSOR_FILE, NN_V7_CAT_PREPROCESSOR_FILE
    # )
    # assets['nn_v7'] = {
    #     "model": model, "cont_prep": cont_prep, 
    #     "cat_prep": cat_prep, "device": device
    # }
    
    print(f"Successfully loaded {len(assets)} model asset groups.")
    return assets

def process_single_instance(task_data):
    """
    Worker function to run all models on a single instance.
    (V4: Added optimal solver time)
    """
    instance_path, solution_path, assets = task_data
    instance_results = []
    
    try:
        inst_data = parse_tsp_instance(instance_path)
        sol_data = parse_tsp_solution(solution_path)
        
        # --- Base Truth ---
        true_optimal_cost = sol_data['optimal_cost']
        coords = inst_data['coordinates']
        n = inst_data['n_customers']
        d = inst_data['dimension']
        grid_size = inst_data['grid_size']
        
        mst_length, mst_time = get_mst_length(coords)
        mst_length_safe = mst_length if mst_length > 1e-9 else 1e-9
        true_alpha = true_optimal_cost / mst_length_safe
        
        # --- (User Req) Extract Optimal Solver Time ---
        optimal_solver = sol_data.get('optimal_solver')
        optimal_time = 0.0
        if optimal_solver == 'concorde':
            optimal_time = sol_data.get('concorde_time_s')
        elif optimal_solver in ['lkh', 'lkh_only', 'lkh_only_timed']:
            optimal_time = sol_data.get('lkh_time_s')
        
        if optimal_time is None: # Handle null/None values
            optimal_time = 0.0
        # --- End ---
        
    except Exception as e:
        print(f"Error loading {instance_path.name}: {e}")
        return [] # Skip this file

    # --- Helper to run and format ---
    def run_and_format(model_name, estimator_func, *args, **kwargs):
        pred_cost, pred_time = estimator_func(*args, **kwargs)
        pred_alpha = pred_cost / mst_length_safe
        return {
            "instance": inst_data['instance_name'],
            "model": model_name,
            "n_customers": n,
            "grid_size": grid_size,
            "distribution": inst_data['distribution_type'],
            "true_cost": true_optimal_cost,
            "optimal_solver": optimal_solver,       # NEW
            "optimal_solve_time_s": optimal_time, # NEW
            "pred_cost": pred_cost,
            "mst_length": mst_length,
            "true_alpha": true_alpha,
            "pred_alpha": pred_alpha,
            "prediction_time_s": pred_time
        }

    # --- Run V1 Feature Models ---
    v1_features_set = assets['lgbm_features'] # "all V1 features" set
    
    instance_results.append(run_and_format(
        'GART_1.0', estimate_tsp_ml_alpha, coords, assets['gart']
    ))
    instance_results.append(run_and_format(
        'Linear', estimate_tsp_from_model, coords, n, d, grid_size, assets['lin_model'], assets['lin_features'], False
    ))
    instance_results.append(run_and_format(
        'Boosted_Linear', estimate_tsp_from_model, coords, n, d, grid_size, assets['boost_model'], assets['boost_features'], True
    ))
    instance_results.append(run_and_format(
        'LGBM', estimate_tsp_lgbm, coords, n, d, grid_size, assets['lgbm_model'], v1_features_set
    ))
    instance_results.append(run_and_format(
        'Piecewise_Linear', estimate_tsp_piecewise, coords, n, d, grid_size, **assets['piecewise']
    ))
    instance_results.append(run_and_format(
        'NN_v4_Varol', estimate_tsp_pytorch_nn_v4, 
        coords, n, d, grid_size, assets['nn_v4']['model'], assets['nn_v4']['preprocessor'], v1_features_set, assets['nn_v4']['device']
    ))
    instance_results.append(run_and_format(
        'NN_v6_TwoTower', estimate_tsp_pytorch_nn_v6,
        coords, n, d, grid_size, assets['nn_v6']['model'], assets['nn_v6']['cont_prep'], assets['nn_v6']['cat_prep'], v1_features_set, assets['nn_v6']['device']
    ))
    
    # --- Run V2 Feature Model ---
    # (User Req: Commented out)
    # instance_results.append(run_and_format(
    #     'NN_v7_TwoPass', estimate_tsp_pytorch_nn_v7,
    #     coords, n, d, grid_size, assets['nn_v7']['model'], assets['nn_v7']['cont_prep'], assets['nn_v7']['cat_prep'], assets['nn_v7']['device']
    # ))

    return instance_results

def calculate_metrics(df):
    """Calculates and prints a summary of performance metrics."""
    
    # --- (User Req) New Optimal Solver Time Summary ---
    print("\n--- Optimal Solver Summary ---")
    # We only need one row per instance, so we can filter for one model
    solver_stats_df = df[df['model'] == 'GART_1.0'].copy()
    
    if not solver_stats_df.empty:
        # Handle cases where solver time might be NaN/None
        solver_stats_df['optimal_solve_time_s'] = pd.to_numeric(solver_stats_df['optimal_solve_time_s'], errors='coerce').fillna(0)
        
        solver_times = solver_stats_df.groupby('optimal_solver')['optimal_solve_time_s'].agg(['mean', 'sum', 'count'])
        print(solver_times.to_string(float_format="{:,.4f}".format))
        
        total_time_s = solver_stats_df['optimal_solve_time_s'].sum()
        total_time_hr = total_time_s / 3600
        print(f"\nTotal Optimal Solve Time (all instances): {total_time_s:,.2f} s ({total_time_hr:,.2f} hrs)")
    else:
        print("No results found to calculate optimal solver stats.")
    # --- End New Summary ---

    print("\n--- Model Performance Summary ---")
    
    # Calculate error metrics
    df['percent_diff'] = ((df['pred_cost'] - df['true_cost']) / df['true_cost']) * 100
    df['abs_percent_diff'] = df['percent_diff'].abs()
    
    grouped = df.groupby('model')
    
    summary = pd.DataFrame()
    summary['R^2 (Cost)'] = grouped.apply(lambda x: r2_score(x['true_cost'], x['pred_cost']))
    summary['R^2 (Alpha)'] = grouped.apply(lambda x: r2_score(x['true_alpha'], x['pred_alpha']))
    summary['MAE (Cost)'] = grouped.apply(lambda x: mean_absolute_error(x['true_cost'], x['pred_cost']))
    summary['Avg. Pred. Time (s)'] = grouped['prediction_time_s'].mean()
    summary['Avg. % Diff'] = grouped['percent_diff'].mean()
    summary['Std. % Diff'] = grouped['percent_diff'].std()
    summary['Avg. Abs. % Diff (MAPE)'] = grouped['abs_percent_diff'].mean()
    
    # Format and print
    summary = summary.sort_values(by='Avg. Abs. % Diff (MAPE)', ascending=True)
    summary['Avg. Pred. Time (s)'] = summary['Avg. Pred. Time (s)'].map('{:,.6f}'.format)
    summary['R^2 (Cost)'] = summary['R^2 (Cost)'].map('{:,.4f}'.format)
    summary['R^2 (Alpha)'] = summary['R^2 (Alpha)'].map('{:,.4f}'.format)
    summary['MAE (Cost)'] = summary['MAE (Cost)'].map('{:,.2f}'.format)
    summary['Avg. % Diff'] = summary['Avg. % Diff'].map('{:,.2f}%'.format)
    summary['Std. % Diff'] = summary['Std. % Diff'].map('{:,.2f}%'.formar)
    summary['Avg. Abs. % Diff (MAPE)'] = summary['Avg. Abs. % Diff (MAPE)'].map('{:,.2f}%'.format)
    
    print(summary.to_string())

def main():
    print("--- 1. Loading All Models ---")
    try:
        loaded_assets = load_all_models()
    except FileNotFoundError as e:
        print(f"\nFATAL: A required model file was not found.")
        print(f"Error: {e}")
        print("Please ensure all models are trained and in their correct folders.")
        sys.exit(1)
        
    print("\n--- 2. Finding Benchmark Tasks ---")
    solution_files = glob(str(SOLUTIONS_DIR / "*.sol.json"))
    tasks = []
    for sol_path in solution_files:
        sol_path = Path(sol_path)
        inst_path = INSTANCES_DIR / sol_path.name.replace(".sol.json", ".json")
        if inst_path.exists():
            tasks.append((inst_path, sol_path, loaded_assets))
        else:
            print(f"Warning: No matching instance for {sol_path.name}, skipping.")
            
    if not tasks:
        print("No instance/solution pairs found. Exiting.")
        return
        
    print(f"Found {len(tasks)} instances to benchmark.")
    
    # --- (User Req) Removed ThreadPoolExecutor ---
    print("\n--- 3. Running Benchmark (Serially) ---")
    all_results = []
    
    # Run in a single-threaded, serial loop
    for task in tqdm(tasks, desc="Benchmarking Instances"):
        # This will re-raise any exception and stop the script
        result_list_for_instance = process_single_instance(task) 
        all_results.extend(result_list_for_instance)

    print("\nBenchmark run complete.")
    # --- END MODIFICATION ---
    
    if not all_results:
        print("No results were generated. This may be due to errors during processing.")
        return

    print(f"\n--- 4. Saving Results ---")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(BENCHMARK_RESULTS_FILE, index=False)
    print(f"Successfully saved {len(results_df)} results to {BENCHMARK_RESULTS_FILE}")
    
    print("\n--- 5. Calculating Final Metrics ---")
    calculate_metrics(results_df)
    
    print("\n✅ Benchmark complete.")

if __name__ == "__main__":
    main()