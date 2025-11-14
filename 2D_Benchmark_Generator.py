# File: 2D_Benchmark_Generator_v7.py
# Purpose: Generate a 2D TSP benchmark dataset.
# Version 7:
# 1. (User Req) Correct execution: Fixes the V5 hang by implementing
#    the user's logic: "scale the cluster size based on the
#    size of the instance."
# 2. (User Req) `generate_clustered` now calculates a *minimum required*
#    radius based on `points_per_cluster` to ensure generation
#    is always possible.
# 3. (User Req) It selects max(min_radius_needed, config_radius),
#    respecting the config only when it's feasible.
# 4. (Internal) This fix makes the V6 "circuit breakers" (max_tries)
#    unnecessary. All of that logic has been REMOVED for a
#    cleaner, non-hanging V5-style implementation.
# 5. Retains hard-fail, correct JSON, and safe temp file names.

import os
import sys
from pathlib import Path
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import time
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from collections import defaultdict

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR / "Generalized_TSP_Analysis"
INSTANCES_DIR = ROOT_DIR / "instances"
SOLUTIONS_DIR = ROOT_DIR / "solutions"
VISUALS_DIR = ROOT_DIR / "visualizations"
SOLVER_SCRATCH_DIR = SCRIPT_DIR / "temp_scratch"

ROOT_DIR.mkdir(exist_ok=True)
INSTANCES_DIR.mkdir(exist_ok=True)
SOLUTIONS_DIR.mkdir(exist_ok=True)
VISUALS_DIR.mkdir(exist_ok=True)
SOLVER_SCRATCH_DIR.mkdir(exist_ok=True)

LKH_EXECUTABLE_PATH = "C:\\LKH\\LKH-3.exe"

# --- Instance Generation Parameters ---
SAMPLES_PER_CONFIG = 5
n_points_list = [5, 8] + list(range(10, 101, 10)) + list(range(200, 1001, 100))
GRID_SIZE_LIST = [1000, 10000]

dist_types = [
    'random', 'normal', 'triangular', 'squeezed_uniform', 'uniform_triangular',
    'triangular_squeezed', 'boundary', 'x_central', 'truncated_exponential',
    'grid', 'correlated'
]

BASE_CONFIGS = [{'n_points': n, 'dist_type': dist} for n in n_points_list for dist in dist_types]
BASE_CONFIGS += [
    {'n_points': 100, 'dist_type': 'clustered', 'clust_n': cn, 'clust_rad': cr}
    for cn, cr in [(5, 0.05), (10, 0.05), (10, 0.10)]
] + [
    {'n_points': 500, 'dist_type': 'clustered', 'clust_n': cn, 'clust_rad': cr}
    for cn, cr in [(10, 0.05), (20, 0.10)]
] + [
    {'n_points': 1000, 'dist_type': 'clustered', 'clust_n': cn, 'clust_rad': cr}
    for cn, cr in [(20, 0.10)]
]

# --- Helper Functions for 2D Instance Generation (V7 - Scaled Cluster) ---

def _add_unique_point(coords_list, unique_set, x, y):
    """Helper to cast, check, and add a point."""
    p = (int(x), int(y))
    if p not in unique_set:
        unique_set.add(p)
        coords_list.append(p)
        return True
    return False

def generate_random(n_points, grid_size, rng):
    coords_list = []
    unique_set = set()
    while len(coords_list) < n_points:
        x, y = rng.random(2) * grid_size
        _add_unique_point(coords_list, unique_set, x, y)
    return np.array(coords_list)

def generate_normal(n_points, grid_size, rng):
    coords_list = []
    unique_set = set()
    while len(coords_list) < n_points:
        x, y = rng.normal(loc=grid_size/2, scale=grid_size/6, size=2)
        _add_unique_point(coords_list, unique_set, np.clip(x, 0, grid_size), np.clip(y, 0, grid_size))
    return np.array(coords_list)

def generate_triangular(n_points, grid_size, rng):
    coords_list = []
    unique_set = set()
    while len(coords_list) < n_points:
        uh, uv = rng.random(2)
        if uh < 0.5: x = grid_size * np.sqrt(uh * 2) / 2
        else: x = grid_size * (1 - np.sqrt((1 - uh) * 2) / 2)
        if uv < 0.5: y = grid_size * np.sqrt(uv * 2) / 2
        else: y = grid_size * (1 - np.sqrt((1 - uv) * 2) / 2)
        _add_unique_point(coords_list, unique_set, x, y)
    return np.array(coords_list)

def generate_squeezed_uniform(n_points, grid_size, rng):
    coords_list = []
    unique_set = set()
    while len(coords_list) < n_points:
        uh, uv, accept = rng.random(3)
        if accept <= uh * uh:
            _add_unique_point(coords_list, unique_set, grid_size * uh, grid_size * uv)
    return np.array(coords_list)

def generate_uniform_triangular(n_points, grid_size, rng):
    coords_list = []
    unique_set = set()
    while len(coords_list) < n_points:
        x_val = grid_size * rng.random()
        uv = rng.random()
        if uv < 0.5: y_val = grid_size * np.sqrt(uv * 2) / 2
        else: y_val = grid_size * (1 - np.sqrt((1 - uv) * 2) / 2)
        _add_unique_point(coords_list, unique_set, x_val, y_val)
    return np.array(coords_list)

def generate_triangular_squeezed(n_points, grid_size, rng):
    coords_list = []
    unique_set = set()
    while len(coords_list) < n_points:
        uh = rng.random()
        if uh < 0.5: x = grid_size * np.sqrt(uh * 2) / 2
        else: x = grid_size * (1 - np.sqrt((1 - uh) * 2) / 2)
        
        y = None
        while True:
            uv, accept = rng.random(2)
            if accept <= uv * uv:
                y = grid_size * uv
                break
        
        _add_unique_point(coords_list, unique_set, x, y)
    return np.array(coords_list)

def generate_boundary(n_points, grid_size, rng):
    coords_list = []
    unique_set = set()
    while len(coords_list) < n_points:
        p, x, y = rng.random(3)
        x *= grid_size
        y *= grid_size
        check = (abs(x - grid_size/2) / (grid_size/2)) * (abs(y - grid_size/2) / (grid_size/2))
        if p <= check:
            _add_unique_point(coords_list, unique_set, x, y)
    return np.array(coords_list)

def generate_x_central(n_points, grid_size, rng):
    coords_list = []
    unique_set = set()
    while len(coords_list) < n_points:
        p, x, y = rng.random(3)
        x *= grid_size
        y *= grid_size
        check = (1 - abs(x - grid_size/2) / (grid_size/2)) * (abs(y - grid_size/2) / (grid_size/2))
        if p <= check:
            _add_unique_point(coords_list, unique_set, x, y)
    return np.array(coords_list)

def generate_truncated_exponential(n_points, grid_size, rng):
    coords_list = []
    unique_set = set()
    while len(coords_list) < n_points:
        u1, u2 = rng.random(2)
        x = (-np.log(u1) / 1.0)
        y = (-np.log(u2) / 1.0)
        x = (x - math.floor(x)) * grid_size
        y = (y - math.floor(y)) * grid_size
        _add_unique_point(coords_list, unique_set, x, y)
    return np.array(coords_list)

def generate_clustered(n_points, grid_size, rng, clust_n=None, clust_rad=None):
    if clust_n is None or clust_rad is None:
        raise ValueError("Clustered distribution requires clust_n and clust_rad parameters.")
    if n_points < clust_n:
        tqdm.write(f"Warning: n_points {n_points} < clust_n {clust_n}. Using n_points as cluster count.")
        clust_n = n_points
    
    cluster_centers = grid_size * rng.random((clust_n, 2))
    points_per_cluster = [n_points // clust_n] * clust_n
    for i in range(n_points % clust_n):
        points_per_cluster[i] += 1
        
    coords_list = []
    unique_set = set()
    
    for i in range(clust_n):
        num_points_in_this_cluster = 0
        
        # --- (User Req) V7 FIX ---
        # Calculate the radius defined by the config
        config_radius = clust_rad * grid_size
        
        # Calculate the *minimum* radius needed to guarantee
        # enough space for unique integer points.
        # Area = pi * r^2, we need Area > points_per_cluster
        # r = sqrt(Area / pi). We add a 20% buffer (1.2*)
        # to ensure generation is fast and not impossibly dense.
        min_radius_needed = math.sqrt(points_per_cluster[i] * 1.2 / math.pi)
        
        # Use the larger of the two radii
        radius = max(config_radius, min_radius_needed)
        # --- END V7 FIX ---
        
        while num_points_in_this_cluster < points_per_cluster[i]:
            center_x, center_y = cluster_centers[i]
            
            angle = 2 * math.pi * rng.random()
            dist = radius * rng.random() # Use the new, safe radius
            
            x = np.clip(center_x + dist * math.cos(angle), 0, grid_size)
            y = np.clip(center_y + dist * math.sin(angle), 0, grid_size)
            
            if _add_unique_point(coords_list, unique_set, x, y):
                num_points_in_this_cluster += 1
                
    # Fill any potential rounding or inter-cluster duplicates
    while len(coords_list) < n_points:
        x, y = rng.random(2) * grid_size
        _add_unique_point(coords_list, unique_set, x, y)
        
    return np.array(coords_list)

def generate_grid(n_points, grid_size, rng):
    coords_list = []
    unique_set = set()
    side_length = math.ceil(math.sqrt(n_points))
    
    points_added = 0
    for i in range(side_length):
        for j in range(side_length):
            if points_added < n_points:
                base_x = (i + 0.5) * (grid_size / side_length)
                base_y = (j + 0.5) * (grid_size / side_length)
                
                while True: # Keep regenerating jitter until unique
                    jitter_x, jitter_y = (rng.random(2) - 0.5) * (grid_size / side_length * 0.1)
                    x = base_x + jitter_x
                    y = base_y + jitter_y
                    if _add_unique_point(coords_list, unique_set, x, y):
                        points_added += 1
                        break
                        
    return np.array(coords_list)

def generate_correlated(n_points, grid_size, rng):
    coords_list = []
    unique_set = set()
    while len(coords_list) < n_points:
        x = rng.uniform(0, grid_size)
        y = x + rng.normal(loc=0, scale=grid_size/10)
        _add_unique_point(coords_list, unique_set, np.clip(x, 0, grid_size), np.clip(y, 0, grid_size))
    return np.array(coords_list)


DIST_MAP = {
    'random': generate_random, 'normal': generate_normal, 'triangular': generate_triangular,
    'squeezed_uniform': generate_squeezed_uniform, 'uniform_triangular': generate_uniform_triangular,
    'triangular_squeezed': generate_triangular_squeezed, 'boundary': generate_boundary,
    'x_central': generate_x_central, 'truncated_exponential': generate_truncated_exponential,
    'clustered': generate_clustered, 'grid': generate_grid, 'correlated': generate_correlated
}

# --- Solver and File Handling Functions ---
# [These are identical to V5 - correct, hard-failing, and clean]

def _encode_id_for_solver(seq_j):
    id_str = str(seq_j)
    if '0' not in id_str:
        return id_str
    encoded_name = []
    replacement_char_code = ord('A')
    for char in id_str:
        if char == '0':
            encoded_name.append(chr(replacement_char_code))
            replacement_char_code += 1
        else:
            encoded_name.append(char)
    return "".join(encoded_name)

def _save_lkh_par(par_path, tsp_path, tour_path, time_limit_s=None):
    with open(par_path, "w") as f:
        f.write(f"PROBLEM_FILE = {tsp_path}\nTOUR_FILE = {tour_path}\n")
        f.write("MTSP_MIN_SIZE = 0\n") 
        if time_limit_s:
            f.write(f"TIME_LIMIT = {time_limit_s}\n")
        f.write("RUNS = 1\n")
        f.write("MAX_TRIALS = 1000\n")

def _compute_distance(p1, p2):
    return int(math.sqrt(np.sum((p1 - p2)**2)) + 0.5)

def _save_as_tsplib(file_path, coords, tsp_name):
    n = len(coords)
    with open(file_path, "w") as f:
        f.write(f"NAME : {tsp_name}\nTYPE : TSP\nCOMMENT : 2D Benchmark Instance\nDIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
        for i in range(n):
            row_distances = [_compute_distance(coords[i], coords[j]) for j in range(n)]
            f.write(" ".join(map(str, row_distances)) + "\n")
        f.write("EOF\n")

def _run_concorde(solver_name, coords):
    """Runs Concorde, hard-fails, and cleans up.
    (V8 Fix: Removed 'timeout=300' to allow Concorde
    to run indefinitely on complex instances.)"""
    scratch_tsp_path = str(SOLVER_SCRATCH_DIR / f"{solver_name}.tsp")
    wsl_tour_file = f"/tmp/{solver_name}.concorde.tour"
    
    try:
        _save_as_tsplib(scratch_tsp_path, coords, solver_name)
        
        clean_scratch_path = scratch_tsp_path.replace('\\', '/')
        wsl_scratch_path = subprocess.run(["wsl", "wslpath", "-a", clean_scratch_path], capture_output=True, text=True, check=True).stdout.strip()
        
        concorde_cmd = ["wsl", "concorde", "-o", wsl_tour_file, wsl_scratch_path]
        
        start_time = time.perf_counter()
        
        # --- THIS IS THE CHANGED LINE ---
        # Removed timeout=300
        subprocess.run(concorde_cmd, capture_output=True, text=True, check=True)
        # --- END OF CHANGE ---
        
        runtime = time.perf_counter() - start_time
        
        tour_content = subprocess.run(["wsl", "cat", wsl_tour_file], capture_output=True, text=True, check=True).stdout.strip()

        lines = tour_content.splitlines()
        if not lines or len(lines) <= 1:
            raise ValueError(f"Concorde ran but tour file was empty for {solver_name}.")
            
        tour_nodes = [int(n) + 1 for n in " ".join(lines[1:]).strip().split()]

        if not tour_nodes:
            raise ValueError(f"Concorde ran but failed to parse a valid tour for {solver_name}.")

        tour_length = sum(_compute_distance(coords[tour_nodes[i]-1], coords[tour_nodes[(i + 1) % len(tour_nodes)]-1]) for i in range(len(tour_nodes)))
        
        subprocess.run(["wsl", "rm", "-f", wsl_tour_file], capture_output=True)
        
        return tour_length, runtime, tour_nodes
        
    finally:
        if os.path.exists(scratch_tsp_path):
            os.remove(scratch_tsp_path)

def _run_lkh(instance_name, solver_name, coords, time_limit_s=None):
    """Runs LKH, hard-fails, and cleans up."""
    tsp_path = str(SOLVER_SCRATCH_DIR / f"{solver_name}.tsp")
    par_path = str(SOLVER_SCRATCH_DIR / f"{solver_name}.par")
    tour_path = str(SOLVER_SCRATCH_DIR / f"{solver_name}.tour")
    
    try:
        _save_as_tsplib(tsp_path, coords, instance_name)
        _save_lkh_par(par_path, tsp_path, tour_path, time_limit_s)
        
        lkh_cmd = [LKH_EXECUTABLE_PATH, par_path]
        start_time = time.perf_counter()
        
        subprocess.run(lkh_cmd, capture_output=True, text=True, check=True, timeout=300)
        lkh_time = time.perf_counter() - start_time
        
        with open(tour_path, 'r') as f:
            tour_content = f.read()
            
        tour_match = re.search(r"TOUR_SECTION\s*([\s\d-]*?)\s*EOF", tour_content, re.DOTALL)
        if not tour_match:
            raise ValueError(f"LKH ran but could not parse tour from output for {instance_name}")
        
        tour_nodes = [int(n) for n in tour_match.group(1).strip().split() if int(n) != -1]
        
        if not tour_nodes:
            raise ValueError(f"LKH ran but tour was empty for {instance_name}")
            
        tour_length = sum(_compute_distance(coords[tour_nodes[i]-1], coords[tour_nodes[(i + 1) % len(tour_nodes)]-1]) for i in range(len(tour_nodes)))
        
        return tour_length, lkh_time, tour_nodes
    finally:
        for f_path in [tsp_path, par_path, tour_path]:
            if os.path.exists(f_path):
                os.remove(f_path)

# --- Instance Generation and Solving Functions ---

def generate_and_save_instance(params):
    """
    Saves a guaranteed-valid instance in the correct JSON format.
    """
    config, grid_size, sample_idx, seq_j, base_seed = params
    n_customers = config['n_points']
    dist_type = config['dist_type']
    
    if dist_type == 'clustered':
        instance_name = f"TSP-{dist_type}-n{n_customers}-g{grid_size}-c{config['clust_n']}-r{int(config['clust_rad'] * 100)}-{sample_idx}"
    else:
        instance_name = f"TSP-{dist_type}-n{n_customers}-g{grid_size}-{sample_idx}"
        
    instance_path = INSTANCES_DIR / f"{instance_name}.json"
    
    if instance_path.exists():
        return
    
    generation_seed = base_seed
    rng = np.random.default_rng(generation_seed)
        
    # Generate coordinates - this is now guaranteed to be valid
    if dist_type == 'clustered':
        final_coords = DIST_MAP[dist_type](
            n_customers, grid_size, rng=rng,
            clust_n=config['clust_n'], clust_rad=config['clust_rad']
        )
    else:
        final_coords = DIST_MAP[dist_type](n_customers, grid_size, rng=rng)

    instance_data = {
        "instance_name": instance_name,
        "n_customers": n_customers,
        "dimension": 2,
        "grid_size": grid_size,
        "distribution_type": dist_type,
        "generation_seed": generation_seed,
        "coordinates": final_coords.tolist()
    }
    if dist_type == 'clustered':
        instance_data['cluster_n'] = config['clust_n']
        instance_data['cluster_rad'] = config['clust_rad']

    with open(instance_path, 'w') as f:
        json.dump(instance_data, f, indent=4)
    
    gc.collect()

def solve_single_instance(params):
    """
    Solves a single instance. Will CRASH if a solver fails.
    """
    config, grid_size, sample_idx, seq_j, base_seed = params
    n_customers = config['n_points']
    dist_type = config['dist_type']
    
    if dist_type == 'clustered':
        instance_name = f"TSP-{dist_type}-n{n_customers}-g{grid_size}-c{config['clust_n']}-r{int(config['clust_rad'] * 100)}-{sample_idx}"
    else:
        instance_name = f"TSP-{dist_type}-n{n_customers}-g{grid_size}-{sample_idx}"
        
    instance_path = INSTANCES_DIR / f"{instance_name}.json"
    solution_path = SOLUTIONS_DIR / f"{instance_name}.sol.json"
    
    if solution_path.exists():
        return
    
    if not instance_path.exists():
        tqdm.write(f"Instance file missing for {instance_name}, skipping solve.")
        return

    with open(instance_path, 'r') as f:
        inst_data = json.load(f)

    coords = np.array(inst_data['coordinates'])
    
    solver_name = _encode_id_for_solver(seq_j)
    lkh_time_limit = min(max(n_customers / 10, 10), 120) if n_customers > 100 else None
        
    lkh_length, lkh_time, lkh_tour = _run_lkh(
        instance_name, solver_name, coords, time_limit_s=lkh_time_limit
    )

    if n_customers <= 5000:
        concorde_length, concorde_time, concorde_tour = _run_concorde(solver_name, coords)

        if concorde_length < lkh_length:
            optimal_cost = concorde_length
            optimal_tour = concorde_tour
            optimal_solver = "concorde"
        else:
            optimal_cost = lkh_length
            optimal_tour = lkh_tour
            optimal_solver = "lkh"
        
        lkh_gap_pct = (lkh_length - concorde_length) / concorde_length * 100.0 if concorde_length > 0 else 0.0

    else:
        tqdm.write(f"Skipping Concorde for {instance_name} (n={n_customers})")
        optimal_cost = lkh_length
        optimal_tour = lkh_tour
        optimal_solver = "lkh_only"
        
        concorde_length, concorde_time, concorde_tour = None, None, None
        lkh_gap_pct = None

    solution_data = {
        "instance_name": instance_name,
        "n_customers": n_customers,
        "dimension": 2,
        "grid_size": grid_size,
        "optimal_cost": optimal_cost,
        "optimal_tour": optimal_tour,
        "optimal_solver": optimal_solver,
        "instance_file_path": str(instance_path),
        "concorde_length": concorde_length,
        "concorde_time_s": concorde_time,
        "concorde_tour": concorde_tour,
        "lkh_length": lkh_length,
        "lkh_time_s": lkh_time,
        "lkh_tour": lkh_tour,
        "lkh_gap_pct": lkh_gap_pct,
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution_data, f, indent=4)

    gc.collect()


# --- Visualization Function ---
def visualize_solutions():
    """Reads all solution files, loads coords, and plots tours."""
    print(f"\nGenerating visualizations in {VISUALS_DIR.resolve()}...")
    solution_files = list(SOLUTIONS_DIR.glob('*.sol.json'))
    
    if not solution_files:
        print("No .sol.json files found to visualize.")
        return
        
    for sol_file in tqdm(solution_files, desc="Plotting Solutions"):
        with open(sol_file, 'r') as f:
            sol_data = json.load(f)
        
        instance_name = sol_data['instance_name']
        vis_path = VISUALS_DIR / f"{instance_name}.png"
        
        if vis_path.exists():
            continue

        instance_path = Path(sol_data['instance_file_path'])
        if not instance_path.exists():
            tqdm.write(f"Missing instance file {instance_path}, skipping viz.")
            continue
            
        with open(instance_path, 'r') as f:
            inst_data = json.load(f)
        
        coords = np.array(inst_data['coordinates'])
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.scatter(coords[0, 0], coords[0, 1], c='red', marker='s', s=80, label='Depot (Node 1)')
        if len(coords) > 1:
            ax.scatter(coords[1:, 0], coords[1:, 1], c='blue', label='Customers', s=20, alpha=0.7)
        
        if sol_data.get('lkh_tour'):
            tour = sol_data['lkh_tour']
            tour_coords = np.array([coords[i-1] for i in tour] + [coords[tour[0]-1]])
            ax.plot(tour_coords[:, 0], tour_coords[:, 1], c='red', ls='--', lw=0.8, label=f"LKH Tour (Cost: {sol_data['lkh_length']})")

        if sol_data.get('concorde_tour'):
            tour = sol_data['concorde_tour']
            tour_coords = np.array([coords[i-1] for i in tour] + [coords[tour[0]-1]])
            ax.plot(tour_coords[:, 0], tour_coords[:, 1], c='blue', ls='-', lw=0.8, label=f"Concorde Tour (Cost: {sol_data['concorde_length']})")

        ax.set_title(f"Instance: {instance_name}", fontsize=10)
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.legend(fontsize=8)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150)
        plt.close(fig)
        
        gc.collect()

# --- Main Generation and Benchmarking Loop ---

def main():
    print("--- 2D TSP Benchmark Generator (V7 - Scaled Cluster) ---")
    print(f"Root directory: {ROOT_DIR.resolve()}")
    if not os.path.exists(LKH_EXECUTABLE_PATH):
        print(f"FATAL: LKH_EXECUTABLE_PATH not found at '{LKH_EXECUTABLE_PATH}'")
        return

    all_params = []
    print("Generating configurations...")
    seq_j = 1
    for grid_size in GRID_SIZE_LIST:
        for config in BASE_CONFIGS:
            for i in range(1, SAMPLES_PER_CONFIG + 1):
                n = config['n_points']
                dist_type = config['dist_type']
                config_num = sum(ord(c) for c in dist_type)
                base_seed = config_num + seq_j * 1000 + n * 100 + grid_size + i
                
                all_params.append((config, grid_size, i, seq_j, base_seed))
                seq_j += 1

    num_workers = os.cpu_count() - 2 if os.cpu_count() > 2 else 1
    
    # 2. Instance Generation Pass
    print(f"\nPreparing to generate {len(all_params)} total instances using {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(generate_and_save_instance, params): params for params in all_params}
        for future in tqdm(as_completed(futures), total=len(all_params), desc="Instance Generation"):
            future.result() # Will re-raise exception from worker

    print("\n--- Instance Generation Complete ---")

    # 3. Solution Generation Pass
    print(f"\nPreparing to solve {len(all_params)} instances using {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(solve_single_instance, params): params for params in all_params}
        for future in tqdm(as_completed(futures), total=len(all_params), desc="Solution Generation"):
            future.result() # Will re-raise exception from worker

    print("\n--- Solution Generation Complete ---")
    
    # 4. Visualization Pass
    visualize_solutions()
    print("\n--- Visualization Complete ---")

if __name__ == "__main__":
    main()