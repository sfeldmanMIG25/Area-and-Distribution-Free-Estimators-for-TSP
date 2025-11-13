import numpy as np
import os
import subprocess
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import re
from itertools import combinations_with_replacement
from tqdm import tqdm
import gc
import lkh
from collections import defaultdict

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCES_DIR = os.path.join(ROOT_DIR, "instances")
SOLUTIONS_DIR = os.path.join(ROOT_DIR, "solutions")

LKH_EXECUTABLE_PATH = "C:\\LKH\\LKH-3.exe"
SOLVER_SCRATCH_DIR = "C:\\Temp_TSP_Scratch"
os.makedirs(SOLVER_SCRATCH_DIR, exist_ok=True)
os.makedirs(INSTANCES_DIR, exist_ok=True)
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
# ---

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def _encode_id_for_solver(seq_j):
    """Encodes a sequential ID to be solver-safe by replacing zeros with sequential letters."""
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

# ====================================================================
# 1D COORDINATE GENERATION FUNCTIONS
# ====================================================================

def _generate_1d_random(n, seed, grid_size):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, grid_size, n)

def _generate_1d_normal(n, seed, grid_size):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(loc=grid_size/2, scale=grid_size/6, size=n), 0, grid_size)

def _generate_1d_clustered(n, seed, grid_size):
    rng = np.random.default_rng(seed)
    clust_n = max(2, int(np.sqrt(n) / 4))
    centers = rng.uniform(0, grid_size, clust_n)
    assignments = rng.integers(0, clust_n, size=n)
    stdev = grid_size * 0.05
    coords = rng.normal(loc=centers[assignments], scale=stdev)
    return np.clip(coords, 0, grid_size)

def _generate_1d_cornered(n, seed, grid_size):
    rng = np.random.default_rng(seed)
    corners = np.array([0, grid_size])
    assignments = rng.integers(0, 2, size=n)
    coords = rng.normal(loc=corners[assignments], scale=grid_size * 0.025)
    return np.clip(coords, 0, grid_size)

def _generate_1d_ring(n, seed, grid_size):
    rng = np.random.default_rng(seed)
    centers = np.array([grid_size * 0.25, grid_size * 0.75])
    assignments = rng.integers(0, 2, size=n)
    coords = rng.normal(loc=centers[assignments], scale=grid_size/20)
    return np.clip(coords, 0, grid_size)

def _generate_1d_powerlaw(n, seed, grid_size):
    rng = np.random.default_rng(seed)
    exponent = 2.5
    u = rng.random(n)
    r = (u ** (1.0 / (1.0 - exponent))) * (grid_size / 2)
    sign = rng.choice([-1, 1], n)
    coords = (grid_size / 2) + r * sign
    return np.clip(coords, 0, grid_size)

def _generate_1d_triangular(n, seed, grid_size):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    coords = grid_size * (np.sin(np.pi * u / 2) ** 2)
    return coords
    
def _generate_1d_grid(n, seed, grid_size):
    rng = np.random.default_rng(seed)
    num_grid_points = n * 2
    all_points = np.linspace(0, grid_size, num_grid_points)
    selected_points = rng.choice(all_points, size=n, replace=False)
    selected_points += rng.normal(scale= (grid_size/num_grid_points) / 4, size=n)
    return np.clip(selected_points, 0, grid_size)
    
def _generate_nd_correlated(n, d, seed, grid_size):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, grid_size, n)
    coords = np.zeros((n, d))
    coords[:, 0] = base
    for j in range(1, d):
        coords[:, j] = base + rng.normal(loc=0, scale=grid_size/10, size=n)
    return np.clip(coords, 0, grid_size).astype(int)

DISTRIBUTION_MAP_1D = {
    "rand": _generate_1d_random, "norm": _generate_1d_normal,
    "clus": _generate_1d_clustered, "corn": _generate_1d_cornered,
    "ring": _generate_1d_ring, "grid": _generate_1d_grid,
    "pow": _generate_1d_powerlaw, "tria": _generate_1d_triangular
}

# ====================================================================
# SOLVER AND FILE HANDLING FUNCTIONS
# ====================================================================
def _save_lkh_par(par_path, tsp_path, tour_path, time_limit_s=None):
    """Writes a simple LKH parameter file."""
    with open(par_path, "w") as f:
        f.write(f"PROBLEM_FILE = {tsp_path}\nTOUR_FILE = {tour_path}\n")
        if time_limit_s:
            f.write(f"TIME_LIMIT = {time_limit_s}\n")

def _compute_distance(p1, p2):
    return int(math.sqrt(np.sum((p1 - p2)**2)) + 0.5)

def _save_as_tsplib(file_path, coords, tsp_name, d):
    """
    Standardized function to save TSP instances.
    ALWAYS writes a full, explicit distance matrix for maximum compatibility.
    """
    n = len(coords)
    with open(file_path, "w") as f:
        f.write(f"NAME : {tsp_name}\nTYPE : TSP\nCOMMENT : Generated Instance\nDIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
        for i in range(n):
            row_distances = [_compute_distance(coords[i], coords[j]) for j in range(n)]
            f.write(" ".join(map(str, row_distances)) + "\n")
        f.write("EOF\n")

def _run_concorde(solver_name, coords, d):
    scratch_tsp_path = os.path.join(SOLVER_SCRATCH_DIR, f"{solver_name}.tsp")
    _save_as_tsplib(scratch_tsp_path, coords, solver_name, d)
    
    clean_scratch_path = scratch_tsp_path.strip().replace('\\', '/')
    wsl_scratch_path = subprocess.run(["wsl", "wslpath", "-a", clean_scratch_path], capture_output=True, text=True, check=True).stdout.strip()
    wsl_tour_file = f"/tmp/{solver_name}.concorde.tour"
    
    concorde_cmd = ["wsl", "concorde", "-o", wsl_tour_file, wsl_scratch_path]
    
    start_time = time.perf_counter()
    # Run and check for errors, will terminate script if it fails
    subprocess.run(concorde_cmd, capture_output=True, text=True, check=True)
    runtime = time.perf_counter() - start_time
    
    tour_content = subprocess.run(["wsl", "cat", wsl_tour_file], capture_output=True, text=True, check=True).stdout.strip()
    
    lines = tour_content.splitlines()
    # Concorde outputs 0-indexed tours, we convert to 1-indexed
    tour_nodes = [int(n) + 1 for n in " ".join(lines[1:]).strip().split()]

    if not tour_nodes:
        raise ValueError(f"Concorde ran but failed to produce a valid tour for {solver_name}.")

    tour_length = sum(_compute_distance(coords[tour_nodes[i]-1], coords[tour_nodes[(i + 1) % len(tour_nodes)]-1]) for i in range(len(tour_nodes)))
    
    subprocess.run(["wsl", "rm", "-f", wsl_tour_file], capture_output=True)
    if os.path.exists(scratch_tsp_path): os.remove(scratch_tsp_path)
    
    return tour_length, runtime, tour_nodes

def _run_lkh(instance_name, coords, d, time_limit_s=None):
    tsp_path = os.path.join(SOLVER_SCRATCH_DIR, f"{instance_name}.tsp")
    par_path = os.path.join(SOLVER_SCRATCH_DIR, f"{instance_name}.par")
    tour_path = os.path.join(SOLVER_SCRATCH_DIR, f"{instance_name}.tour")
    
    try:
        _save_as_tsplib(tsp_path, coords, instance_name, d)
        # Pass the time limit to the parameter file writer
        _save_lkh_par(par_path, tsp_path, tour_path, time_limit_s)
        
        lkh_cmd = [LKH_EXECUTABLE_PATH, par_path]
        start_time = time.perf_counter()
        # Run and check for errors, will terminate script if it fails
        subprocess.run(lkh_cmd, capture_output=True, text=True, check=True)
        lkh_time = time.perf_counter() - start_time
        
        with open(tour_path, 'r') as f:
            tour_content = f.read()
            
        tour_match = re.search(r"TOUR_SECTION\s*([\s\d-]*?)\s*EOF", tour_content, re.DOTALL)
        if not tour_match:
            raise ValueError(f"LKH ran but could not parse tour from output for {instance_name}")
        
        tour_nodes = [int(n) for n in tour_match.group(1).strip().split() if int(n) != -1]
        tour_length = sum(_compute_distance(coords[tour_nodes[i]-1], coords[tour_nodes[(i + 1) % len(tour_nodes)]-1]) for i in range(len(tour_nodes)))
        
        return tour_length, lkh_time, tour_nodes
    finally:
        # Cleanup temporary files regardless of success or failure
        for f_path in [tsp_path, par_path, tour_path]:
            if os.path.exists(f_path):
                os.remove(f_path)

# ====================================================================
# INSTANCE GENERATION
# ====================================================================

def generate_single_instance(params):
    n, d, dist_type, attempt_seed, seq_j, grid_size = params
    
    if dist_type == 'corr':
        coords = _generate_nd_correlated(n, d, attempt_seed, grid_size)
    else:
        coords = np.column_stack([
            DISTRIBUTION_MAP_1D[dist_type](n, attempt_seed + i + 1, grid_size)
            for i in range(d)
        ]).astype(int)

    instance_name = f"N{n}_D{d}_G{grid_size}_{dist_type}_{seq_j}"
    instance_path = os.path.join(INSTANCES_DIR, f"{instance_name}.json")
    
    with open(instance_path, 'w') as f:
        json.dump({"instance_name": instance_name, "n_customers": n, "dimension": d, 
                   "grid_size": grid_size, "distribution_type": dist_type, 
                   "generation_seed": attempt_seed, "coordinates": coords.tolist()}, f, indent=4)

def safely_generate_instance(params):
    n, d, dist_type, original_seed, seq_j, grid_size = params
    instance_name = f"N{n}_D{d}_G{grid_size}_{dist_type}_{seq_j}"
    instance_path = os.path.join(INSTANCES_DIR, f"{instance_name}.json")
    
    if os.path.exists(instance_path):
        return

    generate_single_instance(params)
    gc.collect()

# ====================================================================
# SOLUTION GENERATION
# ====================================================================

def solve_single_instance(params):
    n, d, dist_type, _, seq_j, grid_size = params
    
    instance_name = f"N{n}_D{d}_G{grid_size}_{dist_type}_{seq_j}"
    solution_path = os.path.join(SOLUTIONS_DIR, f"{instance_name}.sol.json")
    if os.path.exists(solution_path):
        return

    instance_path = os.path.join(INSTANCES_DIR, f"{instance_name}.json")
    if not os.path.exists(instance_path):
        tqdm.write(f"Instance file missing for Seq {seq_j}, skipping solve.")
        return

    with open(instance_path, 'r') as f:
        inst_data = json.load(f)

    coords = np.array(inst_data['coordinates'])
    solver_name = _encode_id_for_solver(seq_j)

    # --- NEW LOGIC ---
    # 1. Set LKH time limit for large instances
    lkh_time_limit = n/10 if n > 100 else 10  # seconds
        
    # 2. Always run LKH, now with a potential time limit
    lkh_length, lkh_time, lkh_tour = _run_lkh(instance_name, coords, d, time_limit_s=lkh_time_limit)

    # 3. Conditionally run Concorde only for smaller instances
    if n <= 5000:
        # Run Concorde. Script terminates if this fails.
        concorde_length, concorde_time, concorde_tour = _run_concorde(solver_name, coords, d)

        # Determine the optimal
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
        # 4. For n > 1000, skip Concorde and use LKH results as the "optimum"
        if not lkh_time_limit: # This check is just for safety
             tqdm.write(f"Skipping Concorde for {instance_name} (n={n})")
        else:
             tqdm.write(f"Skipping Concorde and applying LKH {lkh_time_limit}s limit for {instance_name} (n={n})")

        optimal_cost = lkh_length
        optimal_tour = lkh_tour
        optimal_solver = "lkh_only_timed" # New solver name to reflect this
        
        # Set Concorde-specific fields to None
        concorde_length = None
        concorde_time = None
        concorde_tour = None
        lkh_gap_pct = None # Gap is not applicable

    solution_data = {
        "instance_name": instance_name, "n_customers": n, "dimension": d, "grid_size": grid_size,
        "optimal_cost": optimal_cost, "optimal_tour": optimal_tour, "optimal_solver": optimal_solver,
        "instance_file_path": instance_path,
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

# ====================================================================
# MAIN GENERATION SCRIPT
# ====================================================================

def main():
    GRID_SIZE_LIST = [100, 1000, 10000]
    N_CUSTOMERS_LIST = [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    DIMENSION_LIST = [2, 3, 4, 5]
    N_PER_CONFIGURATION = 20

    existing_instance_configs = defaultdict(set)
    existing_solution_configs = defaultdict(set)
    filename_pattern = re.compile(r"^(N\d+_D\d+_G\d+_.+?)_(\d+)\.json$")
    for f in os.listdir(INSTANCES_DIR):
        match = filename_pattern.match(f)
        if match:
            config_base, seq_str = match.group(1), match.group(2)
            existing_instance_configs[config_base].add(int(seq_str))
    for f in os.listdir(SOLUTIONS_DIR):
        match = re.compile(r"^(N\d+_D\d+_G\d+_.+?)_(\d+)\.sol\.json$").match(f)
        if match:
            config_base, seq_str = match.group(1), match.group(2)
            existing_solution_configs[config_base].add(int(seq_str))

    all_params = []
    print("Generating configurations...")
    dist_types = list(DISTRIBUTION_MAP_1D.keys()) + ['corr']
    for n in N_CUSTOMERS_LIST:
        for d in DIMENSION_LIST:
            for grid_size in GRID_SIZE_LIST:
                for dist_type in dist_types:
                    config_base = f"N{n}_D{d}_G{grid_size}_{dist_type}"
                    existing_seqs = existing_instance_configs.get(config_base, set())
                    needed = N_PER_CONFIGURATION - len(existing_seqs)
                    if needed <= 0:
                        continue
                    
                    config_num = sum(ord(c) for c in dist_type)
                    seq_j = 1
                    added = 0
                    while added < needed:
                        if seq_j not in existing_seqs:
                            seed = config_num + seq_j * 100000 + n * 1000 + d * 100 + grid_size
                            all_params.append((n, d, dist_type, seed, seq_j, grid_size))
                            added += 1
                        seq_j += 1
    
    num_workers = os.cpu_count() - 5 if os.cpu_count() > 5 else 1
    print(f"Preparing to generate {len(all_params)} new instances using {num_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(safely_generate_instance, params): params for params in all_params}
        for future in tqdm(as_completed(futures), total=len(all_params), desc="Instance Generation Pass"):
            future.result() # Will re-raise any exception from the worker thread

    print(f"\n--- Instance Generation Complete ---")

    all_params_for_solving = []
    print("Scanning for unsolved instances...")
    for n in N_CUSTOMERS_LIST:
        for d in DIMENSION_LIST:
            for grid_size in GRID_SIZE_LIST:
                for dist_type in dist_types:
                    config_base = f"N{n}_D{d}_G{grid_size}_{dist_type}"
                    existing_insts = existing_instance_configs.get(config_base, set())
                    existing_sols = existing_solution_configs.get(config_base, set())
                    unsolved_seqs = existing_insts - existing_sols
                    
                    for seq_j in sorted(list(unsolved_seqs)):
                        config_num = sum(ord(c) for c in dist_type)
                        seed = config_num + seq_j * 100000 + n * 1000 + d * 100 + grid_size
                        all_params_for_solving.append((n, d, dist_type, seed, seq_j, grid_size))

    print(f"Preparing to solve {len(all_params_for_solving)} instances using {num_workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(solve_single_instance, params): params for params in all_params_for_solving}
        for future in tqdm(as_completed(futures), total=len(all_params_for_solving), desc="Solution Generation Pass"):
            future.result() # Will re-raise any exception from the worker thread

    print("\n--- Solution Generation Complete ---")

if __name__ == '__main__':
    main()