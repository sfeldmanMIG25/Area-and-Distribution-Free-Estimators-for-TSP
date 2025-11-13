# File: Full_Graph_Generator.py
# Purpose: Recreate the full set of instance distributions from the attributed_graph.c
#          file for benchmarking purposes, incorporating all specified modifications.

import os
import sys
from pathlib import Path
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Ensure vrp_utils can be imported
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

# Import LKH_PATH and solve_tsp_lkh from the vrp_utils file.
# The LKH_PATH variable is assumed to be defined in vrp_utils.
from vrp_utils import solve_tsp_lkh, LKH_EXECUTABLE_PATH

# Set the output directory one level above the 'Scripts' folder
OUTPUT_DIR = SCRIPT_DIR.parent / "Generated_TSPs"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR = SCRIPT_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)
VIS_DIR = OUTPUT_DIR / "visualizations"
VIS_DIR.mkdir(exist_ok=True)

# Define the comprehensive sample parameters based on the new batching scheme
SAMPLES_PER_CONFIG = 5  # Number of instances to generate for each configuration

# Node counts to sample
n_points_list = [5, 8] + list(range(10, 101, 10)) + list(range(200, 1001, 100)) + list(range(2000, 10001, 1000))

# Distribution types from the C code
dist_types = [
    'random', 'normal', 'triangular', 'squeezed_uniform', 'uniform_triangular',
    'triangular_squeezed', 'boundary', 'x_central', 'truncated_exponential',
    'grid', 'correlated'
]

# Create a list of all configurations
CONFIGS = [{'n_points': n, 'dist_type': dist} for n in n_points_list for dist in dist_types]

# Adding specific clustered configurations for representation
CONFIGS += [
    {'n_points': 100, 'dist_type': 'clustered', 'clust_n': cn, 'clust_rad': cr}
    for cn, cr in [(5, 0.05), (10, 0.05), (10, 0.10)]
] + [
    {'n_points': 500, 'dist_type': 'clustered', 'clust_n': cn, 'clust_rad': cr}
    for cn, cr in [(10, 0.05), (20, 0.10)]
] + [
    {'n_points': 1000, 'dist_type': 'clustered', 'clust_n': cn, 'clust_rad': cr}
    for cn, cr in [(20, 0.10)]
] + [
    {'n_points': 5000, 'dist_type': 'clustered', 'clust_n': cn, 'clust_rad': cr}
    for cn, cr in [(50, 0.10)]
]


# --- Helper Functions for Instance Generation ---
GRID_SIZE = 1000

def generate_random(n_points, grid_size=GRID_SIZE):
    """Generates n_points with uniform random coordinates."""
    return grid_size * np.random.rand(n_points, 2)

def generate_normal(n_points, grid_size=GRID_SIZE):
    """Generates n_points from a normal (Gaussian) distribution."""
    coords = np.random.normal(loc=grid_size/2, scale=grid_size/6, size=(n_points, 2))
    return np.clip(coords, 0, grid_size)

def generate_triangular(n_points, grid_size=GRID_SIZE):
    """Replicates the C code's triangular distribution logic."""
    coords = np.zeros((n_points, 2))
    for i in range(n_points):
        uh = np.random.rand()
        uv = np.random.rand()
        if uh < 0.5:
            coords[i, 0] = grid_size * np.sqrt(uh * 2) / 2
        else:
            coords[i, 0] = grid_size * (1 - np.sqrt((1 - uh) * 2) / 2)
        if uv < 0.5:
            coords[i, 1] = grid_size * np.sqrt(uv * 2) / 2
        else:
            coords[i, 1] = grid_size * (1 - np.sqrt((1 - uv) * 2) / 2)
    return coords

def generate_squeezed_uniform(n_points, grid_size=GRID_SIZE):
    """Replicates the C code's squeezed uniform distribution logic."""
    coords = np.zeros((n_points, 2))
    i = 0
    while i < n_points:
        uh = np.random.rand()
        uv = np.random.rand()
        accept = np.random.rand()
        if accept <= uh * uv:
            coords[i, 0] = grid_size * uh
            coords[i, 1] = grid_size * uv
            i += 1
    return coords

def generate_uniform_triangular(n_points, grid_size=GRID_SIZE):
    """Replicates the C code's uniform-triangular distribution logic."""
    coords = np.zeros((n_points, 2))
    for i in range(n_points):
        coords[i, 0] = grid_size * np.random.rand()
        uv = np.random.rand()
        if uv < 0.5:
            coords[i, 1] = grid_size * np.sqrt(uv * 2) / 2
        else:
            coords[i, 1] = grid_size * (1 - np.sqrt((1 - uv) * 2) / 2)
    return coords

def generate_triangular_squeezed(n_points, grid_size=GRID_SIZE):
    """Replicates the C code's triangular-squeezed distribution logic."""
    coords = np.zeros((n_points, 2))
    for i in range(n_points):
        uh = np.random.rand()
        if uh < 0.5:
            coords[i, 0] = grid_size * np.sqrt(uh * 2) / 2
        else:
            coords[i, 0] = grid_size * (1 - np.sqrt((1 - uh) * 2) / 2)
        
        j = 0
        while j < 1:
            uv = np.random.rand()
            accept = np.random.rand()
            if accept <= uv:
                coords[i, 1] = grid_size * uv
                j += 1
    return coords

def generate_boundary(n_points, grid_size=GRID_SIZE):
    """Replicates the C code's boundary distribution logic."""
    coords = np.zeros((n_points, 2))
    i = 0
    while i < n_points:
        p = np.random.rand()
        x = grid_size * np.random.rand()
        y = grid_size * np.random.rand()
        check = (abs(x - grid_size/2) / (grid_size/2)) * (abs(y - grid_size/2) / (grid_size/2))
        if p <= check:
            coords[i, 0] = x
            coords[i, 1] = y
            i += 1
    return coords

def generate_x_central(n_points, grid_size=GRID_SIZE):
    """Replicates the C code's X-central distribution logic."""
    coords = np.zeros((n_points, 2))
    i = 0
    while i < n_points:
        p = np.random.rand()
        x = grid_size * np.random.rand()
        y = grid_size * np.random.rand()
        check = (1 - abs(x - grid_size/2) / (grid_size/2)) * (abs(y - grid_size/2) / (grid_size/2))
        if p <= check:
            coords[i, 0] = x
            coords[i, 1] = y
            i += 1
    return coords

def generate_truncated_exponential(n_points, grid_size=GRID_SIZE):
    """Replicates the C code's truncated exponential distribution logic."""
    coords = np.zeros((n_points, 2))
    for i in range(n_points):
        u1 = np.random.rand()
        u2 = np.random.rand()
        coords[i, 0] = (-np.log(u1) / 1.0)
        coords[i, 1] = (-np.log(u2) / 1.0)
        
        coords[i, 0] = (coords[i, 0] - math.floor(coords[i, 0])) * grid_size
        coords[i, 1] = (coords[i, 1] - math.floor(coords[i, 1])) * grid_size
    return coords

def generate_clustered(n_points, grid_size=GRID_SIZE, clust_n=None, clust_rad=None):
    """Generates n_points in clusters."""
    if clust_n is None or clust_rad is None:
        raise ValueError("Clustered distribution requires clust_n and clust_rad parameters.")
    if n_points < clust_n:
        raise ValueError("Number of points must be greater than number of clusters.")
    
    points_per_cluster = n_points // clust_n
    remaining_points = n_points % clust_n
    
    cluster_centers = grid_size * np.random.rand(clust_n, 2)
    coords = []
    
    for i in range(clust_n):
        num_points = points_per_cluster + (1 if i < remaining_points else 0)
        
        center_x, center_y = cluster_centers[i]
        radius = clust_rad * grid_size
        
        for _ in range(num_points):
            angle = 2 * math.pi * np.random.rand()
            dist = radius * np.random.rand()
            
            x = center_x + dist * math.cos(angle)
            y = center_y + dist * math.sin(angle)
            
            x = np.clip(x, 0, grid_size)
            y = np.clip(y, 0, grid_size)
            
            coords.append([x, y])
    return np.array(coords)

def generate_grid(n_points, grid_size=GRID_SIZE):
    """Generates n_points on a uniform grid with slight jitter."""
    side_length = math.ceil(math.sqrt(n_points))
    coords = []
    for i in range(side_length):
        for j in range(side_length):
            if len(coords) < n_points:
                x = (i + 0.5) * (grid_size / side_length) + (np.random.rand() - 0.5) * (grid_size / side_length * 0.1)
                y = (j + 0.5) * (grid_size / side_length) + (np.random.rand() - 0.5) * (grid_size / side_length * 0.1)
                coords.append([x, y])
    return np.array(coords)

def generate_correlated(n_points, grid_size=GRID_SIZE):
    """Generates n_points with correlated x and y coordinates."""
    x = np.random.uniform(0, grid_size, n_points)
    y = x + np.random.normal(loc=0, scale=grid_size/10, size=n_points)
    coords = np.vstack((x, y)).T
    return np.clip(coords, 0, grid_size)

def write_vrp_instance(coords, instance_name, output_dir, tsp_cost):
    """Writes a TSP instance to a .vrp file in the specified format."""
    num_nodes = len(coords)
    file_path = output_dir / f"{instance_name}.vrp"
    
    with open(file_path, 'w') as f:
        f.write(f"NAME : {instance_name}\n")
        f.write("COMMENT : Generated for TSP benchmarking.\n")
        f.write("TYPE : CVRP\n")
        f.write(f"DIMENSION : {num_nodes}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write(f"CAPACITY : {num_nodes}\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords):
            f.write(f"{i + 1}\t{x:.2f}\t{y:.2f}\n")
        f.write("DEMAND_SECTION\n")
        for i in range(num_nodes):
            demand = 0 if i == 0 else 1
            f.write(f"{i + 1}\t{demand}\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")
        f.write(f"TOUR_LENGTH: {tsp_cost:.4f}\n")

# --- New Visualization Function ---
def visualize():
    """Reads all generated .vrp instances and plots them."""
    print(f"Generating visualizations in {VIS_DIR.resolve()}...")
    vrp_files = list(OUTPUT_DIR.glob('*.vrp'))
    
    if not vrp_files:
        print("No .vrp files found to visualize.")
        return
        
    for vrp_file in tqdm(vrp_files, desc="Plotting Instances"):
        # Custom parser for the .vrp file format
        with open(vrp_file, 'r') as f:
            lines = f.readlines()
            
        coords = []
        parsing_coords = False
        instance_name = vrp_file.stem
        
        for line in lines:
            line = line.strip()
            if line.startswith("NAME :"):
                instance_name = line.split(":")[1].strip()
            if line == "NODE_COORD_SECTION":
                parsing_coords = True
                continue
            if line == "DEMAND_SECTION":
                parsing_coords = False
                continue
            
            if parsing_coords and line:
                parts = line.split()
                try:
                    _id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords.append([x, y])
                except (ValueError, IndexError):
                    continue
        
        coords = np.array(coords)
        if len(coords) == 0:
            continue
            
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot all nodes
        ax.scatter(coords[:, 0], coords[:, 1], c='blue', label='Customers', s=50)
        
        # Highlight the depot (assumed to be the first node)
        ax.scatter(coords[0, 0], coords[0, 1], c='red', marker='s', s=100, label='Depot')
        
        ax.set_title(f"Instance: {instance_name}")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        
        plt.savefig(VIS_DIR / f"{instance_name}.png")
        plt.close(fig)

# --- Main Generation and Benchmarking Loop ---

def main():
    """Main function to generate instances, solve with LKH-3, and save."""
    print("--- VRP-to-TSP Instance Generator (Full C Distribution Set) ---")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    
    # Map distribution names to functions
    dist_map = {
        'random': generate_random, 'normal': generate_normal, 'triangular': generate_triangular,
        'squeezed_uniform': generate_squeezed_uniform, 'uniform_triangular': generate_uniform_triangular,
        'triangular_squeezed': generate_triangular_squeezed, 'boundary': generate_boundary,
        'x_central': generate_x_central, 'truncated_exponential': generate_truncated_exponential,
        'clustered': generate_clustered, 'grid': generate_grid, 'correlated': generate_correlated
    }
    
    for config in tqdm(CONFIGS, desc="Generating Instances"):
        n = config['n_points']
        dist_type = config['dist_type']
        
        for i in range(SAMPLES_PER_CONFIG):
            if dist_type == 'clustered':
                coords = dist_map[dist_type](n, clust_n=config['clust_n'], clust_rad=config['clust_rad'])
                instance_name = f"TSP-{dist_type}-n{n}-c{config['clust_n']}-r{int(config['clust_rad'] * 100)}-{i+1}"
            else:
                coords = dist_map[dist_type](n)
                instance_name = f"TSP-{dist_type}-n{n}-{i+1}"
    
            # Solve with LKH-3
            node_ids_for_lkh = list(range(1, len(coords) + 1))
            coords_for_lkh = {i: tuple(c) for i, c in zip(node_ids_for_lkh, coords)}
            depot_id = 1
            all_customer_data_for_lkh = {i: {'coords': c} for i, c in coords_for_lkh.items()}
            tsp_cost = solve_tsp_lkh(
            node_ids_for_tsp=node_ids_for_lkh,
            all_customer_data=all_customer_data_for_lkh,
            depot_id=depot_id
            )
            
            # Write instance to .vrp file with the LKH cost appended
            write_vrp_instance(coords, instance_name, OUTPUT_DIR, tsp_cost)
    
    print("\nInstance generation complete.")
    visualize()
    print("\nVisualizations complete.")

if __name__ == "__main__":
    main()