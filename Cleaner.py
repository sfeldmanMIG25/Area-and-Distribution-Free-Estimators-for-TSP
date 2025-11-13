import os
import re
from tqdm import tqdm

# --- CONFIGURATION (from Dataset_generator.py) ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCES_DIR = os.path.join(ROOT_DIR, "instances")
SOLUTIONS_DIR = os.path.join(ROOT_DIR, "solutions")
# ---

def clean_large_instances(n_threshold=1000):
    """
    Deletes instance and solution files where N > n_threshold.
    """
    
    print(f"Scanning {INSTANCES_DIR} for instances with N > {n_threshold}...")
    
    # Regex to capture the 'N' value from filenames like "N1500_D2_G100_..."
    # It captures the digits immediately following "N" and ending with "_"
    n_pattern = re.compile(r"^N(\d+)_")
    
    instance_files = [f for f in os.listdir(INSTANCES_DIR) if f.endswith('.json')]
    
    if not instance_files:
        print("No instance files found.")
        return

    deleted_instance_count = 0
    deleted_solution_count = 0

    for filename in tqdm(instance_files, desc="Cleaning files"):
        match = n_pattern.match(filename)
        
        if not match:
            # print(f"Skipping file with unexpected name: {filename}")
            continue
            
        try:
            n_customers = int(match.group(1))
            
            if n_customers > n_threshold:
                # 1. Delete the instance file
                instance_path = os.path.join(INSTANCES_DIR, filename)
                if os.path.exists(instance_path):
                    os.remove(instance_path)
                    deleted_instance_count += 1
                
                # 2. Delete the corresponding solution file
                sol_filename = filename.replace('.json', '.sol.json')
                solution_path = os.path.join(SOLUTIONS_DIR, sol_filename)
                
                if os.path.exists(solution_path):
                    os.remove(solution_path)
                    deleted_solution_count += 1
                    
        except ValueError:
            # This handles case if regex captures something not an int (unlikely)
            print(f"Could not parse N-value from: {filename}")
            continue

    print("\n--- Cleanup Complete ---")
    print(f"✅ Deleted {deleted_instance_count} instance files.")
    print(f"✅ Deleted {deleted_solution_count} solution files.")

if __name__ == '__main__':
    # Ensure the directories exist before trying to scan them
    if not os.path.exists(INSTANCES_DIR) or not os.path.exists(SOLUTIONS_DIR):
        print("Error: 'instances' or 'solutions' directory not found.")
        print("Please run this script from the same directory as Dataset_generator.py")
    else:
        clean_large_instances(n_threshold=1000)