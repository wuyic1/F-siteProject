import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import sys # For printing info/errors

# --- Configuration ---
# Input directory: Contains all group-specific (base).txt, (marked).txt, (orig).txt files
INPUT_DIR = Path("/ChEMBL/F_pair") # ADJUST PATH
# Output directory: Where the final merged train/validation files will be saved
OUTPUT_DIR = Path("/sampled_data/Random") # ADJUST PATH
TRAIN_RATIO = 0.9 # Proportion of data for the training set
RANDOM_SEED = 42 # For reproducible splits
# --- End Configuration ---

# --- Substructure Names (Must match file naming convention) ---
SUBSTRUCTURES = [
    "-CFH₂", "-CF₂H", "-CF₃", "-OCF₂H", "-OCF₃",
    "-SCF₂H", "-SCF₃", "-CFH-", "-CF₂-", "-F"
]

def write_data(data_list, file_path):
    """Helper function to write a list of strings to a file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(data_list))
        print(f"Successfully wrote {len(data_list)} lines to {file_path.name}")
        return True
    except IOError as e:
        print(f"Error: Failed to write to file {file_path}: {e}", file=sys.stderr)
        return False

def stratified_split_and_save(input_dir, output_dir, train_ratio=0.9, random_seed=42):
    """
    Reads triples, performs stratified random split by group, merges, and saves
    Model 1 and Model 2 train/validation files with .src/.tgt extensions.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    random.seed(random_seed)
    print(f"Using random seed: {random_seed}")

    # Use defaultdict to store triples grouped by substructure name
    grouped_triples = defaultdict(list)
    total_lines_read = 0
    files_missing_or_error = False

    print("--- Reading and grouping input files by substructure ---")
    for sub in tqdm(SUBSTRUCTURES, desc="Reading & Grouping files"):
        base_file = input_path / f"{sub}(base).txt"
        marked_file = input_path / f"{sub}(marked).txt"
        orig_file = input_path / f"{sub}(orig).txt"

        # Check if all three files exist for the current substructure
        if not (base_file.is_file() and marked_file.is_file() and orig_file.is_file()):
            print(f"Warning: Missing one or more files for substructure '{sub}'. Skipping this group.", file=sys.stderr)
            files_missing_or_error = True
            continue

        try:
            # Read data, strip whitespace, and filter empty lines
            with open(base_file, 'r', encoding='utf-8') as f_base:
                 base_data = [line.strip() for line in f_base if line.strip()]
            with open(marked_file, 'r', encoding='utf-8') as f_marked:
                 marked_data = [line.strip() for line in f_marked if line.strip()]
            with open(orig_file, 'r', encoding='utf-8') as f_orig:
                 orig_data = [line.strip() for line in f_orig if line.strip()]

            # Verify that all files have the same number of non-empty lines
            if not (len(base_data) == len(marked_data) == len(orig_data)):
                print(f"Error: Data length mismatch for substructure '{sub}' ({len(base_data)},{len(marked_data)},{len(orig_data)}). Skipping.", file=sys.stderr)
                files_missing_or_error = True
                continue

            if base_data: # If data was read successfully
                 group_triples = list(zip(base_data, marked_data, orig_data))
                 grouped_triples[sub].extend(group_triples) # Store under the substructure name
                 total_lines_read += len(base_data)
                 # print(f"Read {len(base_data)} valid triples for '{sub}'.") # Optional verbosity
            # else:
                 # print(f"Warning: No data found in files for substructure '{sub}'.", file=sys.stderr)

        except Exception as e:
            print(f"Error: Failed reading files for substructure '{sub}': {e}", file=sys.stderr)
            files_missing_or_error = True

    if files_missing_or_error:
         print("Warning: One or more substructure groups had missing files or read errors.", file=sys.stderr)

    if not grouped_triples:
        print(f"Error: No valid data triples found for any group in {input_path}. Aborting.", file=sys.stderr)
        return

    print(f"--- Total valid triples read across all groups: {total_lines_read} ---")

    # --- Stratified Shuffle and Split ---
    all_train_triples = []
    all_valid_triples = []

    print("--- Performing stratified shuffle and split for each group ---")
    for sub, triples in grouped_triples.items():
        group_size = len(triples)
        if group_size == 0: continue # Should not happen if reading worked, but safe check

        # print(f"Processing group '{sub}' (Size: {group_size})") # Optional verbosity
        # Shuffle the data within the current group
        random.shuffle(triples)

        # Calculate the split point for the current group
        current_train_size = int(group_size * train_ratio)

        # Handle edge cases for very small groups
        if group_size > 0 and current_train_size == 0:
            current_train_size = 1 # Ensure at least one sample in train if possible
        if group_size > 1 and (group_size - current_train_size == 0):
             # Ensure at least one sample in validation if group size > 1
            current_train_size = group_size - 1

        current_valid_size = group_size - current_train_size
        # print(f"  - Splitting '{sub}': {current_train_size} train, {current_valid_size} valid") # Optional

        # Split the data for the current group
        current_train_triples = triples[:current_train_size]
        current_valid_triples = triples[current_train_size:]

        # Add the split data to the overall lists
        all_train_triples.extend(current_train_triples)
        all_valid_triples.extend(current_valid_triples)

    # --- Shuffle the combined sets (Recommended) ---
    # This ensures the final files are not ordered by substructure group
    print("Shuffling combined training set...")
    random.shuffle(all_train_triples)
    print("Shuffling combined validation set...")
    random.shuffle(all_valid_triples)

    print("--- Final dataset sizes ---")
    print(f"Total Training samples: {len(all_train_triples)}")
    print(f"Total Validation samples: {len(all_valid_triples)}")

    # --- Separate data and write output files ---
    print("Separating data and writing output files with .src/.tgt extensions...")

    # Separate training data
    train_base = [triple[0] for triple in all_train_triples]
    train_marked = [triple[1] for triple in all_train_triples]
    train_orig = [triple[2] for triple in all_train_triples]

    # Separate validation data
    valid_base = [triple[0] for triple in all_valid_triples]
    valid_marked = [triple[1] for triple in all_valid_triples]
    valid_orig = [triple[2] for triple in all_valid_triples]

    # --- Write Model 1 Files ---
    print("--- Writing Model 1 data files ---")
    # Use 'and' to check if all writes succeed
    success_m1 = (write_data(train_base, output_path / "train_m1.src") and
                  write_data(train_marked, output_path / "train_m1.tgt") and
                  write_data(valid_base, output_path / "valid_m1.src") and
                  write_data(valid_marked, output_path / "valid_m1.tgt"))
    if success_m1:
        print("Model 1 files written successfully.")
    else:
        print("Error: Failed to write one or more Model 1 files.", file=sys.stderr)

    # --- Write Model 2 Files ---
    print("--- Writing Model 2 data files ---")
    # Model 2 source is Model 1 target
    success_m2 = (write_data(train_marked, output_path / "train_m2.src") and
                  write_data(train_orig, output_path / "train_m2.tgt") and
                  write_data(valid_marked, output_path / "valid_m2.src") and
                  write_data(valid_orig, output_path / "valid_m2.tgt"))
    if success_m2:
        print("Model 2 files written successfully.")
    else:
        print("Error: Failed to write one or more Model 2 files.", file=sys.stderr)

    print("Dataset splitting complete.")


if __name__ == "__main__":
    try:
        # Use configurations defined at the top
        stratified_split_and_save(INPUT_DIR, OUTPUT_DIR, TRAIN_RATIO, RANDOM_SEED)
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}", file=sys.stderr)