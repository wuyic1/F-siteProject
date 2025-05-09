import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem # For fingerprints
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count
import sys # For printing info/errors

# --- Configuration ---
# Input directory: Contains all group-specific (base).txt, (marked).txt, (orig).txt files
INPUT_DIR = Path("/ChEMBL/F_pair") # ADJUST PATH
# Output directory: Where the final merged train/validation files will be saved
OUTPUT_DIR = Path("/sampled_data/Scaffold") # ADJUST PATH
TRAIN_RATIO = 0.9 # Target ratio for training set (applied to scaffolds first)
RANDOM_SEED = 42 # For reproducible splits and t-SNE

# Substructure names (must match file naming)
SUBSTRUCTURES = [
    "-CFH₂", "-CF₂H", "-CF₃", "-OCF₂H", "-OCF₃",
    "-SCF₂H", "-SCF₃", "-CFH-", "-CF₂-", "-F"
]

# Visualization Sample Sizes
VIS_SAMPLE_SIZE_MOLECULE_TSNE = 10000 # Max molecules for molecule t-SNE plot
VIS_SAMPLE_SIZE_SCAFFOLD_TSNE = 1000  # Max scaffolds for scaffold t-SNE plot
VIS_SAMPLE_SIZE_TANIMOTO = 2000    # Max scaffolds per set for Tanimoto calc

# t-SNE Parameters
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_INIT = 'pca' # Faster initialization
TSNE_LEARNING_RATE = 'auto' # Adapts learning rate

# Plotting Parameters
MOL_PLOT_MARKER_SIZE_TRAIN = 8
MOL_PLOT_MARKER_SIZE_VALID = 12
MOL_PLOT_ALPHA = 0.6
MOL_PLOT_EDGE_WIDTH = 0.3
SCAFFOLD_PLOT_MARKER_SIZE_TRAIN = 20
SCAFFOLD_PLOT_MARKER_SIZE_VALID = 25
SCAFFOLD_PLOT_ALPHA = 0.7
SCAFFOLD_PLOT_EDGE_WIDTH = 0.4
# --- End Configuration ---

# Suppress RDKit logs (except errors)
RDLogger.DisableLog('rdApp.*')

# --- Helper Functions ---
def get_scaffold(smiles):
    """Calculate Murcko scaffold SMILES for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        # Generic scaffold (removes side chains)
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        # Filter out very small scaffolds (e.g., single atoms)
        if scaffold_mol.GetNumHeavyAtoms() < 2: return None
        # Return canonical scaffold SMILES (non-isomeric)
        return Chem.MolToSmiles(scaffold_mol, isomericSmiles=False, canonical=True)
    except Exception:
        # print(f"Debug: Failed scaffold calc for {smiles}", file=sys.stderr) # Optional debug
        return None

def calculate_fingerprint(smiles):
    """Calculate Morgan fingerprint for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Morgan fingerprint (similar to ECFP4)
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        else:
            return None
    except Exception:
        # print(f"Debug: Failed fingerprint calc for {smiles}", file=sys.stderr) # Optional debug
        return None

def calculate_tanimoto(fp1, fp2):
    """Calculate Tanimoto similarity between two RDKit fingerprints."""
    if fp1 is None or fp2 is None: return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def compute_fp_worker(smiles):
    """Worker function for parallel fingerprint calculation."""
    return (smiles, calculate_fingerprint(smiles))

def write_data(data_list, file_path):
    """Helper function to write a list of strings to a file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(data_list))
        # print(f"Debug: Successfully wrote {len(data_list)} lines to {file_path.name}") # Optional debug
        return True
    except IOError as e:
        print(f"Error: Failed to write to file {file_path}: {e}", file=sys.stderr)
        return False

# --- Main Splitting Logic ---
def scaffold_split_and_save(input_dir, output_dir, train_ratio=0.9, random_seed=42):
    """
    Performs scaffold splitting on triples, ensuring scaffolds do not overlap
    between train and validation sets (within each substructure group initially).
    Merges results and saves train/validation files. Also generates visualizations.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f"Using random seed: {random_seed}")

    all_train_triples = []
    all_valid_triples = []
    scaffolds_in_train = set() # Track unique scaffolds assigned to train set globally
    scaffolds_in_valid = set() # Track unique scaffolds assigned to validation set globally
    total_lines_read = 0
    total_valid_triples_with_scaffold = 0
    total_scaffolds_found_groups = 0 # Sum of scaffolds found in each group (may have duplicates)
    files_missing_or_error = False

    print("--- 1. Reading, calculating scaffolds, grouping, and splitting by group ---")
    for sub in tqdm(SUBSTRUCTURES, desc="Processing substructures"):
        base_file = input_path / f"{sub}(base).txt"
        marked_file = input_path / f"{sub}(marked).txt"
        orig_file = input_path / f"{sub}(orig).txt"

        if not (base_file.is_file() and marked_file.is_file() and orig_file.is_file()):
            print(f"Warning: Missing one or more files for substructure '{sub}'. Skipping.", file=sys.stderr)
            files_missing_or_error = True
            continue

        # Map: scaffold_smiles -> list of (base, marked, orig) triples for that scaffold in this group
        scaffold_to_triples_map = defaultdict(list)
        # List for triples where scaffold calculation failed or yielded no scaffold
        triples_with_no_scaffold = []

        try:
            # Read data efficiently
            with open(base_file, 'r', encoding='utf-8') as f_base:
                base_data = [line.strip() for line in f_base if line.strip()]
            with open(marked_file, 'r', encoding='utf-8') as f_marked:
                marked_data = [line.strip() for line in f_marked if line.strip()]
            with open(orig_file, 'r', encoding='utf-8') as f_orig:
                orig_data = [line.strip() for line in f_orig if line.strip()]

            # Basic validation
            if not (len(base_data) == len(marked_data) == len(orig_data)):
                print(f"Error: File length mismatch for '{sub}' ({len(base_data)},{len(marked_data)},{len(orig_data)}). Skipping.", file=sys.stderr)
                files_missing_or_error = True
                continue

            current_group_triples_read = len(base_data)
            total_lines_read += current_group_triples_read
            if current_group_triples_read == 0: continue # Skip if group files are empty

            # Calculate scaffolds and group triples
            for i in range(current_group_triples_read):
                base_smi = base_data[i]
                scaffold = get_scaffold(base_smi) # Use base SMILES for scaffold
                triple = (base_smi, marked_data[i], orig_data[i])
                if scaffold:
                    scaffold_to_triples_map[scaffold].append(triple)
                else:
                    triples_with_no_scaffold.append(triple)

            # Split scaffolds *within this group* first
            scaffolds_in_group = list(scaffold_to_triples_map.keys())
            num_scaffolds_in_group = len(scaffolds_in_group)
            num_valid_triples_in_group = sum(len(v) for v in scaffold_to_triples_map.values())
            num_no_scaffold_in_group = len(triples_with_no_scaffold)
            total_valid_triples_with_scaffold += num_valid_triples_in_group
            total_scaffolds_found_groups += num_scaffolds_in_group

            # Assign scaffolds to train/validation for this group
            if num_scaffolds_in_group > 0:
                random.shuffle(scaffolds_in_group)
                train_scaffold_count = int(num_scaffolds_in_group * train_ratio)
                # Edge cases for small numbers of scaffolds
                if num_scaffolds_in_group > 0 and train_scaffold_count == 0: train_scaffold_count = 1
                if num_scaffolds_in_group > 1 and (num_scaffolds_in_group - train_scaffold_count == 0): train_scaffold_count -= 1

                current_train_scaffolds = set(scaffolds_in_group[:train_scaffold_count])
                current_valid_scaffolds = set(scaffolds_in_group[train_scaffold_count:])

                # Add triples to global lists based on scaffold assignment
                for scaffold, triples in scaffold_to_triples_map.items():
                    if scaffold in current_train_scaffolds:
                        all_train_triples.extend(triples)
                        scaffolds_in_train.add(scaffold)
                    else: # Belongs to validation scaffold set for this group
                        all_valid_triples.extend(triples)
                        scaffolds_in_valid.add(scaffold)

            # Handle molecules without scaffolds (split randomly)
            if num_no_scaffold_in_group > 0:
                random.shuffle(triples_with_no_scaffold)
                num_no_scaffold_train = int(num_no_scaffold_in_group * train_ratio)
                # Ensure at least one if possible
                if num_no_scaffold_in_group > 0 and num_no_scaffold_train == 0: num_no_scaffold_train = 1
                if num_no_scaffold_in_group > 1 and (num_no_scaffold_in_group - num_no_scaffold_train == 0): num_no_scaffold_train -=1

                all_train_triples.extend(triples_with_no_scaffold[:num_no_scaffold_train])
                all_valid_triples.extend(triples_with_no_scaffold[num_no_scaffold_train:])

        except Exception as e:
            print(f"Error: An error occurred while processing substructure '{sub}': {e}", file=sys.stderr)
            files_missing_or_error = True

    if files_missing_or_error:
        print("Warning: One or more groups encountered errors during processing.", file=sys.stderr)

    print("--- Initial Data Split Statistics ---")
    print(f"Total lines read from input files: {total_lines_read}")
    print(f"Total triples with valid scaffolds: {total_valid_triples_with_scaffold}")
    print(f"Total scaffolds found (sum across groups): {total_scaffolds_found_groups}")
    print(f"Initial train triples count: {len(all_train_triples)}")
    print(f"Initial validation triples count: {len(all_valid_triples)}")
    print(f"Unique scaffolds assigned to train: {len(scaffolds_in_train)}")
    print(f"Unique scaffolds assigned to validation: {len(scaffolds_in_valid)}")
    # Check for overlap (should be zero if logic is correct)
    overlap = len(scaffolds_in_train.intersection(scaffolds_in_valid))
    print(f"Scaffold overlap between train and validation: {overlap}")
    if overlap > 0:
         print("Error: Scaffold overlap detected! Check splitting logic.", file=sys.stderr)

    # --- 2. Final Shuffle of Combined Data ---
    print("--- 2. Shuffling combined train and validation sets ---")
    random.shuffle(all_train_triples)
    random.shuffle(all_valid_triples)
    print(f"Final training set size: {len(all_train_triples)}")
    print(f"Final validation set size: {len(all_valid_triples)}")

    # --- 3. Separate and Save Output Files ---
    print("--- 3. Separating data and writing output files (.src/.tgt) ---")
    train_base = [triple[0] for triple in all_train_triples]
    train_marked = [triple[1] for triple in all_train_triples]
    train_orig = [triple[2] for triple in all_train_triples]
    valid_base = [triple[0] for triple in all_valid_triples]
    valid_marked = [triple[1] for triple in all_valid_triples]
    valid_orig = [triple[2] for triple in all_valid_triples]

    # Write Model 1 files
    success_m1 = (write_data(train_base, output_path / "train_m1.src") and
                  write_data(train_marked, output_path / "train_m1.tgt") and
                  write_data(valid_base, output_path / "valid_m1.src") and
                  write_data(valid_marked, output_path / "valid_m1.tgt"))
    if not success_m1: print("Error: Failed to write one or more Model 1 files.", file=sys.stderr)

    # Write Model 2 files
    success_m2 = (write_data(train_marked, output_path / "train_m2.src") and
                  write_data(train_orig, output_path / "train_m2.tgt") and
                  write_data(valid_marked, output_path / "valid_m2.src") and
                  write_data(valid_orig, output_path / "valid_m2.tgt"))
    if not success_m2: print("Error: Failed to write one or more Model 2 files.", file=sys.stderr)

    if success_m1 and success_m2: print("Output files saved successfully.")

    # # --- 4. Generate Visualizations ---
    # print("--- 4. Generating visualization plots ---")
    # visualize_molecule_tsne(all_train_triples, all_valid_triples, output_path, sample_size=VIS_SAMPLE_SIZE_MOLECULE_TSNE)
    # visualize_scaffold_tsne(scaffolds_in_train, scaffolds_in_valid, output_path, sample_size=VIS_SAMPLE_SIZE_SCAFFOLD_TSNE)
    # visualize_combined_tanimoto(scaffolds_in_train, scaffolds_in_valid, output_path, sample_size=VIS_SAMPLE_SIZE_TANIMOTO)
    #
    # print("Scaffold splitting and visualization finished.")


# # --- Visualization Functions ---
# def run_tsne(fingerprints_np, random_seed):
#     """Runs t-SNE on the provided fingerprint array."""
#     n_samples = fingerprints_np.shape[0]
#     # Adjust perplexity if number of samples is too small
#     if n_samples <= TSNE_PERPLEXITY:
#         perp_val = max(1, n_samples - 1)
#         print(f"t-SNE Info: Sample size ({n_samples}) <= perplexity ({TSNE_PERPLEXITY}). Adjusting perplexity to {perp_val}.")
#     else:
#         perp_val = TSNE_PERPLEXITY
#
#     tsne = TSNE(n_components=2, random_state=random_seed, n_jobs=-1, # Use all available cores
#                 perplexity=perp_val, n_iter=TSNE_N_ITER,
#                 init=TSNE_INIT, learning_rate=TSNE_LEARNING_RATE,
#                 early_exaggeration=12.0, verbose=0) # Suppress verbose output
#     return tsne.fit_transform(fingerprints_np)
#
# def plot_scatter(tsne_result, labels_filtered, title, filename, output_dir,
#                  marker_size_train, marker_size_valid, alpha, edge_width):
#     """Generates and saves a scatter plot for t-SNE results."""
#     plt.figure(figsize=(8, 6), dpi=300)
#     plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
#     plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
#
#     colors = {'Train': '#1f77b4', 'Validation': '#ff7f0e'} # Blue and Orange
#     markers = {'Train': 'o', 'Validation': 's'} # Circle and Square
#     sizes = {'Train': marker_size_train, 'Validation': marker_size_valid}
#
#     for label in sorted(list(set(labels_filtered))): # Ensure consistent legend order
#         idx = [i for i, l in enumerate(labels_filtered) if l == label]
#         plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1],
#                     c=colors[label], marker=markers[label], label=label,
#                     alpha=alpha, s=sizes[label],
#                     edgecolors='none' if edge_width == 0 else 'w',
#                     linewidths=edge_width if edge_width > 0 else 0)
#
#     plt.xlabel("t-SNE Component 1", fontsize=14)
#     plt.ylabel("t-SNE Component 2", fontsize=14)
#     plt.title(title, fontsize=16, pad=12)
#     legend = plt.legend(fontsize=11, loc='best', frameon=True, shadow=False, facecolor='white', framealpha=0.8, markerscale=1.5)
#     # Ensure legend markers are visible
#     for handle in legend.legendHandles: handle.set_sizes([20.0])
#
#     plt.grid(True, linestyle=':', alpha=0.6, color='lightgrey')
#     plt.xticks(fontsize=11)
#     plt.yticks(fontsize=11)
#     plt.tight_layout(pad=1.0) # Adjust padding
#
#     plot_path = output_dir / filename
#     try:
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         print(f"Plot saved to: {plot_path}")
#     except Exception as e:
#         print(f"Error: Failed to save plot {plot_path}: {e}", file=sys.stderr)
#     plt.close() # Close the plot to free memory
#
# def visualize_molecule_tsne(train_triples, valid_triples, output_dir, sample_size=10000):
#     """Generates t-SNE plot based on base molecule fingerprints."""
#     print(f"Starting molecule-level t-SNE (sample size: ~{sample_size})...")
#     # Extract base SMILES
#     train_b = [t[0] for t in train_triples]
#     valid_b = [t[0] for t in valid_triples]
#
#     # Determine sample counts, ensuring roughly equal representation
#     n_tr = min(len(train_b), sample_size // 2)
#     n_va = min(len(valid_b), sample_size - n_tr) # Use remaining budget for validation
#     n_tr = min(n_tr, sample_size - n_va) # Re-adjust train if validation took more
#
#     if n_tr <= 1 or n_va <= 1: # Need at least 2 points for t-SNE perplexity
#         print("Warning: Insufficient samples in train or validation set for molecule t-SNE.", file=sys.stderr)
#         return
#
#     # Sample SMILES and create labels
#     sampled_smiles = random.sample(train_b, n_tr) + random.sample(valid_b, n_va)
#     labels = ["Train"] * n_tr + ["Validation"] * n_va
#     print(f"Calculating fingerprints for {len(sampled_smiles)} molecules ({n_tr} train, {n_va} valid)...")
#
#     # Calculate fingerprints in parallel
#     num_proc = min(16, cpu_count()) # Limit parallel processes
#     fingerprints_dict = {}
#     valid_indices = []
#     fps_list = []
#     with Pool(processes=num_proc) as pool:
#         results = list(tqdm(pool.imap(compute_fp_worker, sampled_smiles, chunksize=200), total=len(sampled_smiles), desc="Molecule FPs"))
#
#     # Process results and convert to numpy array
#     for i, (smiles, fp) in enumerate(results):
#         if fp:
#             arr = np.zeros((1,), dtype=int)
#             DataStructs.ConvertToNumpyArray(fp, arr)
#             fps_list.append(arr)
#             valid_indices.append(i) # Keep track of which original samples had valid FPs
#
#     if not fps_list:
#         print("Warning: No valid molecule fingerprints generated for t-SNE.", file=sys.stderr)
#         return
#
#     fingerprints_np = np.array(fps_list)
#     labels_filtered = [labels[i] for i in valid_indices] # Filter labels accordingly
#
#     print(f"Performing molecule t-SNE dimensionality reduction ({fingerprints_np.shape[0]} samples)...")
#     try:
#         tsne_result = run_tsne(fingerprints_np, RANDOM_SEED)
#         plot_scatter(tsne_result, labels_filtered, "Molecular Space t-SNE (Base Molecules)",
#                      "tsne_molecule_split.png", output_dir,
#                      MOL_PLOT_MARKER_SIZE_TRAIN, MOL_PLOT_MARKER_SIZE_VALID,
#                      MOL_PLOT_ALPHA, MOL_PLOT_EDGE_WIDTH)
#     except Exception as e:
#         print(f"Error during molecule t-SNE calculation or plotting: {e}", file=sys.stderr)
#
# def visualize_scaffold_tsne(train_scaffolds_set, valid_scaffolds_set, output_dir, sample_size=1000):
#     """Generates t-SNE plot based on scaffold fingerprints."""
#     print(f"Starting scaffold-level t-SNE (sample size: ~{sample_size})...")
#     if not train_scaffolds_set or not valid_scaffolds_set:
#         print("Warning: No train or validation scaffolds provided for t-SNE.", file=sys.stderr)
#         return
#
#     # Sample scaffolds
#     n_tr = min(len(train_scaffolds_set), sample_size // 2)
#     n_va = min(len(valid_scaffolds_set), sample_size - n_tr)
#     n_tr = min(n_tr, sample_size - n_va)
#
#     if n_tr <= 1 or n_va <= 1:
#         print("Warning: Insufficient samples in train or validation scaffold sets for t-SNE.", file=sys.stderr)
#         return
#
#     sampled_scaffolds = random.sample(list(train_scaffolds_set), n_tr) + random.sample(list(valid_scaffolds_set), n_va)
#     labels = ["Train"] * n_tr + ["Validation"] * n_va
#     print(f"Calculating fingerprints for {len(sampled_scaffolds)} scaffolds ({n_tr} train, {n_va} valid)...")
#
#     # Calculate fingerprints in parallel
#     num_proc = min(16, cpu_count())
#     fingerprints_dict = {}
#     valid_indices = []
#     fps_list = []
#     with Pool(processes=num_proc) as pool:
#         results = list(tqdm(pool.imap(compute_fp_worker, sampled_scaffolds, chunksize=200), total=len(sampled_scaffolds), desc="Scaffold FPs"))
#
#     for i, (smiles, fp) in enumerate(results):
#         if fp:
#             arr = np.zeros((1,), dtype=int)
#             DataStructs.ConvertToNumpyArray(fp, arr)
#             fps_list.append(arr)
#             valid_indices.append(i)
#
#     if not fps_list:
#         print("Warning: No valid scaffold fingerprints generated for t-SNE.", file=sys.stderr)
#         return
#
#     fingerprints_np = np.array(fps_list)
#     labels_filtered = [labels[i] for i in valid_indices]
#
#     print(f"Performing scaffold t-SNE dimensionality reduction ({fingerprints_np.shape[0]} samples)...")
#     try:
#         tsne_result = run_tsne(fingerprints_np, RANDOM_SEED)
#         plot_scatter(tsne_result, labels_filtered, "Scaffold Space t-SNE",
#                      "tsne_scaffold_split.png", output_dir,
#                      SCAFFOLD_PLOT_MARKER_SIZE_TRAIN, SCAFFOLD_PLOT_MARKER_SIZE_VALID,
#                      SCAFFOLD_PLOT_ALPHA, SCAFFOLD_PLOT_EDGE_WIDTH)
#     except Exception as e:
#         print(f"Error during scaffold t-SNE calculation or plotting: {e}", file=sys.stderr)
#
#
# def visualize_combined_tanimoto(train_scaffolds_set, valid_scaffolds_set, output_dir, sample_size=2000):
#     """Calculates and plots Tanimoto similarity distribution between train/valid scaffolds."""
#     print(f"Starting Tanimoto similarity visualization (sample size: ~{sample_size} per set)...")
#     if not train_scaffolds_set or not valid_scaffolds_set:
#         print("Warning: No train or validation scaffolds provided for Tanimoto analysis.", file=sys.stderr)
#         return
#
#     # Sample scaffolds from each set
#     n_tr = min(len(train_scaffolds_set), sample_size)
#     n_va = min(len(valid_scaffolds_set), sample_size)
#     if n_tr == 0 or n_va == 0:
#         print("Warning: Insufficient scaffold samples for Tanimoto calculation.", file=sys.stderr)
#         return
#
#     sampled_tr_sc = random.sample(list(train_scaffolds_set), n_tr)
#     sampled_va_sc = random.sample(list(valid_scaffolds_set), n_va)
#
#     print(f"Calculating fingerprints for {len(sampled_tr_sc)} train and {len(sampled_va_sc)} validation scaffolds...")
#     # Calculate fingerprints (can be parallelized for very large sample sizes if needed)
#     tr_fps_valid = [fp for smi, fp in (compute_fp_worker(s) for s in tqdm(sampled_tr_sc, desc="Train Scaffold FPs", leave=False)) if fp is not None]
#     va_fps_valid = [fp for smi, fp in (compute_fp_worker(s) for s in tqdm(sampled_va_sc, desc="Valid Scaffold FPs", leave=False)) if fp is not None]
#
#     if not tr_fps_valid or not va_fps_valid:
#         print("Warning: No valid fingerprints generated for Tanimoto calculation.", file=sys.stderr)
#         return
#
#     print(f"Calculating {len(tr_fps_valid) * len(va_fps_valid)} Tanimoto similarities...")
#     similarities = []
#     # Calculate pairwise similarities (this can be slow for large N)
#     for fp1 in tqdm(tr_fps_valid, desc="Calculating Similarities", leave=False):
#         for fp2 in va_fps_valid:
#             similarities.append(calculate_tanimoto(fp1, fp2))
#
#     if not similarities:
#         print("Warning: Failed to calculate any Tanimoto similarity values.", file=sys.stderr)
#         return
#
#     print("Plotting Tanimoto similarity distribution...")
#     plt.figure(figsize=(8, 6), dpi=300)
#     plt.style.use('seaborn-v0_8-whitegrid')
#     plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
#
#     plt.hist(similarities, bins=np.linspace(0, 1, 51), alpha=0.75, color='#8856a7', density=True, edgecolor='white', linewidth=0.5) # Purpleish color
#     plt.xlabel("Tanimoto Similarity (Train vs Validation Scaffolds)", fontsize=14)
#     plt.ylabel("Density", fontsize=14)
#     plt.title("Scaffold Tanimoto Similarity Distribution", fontsize=16, pad=12)
#     plt.grid(True, linestyle=':', alpha=0.6, color='lightgrey')
#
#     # Add mean and median lines
#     avg_sim = np.mean(similarities)
#     med_sim = np.median(similarities)
#     plt.axvline(avg_sim, color='#d95f02', linestyle='--', linewidth=1.5, label=f'Mean = {avg_sim:.3f}') # Orange dashed
#     plt.axvline(med_sim, color='#7570b3', linestyle=':', linewidth=1.5, label=f'Median = {med_sim:.3f}') # Indigo dotted
#
#     plt.legend(fontsize=11, frameon=True)
#     plt.xlim(0, 1)
#     plt.xticks(fontsize=11)
#     plt.yticks(fontsize=11)
#     plt.tight_layout(pad=1.0)
#
#     plot_path = output_dir / "tanimoto_similarity_distribution.png"
#     try:
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         print(f"Tanimoto plot saved: {plot_path} (Mean: {avg_sim:.3f}, Median: {med_sim:.3f})")
#     except Exception as e:
#         print(f"Error: Failed to save Tanimoto plot: {e}", file=sys.stderr)
#     plt.close()

# --- Main execution block ---
if __name__ == "__main__":
    try:
        scaffold_split_and_save(INPUT_DIR, OUTPUT_DIR, TRAIN_RATIO, RANDOM_SEED)
        print("Script execution finished.")
    except Exception as main_error:
        print(f"Error: An unhandled exception occurred in the main script: {main_error}", file=sys.stderr)
