import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import BulkTanimotoSimilarity
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import random
import time
import argparse

# --- Global Settings ---
FP_RADIUS = 2
FP_BITS = 2048


def get_scaffold(smiles):
    """Generates a Murcko scaffold for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold_mol)
    except:
        return None


def generate_fingerprints(smiles_list):
    """Generates Morgan fingerprints for a list of SMILES."""
    fps = []
    for smiles in tqdm(smiles_list, desc="Generating Fingerprints", leave=False):
        try:
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS) if mol else None
            fps.append(fp)
        except:
            fps.append(None)
    return fps


def init_worker(fps_data_global, smi_data_global):
    """Initializes worker process with global data for multiprocessing."""
    global global_fps, global_smiles
    global_fps = fps_data_global
    global_smiles = smi_data_global


def filter_worker(args):
    """Worker function to find molecules similar to a query molecule."""
    fp_query, smi_query, threshold = args
    if fp_query is None: return []
    similarities = BulkTanimotoSimilarity(fp_query, global_fps)
    high_sim_indices = np.where(np.array(similarities) >= threshold)[0]
    return [(smi_query, global_smiles[idx]) for idx in high_sim_indices]


def get_args():
    """Gets command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Splits molecule data into train, validation, and test sets with similarity filtering.")
    parser.add_argument("--benchmark_dir", type=str, required=True,
                        help="Directory containing benchmark pair files from Cut_save_F.py.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the external test set CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the split data files.")
    parser.add_argument("--smiles_column", type=str, default="SMILES",
                        help="Name of the SMILES column in the test file.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of data to be used for training.")
    parser.add_argument("--similarity_threshold", type=float, default=0.8,
                        help="Tanimoto similarity threshold for blacklisting.")
    parser.add_argument("--rare_class_threshold", type=int, default=50,
                        help="Class size below which is considered 'rare' for random splitting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--processes", type=int, default=max(1, cpu_count() - 2),
                        help="Number of CPU processes to use.")
    return parser.parse_args()


def main():
    """Main function to orchestrate the data splitting process."""
    args = get_args()
    start_time = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"Using {args.processes} CPU cores.")

    # Step 1: Consolidate data sources
    print("\n--- Step 1: Consolidating all data sources ---")
    orig_to_triples = defaultdict(list)
    input_dir = Path(args.benchmark_dir)
    for base_file in tqdm(list(input_dir.glob("*(base).txt")), desc="Reading benchmark pairs"):
        class_name = base_file.stem.replace("(base)", "")
        marked_file = base_file.with_name(f"{class_name}(marked).txt")
        orig_file = base_file.with_name(f"{class_name}(orig).txt")
        if marked_file.exists() and orig_file.exists():
            with open(base_file, 'r', encoding='utf-8') as fb, \
                    open(marked_file, 'r', encoding='utf-8') as fm, \
                    open(orig_file, 'r', encoding='utf-8') as fo:
                bases, markeds, origs = fb.read().splitlines(), fm.read().splitlines(), fo.read().splitlines()
                for i in range(len(bases)):
                    orig_to_triples[origs[i]].append({"class": class_name, "data": (bases[i], markeds[i], origs[i])})

    benchmark_smiles = list(orig_to_triples.keys())
    test_smiles_raw = pd.read_csv(args.test_file)[args.smiles_column].unique().tolist()
    print(f"Loaded {len(benchmark_smiles)} unique molecules for benchmark.")
    print(f"Loaded {len(test_smiles_raw)} unique molecules for external testing.")

    # Step 2: Define initial pools
    print("\n--- Step 2: Defining initial Train, Validation, and Test pools ---")
    class_counts = defaultdict(int)
    for smi in benchmark_smiles:
        for item in orig_to_triples[smi]:
            class_counts[item['class']] += 1

    rare_classes = {cls for cls, count in class_counts.items() if count < args.rare_class_threshold}
    if rare_classes: print(f"Identified {len(rare_classes)} rare classes: {rare_classes}")

    train_pool_smiles, valid_pool_smiles = set(), set()
    remaining_benchmark = set(benchmark_smiles)

    for smi in benchmark_smiles:
        if any(item['class'] in rare_classes for item in orig_to_triples.get(smi, [])):
            if random.random() < args.train_ratio:
                train_pool_smiles.add(smi)
            else:
                valid_pool_smiles.add(smi)
            remaining_benchmark.remove(smi)

    scaffolds = defaultdict(list)
    for smi in tqdm(remaining_benchmark, desc="Generating Scaffolds"):
        scaffold = get_scaffold(smi)
        if scaffold: scaffolds[scaffold].append(smi)

    scaffold_list = list(scaffolds.keys())
    random.shuffle(scaffold_list)
    split_idx = int(len(scaffold_list) * args.train_ratio)
    train_scaffolds = set(scaffold_list[:split_idx])

    for scaffold, smi_list in scaffolds.items():
        if scaffold in train_scaffolds:
            train_pool_smiles.update(smi_list)
        else:
            valid_pool_smiles.update(smi_list)

    test_pool_smiles = set(test_smiles_raw)
    print(
        f"Initial pools created: {len(train_pool_smiles)} Train, {len(valid_pool_smiles)} Validation, {len(test_pool_smiles)} Test.")

    # Step 3: Build global blacklist
    print("\n--- Step 3: Building global blacklist from all pairwise comparisons ---")
    global_blacklist = set()
    pools_to_compare = [
        ("Train", "Valid", list(train_pool_smiles), list(valid_pool_smiles)),
        ("Train", "Test", list(train_pool_smiles), list(test_pool_smiles)),
        ("Valid", "Test", list(valid_pool_smiles), list(test_pool_smiles)),
    ]

    for name1, name2, pool1, pool2 in pools_to_compare:
        print(f"\nComparing {name1} vs {name2}...")
        fps1 = generate_fingerprints(pool1)
        fps2 = generate_fingerprints(pool2)

        data1 = [(smi, fp) for smi, fp in zip(pool1, fps1) if fp]
        data2 = [(smi, fp) for smi, fp in zip(pool2, fps2) if fp]

        query, ref = (data2, data1) if len(data1) > len(data2) else (data1, data2)
        ref_fps, ref_smiles = [item[1] for item in ref], [item[0] for item in ref]

        with Pool(processes=args.processes, initializer=init_worker, initargs=(ref_fps, ref_smiles)) as pool:
            tasks = [(item[1], item[0], args.similarity_threshold) for item in query]
            for pairs in tqdm(pool.imap_unordered(filter_worker, tasks), total=len(tasks),
                              desc=f"Filtering {name1}-{name2}"):
                for s1, s2 in pairs:
                    global_blacklist.add(s1)
                    global_blacklist.add(s2)

    print(f"\nBuilt global blacklist with {len(global_blacklist)} molecules.")

    # Step 4: Apply blacklist
    print("\n--- Step 4: Applying global blacklist to all pools ---")
    final_train = [s for s in train_pool_smiles if s not in global_blacklist]
    final_valid = [s for s in valid_pool_smiles if s not in global_blacklist]
    final_test = [s for s in test_pool_smiles if s not in global_blacklist]
    print(f"Final sizes: {len(final_train)} Train, {len(final_valid)} Validation, {len(final_test)} Test.")

    # Step 5: Generate files
    print("\n--- Step 5: Generating final dataset files ---")
    pd.DataFrame(final_train, columns=["SMILES"]).to_csv(output_path / "train_final.csv", index=False)
    pd.DataFrame(final_valid, columns=["SMILES"]).to_csv(output_path / "valid_final.csv", index=False)
    pd.DataFrame(final_test, columns=["SMILES"]).to_csv(output_path / "test_final.csv", index=False)

    # Stage I files
    stage1_path = output_path / "Stage_I";
    stage1_path.mkdir(exist_ok=True)
    with open(stage1_path / "train-src.txt", 'w', encoding='utf-8') as f_s1_train_src, \
            open(stage1_path / "train-tgt.txt", 'w', encoding='utf-8') as f_s1_train_tgt:
        for smi in tqdm(final_train, desc="Writing Stage I train"):
            for item in orig_to_triples.get(smi, []):
                base, marked, _ = item["data"]
                f_s1_train_src.write(f"{base}\n")
                f_s1_train_tgt.write(f"{marked}\n")

    with open(stage1_path / "valid-src.txt", 'w', encoding='utf-8') as f_s1_valid_src, \
            open(stage1_path / "valid-tgt.txt", 'w', encoding='utf-8') as f_s1_valid_tgt:
        for smi in tqdm(final_valid, desc="Writing Stage I valid"):
            for item in orig_to_triples.get(smi, []):
                base, marked, _ = item["data"]
                f_s1_valid_src.write(f"{base}\n")
                f_s1_valid_tgt.write(f"{marked}\n")

    # Stage II files
    stage2_path = output_path / "Stage_II"
    for cls in class_counts: (stage2_path / "train" / cls).mkdir(parents=True, exist_ok=True)

    train_files = {cls: {"src": open(stage2_path / f"train/{cls}/train-src.txt", 'w', encoding='utf-8'),
                         "tgt": open(stage2_path / f"train/{cls}/train-tgt.txt", 'w', encoding='utf-8')}
                   for cls in class_counts}
    for smi in tqdm(final_train, desc="Writing Stage II train"):
        for item in orig_to_triples.get(smi, []):
            cls, (_, marked, orig) = item["class"], item["data"]
            train_files[cls]["src"].write(f"{marked}\n")
            train_files[cls]["tgt"].write(f"{orig}\n")
    for cls in train_files: train_files[cls]["src"].close(); train_files[cls]["tgt"].close()

    (stage2_path / "valid").mkdir(exist_ok=True)
    with open(stage2_path / "valid/valid-src.txt", 'w', encoding='utf-8') as f_s2_valid_src, \
            open(stage2_path / "valid/valid-tgt.txt", 'w', encoding='utf-8') as f_s2_valid_tgt:
        for smi in tqdm(final_valid, desc="Writing Stage II valid"):
            for item in orig_to_triples.get(smi, []):
                _, marked, orig = item["data"]
                f_s2_valid_src.write(f"{marked}\n")
                f_s2_valid_tgt.write(f"{orig}\n")

    print("Train/Validation data files generated.")
    end_time = time.time()
    print(f"\nTotal time elapsed: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()

