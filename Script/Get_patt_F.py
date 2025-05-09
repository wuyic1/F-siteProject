import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path
import sys # For printing info/errors

# --- START: Definitions (Must match other scripts) ---

# --- Condition Functions (17 total) ---
# (These functions check specific chemical properties beyond the SMARTS match)
def condition_cfh2_sp3(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    return is_sp3 and h_count == 2 and f_count == 1 and not o_or_s

def condition_cfh2_sp2(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    is_sp2 = atom_c.GetHybridization() == Chem.HybridizationType.SP2
    has_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom_c.GetBonds())
    return is_sp2 and has_double_bond and h_count == 1 and f_count == 1 and not o_or_s

def condition_cf2h_sp3(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    non_f_heavy_neighbors = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() not in (1, 9))
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    return is_sp3 and h_count == 1 and f_count == 2 and non_f_heavy_neighbors == 1 and not o_or_s

def condition_cf2h_sp2(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    is_sp2 = atom_c.GetHybridization() == Chem.HybridizationType.SP2
    has_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom_c.GetBonds())
    heavy_neighbors_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() != 1)
    return is_sp2 and has_double_bond and h_count == 0 and f_count == 2 and heavy_neighbors_count == 3 and not o_or_s

def condition_cf3_sp3(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    non_f_heavy_neighbors = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() not in (1, 9))
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    return is_sp3 and f_count == 3 and non_f_heavy_neighbors == 1 and not o_or_s

def condition_ocf2h_sp3(mol, match):
    o_idx, carbon_idx = match[0], match[1]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    o_bond = mol.GetBondBetweenAtoms(o_idx, carbon_idx)
    is_single_o_bond = o_bond and o_bond.GetBondType() == Chem.BondType.SINGLE
    atom_o = mol.GetAtomWithIdx(o_idx)
    o_heavy_degree = sum(1 for n in atom_o.GetNeighbors() if n.GetAtomicNum() != 1)
    return is_sp3 and is_single_o_bond and o_heavy_degree == 2 and h_count == 1 and f_count == 2

def condition_ocf2h_sp2(mol, match):
    o_idx, carbon_idx = match[0], match[1]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    is_sp2 = atom_c.GetHybridization() == Chem.HybridizationType.SP2
    has_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom_c.GetBonds())
    o_bond = mol.GetBondBetweenAtoms(o_idx, carbon_idx)
    is_single_o_bond = o_bond and o_bond.GetBondType() == Chem.BondType.SINGLE
    atom_o = mol.GetAtomWithIdx(o_idx)
    o_heavy_degree = sum(1 for n in atom_o.GetNeighbors() if n.GetAtomicNum() != 1)
    return is_sp2 and has_double_bond and is_single_o_bond and o_heavy_degree == 2 and h_count == 0 and f_count == 2

def condition_ocf3_sp3(mol, match):
    o_idx, carbon_idx = match[0], match[1]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    o_bond = mol.GetBondBetweenAtoms(o_idx, carbon_idx)
    is_single_o_bond = o_bond and o_bond.GetBondType() == Chem.BondType.SINGLE
    atom_o = mol.GetAtomWithIdx(o_idx)
    o_heavy_degree = sum(1 for n in atom_o.GetNeighbors() if n.GetAtomicNum() != 1)
    return is_sp3 and is_single_o_bond and o_heavy_degree == 2 and f_count == 3

def condition_scf2h_sp3(mol, match):
    s_idx, carbon_idx = match[0], match[1]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    s_bond = mol.GetBondBetweenAtoms(s_idx, carbon_idx)
    is_single_s_bond = s_bond and s_bond.GetBondType() == Chem.BondType.SINGLE
    atom_s = mol.GetAtomWithIdx(s_idx)
    s_heavy_degree = sum(1 for n in atom_s.GetNeighbors() if n.GetAtomicNum() != 1)
    return is_sp3 and is_single_s_bond and s_heavy_degree == 2 and h_count == 1 and f_count == 2

def condition_scf2h_sp2(mol, match):
    s_idx, carbon_idx = match[0], match[1]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    is_sp2 = atom_c.GetHybridization() == Chem.HybridizationType.SP2
    has_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom_c.GetBonds())
    s_bond = mol.GetBondBetweenAtoms(s_idx, carbon_idx)
    is_single_s_bond = s_bond and s_bond.GetBondType() == Chem.BondType.SINGLE
    atom_s = mol.GetAtomWithIdx(s_idx)
    s_heavy_degree = sum(1 for n in atom_s.GetNeighbors() if n.GetAtomicNum() != 1)
    return is_sp2 and has_double_bond and is_single_s_bond and s_heavy_degree == 2 and h_count == 0 and f_count == 2

def condition_scf3_sp3(mol, match):
    s_idx, carbon_idx = match[0], match[1]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    s_bond = mol.GetBondBetweenAtoms(s_idx, carbon_idx)
    is_single_s_bond = s_bond and s_bond.GetBondType() == Chem.BondType.SINGLE
    atom_s = mol.GetAtomWithIdx(s_idx)
    s_heavy_degree = sum(1 for n in atom_s.GetNeighbors() if n.GetAtomicNum() != 1)
    return is_sp3 and is_single_s_bond and s_heavy_degree == 2 and f_count == 3

def condition_cfh_sp3(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s_neighbor = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    heavy_neighbors_count_excl_f = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() != 1 and n.GetIdx() != match[1])
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    return is_sp3 and h_count == 1 and f_count == 1 and heavy_neighbors_count_excl_f == 2 and not o_or_s_neighbor

def condition_cfh_sp2(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s_neighbor = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    is_sp2 = atom_c.GetHybridization() == Chem.HybridizationType.SP2
    has_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom_c.GetBonds())
    heavy_neighbors_count_excl_f = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() != 1 and n.GetIdx() != match[1])
    return is_sp2 and has_double_bond and h_count == 1 and f_count == 1 and heavy_neighbors_count_excl_f == 1 and not o_or_s_neighbor

def condition_cfh_sp(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s_neighbor = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    is_sp = atom_c.GetHybridization() == Chem.HybridizationType.SP
    has_triple_bond = any(bond.GetBondType() == Chem.BondType.TRIPLE for bond in atom_c.GetBonds())
    heavy_neighbors_count_excl_f = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() != 1 and n.GetIdx() != match[1])
    return is_sp and has_triple_bond and h_count == 0 and f_count == 1 and heavy_neighbors_count_excl_f == 1 and not o_or_s_neighbor

def condition_cf2_sp3(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s_neighbor = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    heavy_neighbors_count_excl_f = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() != 1 and n.GetIdx() not in match[1:])
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    return is_sp3 and h_count == 0 and f_count == 2 and heavy_neighbors_count_excl_f == 2 and not o_or_s_neighbor

def condition_cf2_sp2(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s_neighbor = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    is_sp2 = atom_c.GetHybridization() == Chem.HybridizationType.SP2
    has_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom_c.GetBonds())
    heavy_neighbors_count_excl_f = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() != 1 and n.GetIdx() not in match[1:])
    return is_sp2 and has_double_bond and h_count == 0 and f_count == 2 and heavy_neighbors_count_excl_f == 1 and not o_or_s_neighbor

def condition_f_terminal(mol, match):
    f_idx = match[0]
    atom_f = mol.GetAtomWithIdx(f_idx)
    if atom_f.GetDegree() != 1:
        return False
    return True

# --- Substructure Definitions (10 Types) ---
substructures = [
    {"name": "-CFH₂", "variants": [
        {"smarts": "[CH2;X4;!$(C=[O,S,N]);!$(C#N)](F)", "condition": condition_cfh2_sp3},
        {"smarts": "[CH1;X3;$(C=A)](F)", "condition": condition_cfh2_sp2}
    ]},
    {"name": "-CF₂H", "variants": [
        {"smarts": "[CH1;X4;!$(C=[O,S,N]);!$(C#N)](F)(F)", "condition": condition_cf2h_sp3},
        {"smarts": "[C;X3;H0;$(C=A)](F)(F)", "condition": condition_cf2h_sp2}
    ]},
    {"name": "-CF₃", "variants": [
        {"smarts": "[C;X4;!$(C=[O,S,N]);!$(C#N)](F)(F)(F)", "condition": condition_cf3_sp3}
    ]},
    {"name": "-OCF₂H", "variants": [
        {"smarts": "[O;X2;H0][CH1;X4](F)(F)", "condition": condition_ocf2h_sp3},
        {"smarts": "[O;X2;H0][C;X3;H0;$(C=A)](F)(F)", "condition": condition_ocf2h_sp2}
    ]},
    {"name": "-OCF₃", "variants": [
        {"smarts": "[O;X2;H0][C;X4](F)(F)(F)", "condition": condition_ocf3_sp3}
    ]},
    {"name": "-SCF₂H", "variants": [
        {"smarts": "[S;X2;H0][CH1;X4](F)(F)", "condition": condition_scf2h_sp3},
        {"smarts": "[S;X2;H0][C;X3;H0;$(C=A)](F)(F)", "condition": condition_scf2h_sp2}
    ]},
    {"name": "-SCF₃", "variants": [
        {"smarts": "[S;X2;H0][C;X4](F)(F)(F)", "condition": condition_scf3_sp3}
    ]},
    {"name": "-CFH-", "variants": [
        {"smarts": "[CH1;X4;!$(C=[O,S,N]);!$(C#N)](F)", "condition": condition_cfh_sp3},
        {"smarts": "[CH1;X3;$(C=A)](F)", "condition": condition_cfh_sp2},
        {"smarts": "[C;X2;H0;$(C#A)](F)", "condition": condition_cfh_sp}
    ]},
    {"name": "-CF₂-", "variants": [
        {"smarts": "[C;X4;H0;!$(C=[O,S,N]);!$(C#N)](F)(F)", "condition": condition_cf2_sp3},
        {"smarts": "[C;X3;H0;$(C=A)](F)(F)", "condition": condition_cf2_sp2}
    ]},
    {"name": "-F", "variants": [
        {"smarts": "[F;X1]", "condition": condition_f_terminal}
    ]},
]
# --- END: Definitions ---

def get_atoms_in_complex_groups(mol):
    """Helper function to find all atom indices belonging to the 9 complex fluoro-groups."""
    complex_group_atoms = set()
    # Iterate over all groups EXCEPT '-F'
    for substruct_info in substructures:
        if substruct_info['name'] == '-F':
            continue

        # group_name = substruct_info["name"] # Name not used here
        for variant in substruct_info["variants"]:
            substruct_smarts = variant["smarts"]
            condition_func = variant.get("condition")
            substructure_pattern = Chem.MolFromSmarts(substruct_smarts)
            if not substructure_pattern: continue

            try:
                matches = mol.GetSubstructMatches(substructure_pattern)
            except Exception:
                continue # Ignore matching errors

            for match_indices_tuple in matches:
                passes_condition = True
                if condition_func:
                    try:
                        if not condition_func(mol, match_indices_tuple): passes_condition = False
                    except Exception:
                        passes_condition = False # Ignore condition check errors

                if passes_condition:
                    complex_group_atoms.update(match_indices_tuple)
    return complex_group_atoms

# Worker function for multiprocessing
def worker(args):
    """
    Checks if a SMILES string contains the specified fluoro-group.
    Handles the '-F' case specifically to only match independent F atoms.
    Returns True if a valid match is found, False otherwise.
    """
    smiles, substruct_info = args
    group_name = substruct_info["name"]
    variants = substruct_info["variants"]

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return False

        # --- Special handling for '-F' group ---
        if group_name == '-F':
            complex_group_atoms = get_atoms_in_complex_groups(mol) # Find atoms in other groups
            f_variant = variants[0] # Assuming one variant for -F
            f_smarts = f_variant["smarts"]
            f_condition = f_variant.get("condition")
            f_pattern = Chem.MolFromSmarts(f_smarts)
            if not f_pattern: return False

            try:
                f_matches = mol.GetSubstructMatches(f_pattern)
            except Exception:
                return False # Error in matching

            for f_match_tuple in f_matches:
                f_idx = f_match_tuple[0]
                # Check 1: Is this F independent (not part of another complex group)?
                if f_idx not in complex_group_atoms:
                    # Check 2: Does it pass the terminal condition?
                    passes_condition = True
                    if f_condition:
                        try:
                            if not f_condition(mol, f_match_tuple): passes_condition = False
                        except Exception:
                            passes_condition = False # Error in condition check
                    # If independent and passes condition, molecule belongs to -F group
                    if passes_condition:
                        return True # Found at least one independent -F
            return False # No independent -F found

        # --- Standard handling for other groups ---
        else:
            for variant in variants:
                substruct_smarts = variant["smarts"]
                condition_func = variant.get("condition")
                substructure_pattern = Chem.MolFromSmarts(substruct_smarts)
                if not substructure_pattern: continue

                try:
                    matches = mol.GetSubstructMatches(substructure_pattern)
                except Exception:
                    continue # Error in matching

                if not matches: continue # No matches for this variant

                # If no condition function, any match is sufficient
                if condition_func is None:
                    return True

                # Check condition for each match
                for match in matches:
                    try:
                        if condition_func(mol, match):
                            return True # Condition met for this match
                    except Exception:
                        continue # Ignore condition errors for robustness
            return False # No variant matched successfully

    except Exception as e:
        print(f"Warning: Error processing SMILES '{smiles}' for group '{group_name}' in worker: {e}", file=sys.stderr)
        return False

# Main function
def main(input_csv, output_dir):
    """Finds molecules containing each fluoro-group and saves them to separate files."""
    input_path = Path(input_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Reading filtered SMILES from: {input_path}")
    try:
        data = pd.read_csv(input_path)
        if 'SMILES' not in data.columns:
             print(f"Error: 'SMILES' column not found in {input_path}.", file=sys.stderr)
             return
        smiles_list = data['SMILES'].astype(str).tolist()
        print(f"Loaded {len(smiles_list)} SMILES for substructure search.")
    except Exception as e:
        print(f"Error reading input CSV {input_path}: {e}", file=sys.stderr)
        return

    if not smiles_list:
        print("Warning: No SMILES loaded from input file.", file=sys.stderr)
        return

    num_workers = min(32, cpu_count()) # Use up to 32 cores
    print(f"Using {num_workers} processes for matching.")

    # Process each substructure group
    for substruct_info in substructures: # Now includes "-F"
        name = substruct_info["name"]
        print(f"Searching for group: {name}")

        # Prepare arguments for multiprocessing
        worker_args = [(smiles, substruct_info) for smiles in smiles_list]

        matched_smiles_for_group = []
        with Pool(num_workers) as pool:
            # The worker handles the logic for the specific group passed in args
            results = list(tqdm(pool.imap(worker, worker_args, chunksize=1000),
                                total=len(smiles_list),
                                desc=f"Matching {name}"))

        # Collect SMILES that had a match for this group
        for i, result in enumerate(results):
            if result:
                matched_smiles_for_group.append(smiles_list[i])

        # Save results for this group (e.g., -F.csv)
        output_csv_path = output_path / f"{name}.csv"
        count = len(matched_smiles_for_group)
        if count > 0:
            try:
                pd.DataFrame(matched_smiles_for_group, columns=['SMILES']).to_csv(output_csv_path, index=False, header=True)
                print(f"Found {count} molecules containing {name}. Saved to {output_csv_path}")
            except IOError as e:
                print(f"Error: Could not write output file {output_csv_path}: {e}", file=sys.stderr)
        else:
            print(f"Found 0 molecules containing {name}.")


if __name__ == "__main__":
    # --- Configuration ---
    INPUT_CSV = "/ChEMBL/F-filter.csv" # ADJUST PATH AS NEEDED
    OUTPUT_DIR = "/ChEMBL/F_sub"       # ADJUST PATH AS NEEDED
    # --- End Configuration ---

    try:
        main(INPUT_CSV, OUTPUT_DIR)
        print("Substructure matching finished.")
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}", file=sys.stderr)