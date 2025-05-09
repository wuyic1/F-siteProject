import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path
import sys # For printing info/errors

# --- Filter Settings ---
# Adjust thresholds as needed
MIN_HEAVY_ATOMS = 3     # Exclude very small fragments
MAX_HEAVY_ATOMS = 70    # Exclude very large molecules/polymers
MIN_MW = 50
MAX_MW = 800
MAX_FLUORINE_ATOMS = 25 # Arbitrary upper limit for F count
MAX_FLUORINE_RATIO = 0.75 # Max ratio F atoms / Heavy atoms
REMOVE_PAINS = True     # Use RDKit's PAINS filters

# Initialize PAINS filter catalog (do this once globally)
pains_catalog = None
if REMOVE_PAINS:
    try:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        pains_catalog = FilterCatalog.FilterCatalog(params)
        print("PAINS filter catalog initialized.")
    except Exception as e:
        print(f"Warning: Could not initialize PAINS filter catalog: {e}", file=sys.stderr)
        pains_catalog = None

# --- START: Definitions for Fluoro-Groups (10 Types) ---
# IMPORTANT: These definitions MUST be consistent across all scripts.

# --- Condition Functions (18 total) ---
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
    # Additional checks happen later in has_multiple_target_fluoro_groups
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

# Separate multi-atom groups from the single -F group
multi_atom_substructs = [s for s in substructures if s["name"] != "-F"]
f_substruct_info = next((s for s in substructures if s["name"] == "-F"), None)

# --- END: Definitions for Fluoro-Groups ---

# --- Filtering Logic ---

def calculate_properties(mol):
    """Calculates properties needed for filtering."""
    try:
        mw = Descriptors.MolWt(mol)
        hac = mol.GetNumHeavyAtoms()
        f_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
        f_ratio = f_count / hac if hac > 0 else 0
        return {"MW": mw, "HAC": hac, "F_count": f_count, "F_ratio": f_ratio}
    except Exception as e:
        # print(f"Error calculating properties: {e}", file=sys.stderr) # Optional
        return None

def check_property_filters(properties):
    """Checks if molecule passes property-based filters."""
    if properties is None: return False
    if not (MIN_HEAVY_ATOMS <= properties["HAC"] <= MAX_HEAVY_ATOMS): return False
    if not (MIN_MW <= properties["MW"] <= MAX_MW): return False
    if properties["F_count"] > MAX_FLUORINE_ATOMS: return False
    if properties["F_ratio"] > MAX_FLUORINE_RATIO: return False
    return True

def check_pains_filter(mol):
    """Checks if molecule triggers PAINS filters."""
    if not pains_catalog:
        return False # Filter not enabled or failed to load
    try:
        entry = pains_catalog.GetFirstMatch(mol)
        if entry:
             # print(f"PAINS Filter Match: {entry.GetDescription()}") # Optional debug
             return True # Found a PAINS match
        return False # No PAINS match
    except Exception as e:
        # print(f"Error during PAINS check: {e}", file=sys.stderr) # Optional
        return False # Treat error as non-match for safety, or True to filter out

def has_multiple_target_fluoro_groups(mol):
    """
    Checks if the molecule contains more than one instance
    of the *same* targeted fluoro-group type. Handles -F correctly.
    """
    multi_group_atom_indices = set() # Stores atom indices belonging to any identified multi-atom group

    # Check multi-atom groups first
    for substruct_info in multi_atom_substructs:
        group_name = substruct_info["name"]
        group_match_count = 0
        atoms_matched_this_group = set() # Track non-overlapping atoms for *this* group type

        for variant in substruct_info["variants"]:
            substruct_smarts = variant["smarts"]
            condition_func = variant.get("condition")
            substructure_pattern = Chem.MolFromSmarts(substruct_smarts)
            if not substructure_pattern: continue

            try:
                matches = mol.GetSubstructMatches(substructure_pattern)
            except Exception:
                continue # Ignore SMARTS matching errors

            for match_indices_tuple in matches:
                passes_condition = True
                if condition_func:
                    try:
                        if not condition_func(mol, match_indices_tuple): passes_condition = False
                    except Exception:
                        passes_condition = False # Ignore condition check errors

                if passes_condition:
                    current_match_indices = set(match_indices_tuple)
                    # Check for overlap with atoms already assigned to *this* specific group type
                    if not current_match_indices.intersection(atoms_matched_this_group):
                        group_match_count += 1
                        atoms_matched_this_group.update(current_match_indices)
                        # Add these atoms to the global set for the later -F check
                        multi_group_atom_indices.update(current_match_indices)

                        # If this specific multi-atom group appears more than once, filter immediately
                        if group_match_count > 1:
                            # print(f"Molecule has multiple '{group_name}' groups.") # Optional debug
                            return True

    # Check for multiple "independent" -F groups
    if f_substruct_info:
        f_group_count = 0
        f_variant = f_substruct_info["variants"][0] # Assumes only one variant for -F
        f_smarts = f_variant["smarts"]
        f_condition = f_variant.get("condition")
        f_pattern = Chem.MolFromSmarts(f_smarts)

        if f_pattern:
            try:
                f_matches = mol.GetSubstructMatches(f_pattern)
            except Exception:
                f_matches = []

            for f_match_tuple in f_matches: # Should be tuples like (f_idx,)
                f_idx = f_match_tuple[0]

                # Check 1: Is this F already part of a complex group identified above?
                if f_idx in multi_group_atom_indices:
                    continue # Skip this F, it belongs to a CF3, CF2H, etc.

                # Check 2: Does it pass the specific -F condition (e.g., terminal)?
                passes_condition = True
                if f_condition:
                    try:
                        if not f_condition(mol, f_match_tuple): passes_condition = False
                    except Exception:
                        passes_condition = False

                if passes_condition:
                    # This F is considered an "independent" -F group instance
                    f_group_count += 1

                    # If we find more than one independent -F, filter the molecule
                    if f_group_count > 1:
                        # print(f"Molecule has multiple independent '-F' groups.") # Optional debug
                        return True

    # If no filter triggered, the molecule passes
    return False

def filter_molecule(smiles):
    """
    Applies all filters to a single SMILES string.
    Returns the canonical SMILES if it passes all filters, otherwise None.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Apply Property Filters
        properties = calculate_properties(mol)
        if not check_property_filters(properties):
            return None

        # Apply PAINS Filter
        if REMOVE_PAINS and check_pains_filter(mol):
            return None

        # Apply Multiple *Same* Target Fluoro-Group Filter
        if has_multiple_target_fluoro_groups(mol):
            return None

        # If all filters pass, return the canonical SMILES
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    except Exception as e:
        print(f"Warning: General error filtering SMILES '{smiles}': {e}", file=sys.stderr)
        return None


def main(input_csv, output_csv):
    """
    Main function to read, filter, and save SMILES data.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.is_file():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return

    print(f"Reading standardized SMILES from: {input_path}")
    try:
        df = pd.read_csv(input_path)
        if 'SMILES' not in df.columns:
            print(f"Error: 'SMILES' column not found in {input_path}. Ensure the input CSV has a 'SMILES' header.", file=sys.stderr)
            return
        # Get unique SMILES to avoid redundant processing
        smiles_list = df['SMILES'].astype(str).unique().tolist()
        print(f"Read {len(smiles_list)} unique standardized SMILES.")
    except Exception as e:
        print(f"Error reading input CSV '{input_path}': {e}", file=sys.stderr)
        return

    if not smiles_list:
        print("Warning: No SMILES found in the input file.", file=sys.stderr)
        return

    num_processes = min(32, cpu_count()) # Use up to 32 cores
    print(f"Filtering SMILES using {num_processes} processes...")

    filtered_smiles_list = []
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(filter_molecule, smiles_list, chunksize=500),
                            total=len(smiles_list), desc="Filtering SMILES"))

    filtered_smiles_list = [s for s in results if s is not None]

    num_original = len(smiles_list)
    num_filtered = len(filtered_smiles_list)
    num_removed = num_original - num_filtered

    print("--- Filtering Summary ---")
    print(f"Original unique SMILES: {num_original}")
    print(f"SMILES after filtering: {num_filtered}")
    print(f"SMILES removed by filters: {num_removed}")

    if num_filtered > 0:
        print(f"Saving {num_filtered} filtered SMILES to: {output_path}")
        try:
            df_out = pd.DataFrame(filtered_smiles_list, columns=['SMILES'])
            df_out.to_csv(output_path, index=False)
            print("Filtering complete.")
        except IOError as e:
             print(f"Error writing output file {output_path}: {e}", file=sys.stderr)
    else:
        print("No SMILES passed the filters. No output file written.")


if __name__ == '__main__':
    # --- Configuration ---
    INPUT_CSV = "/ChEMBL/F-clear.csv" # ADJUST PATH AS NEEDED
    OUTPUT_CSV = "/ChEMBL/F-filter.csv" # ADJUST PATH AS NEEDED
    # --- End Configuration ---

    try:
        Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        main(INPUT_CSV, OUTPUT_CSV)
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}", file=sys.stderr)