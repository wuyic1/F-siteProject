import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from pathlib import Path
import sys # For printing info/errors

# --- Marker Definition ---
MARKER_ATOM_SYMBOL = "Se"  # Temporary atom symbol used during processing
MARKER_SMILES_TOKEN_TEMP = "[Se]" # SMILES representation of the temporary marker
MARKER_ATOM_SMARTS_FINAL = "[*]" # Final wildcard marker in output SMILES

# --- START: COMPLETE Definitions (Copied from Get_patt_F.py) ---
# (Includes all 17 condition functions and substructure definitions)

# --- Condition Functions (17 total) ---
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

# --- Map condition function names to functions ---
condition_map = {
    "condition_cfh2_sp3": condition_cfh2_sp3, "condition_cfh2_sp2": condition_cfh2_sp2,
    "condition_cf2h_sp3": condition_cf2h_sp3, "condition_cf2h_sp2": condition_cf2h_sp2,
    "condition_cf3_sp3": condition_cf3_sp3,
    "condition_ocf2h_sp3": condition_ocf2h_sp3, "condition_ocf2h_sp2": condition_ocf2h_sp2,
    "condition_ocf3_sp3": condition_ocf3_sp3,
    "condition_scf2h_sp3": condition_scf2h_sp3, "condition_scf2h_sp2": condition_scf2h_sp2,
    "condition_scf3_sp3": condition_scf3_sp3,
    "condition_cfh_sp3": condition_cfh_sp3, "condition_cfh_sp2": condition_cfh_sp2, "condition_cfh_sp": condition_cfh_sp,
    "condition_cf2_sp3": condition_cf2_sp3, "condition_cf2_sp2": condition_cf2_sp2,
    "condition_f_terminal": condition_f_terminal
}

# --- COMPLETE Substructure Definitions (10 Types with condition_name) ---
substructure_defs = [
    {"name": "-CFH₂", "variants": [
        {"smarts": "[CH2;X4;!$(C=[O,S,N]);!$(C#N)](F)", "condition_name": "condition_cfh2_sp3"},
        {"smarts": "[CH1;X3;$(C=A)](F)", "condition_name": "condition_cfh2_sp2"}
    ]},
    {"name": "-CF₂H", "variants": [
        {"smarts": "[CH1;X4;!$(C=[O,S,N]);!$(C#N)](F)(F)", "condition_name": "condition_cf2h_sp3"},
        {"smarts": "[C;X3;H0;$(C=A)](F)(F)", "condition_name": "condition_cf2h_sp2"}
    ]},
    {"name": "-CF₃", "variants": [
        {"smarts": "[C;X4;!$(C=[O,S,N]);!$(C#N)](F)(F)(F)", "condition_name": "condition_cf3_sp3"}
    ]},
    {"name": "-OCF₂H", "variants": [
        {"smarts": "[O;X2;H0][CH1;X4](F)(F)", "condition_name": "condition_ocf2h_sp3"},
        {"smarts": "[O;X2;H0][C;X3;H0;$(C=A)](F)(F)", "condition_name": "condition_ocf2h_sp2"}
    ]},
    {"name": "-OCF₃", "variants": [
        {"smarts": "[O;X2;H0][C;X4](F)(F)(F)", "condition_name": "condition_ocf3_sp3"}
    ]},
    {"name": "-SCF₂H", "variants": [
        {"smarts": "[S;X2;H0][CH1;X4](F)(F)", "condition_name": "condition_scf2h_sp3"},
        {"smarts": "[S;X2;H0][C;X3;H0;$(C=A)](F)(F)", "condition_name": "condition_scf2h_sp2"}
    ]},
    {"name": "-SCF₃", "variants": [
        {"smarts": "[S;X2;H0][C;X4](F)(F)(F)", "condition_name": "condition_scf3_sp3"}
    ]},
    {"name": "-CFH-", "variants": [
        {"smarts": "[CH1;X4;!$(C=[O,S,N]);!$(C#N)](F)", "condition_name": "condition_cfh_sp3"},
        {"smarts": "[CH1;X3;$(C=A)](F)", "condition_name": "condition_cfh_sp2"},
        {"smarts": "[C;X2;H0;$(C#A)](F)", "condition_name": "condition_cfh_sp"}
    ]},
    {"name": "-CF₂-", "variants": [
        {"smarts": "[C;X4;H0;!$(C=[O,S,N]);!$(C#N)](F)(F)", "condition_name": "condition_cf2_sp3"},
        {"smarts": "[C;X3;H0;$(C=A)](F)(F)", "condition_name": "condition_cf2_sp2"}
    ]},
    {"name": "-F", "variants": [
        {"smarts": "[F;X1]", "condition_name": "condition_f_terminal"}
    ]},
]
# Create a map for easy lookup by name
substructure_map = {s['name']: s for s in substructure_defs}
# --- END: Definitions ---


# --- Helper function ---
def get_atoms_in_complex_groups(mol):
    """Helper function to find all atom indices belonging to the 9 complex fluoro-groups."""
    complex_group_atoms = set()
    for s_info in substructure_defs:
        if s_info['name'] == '-F': continue
        # group_name = s_info["name"] # Not used here
        for variant in s_info["variants"]:
            substruct_smarts = variant["smarts"]
            condition_func_name = variant.get("condition_name")
            condition_func = condition_map.get(condition_func_name)
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
                        passes_condition = False # Ignore condition errors
                if passes_condition:
                    complex_group_atoms.update(match_indices_tuple)
    return complex_group_atoms

# --- Function to find matches ---
def get_fluoro_group_info(mol, substructure_info):
    """
    Finds VALID matches for the specific target group for cutting/marking. Handles '-F' specifically.
    Returns a list of tuples: (match_indices_tuple, attachment_atom_idx, fluoro_group_atom_indices, bond_type_to_break)
    """
    fluoro_group_matches = []
    name = substructure_info['name']
    variants = substructure_info['variants']
    processed_matches_tuples = set() # Avoid processing the same match tuple multiple times

    if name == '-F':
        complex_group_atoms = get_atoms_in_complex_groups(mol)
        f_variant = variants[0]
        f_smarts = f_variant["smarts"]
        f_condition_name = f_variant.get("condition_name")
        f_condition = condition_map.get(f_condition_name)
        f_pattern = Chem.MolFromSmarts(f_smarts)
        if not f_pattern: return []
        try:
            f_matches = mol.GetSubstructMatches(f_pattern)
        except Exception:
            f_matches = []

        for f_match_tuple in f_matches:
            if f_match_tuple in processed_matches_tuples: continue
            f_idx = f_match_tuple[0]
            if f_idx in complex_group_atoms: continue # Skip if F is part of a complex group
            passes_condition = True
            if f_condition:
                try:
                    if not f_condition(mol, f_match_tuple): passes_condition = False
                except Exception:
                    passes_condition = False
            if passes_condition:
                try:
                    atom_f = mol.GetAtomWithIdx(f_idx)
                    if atom_f.GetDegree() == 1:
                        neighbor = atom_f.GetNeighbors()[0]
                        attachment_atom_idx = neighbor.GetIdx()
                        fluoro_group_atom_indices = {f_idx}
                        bond = mol.GetBondBetweenAtoms(f_idx, attachment_atom_idx)
                        bond_type = bond.GetBondType() if bond else Chem.BondType.SINGLE
                        fluoro_group_matches.append(
                            (f_match_tuple, attachment_atom_idx, list(fluoro_group_atom_indices), bond_type)
                        )
                        processed_matches_tuples.add(f_match_tuple)
                except Exception as e:
                    # print(f"Debug: Error getting neighbor/bond for independent F match {f_match_tuple}: {e}", file=sys.stderr)
                    continue
    else: # Handling for complex groups
        for variant in variants:
            substructure_smarts = variant['smarts']
            condition_func_name = variant.get("condition_name")
            condition_func = condition_map.get(condition_func_name)
            substructure = Chem.MolFromSmarts(substructure_smarts)
            if not substructure: continue
            try:
                matches = mol.GetSubstructMatches(substructure)
            except Exception:
                continue # Ignore matching errors

            for match_indices_tuple in matches:
                if match_indices_tuple in processed_matches_tuples: continue
                if condition_func:
                    try:
                        if not condition_func(mol, match_indices_tuple): continue
                    except Exception:
                        continue # Ignore condition check errors

                # Determine attachment point and atoms to remove based on group type
                attachment_atom_idx = -1
                fluoro_group_atom_indices = set()
                bond_to_break_type = Chem.BondType.SINGLE

                try:
                    # Logic to find the attachment point and the atoms forming the fluoro group
                    # This depends heavily on the SMARTS pattern and the group structure
                    if name in ["-CF₃", "-CF₂H", "-CFH₂"]: # C-based terminal groups
                        c_idx = match_indices_tuple[0]
                        fluoro_group_atom_indices = set(match_indices_tuple) # C and attached F(s)
                        for neighbor in mol.GetAtomWithIdx(c_idx).GetNeighbors():
                             if neighbor.GetAtomicNum() != 9 and neighbor.GetAtomicNum() != 1: # Find non-F, non-H neighbor
                                attachment_atom_idx = neighbor.GetIdx()
                                bond = mol.GetBondBetweenAtoms(c_idx, attachment_atom_idx)
                                if bond: bond_to_break_type = bond.GetBondType()
                                break
                    elif name in ["-OCF₃", "-SCF₃", "-OCF₂H", "-SCF₂H"]: # Heteroatom-linked groups
                        hetero_idx, c_idx = match_indices_tuple[0], match_indices_tuple[1]
                        fluoro_group_atom_indices = set(match_indices_tuple) # Heteroatom, C, F(s)
                        for neighbor in mol.GetAtomWithIdx(hetero_idx).GetNeighbors():
                            if neighbor.GetIdx() != c_idx: # Find the neighbor *not* part of the fluoro group C
                                attachment_atom_idx = neighbor.GetIdx()
                                bond = mol.GetBondBetweenAtoms(hetero_idx, attachment_atom_idx)
                                if bond: bond_to_break_type = bond.GetBondType()
                                break
                    elif name == "-CFH-": # Internal CFH group
                        c_idx, f_idx = match_indices_tuple[0], match_indices_tuple[1]
                        fluoro_group_atom_indices = {f_idx} # Remove only the F atom
                        attachment_atom_idx = c_idx # Attach marker/H to the Carbon
                        bond = mol.GetBondBetweenAtoms(c_idx, f_idx)
                        if bond: bond_to_break_type = bond.GetBondType()
                    elif name == "-CF₂-": # Internal CF2 group
                        c_idx = match_indices_tuple[0]
                        f1_idx, f2_idx = match_indices_tuple[1], match_indices_tuple[2]
                        fluoro_group_atom_indices = {f1_idx, f2_idx} # Remove only the two F atoms
                        attachment_atom_idx = c_idx # Attach marker/H to the Carbon
                        bond1 = mol.GetBondBetweenAtoms(c_idx, f1_idx) # Use bond type of one C-F bond
                        if bond1: bond_to_break_type = bond1.GetBondType()
                    else:
                         print(f"Warning: Unhandled group name '{name}' in get_fluoro_group_info logic.", file=sys.stderr)
                         continue

                except Exception as e:
                    # print(f"Debug: Error finding attach/group atoms for {name}, match {match_indices_tuple}: {e}", file=sys.stderr)
                    continue

                # Validation checks
                if attachment_atom_idx != -1 and fluoro_group_atom_indices:
                    if attachment_atom_idx in fluoro_group_atom_indices:
                        # print(f"Debug: Attach point {attachment_atom_idx} in group atoms {fluoro_group_atom_indices} for {name}. Skipping.", file=sys.stderr)
                        continue
                    fluoro_group_matches.append(
                        (match_indices_tuple, attachment_atom_idx, list(fluoro_group_atom_indices), bond_to_break_type)
                    )
                    processed_matches_tuples.add(match_indices_tuple) # Mark this specific SMARTS match as processed

    return fluoro_group_matches

# --- Function to generate modified SMILES (with explicit marker atom properties) ---
def generate_modified_smiles(original_smiles, substructure_info):
    """
    Generates (base_smiles, marked_smiles, original_smiles) triples.
    Uses a temporary real atom marker ([Se]) and replaces it with the final marker ([*]).
    Returns a list of valid triples.
    """
    if not original_smiles or not isinstance(original_smiles, str): return []
    try:
        mol = Chem.MolFromSmiles(original_smiles)
        if not mol:
            # print(f"Debug: Skipping invalid input SMILES: {original_smiles}", file=sys.stderr)
            return []

        match_details_list = get_fluoro_group_info(mol, substructure_info)
        if not match_details_list:
            # print(f"Debug: No valid matches found for group '{substructure_info['name']}' in SMILES: {original_smiles}", file=sys.stderr)
            return []

        results = []
        MAX_CUTS_PER_MOL = 5 # Limit cuts per molecule to avoid excessive fragmentation
        cuts_done = 0

        try:
            # Ensure we have a canonical version of the original SMILES for comparison
            original_smiles_canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        except Exception as e:
             print(f"Warning: Failed to canonicalize original SMILES: {original_smiles}. Error: {e}", file=sys.stderr)
             return [] # Cannot proceed without canonical original

        for match_indices_tuple, attachment_idx, group_indices, bond_type in match_details_list:
            if cuts_done >= MAX_CUTS_PER_MOL:
                 # print(f"Debug: Reached max cuts ({MAX_CUTS_PER_MOL}) for SMILES: {original_smiles}", file=sys.stderr)
                 break

            base_smiles = None
            marked_smiles_final = None

            # --- Create Marked Molecule with Temporary Marker ([Se]) ---
            marked_smiles_temp = None
            try:
                edit_mol_marked = Chem.RWMol(mol)
                marker_atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(MARKER_ATOM_SYMBOL)
                marker_idx = edit_mol_marked.AddAtom(Chem.Atom(marker_atomic_num))

                # Set explicit properties for the marker atom to avoid implicit hydrogens etc.
                marker_atom = edit_mol_marked.GetAtomWithIdx(marker_idx)
                marker_atom.SetNoImplicit(True)
                marker_atom.SetNumExplicitHs(0)
                # marker_atom.SetFormalCharge(0) # Usually default is 0

                # Ensure bond type is valid before adding
                valid_bond_type = bond_type if bond_type in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC] else Chem.BondType.SINGLE
                edit_mol_marked.AddBond(attachment_idx, marker_idx, valid_bond_type)

                # Validate and remove group atoms carefully
                max_idx_mol = mol.GetNumAtoms() - 1
                valid_group_indices = [idx for idx in group_indices if 0 <= idx <= max_idx_mol]
                if len(valid_group_indices) != len(group_indices):
                    raise ValueError("Invalid atom index found in group_indices")

                indices_to_remove_sorted = sorted(valid_group_indices, reverse=True)
                for atom_idx in indices_to_remove_sorted:
                    if atom_idx < edit_mol_marked.GetNumAtoms():
                         edit_mol_marked.RemoveAtom(atom_idx)
                    else:
                         # This should ideally not happen if indices are validated first
                         raise ValueError("Atom index out of bounds during removal")

                # Sanitize AFTER modifications
                Chem.SanitizeMol(edit_mol_marked)
                marked_mol = edit_mol_marked.GetMol()

                # Check for fragmentation
                if len(rdmolops.GetMolFrags(marked_mol)) > 1:
                    # print(f"Debug: Marked molecule fragmented for {original_smiles}, match {match_indices_tuple}", file=sys.stderr)
                    continue # Skip fragmented results

                # Generate SMILES with temporary marker
                marked_smiles_temp = Chem.MolToSmiles(marked_mol, isomericSmiles=True, canonical=True)

                # Replace temporary marker with final marker and validate
                if marked_smiles_temp and MARKER_SMILES_TOKEN_TEMP in marked_smiles_temp:
                    marked_smiles_final = marked_smiles_temp.replace(MARKER_SMILES_TOKEN_TEMP, MARKER_ATOM_SMARTS_FINAL)
                    # Check if the final SMILES is valid
                    mol_check = Chem.MolFromSmiles(marked_smiles_final, sanitize=True)
                    if mol_check is None:
                         # print(f"Debug: Final marked SMILES {marked_smiles_final} failed RDKit parsing/sanitization for {original_smiles}", file=sys.stderr)
                         marked_smiles_final = None
                else:
                    # This indicates a potential issue in the marking process
                    print(f"Warning: Temporary marker {MARKER_SMILES_TOKEN_TEMP} missing in intermediate marked SMILES '{marked_smiles_temp}' for {original_smiles}", file=sys.stderr)
                    marked_smiles_final = None

            except ValueError as ve:
                # print(f"Debug: ValueError during Marked SMILES gen for {original_smiles}, match {match_indices_tuple}: {ve}", file=sys.stderr)
                marked_smiles_final = None
            except Exception as e:
                # print(f"Debug: Marked SMILES failed RDKit op for {original_smiles}, match {match_indices_tuple}. Error: {e}", file=sys.stderr)
                marked_smiles_final = None


            # --- Create Base SMILES (Only if marked SMILES was successful) ---
            if marked_smiles_final:
                try:
                    edit_mol_base = Chem.RWMol(mol)
                    h_idx = edit_mol_base.AddAtom(Chem.Atom(1)) # Add a Hydrogen atom
                    # Ensure bond type is valid
                    valid_bond_type = bond_type if bond_type in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC] else Chem.BondType.SINGLE
                    edit_mol_base.AddBond(attachment_idx, h_idx, valid_bond_type)

                    # Validate and remove group atoms (same logic as for marked)
                    max_idx_mol = mol.GetNumAtoms() - 1
                    valid_group_indices = [idx for idx in group_indices if 0 <= idx <= max_idx_mol]
                    if len(valid_group_indices) != len(group_indices):
                        raise ValueError("Invalid atom index found in group_indices for base mol")

                    indices_to_remove_sorted = sorted(valid_group_indices, reverse=True)
                    for atom_idx in indices_to_remove_sorted:
                         if atom_idx < edit_mol_base.GetNumAtoms():
                             edit_mol_base.RemoveAtom(atom_idx)
                         else:
                             raise ValueError("Atom index out of bounds during removal for base mol")

                    Chem.SanitizeMol(edit_mol_base)
                    base_mol = edit_mol_base.GetMol()

                    # Check for fragmentation
                    if len(rdmolops.GetMolFrags(base_mol)) > 1:
                         # print(f"Debug: Base molecule fragmented for {original_smiles}, match {match_indices_tuple}", file=sys.stderr)
                         continue # Skip fragmented results

                    # Remove explicit Hs added during modification, sanitize again
                    base_mol_clean = rdmolops.RemoveHs(base_mol, sanitize=True)
                    if base_mol_clean is None:
                        # print(f"Debug: RemoveHs failed for base mol of {original_smiles}, trying without RemoveHs.", file=sys.stderr)
                        try:
                            Chem.SanitizeMol(base_mol) # Ensure original base mol is valid
                            base_mol_clean = base_mol # Use the version with potentially explicit H
                        except Exception as sanitize_err:
                             # print(f"Debug: Sanitization failed even on original base_mol for {original_smiles}: {sanitize_err}", file=sys.stderr)
                             base_mol_clean = None # Cannot get a valid base molecule

                    if base_mol_clean:
                        base_smiles = Chem.MolToSmiles(base_mol_clean, isomericSmiles=True, canonical=True)
                    else:
                         base_smiles = None # Failed to get a valid base SMILES

                except ValueError as ve:
                    # print(f"Debug: ValueError during Base SMILES gen for {original_smiles}, match {match_indices_tuple}: {ve}", file=sys.stderr)
                    base_smiles = None
                except Exception as e:
                    # print(f"Debug: Base SMILES failed RDKit op for {original_smiles}, match {match_indices_tuple}. Error: {e}", file=sys.stderr)
                    base_smiles = None

            # --- Append the valid triple ---
            if base_smiles and marked_smiles_final:
                # Ensure generated SMILES are different from the original
                if base_smiles != original_smiles_canonical and marked_smiles_final != original_smiles_canonical:
                    results.append((base_smiles, marked_smiles_final, original_smiles_canonical))
                    cuts_done += 1
                # else:
                    # print(f"Debug: Generated base/marked identical to original for {original_smiles}, match {match_indices_tuple}. Skipping.", file=sys.stderr)
            # elif marked_smiles_final and not base_smiles:
                 # print(f"Debug: Marked SMILES generated but Base SMILES failed for {original_smiles}, match {match_indices_tuple}", file=sys.stderr)
            # elif not marked_smiles_final and not base_smiles:
                 # print(f"Debug: Both Marked and Base SMILES failed for {original_smiles}, match {match_indices_tuple}", file=sys.stderr)

        return results # List of valid (base, marked, orig) triples
    except Exception as e:
        print(f"Warning: General error processing {original_smiles} in generate_modified_smiles: {e}", file=sys.stderr)
        return []


# --- Process a single file function ---
def process_file(input_file_path, output_dir_path):
    """Processes a single CSV file containing SMILES for one substructure group."""
    sub_name = input_file_path.stem # Get filename without extension
    sub_info = substructure_map.get(sub_name)
    # Handle potential differences in filename format (e.g., _ vs -)
    if not sub_info:
        sub_name_alt = sub_name.replace('_', '-')
        sub_info = substructure_map.get(sub_name_alt)
    if not sub_info:
        sub_name_alt = sub_name.replace('-', '_')
        sub_info = substructure_map.get(sub_name_alt)

    if not sub_info:
        print(f"Warning: No definition found for substructure '{sub_name}' (or variations). Skipping file {input_file_path}", file=sys.stderr)
        return

    print(f"Processing file: {input_file_path} for target group: {sub_info['name']}")
    output_base = output_dir_path / f"{sub_name}(base).txt"
    output_marked = output_dir_path / f"{sub_name}(marked).txt"
    output_orig = output_dir_path / f"{sub_name}(orig).txt"

    try:
        smiles_data = pd.read_csv(input_file_path)
        if 'SMILES' not in smiles_data.columns:
             print(f"Error: 'SMILES' column not found in {input_file_path}.", file=sys.stderr)
             return
        # Read unique SMILES to avoid redundant processing if duplicates exist
        smiles_list = smiles_data['SMILES'].drop_duplicates().astype(str).tolist()
        print(f"Read {len(smiles_list)} unique SMILES from {input_file_path.name}.")
    except Exception as e:
        print(f"Error reading or processing SMILES column from {input_file_path}: {e}", file=sys.stderr)
        return

    if not smiles_list:
         print(f"Warning: No SMILES found in {input_file_path} after processing. Skipping.", file=sys.stderr)
         return

    base_data, marked_data, orig_data = [], [], []
    num_processes = min(32, cpu_count()) # Adjust max cores if needed
    # Create a partial function with substructure_info fixed
    processor = partial(generate_modified_smiles, substructure_info=sub_info)

    print(f"Starting multiprocessing pool ({num_processes} workers) for {len(smiles_list)} SMILES...")
    with Pool(num_processes) as pool:
        # imap processes iterables lazily, chunksize improves performance
        results_list = list(tqdm(pool.imap(processor, smiles_list, chunksize=100),
                                total=len(smiles_list), desc=f"Generating pairs for {sub_info['name']}"))

    processed_count = 0
    valid_original_smiles_processed = set() # Track unique originals processed
    for result_triples in results_list: # Each item in results_list is a list of triples from one original SMILES
        if result_triples:
            # Add the original SMILES from the first triple (they are all the same)
            first_original = result_triples[0][2]
            valid_original_smiles_processed.add(first_original)
            # Add all valid generated triples
            for base_smi, marked_smi, orig_smi in result_triples:
                if base_smi and marked_smi and orig_smi:
                    base_data.append(base_smi)
                    marked_data.append(marked_smi)
                    orig_data.append(orig_smi)
                    processed_count += 1

    print(f"Finished processing {input_file_path.name}:")
    print(f"  - Generated {processed_count} valid (base, marked, orig) triples.")
    print(f"  - Corresponding to {len(valid_original_smiles_processed)} unique original SMILES.")

    if processed_count > 0:
        try:
            # Write output files
            with open(output_base, 'w', encoding='utf-8') as f: f.write('\n'.join(base_data))
            with open(output_marked, 'w', encoding='utf-8') as f: f.write('\n'.join(marked_data))
            with open(output_orig, 'w', encoding='utf-8') as f: f.write('\n'.join(orig_data))
            print(f"Successfully saved data for {sub_info['name']} to .txt files in {output_dir_path}")
        except IOError as e:
            print(f"Error writing output files for {sub_info['name']}: {e}", file=sys.stderr)
    else:
        print(f"Warning: No valid triples were generated for group {sub_info['name']} from file {input_file_path.name}. No output files created.", file=sys.stderr)


# --- main function ---
def main():
    input_dir = Path("/ChEMBL/F_sub") # ADJUST PATH
    output_dir = Path("/ChEMBL/F_pair") # ADJUST PATH

    print(f"Starting Cut_save_F script.")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
    # Find all CSV files in the input directory
    csv_files = list(input_dir.glob('*.csv'))
    if not csv_files:
        print(f"Error: No CSV files found in the specified input directory: {input_dir}", file=sys.stderr)
        return

    print(f"Found {len(csv_files)} CSV files to process in {input_dir}.")
    # Process each file found
    for csv_file in csv_files:
        process_file(csv_file, output_dir)

    print("All file processing finished.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}", file=sys.stderr)