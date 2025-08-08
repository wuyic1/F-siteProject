import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog
from multiprocessing import Pool, cpu_count
import logging
from tqdm import tqdm
from pathlib import Path
import argparse

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Filter Settings ---
MIN_HEAVY_ATOMS = 3  # Exclude very small fragments
MAX_HEAVY_ATOMS = 70  # Exclude very large molecules/polymers
MIN_MW = 50
MAX_MW = 800
MAX_FLUORINE_ATOMS = 25  # Arbitrary upper limit for F count
MAX_FLUORINE_RATIO = 0.75  # Max ratio of F atoms to Heavy atoms
REMOVE_PAINS = True  # Use RDKit's PAINS filters

# Initialize PAINS filter catalog (do this once globally)
if REMOVE_PAINS:
    try:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        pains_catalog = FilterCatalog.FilterCatalog(params)
        logging.info("PAINS filter catalog initialized.")
    except Exception as e:
        logging.error(f"Could not initialize PAINS filter catalog: {e}")
        pains_catalog = None
else:
    pains_catalog = None


# --- START: Definitions for Fluoro-Groups (10 Types) ---
# IMPORTANT: These definitions MUST be consistent across all related scripts.

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
    return is_sp2 and has_double_bond and h_count == 0 and f_count == 2 and not o_or_s


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
    o_neighbors_count = atom_o.GetDegree()
    return is_sp3 and is_single_o_bond and o_neighbors_count == 2 and h_count == 1 and f_count == 2


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
    o_neighbors_count = atom_o.GetDegree()
    return is_sp2 and has_double_bond and is_single_o_bond and o_neighbors_count == 2 and h_count == 0 and f_count == 2


def condition_ocf3_sp3(mol, match):
    o_idx, carbon_idx = match[0], match[1]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    o_bond = mol.GetBondBetweenAtoms(o_idx, carbon_idx)
    is_single_o_bond = o_bond and o_bond.GetBondType() == Chem.BondType.SINGLE
    atom_o = mol.GetAtomWithIdx(o_idx)
    o_neighbors_count = atom_o.GetDegree()
    return is_sp3 and is_single_o_bond and o_neighbors_count == 2 and f_count == 3


def condition_scf2h_sp3(mol, match):
    s_idx, carbon_idx = match[0], match[1]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    s_bond = mol.GetBondBetweenAtoms(s_idx, carbon_idx)
    is_single_s_bond = s_bond and s_bond.GetBondType() == Chem.BondType.SINGLE
    atom_s = mol.GetAtomWithIdx(s_idx)
    s_neighbors_count = atom_s.GetDegree()
    return is_sp3 and is_single_s_bond and s_neighbors_count == 2 and h_count == 1 and f_count == 2


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
    s_neighbors_count = atom_s.GetDegree()
    return is_sp2 and has_double_bond and is_single_s_bond and s_neighbors_count == 2 and h_count == 0 and f_count == 2


def condition_scf3_sp3(mol, match):
    s_idx, carbon_idx = match[0], match[1]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    s_bond = mol.GetBondBetweenAtoms(s_idx, carbon_idx)
    is_single_s_bond = s_bond and s_bond.GetBondType() == Chem.BondType.SINGLE
    atom_s = mol.GetAtomWithIdx(s_idx)
    s_neighbors_count = atom_s.GetDegree()
    return is_sp3 and is_single_s_bond and s_neighbors_count == 2 and f_count == 3


def condition_cfh_sp3(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s_neighbor = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    heavy_neighbors_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() != 1)
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    return is_sp3 and h_count == 1 and f_count == 1 and heavy_neighbors_count == 3 and not o_or_s_neighbor


def condition_cfh_sp2(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s_neighbor = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    is_sp2 = atom_c.GetHybridization() == Chem.HybridizationType.SP2
    has_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom_c.GetBonds())
    heavy_neighbors_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() != 1)
    return is_sp2 and has_double_bond and h_count == 1 and f_count == 1 and heavy_neighbors_count == 2 and not o_or_s_neighbor


def condition_cfh_sp(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s_neighbor = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    is_sp = atom_c.GetHybridization() == Chem.HybridizationType.SP
    has_triple_bond = any(bond.GetBondType() == Chem.BondType.TRIPLE for bond in atom_c.GetBonds())
    heavy_neighbors_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() != 1)
    return is_sp and has_triple_bond and h_count == 0 and f_count == 1 and heavy_neighbors_count == 2 and not o_or_s_neighbor


def condition_cf2_sp3(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s_neighbor = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    heavy_neighbors_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() != 1)
    is_sp3 = atom_c.GetHybridization() == Chem.HybridizationType.SP3
    return is_sp3 and h_count == 0 and f_count == 2 and heavy_neighbors_count == 4 and not o_or_s_neighbor


def condition_cf2_sp2(mol, match):
    carbon_idx = match[0]
    atom_c = mol.GetAtomWithIdx(carbon_idx)
    h_count = atom_c.GetTotalNumHs(includeNeighbors=True)
    f_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() == 9)
    o_or_s_neighbor = any(n.GetAtomicNum() in (8, 16) for n in atom_c.GetNeighbors())
    is_sp2 = atom_c.GetHybridization() == Chem.HybridizationType.SP2
    has_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom_c.GetBonds())
    heavy_neighbors_count = sum(1 for n in atom_c.GetNeighbors() if n.GetAtomicNum() != 1)
    return is_sp2 and has_double_bond and h_count == 0 and f_count == 2 and heavy_neighbors_count == 3 and not o_or_s_neighbor


def condition_f_terminal(mol, match):
    f_idx = match[0]
    atom_f = mol.GetAtomWithIdx(f_idx)
    if atom_f.GetDegree() != 1:
        return False
    return True


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

multi_atom_substructs = [s for s in substructures if s["name"] != "-F"]
f_substruct_info = next((s for s in substructures if s["name"] == "-F"), None)


# --- END: Definitions for Fluoro-Groups ---


def calculate_properties(mol):
    """Calculates properties needed for filtering."""
    try:
        mw = Descriptors.MolWt(mol)
        hac = mol.GetNumHeavyAtoms()
        f_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
        f_ratio = f_count / hac if hac > 0 else 0
        return {"MW": mw, "HAC": hac, "F_count": f_count, "F_ratio": f_ratio}
    except Exception as e:
        logging.debug(f"Error calculating properties: {e}")
        return None


def check_property_filters(properties):
    """Checks if a molecule passes property-based filters."""
    if properties is None: return False
    if not (MIN_HEAVY_ATOMS <= properties["HAC"] <= MAX_HEAVY_ATOMS): return False
    if not (MIN_MW <= properties["MW"] <= MAX_MW): return False
    if properties["F_count"] > MAX_FLUORINE_ATOMS: return False
    if properties["F_ratio"] > MAX_FLUORINE_RATIO: return False
    return True


def check_pains_filter(mol):
    """Checks if a molecule triggers PAINS filters."""
    if not pains_catalog:
        return False
    try:
        if pains_catalog.GetFirstMatch(mol):
            return True
        return False
    except Exception as e:
        logging.debug(f"Error during PAINS check: {e}")
        return False


def has_multiple_target_fluoro_groups(mol):
    """
    Checks if the molecule contains more than one instance of the same targeted fluoro-group type.
    This logic ensures independent -F atoms are counted correctly.
    """
    multi_group_atom_indices = set()

    for substruct_info in multi_atom_substructs:
        group_match_count = 0
        atoms_matched_this_group = set()

        for variant in substruct_info["variants"]:
            patt = Chem.MolFromSmarts(variant["smarts"])
            if not patt: continue
            try:
                matches = mol.GetSubstructMatches(patt)
            except:
                continue

            for match in matches:
                if variant.get("condition") and not variant["condition"](mol, match):
                    continue
                current_match_indices = set(match)
                if not current_match_indices.intersection(atoms_matched_this_group):
                    group_match_count += 1
                    atoms_matched_this_group.update(current_match_indices)
                    multi_group_atom_indices.update(current_match_indices)
                    if group_match_count > 1:
                        return True

    if f_substruct_info:
        f_group_count = 0
        f_variant = f_substruct_info["variants"][0]
        f_patt = Chem.MolFromSmarts(f_variant["smarts"])
        if f_patt:
            try:
                f_matches = mol.GetSubstructMatches(f_patt)
            except:
                f_matches = []

            for f_match in f_matches:
                f_idx = f_match[0]
                if f_idx in multi_group_atom_indices:
                    continue
                if f_variant.get("condition") and not f_variant["condition"](mol, f_match):
                    continue
                f_group_count += 1
                if f_group_count > 1:
                    return True

    return False


def filter_molecule(smiles):
    """
    Applies all filters to a single SMILES string.
    Returns the canonical SMILES if it passes, otherwise None.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        if not check_property_filters(calculate_properties(mol)): return None
        if REMOVE_PAINS and check_pains_filter(mol): return None
        if has_multiple_target_fluoro_groups(mol): return None

        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception as e:
        logging.warning(f"General error filtering SMILES '{smiles}': {e}")
        return None


def get_args():
    """Gets command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filters a CSV of sanitized SMILES based on physicochemical properties and structural alerts.")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the input CSV file (e.g., from Sanitization_smi.py).")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the filtered CSV file.")
    parser.add_argument("--processes", type=int, default=cpu_count(), help="Number of CPU processes to use.")
    return parser.parse_args()


def main():
    """Main function to read, filter, and save SMILES data."""
    args = get_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    if not input_path.is_file():
        logging.error(f"Input file not found: {input_path}")
        return

    logging.info(f"Reading standardized SMILES from: {input_path}")
    try:
        df = pd.read_csv(input_path)
        if 'SMILES' not in df.columns:
            logging.error(f"'SMILES' column not found in {input_path}. Ensure it has a 'SMILES' header.")
            return
        smiles_list = df['SMILES'].astype(str).unique().tolist()
        logging.info(f"Read {len(smiles_list)} unique standardized SMILES.")
    except Exception as e:
        logging.error(f"Error reading input CSV '{input_path}': {e}")
        return

    num_processes = min(args.processes, cpu_count())
    logging.info(f"Filtering SMILES using {num_processes} processes...")

    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(filter_molecule, smiles_list, chunksize=500),
                            total=len(smiles_list), desc="Filtering SMILES"))

    filtered_smiles_list = [s for s in results if s is not None]

    num_original = len(smiles_list)
    num_filtered = len(filtered_smiles_list)
    num_removed = num_original - num_filtered

    logging.info("--- Filtering Summary ---")
    logging.info(f"Original unique SMILES: {num_original}")
    logging.info(f"SMILES after filtering: {num_filtered}")
    logging.info(f"SMILES removed by filters: {num_removed}")

    logging.info(f"Saving {num_filtered} filtered SMILES to: {output_path}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(filtered_smiles_list, columns=['SMILES']).to_csv(output_path, index=False)
        logging.info("Filtering complete.")
    except IOError as e:
        logging.error(f"Error writing output file {output_path}: {e}")


if __name__ == '__main__':
    main()

