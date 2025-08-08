import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging
from pathlib import Path
import argparse

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- START: Definitions (Must match other scripts) ---

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


# --- END: Definitions ---

def get_atoms_in_complex_groups(mol):
    """Helper function to find all atom indices belonging to the 9 complex fluoro-groups."""
    complex_group_atoms = set()
    for substruct_info in substructures:
        if substruct_info['name'] == '-F':
            continue
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
                complex_group_atoms.update(match)
    return complex_group_atoms


def worker(args):
    """
    Checks if a SMILES string contains the specified fluoro-group, handling the '-F' case specifically.
    """
    smiles, substruct_info = args
    group_name = substruct_info["name"]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return False

        if group_name == '-F':
            complex_group_atoms = get_atoms_in_complex_groups(mol)
            f_variant = substruct_info["variants"][0]
            f_patt = Chem.MolFromSmarts(f_variant["smarts"])
            if not f_patt: return False
            try:
                f_matches = mol.GetSubstructMatches(f_patt)
            except:
                return False
            for f_match in f_matches:
                f_idx = f_match[0]
                if f_idx not in complex_group_atoms:
                    if f_variant.get("condition") and not f_variant["condition"](mol, f_match):
                        continue
                    return True  # Found an independent -F
            return False

        else:  # Standard handling for other groups
            for variant in substruct_info["variants"]:
                patt = Chem.MolFromSmarts(variant["smarts"])
                if not patt: continue
                try:
                    matches = mol.GetSubstructMatches(patt)
                except:
                    continue
                if not matches: continue
                if variant.get("condition") is None: return True

                for match in matches:
                    if variant["condition"](mol, match):
                        return True
            return False

    except Exception as e:
        logging.warning(f"Error processing SMILES {smiles} for group {group_name}: {e}")
        return False


def get_args():
    """Gets command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Identifies molecules containing specific fluorine-containing substructures.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the filtered input CSV file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output CSV files for each substructure.")
    parser.add_argument("--processes", type=int, default=cpu_count(), help="Number of CPU processes to use.")
    return parser.parse_args()


def main():
    """Finds molecules containing each fluoro-group and saves them to separate files."""
    args = get_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading filtered SMILES from: {input_path}")
    try:
        data = pd.read_csv(input_path)
        if 'SMILES' not in data.columns:
            logging.error(f"'SMILES' column not found in {input_path}.")
            return
        smiles_list = data['SMILES'].astype(str).tolist()
        logging.info(f"Loaded {len(smiles_list)} SMILES for substructure search.")
    except Exception as e:
        logging.error(f"Error reading input CSV {input_path}: {e}")
        return

    num_workers = min(args.processes, cpu_count())
    logging.info(f"Using {num_workers} processes for matching.")

    for substruct_info in substructures:
        name = substruct_info["name"]
        logging.info(f"Searching for group: {name}")

        worker_args = [(smiles, substruct_info) for smiles in smiles_list]
        matched_smiles = []
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(worker, worker_args, chunksize=1000),
                                total=len(smiles_list),
                                desc=f"Matching {name}"))

        matched_smiles = [smiles for smiles, result in zip(smiles_list, results) if result]

        output_csv_path = output_path / f"{name}.csv"
        try:
            pd.DataFrame(matched_smiles, columns=['SMILES']).to_csv(output_csv_path, index=False, header=True)
            logging.info(f"Found {len(matched_smiles)} molecules containing '{name}'. Saved to {output_csv_path}")
        except IOError as e:
            logging.error(f"Could not write output file {output_csv_path}: {e}")


if __name__ == "__main__":
    main()

