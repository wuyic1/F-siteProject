import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from multiprocessing import Pool, cpu_count
from functools import partial
import logging
from tqdm import tqdm
from pathlib import Path
import argparse

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- Marker Definition ---
MARKER_ATOM_SYMBOL = "Se"  # Temporary atom for robust SMILES generation
MARKER_SMILES_TOKEN_TEMP = "[Se]"
MARKER_ATOM_SMARTS_FINAL = "[*]"  # Final wildcard marker


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


condition_map = {
    "condition_cfh2_sp3": condition_cfh2_sp3, "condition_cfh2_sp2": condition_cfh2_sp2,
    "condition_cf2h_sp3": condition_cf2h_sp3, "condition_cf2h_sp2": condition_cf2h_sp2,
    "condition_cf3_sp3": condition_cf3_sp3,
    "condition_ocf2h_sp3": condition_ocf2h_sp3, "condition_ocf2h_sp2": condition_ocf2h_sp2,
    "condition_ocf3_sp3": condition_ocf3_sp3,
    "condition_scf2h_sp3": condition_scf2h_sp3, "condition_scf2h_sp2": condition_scf2h_sp2,
    "condition_scf3_sp3": condition_scf3_sp3,
    "condition_cfh_sp3": condition_cfh_sp3, "condition_cfh_sp2": condition_cfh_sp2,
    "condition_cfh_sp": condition_cfh_sp,
    "condition_cf2_sp3": condition_cf2_sp3, "condition_cf2_sp2": condition_cf2_sp2,
    "condition_f_terminal": condition_f_terminal
}

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
substructure_map = {s['name']: s for s in substructure_defs}


# --- END: Definitions ---


def get_atoms_in_complex_groups(mol):
    """Helper function to find all atom indices belonging to the 9 complex fluoro-groups."""
    complex_group_atoms = set()
    for s_info in substructure_defs:
        if s_info['name'] == '-F': continue
        for variant in s_info["variants"]:
            patt = Chem.MolFromSmarts(variant["smarts"])
            if not patt: continue
            try:
                matches = mol.GetSubstructMatches(patt)
            except:
                continue
            for match in matches:
                condition_func = condition_map.get(variant.get("condition_name"))
                if condition_func and not condition_func(mol, match):
                    continue
                complex_group_atoms.update(match)
    return complex_group_atoms


def get_fluoro_group_info(mol, substructure_info):
    """
    Finds valid matches for the target group for cutting/marking.
    Returns a list of tuples: (match_indices, attachment_atom_idx, group_atom_indices, bond_type)
    """
    fluoro_group_matches = []
    name = substructure_info['name']
    processed_matches = set()

    if name == '-F':
        complex_atoms = get_atoms_in_complex_groups(mol)
        variant = substructure_info['variants'][0]
        patt = Chem.MolFromSmarts(variant['smarts'])
        if not patt: return []
        try:
            matches = mol.GetSubstructMatches(patt)
        except:
            return []

        for match in matches:
            if match in processed_matches: continue
            f_idx = match[0]
            if f_idx in complex_atoms: continue

            condition = condition_map.get(variant.get('condition_name'))
            if condition and not condition(mol, match): continue

            try:
                atom_f = mol.GetAtomWithIdx(f_idx)
                if atom_f.GetDegree() == 1:
                    neighbor = atom_f.GetNeighbors()[0]
                    attach_idx = neighbor.GetIdx()
                    group_indices = {f_idx}
                    bond = mol.GetBondBetweenAtoms(f_idx, attach_idx)
                    bond_type = bond.GetBondType() if bond else Chem.BondType.SINGLE
                    fluoro_group_matches.append((match, attach_idx, list(group_indices), bond_type))
                    processed_matches.add(match)
            except Exception as e:
                logger.debug(f"Error in -F match processing {match}: {e}")
    else:  # For complex groups
        for variant in substructure_info['variants']:
            patt = Chem.MolFromSmarts(variant['smarts'])
            if not patt: continue
            try:
                matches = mol.GetSubstructMatches(patt)
            except:
                continue

            for match in matches:
                if match in processed_matches: continue
                condition = condition_map.get(variant.get('condition_name'))
                if condition and not condition(mol, match): continue

                attach_idx = -1
                group_indices = set()
                bond_type = Chem.BondType.SINGLE

                try:
                    if name in ["-CFH₂", "-CF₂H", "-CF₃"]:
                        c_idx = match[0]
                        group_indices = set(match)
                        for n in mol.GetAtomWithIdx(c_idx).GetNeighbors():
                            if n.GetAtomicNum() != 9 and n.GetAtomicNum() != 1:
                                attach_idx = n.GetIdx()
                                bond = mol.GetBondBetweenAtoms(c_idx, attach_idx)
                                if bond: bond_type = bond.GetBondType()
                                break
                    elif name in ["-OCF₂H", "-OCF₃", "-SCF₂H", "-SCF₃"]:
                        hetero_idx, c_idx = match[0], match[1]
                        group_indices = set(match)
                        for n in mol.GetAtomWithIdx(hetero_idx).GetNeighbors():
                            if n.GetIdx() != c_idx:
                                attach_idx = n.GetIdx()
                                bond = mol.GetBondBetweenAtoms(hetero_idx, attach_idx)
                                if bond: bond_type = bond.GetBondType()
                                break
                    elif name == "-CFH-":
                        c_idx, f_idx = match[0], match[1]
                        group_indices = {f_idx}
                        attach_idx = c_idx
                        bond = mol.GetBondBetweenAtoms(c_idx, f_idx)
                        if bond: bond_type = bond.GetBondType()
                    elif name == "-CF₂-":
                        c_idx, f_idx1, f_idx2 = match[0], match[1], match[2]
                        group_indices = {f_idx1, f_idx2}
                        attach_idx = c_idx
                        bond = mol.GetBondBetweenAtoms(c_idx, f_idx1)
                        if bond: bond_type = bond.GetBondType()

                    if attach_idx != -1 and group_indices:
                        if attach_idx not in group_indices:
                            fluoro_group_matches.append((match, attach_idx, list(group_indices), bond_type))
                            processed_matches.add(match)

                except Exception as e:
                    logger.debug(f"Error finding attach point for {name}, match {match}: {e}")

    return fluoro_group_matches


def generate_modified_smiles(original_smiles, substructure_info):
    """
    Generates (base_smiles, marked_smiles, original_smiles) triples for a given SMILES.
    """
    if not original_smiles or not isinstance(original_smiles, str): return []
    try:
        mol = Chem.MolFromSmiles(original_smiles)
        if not mol: return []

        match_details_list = get_fluoro_group_info(mol, substructure_info)
        if not match_details_list: return []

        results = []
        original_smiles_canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

        for _, attach_idx, group_indices, bond_type in match_details_list:
            base_smi, marked_smi_final = None, None
            valid_bond = bond_type if bond_type in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE,
                                                    Chem.BondType.AROMATIC] else Chem.BondType.SINGLE

            # Generate marked SMILES
            try:
                rw_mol = Chem.RWMol(mol)
                marker_idx = rw_mol.AddAtom(Chem.Atom(Chem.GetPeriodicTable().GetAtomicNumber(MARKER_ATOM_SYMBOL)))
                rw_mol.GetAtomWithIdx(marker_idx).SetNoImplicit(True)
                rw_mol.AddBond(attach_idx, marker_idx, valid_bond)

                indices_to_remove = sorted([idx for idx in group_indices if idx < rw_mol.GetNumAtoms()], reverse=True)
                for idx in indices_to_remove: rw_mol.RemoveAtom(idx)

                Chem.SanitizeMol(rw_mol)
                marked_mol = rw_mol.GetMol()
                if len(rdmolops.GetMolFrags(marked_mol)) == 1:
                    marked_smi_temp = Chem.MolToSmiles(marked_mol, isomericSmiles=True, canonical=True)
                    if MARKER_SMILES_TOKEN_TEMP in marked_smi_temp:
                        marked_smi_final = marked_smi_temp.replace(MARKER_SMILES_TOKEN_TEMP, MARKER_ATOM_SMARTS_FINAL)
                        if Chem.MolFromSmiles(marked_smi_final) is None: marked_smi_final = None
            except Exception as e:
                logger.debug(f"Marked SMILES generation failed for {original_smiles}: {e}")

            # Generate base SMILES (if marked SMILES was successful)
            if marked_smi_final:
                try:
                    rw_mol = Chem.RWMol(mol)
                    h_idx = rw_mol.AddAtom(Chem.Atom(1))
                    rw_mol.AddBond(attach_idx, h_idx, valid_bond)

                    indices_to_remove = sorted([idx for idx in group_indices if idx < rw_mol.GetNumAtoms()],
                                               reverse=True)
                    for idx in indices_to_remove: rw_mol.RemoveAtom(idx)

                    Chem.SanitizeMol(rw_mol)
                    base_mol = rdmolops.RemoveHs(rw_mol.GetMol(), sanitize=True)
                    if base_mol and len(rdmolops.GetMolFrags(base_mol)) == 1:
                        base_smi = Chem.MolToSmiles(base_mol, isomericSmiles=True, canonical=True)
                except Exception as e:
                    logger.debug(f"Base SMILES generation failed for {original_smiles}: {e}")

            if base_smi and marked_smi_final and base_smi != original_smiles_canonical:
                results.append((base_smi, marked_smi_final, original_smiles_canonical))

        return results
    except Exception as e:
        logger.warning(f"General error processing {original_smiles}: {e}")
        return []


def process_file(file_path, output_dir_path):
    """Processes a single CSV file to generate molecule pairs."""
    sub_name = file_path.stem.replace('_', '-')
    sub_info = substructure_map.get(sub_name)
    if not sub_info:
        logger.warning(f"No definition for '{sub_name}'. Skipping {file_path}")
        return

    logger.info(f"Processing file: {file_path} for target group: {sub_info['name']}")

    try:
        smiles_list = pd.read_csv(file_path)['SMILES'].drop_duplicates().astype(str).tolist()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return

    processor = partial(generate_modified_smiles, substructure_info=sub_info)

    with Pool(cpu_count()) as pool:
        results_list = list(tqdm(pool.imap(processor, smiles_list, chunksize=100),
                                 total=len(smiles_list), desc=f"Generating pairs for {sub_info['name']}"))

    base_data, marked_data, orig_data = [], [], []
    for result_triples in results_list:
        for base, marked, orig in result_triples:
            base_data.append(base)
            marked_data.append(marked)
            orig_data.append(orig)

    if base_data:
        try:
            with open(output_dir_path / f"{sub_name}(base).txt", 'w') as f:
                f.write('\n'.join(base_data))
            with open(output_dir_path / f"{sub_name}(marked).txt", 'w') as f:
                f.write('\n'.join(marked_data))
            with open(output_dir_path / f"{sub_name}(orig).txt", 'w') as f:
                f.write('\n'.join(orig_data))
            logger.info(f"Generated {len(base_data)} pairs for {sub_info['name']}.")
        except IOError as e:
            logger.error(f"Error writing output files for {sub_info['name']}: {e}")
    else:
        logger.warning(f"No valid pairs generated for {sub_info['name']}.")


def get_args():
    """Gets command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generates (base, marked, original) molecule pairs by cutting specific substructures.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the input CSV files, one for each substructure.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the generated text file pairs.")
    return parser.parse_args()


def main():
    """Main function to orchestrate the pair generation process."""
    args = get_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_dir.glob('*.csv'))
    if not csv_files:
        logging.error(f"No CSV files found in input directory: {input_dir}")
        return

    for csv_file in csv_files:
        process_file(csv_file, output_dir)

    logging.info("All files processed.")


if __name__ == "__main__":
    main()

