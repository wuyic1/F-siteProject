import sys
import os
from pathlib import Path
from typing import Union
from rdkit import Chem
from tqdm import tqdm


def smiles_to_ggnn_format(smiles: str) -> Union[str, None]:
    """
    Converts a single SMILES string to the OpenNMT GGNN input format.
    The format is:
    AtomTokens <EOT> AtomFeatures <EOT> EdgeType1_Edges,EdgeType2_Edges,...

    - AtomTokens are chemical symbols of atoms.
    - AtomFeatures are their atomic numbers.
    - Edges are grouped by bond type (SINGLE, DOUBLE, TRIPLE, AROMATIC).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Warning: RDKit could not parse SMILES '{smiles}'. Skipping.", file=sys.stderr)
        return None

    atom_tokens = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if not atom_tokens:
        print(f"Warning: Mol from SMILES '{smiles}' has no atoms. Skipping.", file=sys.stderr)
        return None

    atom_features = [str(atom.GetAtomicNum()) for atom in mol.GetAtoms()]

    edge_lists = {1: [], 2: [], 3: [], 12: []}  # Keys for SINGLE, DOUBLE, TRIPLE, AROMATIC

    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type_float = bond.GetBondTypeAsDouble()

        if bond_type_float == 1.0:
            bond_key = 1
        elif bond_type_float == 2.0:
            bond_key = 2
        elif bond_type_float == 3.0:
            bond_key = 3
        elif bond_type_float == 1.5:
            bond_key = 12
        else:
            continue

        edge_lists[bond_key].append(f"{start_idx} {end_idx}")
        edge_lists[bond_key].append(f"{end_idx} {start_idx}")

    tokens_str = " ".join(atom_tokens)
    features_str = " ".join(atom_features)

    edge_group_strs = [" ".join(edge_lists[bond_key]) for bond_key in [1, 2, 3, 12]]
    edges_str = ",".join(edge_group_strs)

    return f"{tokens_str} <EOT> {features_str} <EOT> {edges_str}"


def process_file(input_path: Path):
    """
    Reads a file of SMILES, converts each, and writes to a new file
    named 'test-src-ggnn.txt' in the same directory.
    Includes a progress bar.
    """
    output_path = input_path.parent / "test-src-ggnn.txt"

    print(f"Processing {input_path} -> {output_path}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for line in f)

        with open(input_path, 'r', encoding='utf-8') as infile, \
                open(output_path, 'w', encoding='utf-8') as outfile:

            for line in tqdm(infile, total=num_lines, desc=f"Converting {input_path.name}"):
                smiles = line.strip()
                if not smiles:
                    continue
                ggnn_line = smiles_to_ggnn_format(smiles)
                if ggnn_line:
                    outfile.write(ggnn_line + '\n')
    except Exception as e:
        print(f"Error processing file {input_path}: {e}", file=sys.stderr)


def find_and_convert_test_files(root_directory: str):
    """
    Recursively finds all files named 'test-src.txt' in a given directory
    and converts them to GGNN format.
    """
    root_path = Path(root_directory)
    if not root_path.is_dir():
        print(f"Error: Provided path '{root_directory}' is not a valid directory.", file=sys.stderr)
        return

    print(f"Starting search for 'test-src.txt' files in '{root_path}'...")

    # Use rglob to find all matching files recursively
    test_files = list(root_path.rglob('test-src.txt'))

    if not test_files:
        print("No 'test-src.txt' files found.")
        return

    print(f"Found {len(test_files)} file(s) to convert:")
    for f in test_files:
        print(f" - {f}")

    for file_path in test_files:
        process_file(file_path)

    print("\nAll conversions complete.")


if __name__ == '__main__':

    # Provide the root directory you want to search as a command-line argument.
    # If no argument is given, it will search the current directory.

    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        target_directory = '.'  # Default to current directory
        print("No directory provided. Searching in the current directory.")

    find_and_convert_test_files(target_directory)