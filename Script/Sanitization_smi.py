import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from rdkit import RDLogger # Keep for RDKit log level control
from pathlib import Path
import sys # For printing errors

# --- Configuration ---
INPUT_FILE = Path("/ChEMBL/F-compound.csv") # ADJUST THIS PATH
OUTPUT_FILE = Path("/ChEMBL/F-clear.csv") # ADJUST THIS PATH
# --- End Configuration ---

# Suppress RDKit's verbose warnings, show only errors
rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.ERROR)

# Global Uncharger instance (reusable for efficiency)
uncharger = rdMolStandardize.Uncharger()
# Global LargestFragmentChooser instance
lfc = rdMolStandardize.LargestFragmentChooser()

def standardize_and_clean_smiles(smiles):
    """
    Cleans, standardizes, neutralizes charge, and gets the largest fragment.
    Returns canonical, isomeric SMILES, or None if processing fails.
    """
    if not isinstance(smiles, str) or not smiles:
        # print(f"Invalid input: {smiles}", file=sys.stderr) # Optional: print errors
        return None
    try:
        # 1. Initial parsing, sanitize=False to catch more raw errors
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            # print(f"Initial parsing failed: {smiles}", file=sys.stderr)
            return None

        # 2. Handle fragments/salts, keep the largest fragment
        mol = lfc.choose(mol)
        if mol is None:
             # print(f"Choosing largest fragment failed: {smiles}", file=sys.stderr)
             return None

        # 3. Attempt basic sanitize (fix valence, etc.) before uncharging
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            # Even if this fails, try to continue with uncharging
            # print(f"Sanitize after fragmentation failed (attempting to continue): {smiles} - {e}", file=sys.stderr)
            pass # Allow the process to continue

        # 4. Neutralize charge
        try:
            mol = uncharger.uncharge(mol)
        except Exception as e:
             # print(f"Error during charge neutralization: {smiles} - {e}", file=sys.stderr)
             # Uncharging failure often means problematic structure
             return None
        if mol is None: # uncharge might return None
            # print(f"Charge neutralization failed (resulted in None): {smiles}", file=sys.stderr)
            return None

        # 5. Final Sanitize after all modifications
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            # print(f"Final Sanitize failed: {smiles} - {e}", file=sys.stderr)
            return None

        # 6. Generate canonical, isomeric SMILES
        # canonical=True is crucial for sequence models
        smiles_out = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return smiles_out

    except Exception as e:
        # Catch any other unexpected errors
        print(f"Generic error processing SMILES '{smiles}': {e}", file=sys.stderr)
        return None

# Main function
def main(input_csv_path, output_csv_path):
    print(f"Starting to read SMILES from file: {input_csv_path}")
    try:
        # Adjust skiprows/header based on your CSV file format
        # header=None: no header row; skiprows=0: read from the first line
        # low_memory=False avoids DtypeWarning
        # keep_default_na=False, na_values=[''] handles empty strings explicitly
        df = pd.read_csv(input_csv_path, usecols=[0], header=None, skiprows=0, low_memory=False, keep_default_na=False, na_values=[''])
        # Ensure data is read as string, handle potential nulls or non-strings
        smiles_list = df[df.columns[0]].astype(str).dropna().tolist()
        print(f"Read {len(smiles_list)} SMILES.")
    except FileNotFoundError:
        print(f"Error: Input file not found {input_csv_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error reading input CSV: {e}", file=sys.stderr)
        return

    if not smiles_list:
        print("Warning: No valid SMILES data found in the input file.", file=sys.stderr)
        return

    # Set number of processes
    num_processes = min(32, cpu_count()) # Use up to 32 cores or available cores
    print(f"Using {num_processes} processes for standardization...")

    # Use multiprocessing Pool
    cleaned_smiles_list = []
    with Pool(num_processes) as pool:
        # Use tqdm for progress bar, chunksize for optimization
        results = list(tqdm(pool.imap(standardize_and_clean_smiles, smiles_list, chunksize=1000),
                            total=len(smiles_list), desc="Cleaning and Standardizing SMILES"))

    # Filter out None values from failed processing
    cleaned_smiles_list = [s for s in results if s is not None]
    num_failed = len(smiles_list) - len(cleaned_smiles_list)

    print(f"Original SMILES count: {len(smiles_list)}")
    print(f"Successfully cleaned and standardized SMILES count: {len(cleaned_smiles_list)}")
    print(f"Failed or discarded SMILES count: {num_failed}")

    # Save cleaned data
    print(f"Saving cleaned SMILES to: {output_csv_path}")
    # Include a 'SMILES' header for clarity
    try:
        pd.DataFrame(cleaned_smiles_list, columns=['SMILES']).to_csv(output_csv_path, index=False, header=True)
        print("Processing complete.")
    except IOError as e:
        print(f"Error writing output file {output_csv_path}: {e}", file=sys.stderr)


if __name__ == '__main__':
    # Ensure output directory exists
    try:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        main(INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}", file=sys.stderr)