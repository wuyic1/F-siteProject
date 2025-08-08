import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from multiprocessing import Pool, cpu_count
import logging
from tqdm import tqdm
from rdkit import RDLogger
from pathlib import Path
import argparse

# Suppress RDKit's verbose warnings, show only errors
rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.ERROR)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=logging.StreamHandler())
logger = logging.getLogger()

# Reusable global instances for efficiency
uncharger = rdMolStandardize.Uncharger()
lfc = rdMolStandardize.LargestFragmentChooser()


def standardize_and_clean_smiles(smiles):
    """
    Cleans, standardizes, neutralizes charges, and retains the largest fragment of a SMILES string.

    Returns a canonical, isomeric SMILES string, or None if processing fails.
    """
    if not isinstance(smiles, str) or not smiles:
        logger.debug(f"Invalid input: {smiles}")
        return None
    try:
        # 1. Initial parsing without sanitization to catch more raw errors.
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            logger.debug(f"Initial parsing failed: {smiles}")
            return None

        # 2. Handle fragments/salts by retaining the largest fragment.
        mol = lfc.choose(mol)
        if mol is None:
            logger.debug(f"Choosing largest fragment failed: {smiles}")
            return None

        # 3. Attempt basic sanitization to fix valency, etc., before uncharging.
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            logger.debug(f"Initial SanitizeMol after fragmentation failed (will attempt to continue): {smiles} - {e}")

        # 4. Neutralize charges.
        try:
            mol = uncharger.uncharge(mol)
        except Exception as e:
            logger.debug(f"Error during charge neutralization: {smiles} - {e}")
            return None
        if mol is None:
            logger.debug(f"Charge neutralization resulted in None: {smiles}")
            return None

        # 5. Final sanitization after all modifications.
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            logger.debug(f"Final SanitizeMol failed: {smiles} - {e}")
            return None

        # 6. Generate canonical, isomeric SMILES.
        smiles_out = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return smiles_out

    except Exception as e:
        logger.warning(f"An unexpected error occurred while processing SMILES '{smiles}': {e}")
        return None


def get_args():
    """Gets command-line arguments."""
    parser = argparse.ArgumentParser(description="Sanitizes a CSV file of SMILES strings.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input CSV file with raw SMILES in the first column.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the cleaned CSV file.")
    parser.add_argument("--processes", type=int, default=cpu_count(), help="Number of CPU processes to use.")
    return parser.parse_args()


def main():
    """Main function to orchestrate the SMILES sanitization process."""
    args = get_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    logger.info(f"Starting to read SMILES from file: {input_path}")
    try:
        # Assuming SMILES are in the first column with no header.
        df = pd.read_csv(input_path, usecols=[0], header=None, low_memory=False, keep_default_na=False, na_values=[''])
        smiles_list = df[df.columns[0]].astype(str).dropna().tolist()
        logger.info(f"Read {len(smiles_list)} SMILES strings.")
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {input_path}")
        return
    except Exception as e:
        logger.error(f"Error reading input CSV: {e}")
        return

    if not smiles_list:
        logger.warning("No valid SMILES data found in the input file.")
        return

    num_processes = min(args.processes, cpu_count())
    logger.info(f"Using {num_processes} processes for standardization...")

    cleaned_smiles_list = []
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(standardize_and_clean_smiles, smiles_list, chunksize=1000),
                            total=len(smiles_list), desc="Sanitizing and standardizing SMILES"))

    cleaned_smiles_list = [s for s in results if s is not None]
    num_failed = len(smiles_list) - len(cleaned_smiles_list)

    logger.info(f"Original SMILES count: {len(smiles_list)}")
    logger.info(f"Successfully cleaned and standardized SMILES count: {len(cleaned_smiles_list)}")
    logger.info(f"Failed or discarded SMILES count: {num_failed}")

    logger.info(f"Saving cleaned SMILES to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cleaned_smiles_list, columns=['SMILES']).to_csv(output_path, index=False, header=True)
    logger.info("Processing complete.")


if __name__ == '__main__':
    main()

