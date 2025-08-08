from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
import time
from pathlib import Path
import traceback
import argparse

# --- Global Vocabulary ---
VOCAB_SORTED = None


def setup_logging(output_dir):
    """Sets up logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_filename = output_dir / f"tokenization_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def load_vocab(vocab_path):
    """Loads vocabulary from a file and sorts it by length, descending."""
    global VOCAB_SORTED
    if not vocab_path.is_file():
        logging.error(f"Vocabulary file not found: {vocab_path}")
        return False
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f if line.strip()]
        VOCAB_SORTED = sorted(vocab_list, key=len, reverse=True)
        logging.info(f"Loaded {len(VOCAB_SORTED)} tokens from {vocab_path}")
        return True
    except Exception as e:
        logging.error(f"Error loading vocabulary from {vocab_path}: {e}", exc_info=True)
        return False


def tokenize_smiles_by_vocab(smiles, vocab_sorted):
    """Performs greedy tokenization using a pre-loaded, sorted vocabulary."""
    if not isinstance(smiles, str) or not smiles: return []

    tokens = []
    idx = 0
    smiles_len = len(smiles)

    while idx < smiles_len:
        match_found = False
        # Attempt to match the longest possible token from the vocabulary
        for token in vocab_sorted:
            if smiles.startswith(token, idx):
                tokens.append(token)
                idx += len(token)
                match_found = True
                break

        # Fallback: if no token matches, treat the current character as a single token
        if not match_found:
            unknown_char = smiles[idx]
            tokens.append(unknown_char)
            idx += 1

    return tokens


def process_file_tokenization(args):
    """Worker function to read, tokenize, and write a single file."""
    file_path_str, input_dir_str, output_dir_str, vocab_shared = args
    file_path = Path(file_path_str)
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    try:
        relative_path = file_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tokenized_lines = []
        with open(file_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()

        for line in lines:
            smiles = line.strip()
            if smiles:
                tokens = tokenize_smiles_by_vocab(smiles, vocab_shared)
                if tokens:
                    # Use space as the delimiter for OpenNMT input format
                    tokenized_lines.append(' '.join(tokens))

        if tokenized_lines:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                f_out.write('\n'.join(tokenized_lines) + '\n')

        return {'file_name': file_path.name, 'lines_processed': len(lines), 'success': True}

    except Exception as e:
        return {'file_name': file_path.name, 'error_info': str(e) + '\n' + traceback.format_exc(), 'success': False}


def get_args():
    """Gets command-line arguments."""
    parser = argparse.ArgumentParser(description="Tokenizes SMILES files based on a shared vocabulary.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the split data files (*-src.txt, *-tgt.txt).")
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to the shared vocabulary file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the tokenized output files.")
    parser.add_argument("--processes", type=int, default=cpu_count(), help="Number of CPU processes to use.")
    return parser.parse_args()


def main():
    """Main function to orchestrate the tokenization process."""
    args = get_args()
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)

    if not load_vocab(Path(args.vocab_file)):
        return

    input_dir = Path(args.data_dir)
    logger.info(f"Input data directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    logger.info(f"Searching for all '*-src.txt' and '*-tgt.txt' files in {input_dir}...")
    files_to_process = list(input_dir.rglob("*-src.txt")) + list(input_dir.rglob("*-tgt.txt"))

    if not files_to_process:
        logger.error(f"No source or target files found in {input_dir}.")
        return

    logger.info(f"Found {len(files_to_process)} files to tokenize.")
    num_cores = min(args.processes, cpu_count())
    logger.info(f"Using {num_cores} CPU cores for parallel processing.")

    file_args = [(str(file), str(input_dir), str(output_dir), VOCAB_SORTED) for file in files_to_process]

    results = []
    try:
        with Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.imap(process_file_tokenization, file_args),
                                total=len(files_to_process),  # <-- CORRECTED THIS LINE
                                desc="Tokenizing files"))
    except Exception as pool_err:
        logger.error(f"Multiprocessing pool error: {pool_err}", exc_info=True)

    logger.info("--- Tokenization Summary ---")
    success_count, fail_count = 0, 0
    for result in results:
        if result and result.get('success'):
            success_count += 1
            logger.info(f"  Processed {result['file_name']}: {result['lines_processed']} lines.")
        else:
            fail_count += 1
            logger.error(
                f"  FAILED {result.get('file_name', 'Unknown file')}: {result.get('error_info', 'Unknown error')}")

    logger.info(f"Successfully processed: {success_count} files")
    logger.info(f"Failed to process: {fail_count} files")
    logger.info(f"Tokenized files saved in: {output_dir}")
    logger.info("Tokenization script finished.")


if __name__ == "__main__":
    main()

