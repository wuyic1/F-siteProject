import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re
from pathlib import Path
import sys # For printing info/errors

# --- Configuration ---
# Input directory: Contains the raw .src and .tgt files (e.g., from Random or Scaffold split)
INPUT_DIR = Path("/sampled_data/Scaffold") # ADJUST PATH
# Output directory: Where the tokenized files will be saved
OUTPUT_DIR = Path("/sampled_data/Scaffold/token") # ADJUST PATH
# Vocabulary file path (must exist)
VOCAB_FILE = Path("/run_directory/shared.vocab") # ADJUST PATH
# --- End Configuration ---


# --- Load Vocabulary ---
def load_vocab(vocab_path):
    """Loads vocabulary from a file, returning a list sorted by length descending."""
    if not vocab_path.is_file():
        print(f"Error: Vocabulary file not found: {vocab_path}", file=sys.stderr)
        return None
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            # Read lines, strip whitespace, filter empty lines
            vocab_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(vocab_list)} tokens from {vocab_path}")
        # Sort by length descending to ensure longest match priority in tokenizer
        vocab_list_sorted = sorted(vocab_list, key=len, reverse=True)
        return vocab_list_sorted
    except Exception as e:
        print(f"Error loading vocabulary from {vocab_path}: {e}", file=sys.stderr)
        return None

# --- Global Vocabulary ---
# Load vocab once when the script starts
VOCAB_SORTED = load_vocab(VOCAB_FILE)
if VOCAB_SORTED is None:
    print("Error: Failed to load vocabulary. Exiting.", file=sys.stderr)
    sys.exit(1) # Exit if vocab is essential and failed to load


# --- Tokenizer Function (Greedy, Vocab-Based) ---
def tokenize_smiles_by_vocab(smiles, vocab_sorted):
    """Tokenizes a SMILES string using a pre-sorted vocabulary list (longest first)."""
    if not isinstance(smiles, str): return [] # Handle non-string input
    if not smiles: return [] # Handle empty string

    tokens = []
    idx = 0
    smiles_len = len(smiles)

    while idx < smiles_len:
        match_found = False
        # Iterate through sorted vocab (longest tokens first)
        for token in vocab_sorted:
            token_len = len(token)
            # Check bounds and if the token matches at the current position
            if idx + token_len <= smiles_len and smiles[idx : idx + token_len] == token:
                tokens.append(token)
                idx += token_len
                match_found = True
                break # Found the longest possible match, move to next position

        # Fallback: If no token from vocab matches, treat the current character as a single token
        if not match_found:
            unknown_char = smiles[idx]
            # print(f"Debug: Character '{unknown_char}' at index {idx} in '{smiles}' not found in vocab. Treating as single token.", file=sys.stderr)
            tokens.append(unknown_char)
            idx += 1

    # --- Verification (Optional but Recommended) ---
    reconstructed_smiles = "".join(tokens)
    if reconstructed_smiles != smiles:
        # This warning usually indicates an incomplete vocabulary or tokenizer bug
        print(f"Warning: Tokenization verification failed!", file=sys.stderr)
        print(f"Original:      '{smiles}'", file=sys.stderr)
        print(f"Tokens:        {tokens}", file=sys.stderr)
        print(f"Reconstructed: '{reconstructed_smiles}'", file=sys.stderr)
        # Depending on severity, you might want to handle this differently

    return tokens


# --- File Processing Function for Multiprocessing ---
def process_file_tokenization(args):
    """Reads a file, tokenizes each line, and writes the tokenized output."""
    file_path_str, input_dir_str, output_dir_str = args
    file_path = Path(file_path_str)
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    try:
        # Determine output path, preserving relative structure
        relative_path = file_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        output_folder = output_path.parent
        output_folder.mkdir(parents=True, exist_ok=True) # Create output folder if needed

        tokenized_lines = []
        processed_count = 0
        error_count = 0
        # print(f"Tokenizing file: {file_path.name}") # Can be verbose with many files

        # Read all lines at once for potentially faster processing
        with open(file_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()

        # Tokenize each line
        # Add tqdm here if individual file processing is slow and needs progress
        # for line in tqdm(lines, desc=f"Tokenizing {file_path.name}", leave=False):
        for line in lines:
            smiles = line.strip()
            if smiles:
                try:
                    tokens = tokenize_smiles_by_vocab(smiles, VOCAB_SORTED) # Use pre-loaded vocab
                    if tokens:
                        tokenized_line = ' '.join(tokens) # Join tokens with space
                        tokenized_lines.append(tokenized_line)
                        processed_count += 1
                    # else: # Tokenizer might return empty list for valid reasons
                        # print(f"Warning: Tokenizer returned empty tokens for SMILES: '{smiles}' in {file_path.name}", file=sys.stderr)
                except Exception as e:
                    print(f"Error tokenizing SMILES '{smiles}' in {file_path.name}: {e}", file=sys.stderr)
                    error_count += 1

        # Write tokenized lines to the output file
        if tokenized_lines:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                f_out.write('\n'.join(tokenized_lines))
            # Return status message
            status = f"Tokenized {processed_count} lines from {file_path.name} -> {output_path.name}"
            if error_count > 0: status += f" ({error_count} errors)"
            return status
        else:
            status = f"No lines tokenized or written for {file_path.name}"
            if error_count > 0: status += f" ({error_count} errors)"
            return status

    except Exception as e:
        print(f"Error processing file {file_path}: {e}", file=sys.stderr)
        return f"FAILED processing {file_path.name}: {e}"


# --- Main Function ---
def main():
    # Use configured paths
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    print(f"Starting SMILES tokenization.")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Vocabulary file: {VOCAB_FILE}")

    if VOCAB_SORTED is None: # Double-check vocab loaded
        print("Error: Cannot proceed without a valid vocabulary.", file=sys.stderr)
        return

    # Find all .src and .tgt files recursively in the input directory
    print(f"Searching for all '.src' and '.tgt' files recursively in {input_dir}...")
    # Use Path.glob for recursive search (requires Python 3.5+)
    files_to_process = list(input_dir.glob('**/*.src')) + list(input_dir.glob('**/*.tgt'))
    # You might want to use *.txt if your split files use that extension:
    # files_to_process = list(input_dir.glob('**/*.txt'))

    if not files_to_process:
        print(f"Error: No '.src' or '.tgt' files found in {input_dir} (recursive search). Please check the input directory and file extensions.", file=sys.stderr)
        return

    print(f"Found {len(files_to_process)} files to tokenize.")
    # print(f"Debug: Files to process: {[str(f) for f in files_to_process]}") # Optional debug

    # Determine number of cores to use
    num_cores = cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing.")

    # Prepare arguments for each file processing task
    file_args = [(str(file), str(input_dir), str(output_dir)) for file in files_to_process]

    # Run tokenization in parallel using multiprocessing pool
    results = []
    print("Starting tokenization process...")
    try:
        with Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.imap(process_file_tokenization, file_args),
                                total=len(files_to_process),
                                desc="Tokenizing files"))
    except Exception as pool_err:
         print(f"Error: Multiprocessing pool failed: {pool_err}", file=sys.stderr)


    # Print summary of results
    print("--- Tokenization Summary ---")
    success_count = 0
    fail_count = 0
    for result in results:
        if result and "FAILED" not in result:
            success_count += 1
            # print(f"Debug: {result}") # Optional detailed success message
        else:
            fail_count += 1
            print(f"Error Summary: {result}", file=sys.stderr) # Print failure messages
    print(f"Successfully processed: {success_count} files")
    print(f"Failed to process: {fail_count} files")
    print(f"Tokenized files saved in: {output_dir}")
    print("Tokenization script finished.")


if __name__ == "__main__":
    # Ensure INPUT_DIR, OUTPUT_DIR, and VOCAB_FILE are correctly set in Configuration
    try:
        main()
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}", file=sys.stderr)