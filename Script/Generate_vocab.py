import os
from collections import Counter
from pathlib import Path
import re
from tqdm import tqdm
import sys # For printing info/errors

# --- Configuration ---
# Directory containing the processed (base).txt, (marked).txt, (orig).txt files
PROCESSED_DATA_DIR = Path("/ChEMBL/F_pair") # ADJUST PATH
# Directory where vocabularies will be saved
VOCAB_OUTPUT_DIR = Path("/run_directory") # ADJUST PATH
# --- Other Configuration ---
MIN_FREQUENCY = 1 # Minimum token frequency to be included in vocab
# --- End Configuration ---

# --- Tokenizer ---
def smiles_tokenizer(s):
    """Basic SMILES tokenizer using regex."""
    if not isinstance(s, str):
        print(f"Tokenizer received non-string input: {type(s)} - {s}", file=sys.stderr)
        return []
    # Regex to capture bracket elements (e.g., [nH]), multi-char elements (Br, Cl), single chars, and symbols
    pattern = r"(\[\*\]|\[[^\]]+\]|Br|Cl|Si|Se|Mg|Na|Ca|Fe|As|Al|I|B|K|Li|Zn|Au|Ag|Cu|Ni|Cd|Mn|Cr|Co|Sn|Ba|Ti|H[1-9]?|b|c|n|o|s|p|f|i|k|C|N|O|S|P|F|I|K|\(|\)|\.|=|#|-|\+|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = regex.findall(s)
    # Verification: Check if joining tokens reconstructs the original string
    if ''.join(tokens) != s:
        print(f"Warning: Tokenizer regex mismatch for SMILES: '{s}'.", file=sys.stderr)
        print(f"Original length: {len(s)}, Token joined length: {len(''.join(tokens))}", file=sys.stderr)
        print(f"Recovered tokens: {tokens}", file=sys.stderr)
        # Fallback: treat as individual characters if mismatch is significant
        # This simple fallback might split bracket atoms incorrectly.
        # A more robust approach might be needed if this happens often.
        if abs(len(s) - len("".join(tokens))) > 2: # Arbitrary threshold for significant mismatch
             print("Significant mismatch detected, using character-level fallback.", file=sys.stderr)
             tokens = list(s)
        # else: # Keep regex tokens if mismatch is minor (e.g., whitespace)
             # print("Minor mismatch, keeping regex tokens.", file=sys.stderr)

    return tokens

# --- Vocabulary Generation Function ---
def generate_vocab(data_files, output_vocab_file, min_frequency=1):
    """Generates a vocabulary file from a list of input text files."""
    if not data_files:
        print(f"Warning: No input files provided for vocabulary: {output_vocab_file.name}. Skipping.", file=sys.stderr)
        return False

    print(f"Starting vocabulary generation for {output_vocab_file.name} from {len(data_files)} files...")
    # print(f"Debug: Input files list: {[str(f) for f in data_files]}") # Optional debug

    token_counts = Counter()
    total_lines_read = 0
    processed_smiles_count = 0
    files_with_errors = 0

    for file_path in data_files:
        if not file_path.is_file():
            print(f"Error: Input file path does not exist or is not a file: {file_path}. Skipping.", file=sys.stderr)
            files_with_errors += 1
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            file_line_count = len(lines)
            total_lines_read += file_line_count
            print(f"Tokenizing SMILES from {file_path.name} ({file_line_count} lines)...")

            for line in tqdm(lines, desc=f"Tokenizing {file_path.name}", unit=" lines", leave=False):
                smiles = line.strip()
                if smiles:
                    try:
                        tokens = smiles_tokenizer(smiles)
                        if tokens:
                            token_counts.update(tokens)
                            processed_smiles_count += 1
                        else:
                            print(f"Warning: Tokenizer returned empty list for SMILES: '{smiles}' in {file_path.name}", file=sys.stderr)
                    except Exception as tokenize_err:
                        print(f"Error: Tokenization failed for SMILES: '{smiles}' in {file_path.name} - Error: {tokenize_err}", file=sys.stderr)

        except Exception as read_err:
            print(f"Error reading or processing file {file_path}: {read_err}", file=sys.stderr)
            files_with_errors += 1

    if files_with_errors > 0:
        print(f"Warning: Errors occurred while reading {files_with_errors} input file(s) for {output_vocab_file.name}. Vocabulary might be incomplete.", file=sys.stderr)

    print(f"Tokenization complete for {output_vocab_file.name}. Read {total_lines_read} lines in total.")
    print(f"Processed {processed_smiles_count} non-empty SMILES strings.")
    print(f"Found {len(token_counts)} unique tokens before filtering.")

    # Filter tokens by minimum frequency
    vocab = {token: count for token, count in token_counts.items() if count >= min_frequency}
    filtered_count = len(vocab)
    print(f"Vocabulary size after filtering (min_frequency={min_frequency}): {filtered_count}")

    if filtered_count == 0 and len(token_counts) > 0:
         print(f"Warning: All tokens were filtered out for {output_vocab_file.name}.", file=sys.stderr)
         print("Top 10 tokens before filtering:")
         for t, c in token_counts.most_common(10):
             print(f"  - '{t}': {c}")

    # Sort vocabulary alphabetically for consistency
    sorted_vocab = sorted(list(vocab.keys()))

    if sorted_vocab:
        try:
            output_vocab_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_vocab_file, 'w', encoding='utf-8') as f:
                for token in sorted_vocab:
                    f.write(token + '\n')
            print(f"Vocabulary successfully saved to: {output_vocab_file}")
            return True
        except IOError as write_err:
            print(f"Error: Failed to write vocabulary file {output_vocab_file}: {write_err}", file=sys.stderr)
            return False
    else:
         print(f"Warning: Final vocabulary for {output_vocab_file.name} is empty. No file written.", file=sys.stderr)
         # Return True if processing finished, even if vocab is empty, unless read errors occurred
         return files_with_errors == 0

# --- Main function with manual file filtering ---
def main():
    processed_data_path = PROCESSED_DATA_DIR
    vocab_output_path = VOCAB_OUTPUT_DIR
    vocab_output_path.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    print(f"Starting vocabulary generation.")
    print(f"Reading processed data from: {processed_data_path}")
    print(f"Saving vocabularies to: {vocab_output_path}")

    # --- Explicit Directory Listing and Manual Filtering ---
    print(f"--- Scanning directory {processed_data_path} for .txt files ---")
    if not processed_data_path.is_dir():
        print(f"Error: Path {processed_data_path} is not a valid directory.", file=sys.stderr)
        return

    model1_src_files = [] # Files ending in (base).txt
    model1_tgt_files = [] # Files ending in (marked).txt (also Model 2 source)
    model2_tgt_files = [] # Files ending in (orig).txt
    all_txt_files_in_dir = []

    try:
        for item in processed_data_path.iterdir():
            # Check if it's a file and ends with .txt (case-insensitive)
            if item.is_file() and item.suffix.lower() == '.txt':
                all_txt_files_in_dir.append(item.name) # Track filename found
                # Use string methods to check filename patterns
                if item.name.endswith('(base).txt'):
                    model1_src_files.append(item)
                elif item.name.endswith('(marked).txt'):
                    model1_tgt_files.append(item)
                elif item.name.endswith('(orig).txt'):
                    model2_tgt_files.append(item)

        print(f"Scan: Found {len(all_txt_files_in_dir)} total .txt files.")
        # print(f"Debug Scan: All .txt filenames found: {all_txt_files_in_dir}") # Optional debug
        print(f"Scan: Matched {len(model1_src_files)} files for Model 1 source (base).")
        print(f"Scan: Matched {len(model1_tgt_files)} files for Model 1 target / Model 2 source (marked).")
        print(f"Scan: Matched {len(model2_tgt_files)} files for Model 2 target (orig).")

    except Exception as e:
        print(f"Error: Failed scanning directory {processed_data_path}: {e}", file=sys.stderr)
        return
    print(f"--- End Scan ---")
    # --- End Explicit Directory Listing ---

    # --- Verify Files Found ---
    file_check_ok = True
    if not model1_src_files:
        print(f"Error: Failed to find any files ending with '(base).txt' in {processed_data_path}", file=sys.stderr)
        file_check_ok = False
    if not model1_tgt_files:
        print(f"Error: Failed to find any files ending with '(marked).txt' in {processed_data_path}", file=sys.stderr)
        file_check_ok = False
    # Model 2 target files might be optional depending on workflow, but check anyway
    if not model2_tgt_files:
        print(f"Warning: Failed to find any files ending with '(orig).txt' in {processed_data_path}", file=sys.stderr)
        # file_check_ok = False # Decide if this is critical

    # --- Generate Vocabularies ---
    if file_check_ok:
        print("Found necessary files. Proceeding with vocabulary generation...")
        print("--- Generating Vocab for Model 1 ---")
        success1_src = generate_vocab(model1_src_files, vocab_output_path / "model1.src.vocab", MIN_FREQUENCY)
        success1_tgt = generate_vocab(model1_tgt_files, vocab_output_path / "model1.tgt.vocab", MIN_FREQUENCY)

        print("--- Generating Vocab for Model 2 ---")
        # Re-use model1_tgt_files list for Model 2 source
        success2_src = generate_vocab(model1_tgt_files, vocab_output_path / "model2.src.vocab", MIN_FREQUENCY)
        success2_tgt = generate_vocab(model2_tgt_files, vocab_output_path / "model2.tgt.vocab", MIN_FREQUENCY)

        print("--- Generating Shared Vocab (All files combined) ---")
        all_files_manual = model1_src_files + model1_tgt_files + model2_tgt_files
        # Ensure unique file paths if a file matched multiple patterns (unlikely here)
        all_unique_files_manual = list(set(all_files_manual))
        if all_unique_files_manual:
             success_shared = generate_vocab(all_unique_files_manual, vocab_output_path / "shared.vocab", MIN_FREQUENCY)
        else:
             print("Warning: No files matched filtering criteria to generate shared vocabulary.", file=sys.stderr)
             success_shared = True # Consider this success if no files were expected

        if not (success1_src and success1_tgt and success2_src and success2_tgt and success_shared):
             print("Error: One or more vocabulary generation steps failed or produced empty results. Check output.", file=sys.stderr)
        else:
             print("All vocabulary generation steps completed.")

    else:
        print("Error: Vocabulary generation skipped due to missing input files.", file=sys.stderr)

    print("Vocabulary generation script finished.")

if __name__ == "__main__":
    # Ensure PROCESSED_DATA_DIR and VOCAB_OUTPUT_DIR are set correctly in Configuration
    try:
        main()
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}", file=sys.stderr)