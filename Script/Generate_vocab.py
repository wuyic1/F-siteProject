from collections import Counter
from pathlib import Path
import re
from tqdm import tqdm
import argparse


def smiles_tokenizer(s):
    """
    A robust SMILES tokenizer that handles complex tokens and special cases,
    including all stereochemistry symbols.
    """
    if not isinstance(s, str):
        print(f"Error: Tokenizer received non-string input: {type(s)} - {s}")
        return []
    # Regex to capture bracketed terms, common elements, and special symbols
    pattern = r"(\[\*\]|\[[^\]]+\]|Br?|Cl?|Si?|Se?|Mg?|Na?|Ca?|Fe?|As?|Al?|I|B|K|Li?|Zn?|Au?|Ag?|Cu?|Ni?|Cd?|Mn?|Cr?|Co?|Sn?|Ba?|Ti?|H[1-9]?|b|c|n|o|s|p|f|i|k|C|N|O|S|P|F|I|K|\(|\)|\.|=|#|-|\+|\%[0-9]{2}|[0-9]|\\|/|@@?)"
    regex = re.compile(pattern)
    tokens = regex.findall(s)

    # Verification to ensure the tokenizer captures the entire string
    if ''.join(tokens) != s:
        print(f"Warning: Tokenizer validation failed!")
        print(f"  - Original SMILES: '{s}'")
        print(f"  - Reconstructed SMILES: '{''.join(tokens)}'")
    return tokens


def generate_shared_vocab(data_files, output_vocab_file, min_frequency=1):
    """
    Generates a shared vocabulary from a list of text files.
    """
    if not data_files:
        print("Error: No input files provided for vocabulary generation. Aborting.")
        return False

    print(f"Generating shared vocabulary from {len(data_files)} source files.")
    token_counts = Counter()
    total_lines = 0

    for file_path in tqdm(data_files, desc="Reading and tokenizing"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    smiles = line.strip()
                    if smiles:
                        token_counts.update(smiles_tokenizer(smiles))
                        total_lines += 1
        except Exception as e:
            print(f"Error reading or processing file {file_path}: {e}")
            return False

    print(f"Tokenization complete. Processed {total_lines} lines.")
    print(f"Found {len(token_counts)} unique tokens before filtering.")

    # Filter by minimum frequency
    vocab = {token for token, count in token_counts.items() if count >= min_frequency}
    sorted_vocab = sorted(list(vocab))

    print(f"After filtering (min_frequency={min_frequency}), final vocabulary size is: {len(sorted_vocab)}")

    if not sorted_vocab:
        print("Warning: Final vocabulary is empty. No file will be written.")
        return False

    try:
        output_vocab_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_vocab_file, 'w', encoding='utf-8') as f:
            for token in sorted_vocab:
                f.write(token + '\n')
        print(f"Shared vocabulary successfully saved to: {output_vocab_file}")
        return True
    except IOError as e:
        print(f"Error writing vocabulary file {output_vocab_file}: {e}")
        return False


def get_args():
    """Gets command-line arguments."""
    parser = argparse.ArgumentParser(description="Generates a shared vocabulary from tokenized SMILES data.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Top-level directory containing all *-src.txt and *-tgt.txt files.")
    parser.add_argument("--vocab_dir", type=str, required=True,
                        help="Directory where the 'shared.vocab' file will be saved.")
    parser.add_argument("--min_frequency", type=int, default=1,
                        help="Minimum frequency for a token to be included in the vocabulary.")
    return parser.parse_args()


def main():
    """Main function to orchestrate vocabulary generation."""
    args = get_args()
    data_path = Path(args.data_dir)
    vocab_path = Path(args.vocab_dir)

    print("--- Starting Shared Vocabulary Generation Script ---")
    print(f"Data source directory: {data_path}")
    print(f"Vocabulary output directory: {vocab_path}")

    if not data_path.is_dir():
        print(f"Fatal Error: The specified data directory does not exist: {data_path}")
        return

    # Recursively find all relevant source and target files
    files_to_process = list(data_path.rglob("*-src.txt")) + list(data_path.rglob("*-tgt.txt"))
    if not files_to_process:
        print(f"Error: No '*-src.txt' or '*-tgt.txt' files found in {data_path} or its subdirectories.")
        print("Please ensure the data splitting script ran successfully and paths are correct.")
        return

    print(f"Found {len(files_to_process)} files for vocabulary creation.")

    output_file = vocab_path / "shared.vocab"
    success = generate_shared_vocab(files_to_process, output_file, args.min_frequency)

    if success:
        print("\nScript finished successfully.")
    else:
        print("\nScript encountered errors. Please review the output above.")


if __name__ == "__main__":
    main()

