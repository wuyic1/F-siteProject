# build_ggnn_vocab.py

"""
Builds a vocabulary file for OpenNMT's Gated Graph Neural Network (GGNN) encoder
from pre-processed source files.

This script recursively searches a directory for files ending in '-src-ggnn.txt',
parses them, and creates a source vocabulary file with the specific ordering
required by the GGNN implementation.

Required ordering:
1. Special tokens like <unk>, <blank>, <s>, </s>.
2. Tokens representing node labels (e.g., atom symbols 'C', 'N', 'O').
3. The graph structure separator token '<EOT>'.
4. All other tokens used for features or edges (e.g., numbers, ',').

Usage:
    python build_ggnn_vocab.py --data_dir /path/to/your/data \
                               --output_vocab /path/to/save/ggnn-src-vocab.txt
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Counter as CounterType

# --- Special Tokens ---
UNK_TOKEN = "<unk>"
BLANK_TOKEN = "<blank>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
GGNN_SEP_TOKEN = "<EOT>"
GGNN_EDGE_SEP_TOKEN = ","


def find_ggnn_source_files(root_dir: Path) -> List[Path]:
    """Recursively finds all files matching '*-src-ggnn.txt' in a directory."""
    if not root_dir.is_dir():
        print(f"Error: Provided data directory does not exist: {root_dir}", file=sys.stderr)
        return []

    print(f"Searching for '*-src-ggnn.txt' files in '{root_dir}'...")
    files = list(root_dir.rglob("*-src-ggnn.txt"))
    print(f"Found {len(files)} matching source files.")
    return files


def parse_ggnn_files(file_paths: List[Path]) -> CounterType[str]:
    """Parses GGNN-formatted files and counts all unique tokens."""
    token_counter: CounterType[str] = Counter()

    for file_path in file_paths:
        print(f"--> Parsing file: {file_path.name}")
        try:
            with file_path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(GGNN_SEP_TOKEN)
                    if len(parts) != 3:
                        print(
                            f"Warning: Malformed line #{i} in {file_path.name}. "
                            f"Expected 3 parts separated by '<EOT>', got {len(parts)}. "
                            f"Line: '{line}'", file=sys.stderr
                        )
                        continue

                    atom_part, feature_part, edge_part = parts
                    token_counter.update(atom_part.strip().split())
                    token_counter.update(feature_part.strip().split())
                    # Correctly handle comma as a separator, not a token
                    token_counter.update(edge_part.replace(GGNN_EDGE_SEP_TOKEN, " ").split())

        except Exception as e:
            print(f"Error: Failed to read or parse {file_path}: {e}", file=sys.stderr)

    return token_counter


def write_ordered_ggnn_vocab(
        output_path: Path, token_counter: CounterType[str]
) -> None:
    """
    Writes a vocabulary file in the specific order required by OpenNMT's GGNN.
    """
    atom_symbols: CounterType[str] = Counter()
    numeric_tokens: CounterType[str] = Counter()

    # --- 1. Classify all tokens ---
    for token, count in token_counter.items():
        if token.isdigit():
            numeric_tokens[token] = count
        elif re.fullmatch(r"^[A-Za-z][a-z]?$", token):
            atom_symbols[token] = count
        elif token not in [
            UNK_TOKEN, BLANK_TOKEN, BOS_TOKEN, EOS_TOKEN,
            GGNN_SEP_TOKEN, GGNN_EDGE_SEP_TOKEN
        ]:
            print(f"Warning: Found non-standard atom symbol: '{token}'. Classifying as atom.", file=sys.stderr)
            atom_symbols[token] = count

    # --- 2. Sort tokens within categories ---
    sorted_atoms = sorted(atom_symbols.items(), key=lambda x: x[1], reverse=True)
    sorted_numbers = sorted(numeric_tokens.items(), key=lambda x: int(x[0]))

    # --- 3. Write file in the required order ---
    print(f"Writing correctly ordered vocabulary to {output_path}...")
    try:
        with output_path.open("w", encoding="utf-8") as f:
            # Part 1: Standard special tokens
            f.write(f"{UNK_TOKEN}\n")
            f.write(f"{BLANK_TOKEN}\n")
            f.write(f"{BOS_TOKEN}\n")
            f.write(f"{EOS_TOKEN}\n")

            # Part 2: Node Label Vocabulary (Atom Symbols)
            for token, _ in sorted_atoms:
                f.write(f"{token}\n")

            # Part 3: The critical GGNN separator token
            f.write(f"{GGNN_SEP_TOKEN}\n")

            # Part 4: All other non-label tokens (Features, Edge info)
            for token, _ in sorted_numbers:
                f.write(f"{token}\n")

            f.write(f"{GGNN_EDGE_SEP_TOKEN}\n")

        print("Vocabulary file successfully created.")
    except IOError as e:
        print(f"Error: Failed to write vocabulary file to {output_path}: {e}", file=sys.stderr)


def main() -> None:
    """Main function to orchestrate the vocabulary building process."""
    parser = argparse.ArgumentParser(
        description="Build a source vocabulary for OpenNMT-py's GGNN encoder.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=Path,
        help="Path to the root directory containing '*-src-ggnn.txt' files.",
    )
    parser.add_argument(
        "--output_vocab",
        required=True,
        type=Path,
        help="Path to save the generated source vocabulary file.",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_vocab.parent.mkdir(parents=True, exist_ok=True)

    source_files = find_ggnn_source_files(args.data_dir)
    if not source_files:
        print("Error: No source files found. Aborting.", file=sys.stderr)
        return

    token_counts = parse_ggnn_files(source_files)
    if not token_counts:
        print("Error: No tokens were extracted from the source files. Aborting.", file=sys.stderr)
        return

    write_ordered_ggnn_vocab(args.output_vocab, token_counts)

    print("\n--- Process Complete ---")
    print(f"Generated vocabulary file: {args.output_vocab}")
    print("Next Step: Update your 'config-ggnn.yaml' to use this new vocabulary.")


if __name__ == "__main__":
    main()