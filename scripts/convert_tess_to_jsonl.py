import os
import json
import argparse
import random


def read_blocks(in_path: str) -> list[str]:
    """Read blank-line separated blocks from tess_train1.txt and return a list of sequences (as strings).
    Each block consists of one or more lines; empty line ends a block.
    """
    blocks = []
    buf = []
    with open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n\r")
            if line.strip() == "":
                if buf:
                    blocks.append(buf)
                    buf = []
            else:
                buf.append(line)
    if buf:
        blocks.append(buf)
    return blocks


def to_jsonl(records: list[str], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fout:
        for text in records:
            json.dump({"text": text}, fout, ensure_ascii=False)
            fout.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Convert tess_train1.txt to JSONL train/valid files.")
    parser.add_argument("--input", type=str, default=None, help="Path to tess_train1.txt")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for .json files")
    parser.add_argument("--valid_ratio", type=float, default=0.01, help="Validation split ratio (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle before split")
    parser.add_argument("--joiner", type=str, default=" ||| ", help="Separator used to join lines within a block")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = args.input or os.path.join(repo_root, "tess_train1.txt")
    out_dir = args.output_dir or repo_root

    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading blocks from {in_path} ...")
    blocks = read_blocks(in_path)
    print(f"Found {len(blocks)} sequences (blocks).")

    # Join lines in each block into one string using joiner
    sequences = [args.joiner.join(lines) for lines in blocks]

    # Shuffle then split
    rng = random.Random(args.seed)
    rng.shuffle(sequences)
    n_total = len(sequences)
    n_valid = max(1, int(n_total * args.valid_ratio))
    valid = sequences[:n_valid]
    train = sequences[n_valid:]

    train_path = os.path.join(out_dir, "tess_train1_train.json")
    valid_path = os.path.join(out_dir, "tess_train1_valid.json")

    print(f"Writing train ({len(train)}) to {train_path}")
    to_jsonl(train, train_path)
    print(f"Writing valid ({len(valid)}) to {valid_path}")
    to_jsonl(valid, valid_path)
    print("Done.")


if __name__ == "__main__":
    main()
