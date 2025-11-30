import os


def convert(in_path: str, out_path: str, joiner: str = " ||| "):
    """Convert blank-line separated multi-line sequences into one-sequence-per-line.

    - Each non-empty line belongs to the current sequence.
    - Empty line ends the current sequence.
    - Lines within a sequence are joined by `joiner`.
    """
    total_seq = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8", newline="") as fout:
        buf = []
        for line in fin:
            line = line.rstrip("\n\r")
            if line.strip() == "":
                if buf:
                    fout.write(joiner.join(buf) + "\n")
                    total_seq += 1
                    buf = []
            else:
                buf.append(line)
        # flush last
        if buf:
            fout.write(joiner.join(buf) + "\n")
            total_seq += 1
    print(f"Wrote {total_seq} sequences to {out_path}")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(repo_root, "tess_test1.txt")
    out_path = os.path.join(repo_root, "tess_test1_oneline.txt")
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)
    convert(in_path, out_path)
