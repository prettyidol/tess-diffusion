import argparse

def load_nonempty_lines(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def process(oneline_in: str, base_facts: str, out_path: str, joiner: str = " ||| ", skip_if_single: bool = True):
    base = load_nonempty_lines(base_facts)
    total = 0
    appended = 0
    with open(oneline_in, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8", newline="") as fout:
        seq_idx = 0  # count non-empty sequences to align with base facts
        for _, line in enumerate(fin):
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
            total += 1
            if seq_idx >= len(base):
                raise IndexError(f"Base facts count ({len(base)}) is less than sequence lines (need at least {seq_idx+1}).")
            seq = line
            if skip_if_single:
                parts = [seg.strip() for seg in line.split(joiner)]
                if len(parts) <= 1:
                    fout.write(seq + "\n")
                    seq_idx += 1
                    continue
            # append base quad for this query
            seq = seq + joiner + base[seq_idx]
            appended += 1
            fout.write(seq + "\n")
            seq_idx += 1
    print({"wrote": total, "appended": appended, "out": out_path})


def main():
    ap = argparse.ArgumentParser(description="Append per-query base quadruple (from tess_all.txt) to end of each oneline sequence.")
    ap.add_argument("--oneline_in", type=str, required=True, help="Input oneline file (e.g., tess_train1_oneline.txt)")
    ap.add_argument("--base_facts", type=str, required=True, help="Base facts file (e.g., tess_all.txt), one quadruple per query")
    ap.add_argument("--out", type=str, required=True, help="Output oneline file path")
    ap.add_argument("--joiner", type=str, default=" ||| ", help="Joiner token used in oneline file")
    ap.add_argument("--no_skip_if_single", action="store_true", help="Do not skip single-quad sequences; append anyway")
    args = ap.parse_args()

    process(
        oneline_in=args.oneline_in,
        base_facts=args.base_facts,
        out_path=args.out,
        joiner=args.joiner,
        skip_if_single=not args.no_skip_if_single,
    )


if __name__ == "__main__":
    main()
