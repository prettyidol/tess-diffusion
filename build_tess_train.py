import os
import sys
from datetime import date
from bisect import bisect_left


def parse_date(s: str) -> date:
    """Parse a date string like YYYY-MM-DD or YYYY-M-D into a date object.
    Falls back to stripping whitespace and handling common variants.
    """
    s = s.strip()
    # Handle possible trailing characters
    s = s.replace("/", "-")
    parts = s.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid date format: {s}")
    y, m, d = parts
    y = int(y)
    m = int(m)
    d = int(d)
    return date(y, m, d)


def read_train(path: str):
    """Read train.txt containing quadruples: head\trelation\ttail\tYYYY-MM-DD.
    Returns a list of dicts with fields and preserves the original line.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
            # Prefer TSV
            parts = line.split("\t")
            if len(parts) == 4:
                h, r, t, ds = parts
            else:
                # Fallback: try whitespace split assuming last token is date
                ws = line.split()
                if len(ws) < 4:
                    # Skip malformed line
                    continue
                ds = ws[-1]
                # Heuristic: the rest split into three by tabs/spaces is ambiguous; skip to avoid corruption
                # Prefer strict TSV to avoid mis-parsing entities with spaces
                # If needed, user can re-save as TSV.
                # Here we skip non-TSV lines.
                continue
            try:
                d = parse_date(ds)
            except Exception:
                # Skip lines with invalid date
                continue
            records.append({
                "head": h,
                "relation": r,
                "tail": t,
                "date": d,
                "date_str": f"{d.year:04d}-{d.month:02d}-{d.day:02d}",
                "line": f"{h}\t{r}\t{t}\t{d.year:04d}-{d.month:02d}-{d.day:02d}",
                "index": idx,
            })
    return records


def build_index_by_head(records):
    """Build per-head sorted lists by (date asc, index asc) for efficient lookup."""
    by_head = {}
    for rec in records:
        h = rec["head"]
        by_head.setdefault(h, []).append((rec["date"], rec["index"]))
    # Sort each list ascending by date, then by original index to keep stable file order per day
    for h in by_head:
        by_head[h].sort(key=lambda x: (x[0], x[1]))
    return by_head


def build_position_lookup(records):
    """Map from (head, index) -> position in the per-head sorted array to enable bisect lookup."""
    pos = {}
    by_head = {}
    for rec in records:
        h = rec["head"]
        by_head.setdefault(h, []).append((rec["date"], rec["index"]))
    for h in by_head:
        arr = sorted(by_head[h], key=lambda x: (x[0], x[1]))
        for i, (_, idx) in enumerate(arr):
            pos[(h, idx)] = (arr, i)
    return pos


def select_prev_five(records, pos_lookup, rec, idx_to_line):
    """Select up to 5 previous quadruples for the same head strictly before rec.date.
    Selection order: from most recent day going backward (date desc), within the same day keep original order.
    Returns a list of lines to output (strings). If none exist, returns [rec.line] itself.
    """
    h = rec["head"]
    key = (h, rec["index"])  # current record identifier
    if key not in pos_lookup:
        return [rec["line"]]
    arr, _ = pos_lookup[key]
    # arr is sorted ascending (date, index). Find cutoff for strictly earlier than rec.date
    # Since we don't have timestamps, use bisect_left on date to find first >= rec.date
    # Build a list of just dates for bisect
    dates_only = [dt for (dt, _) in arr]
    cutoff = bisect_left(dates_only, rec["date"])  # first position with date >= rec.date
    if cutoff <= 0:
        # No earlier records
        return [rec["line"]]
    prev_slice = arr[:cutoff]
    # Now take up to 5 most recent: reverse by date desc, within same day keep original file order.
    # To keep within-day order while reversing days, we can group by date.
    # Simpler: sort by (date desc, index asc) and take first 5.
    prev_sorted = sorted(prev_slice, key=lambda x: (x[0], x[1]))  # asc
    prev_sorted = prev_sorted[::-1]  # now date desc, index desc; adjust to index asc within day
    # To enforce index asc within day, re-sort stable by index ascending within equal dates.
    # We can do a stable sort: first by index asc, then stable sort by date desc.
    prev_sorted = sorted(prev_slice, key=lambda x: x[1])
    prev_sorted = sorted(prev_sorted, key=lambda x: x[0], reverse=True)
    chosen = prev_sorted[:5]
    # Map chosen indices back to lines in original records
    return [idx_to_line[i] for (_, i) in chosen]


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(repo_root, "all_facts1.txt")
    out_path = os.path.join(repo_root, "tess_train_all.txt")

    if not os.path.exists(in_path):
        print(f"Input file not found: {in_path}")
        sys.exit(1)

    print("Reading train.txt ...")
    records = read_train(in_path)
    if not records:
        print("No valid records found in train.txt")
        sys.exit(1)
    print(f"Loaded {len(records)} quadruples.")

    print("Indexing by head ...")
    pos_lookup = build_position_lookup(records)

    print("Building sequences ...")
    # Precompute index -> line mapping once for efficiency
    idx_to_line = {r["index"]: r["line"] for r in records}
    with open(out_path, "w", encoding="utf-8", newline="") as out:
        for rec in records:
            seq_lines = select_prev_five(records, pos_lookup, rec, idx_to_line)
            for line in seq_lines:
                out.write(line + "\n")
            out.write("\n")  # blank line between sequences

    print(f"Done. Wrote sequences to {out_path}")


if __name__ == "__main__":
    main()
