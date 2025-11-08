# parse_attack.py
from pathlib import Path
import pandas as pd
import numpy as np

# Inputs and outputs
INPUT_FILES = [
    "datasets/DoS_dataset.csv",
    "datasets/Fuzzy_dataset.csv",
    "datasets/gear_dataset.csv",
    "datasets/RPM_dataset.csv",
]
OUTPUT_DIR = Path("datasets")

# Final output schema
OUT_COLUMNS = [
    "timestamp",
    "can_id",
    "flags",
    "dlc",
    "data_0",
    "data_1",
    "data_2",
    "data_3",
    "data_4",
    "data_5",
    "data_6",
    "data_7",
    "label",
]

# Expected raw schema after robust load (12 cols)
RAW_COLUMNS = [
    "timestamp",
    "id",
    "dlc",
    "b0",
    "b1",
    "b2",
    "b3",
    "b4",
    "b5",
    "b6",
    "b7",
    "label",  # kept in output
]


def looks_hex(s: str) -> bool:
    s = s.strip().lower()
    return s.startswith("0x") or any(c in s for c in "abcdef")


def to_int_byte(x):
    if x is None:
        return 0
    s = str(x).strip()
    if s == "":
        return 0
    try:
        return int(s, 16) if looks_hex(s) else int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return 0


def to_can_id(val):
    """Parse CAN id from hex-like or decimal."""
    if val is None:
        return 0
    s = str(val).strip().lower()
    if s == "":
        return 0
    try:
        if looks_hex(s) or (len(s) <= 4 and s.startswith("0")):
            return int(s.replace("0x", ""), 16)
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return 0


def read_messy_csv(path: Path) -> pd.DataFrame:
    """
    Robust reader:
    - Accept optional header line (if first row contains 'timestamp').
    - Split each subsequent line by ','.
    - Pad/truncate to exactly 12 fields.
    """
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        header_like = "timestamp" in first.lower() and "," in first
        if not header_like:
            parts = [p.strip() for p in first.strip().split(",")]
            if parts and any(parts):
                rows.append(parts)

        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if not parts or all(p == "" for p in parts):
                continue
            if len(parts) < 12:
                parts += [""] * (12 - len(parts))
            elif len(parts) > 12:
                parts = parts[:12]
            rows.append(parts)

    df = pd.DataFrame(rows, columns=RAW_COLUMNS, dtype=str)
    return df


def convert(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    out["can_id"] = df["id"].apply(to_can_id).astype("int64")
    out["dlc"] = df["dlc"].apply(to_int_byte).clip(lower=0, upper=8).astype("int64")

    # data bytes [0..255]
    for i, col in enumerate(["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]):
        out[f"data_{i}"] = (
            df[col].apply(to_int_byte).clip(lower=0, upper=255).astype("int64")
        )

    # flags not present -> set to 0
    out["flags"] = 0

    # normalize label (keep as string, uppercase)
    out["label"] = df["label"].astype(str).str.strip().str.upper().replace({"": np.nan})

    # reorder to final schema
    out = out[OUT_COLUMNS]

    # zero bytes beyond DLC
    mask = out["dlc"].to_numpy()
    data = out[[f"data_{i}" for i in range(8)]].to_numpy()
    for i in range(8):
        data[:, i] = np.where(mask > i, data[:, i], 0)
    out[[f"data_{i}" for i in range(8)]] = data

    # drop bad timestamps
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    return out


def main():
    root = Path.cwd()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname in INPUT_FILES:
        src = root / fname
        stem = src.stem.split("_")[0]  # 'DoS_attack_dataset_raw' -> 'DoS'
        dst = OUTPUT_DIR / f"{stem}_attack_dataset.csv"

        # Check if output already exists
        if dst.exists():
            print(f"[exists] Skipping {src.name} — output already present: {dst}")
            continue

        # Ensure input file exists before attempting to read
        if not src.exists():
            print(f"[missing] Input not found: {src} — skipping")
            continue

        print(f"[start] Reading {src}...")
        try:
            df_raw = read_messy_csv(src)
            print(f"[loaded] Raw rows: {len(df_raw)} — converting...")
            df_out = convert(df_raw)
            df_out.to_csv(dst, index=False)
            print(f"[ok]  {src.name} -> {dst} (rows: {len(df_out)})")
        except Exception as e:
            print(f"[err] {src.name}: {e}")


if __name__ == "__main__":
    main()
