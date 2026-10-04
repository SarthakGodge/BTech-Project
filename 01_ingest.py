"""
Stage 1 - Raw CSV ingest and cleaning.

Reads each CIC-IDS2018 day-file in chunks, cleans it, and writes one
Parquet file per day. Parquet because it is ~8x smaller, keeps dtypes,
and loads ~20x faster than re-parsing CSV on every later run.

Real quirks in these files this script handles:
  * Header rows repeat in the middle of several files (the CSVs were
    concatenated by the authors). Any row where Dst Port == "Dst Port"
    is a stray header and is dropped.
  * The 20-02 file has four extra leading columns (Flow ID, Src IP,
    Src Port, Dst IP) that no other file has.
  * Rate columns contain "Infinity" and "NaN" as literal strings.
  * Timestamp format is inconsistent: dayfirst 24h in most files,
    12h with AM/PM in the 20-02 file.

Run:  python 01_ingest.py
"""
import sys
import numpy as np
import pandas as pd

from config import RAW_DIR, PARQUET_DIR, DAY_SPLITS, INCLUDE_0220

CHUNK = 500_000


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    # Drop the identity columns present only in the 20-02 file.
    for col in ("Flow ID", "Src IP", "Src Port", "Dst IP"):
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = normalise_columns(df)

    # 1. Stray repeated header rows.
    if "Dst Port" in df.columns:
        df = df[df["Dst Port"].astype(str).str.strip() != "Dst Port"]
    if df.empty:
        return df

    # 2. Timestamp -> datetime (needed for time ordering later).
    ts = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True,
                        format="mixed")
    df = df.assign(Timestamp=ts)
    df = df[df["Timestamp"].notna()]

    # 3. Everything except Timestamp/Label must be numeric.
    feature_cols = [c for c in df.columns if c not in ("Timestamp", "Label")]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # 4. Inf -> NaN. CICFlowMeter divides by Flow Duration; zero-duration
    #    flows produce Infinity in Flow Byts/s and Flow Pkts/s.
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # 5. A row missing its core timing field is unusable - drop it.
    dur_col = "Flow Duration" if "Flow Duration" in df.columns else None
    if dur_col:
        df = df[df[dur_col].notna()]

    # 6. Downcast to float32 - halves memory and matches what TF wants.
    df[feature_cols] = df[feature_cols].astype("float32")

    df["Label"] = df["Label"].astype(str).str.strip()
    return df


def ingest_day(csv_path, day_key):
    out = PARQUET_DIR / f"{day_key}.parquet"
    if out.exists():
        print(f"  skip (exists): {out.name}")
        return

    frames, n_raw = [], 0
    reader = pd.read_csv(csv_path, chunksize=CHUNK, low_memory=False,
                         na_values=["Infinity", "-Infinity", "NaN", "nan", ""])
    for i, chunk in enumerate(reader):
        n_raw += len(chunk)
        cleaned = clean_chunk(chunk)
        if not cleaned.empty:
            frames.append(cleaned)
        print(f"    chunk {i}: {len(chunk):>8,} read -> {len(cleaned):>8,} kept",
              end="\r")

    if not frames:
        print(f"  !! nothing survived cleaning in {csv_path.name}")
        return

    df = pd.concat(frames, ignore_index=True)

    # Median-fill the remaining NaNs (from the Inf conversion). Median, not
    # mean: flow features are heavily right-skewed and the mean is dragged
    # around by outliers.
    feature_cols = [c for c in df.columns if c not in ("Timestamp", "Label")]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    # Time ordering is the whole point of this project - sort once, here.
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)

    df.to_parquet(out, index=False, compression="snappy")
    print(f"\n  wrote {out.name}: {len(df):,} rows "
          f"({n_raw - len(df):,} dropped), {df.shape[1]} cols")
    print("    labels:", dict(df["Label"].value_counts()))


def main():
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        sys.exit(f"No CSVs in {RAW_DIR}. Run 00_download.sh first.")

    for csv_path in csvs:
        # Match "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv" to a key.
        day_key = next((k for k in DAY_SPLITS if k in csv_path.name), None)
        if day_key is None:
            print(f"  ?? unrecognised file, skipping: {csv_path.name}")
            continue
        if day_key == "Thursday-20-02-2018" and not INCLUDE_0220:
            print(f"  skip (INCLUDE_0220=False): {csv_path.name}")
            continue

        print(f"\n[{day_key}] {csv_path.name}")
        ingest_day(csv_path, day_key)


if __name__ == "__main__":
    main()
