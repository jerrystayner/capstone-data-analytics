# 01_clean_and_merge.py
# Jerry Stayner / CIS 480
#
# Loads the six tract-level CSVs from PolicyMap and merges them on the FIPS
# code into one file. Grocery density gets added later in script 02.

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"
OUT_FILE = DATA_DIR / "mesa_tract_data.csv"

# Each raw file plus the column inside it I actually want, and what I want
# to rename it to. PolicyMap uses short codes like 'pbachp' that aren't
# obvious later, so I'm renaming everything up front.
TRACT_FILES = {
    "obesity_raw.csv":    ("p_crd_obesity", "obesity_pct"),
    "income_raw.csv":     ("mhhinc",        "median_income"),
    "uninsured_raw.csv":  ("ppopwoins",     "uninsured_pct"),
    "bachelors_raw.csv":  ("pbachp",        "bachelors_pct"),
    "drove_raw.csv":      ("ptranmv",       "drove_pct"),
    "population_raw.csv": ("cpop",          "population"),
}


def load_tract_csv(filename, value_col, new_name):
    # Read one PolicyMap export and return just [fips, value].
    # IMPORTANT: dtype=str on the read so the leading 0 in Arizona's
    # state code (04...) doesn't get dropped. Took me a while to catch
    # that the first time I ran this.
    df = pd.read_csv(DATA_DIR / filename, dtype=str)
    df = df.rename(columns={"GeoID": "fips"})
    df["fips"] = df["fips"].str.strip().str.zfill(11)
    df[new_name] = pd.to_numeric(df[value_col], errors="coerce")
    return df[["fips", new_name]]


def main():
    print("Merging tract-level files...")
    merged = None
    for fname, (value_col, new_name) in TRACT_FILES.items():
        part = load_tract_csv(fname, value_col, new_name)
        print(f"  {fname:25s} -> {new_name:18s} ({len(part)} rows)")
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on="fips", how="outer")

    print(f"\nMerged: {len(merged)} tracts, {merged.shape[1]} columns")
    print("Missing values per column:")
    print(merged.isna().sum().to_string())

    # Strip out any junk rows where FIPS isn't actually an 11-digit number
    # (PolicyMap occasionally tacks on a header echo at the bottom)
    merged = merged[merged["fips"].str.match(r"^\d{11}$", na=False)].copy()

    merged.to_csv(OUT_FILE, index=False)
    print(f"\nSaved {OUT_FILE} ({len(merged)} rows)")
    print(merged.head().to_string(index=False))


if __name__ == "__main__":
    main()
