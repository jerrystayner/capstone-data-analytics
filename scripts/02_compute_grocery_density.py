# 02_compute_grocery_density.py
# Jerry Stayner / CIS 480
#
# Builds the grocery store variable. The SNAP file from PolicyMap is point
# data (one row per store) joined to the tract FIPS each store sits in.
# I count grocery-type stores per tract and compute density per 10k people.
#
# I'm only counting Supermarkets, Super Stores, and actual Grocery Stores.
# Convenience stores, restaurant-meals-program places, and "Other" get
# dropped because Hypothesis 2 is specifically about whether more grocery
# access lowers obesity, not the food-swamp side of things.

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"
SNAP_FILE  = DATA_DIR / "snap_with_tracts_raw.csv"
TRACT_FILE = DATA_DIR / "mesa_tract_data.csv"
OUT_FILE   = DATA_DIR / "mesa_tract_master.csv"

GROCERY_TYPES = {"Supermarket", "Super Store", "Grocery Store"}


def main():
    # Load all SNAP retailers in Maricopa County (already joined to tracts)
    snap = pd.read_csv(SNAP_FILE, dtype=str)
    snap = snap.rename(columns={"GeoID": "fips"})
    snap["fips"] = snap["fips"].str.strip().str.zfill(11)

    print(f"SNAP retailers in county: {len(snap)}")
    print("Store types in the county:")
    print(snap["store_type"].value_counts().to_string())

    # Keep only the grocery-style stores
    grocery = snap[snap["store_type"].isin(GROCERY_TYPES)].copy()
    print(f"\nKept after filtering to grocery types: {len(grocery)}")

    # Count stores per tract
    counts = grocery.groupby("fips").size().rename("grocery_count").reset_index()

    # Join onto the Mesa tract list. Tracts with no stores need to be 0,
    # not NaN, so the regression treats them correctly.
    tracts = pd.read_csv(TRACT_FILE, dtype={"fips": str})
    master = tracts.merge(counts, on="fips", how="left")
    master["grocery_count"] = master["grocery_count"].fillna(0).astype(int)

    # Density per 10,000 people. The 4 zero-pop tracts (airport, industrial)
    # would divide-by-zero, so swap their pop to NaN first.
    pop = master["population"].replace(0, np.nan)
    master["grocery_per_10k"] = ((master["grocery_count"] / pop) * 10_000).round(3)

    # Order columns the way the regression expects
    master = master[[
        "fips", "obesity_pct",
        "median_income", "grocery_per_10k", "drove_pct",
        "bachelors_pct", "uninsured_pct",
        "grocery_count", "population",
    ]]
    master.to_csv(OUT_FILE, index=False)

    print(f"\nSaved {OUT_FILE} ({len(master)} tracts)")
    print(f"Mesa tracts with at least 1 grocery store: {(master['grocery_count'] > 0).sum()}")
    print(f"Total grocery stores in Mesa: {master['grocery_count'].sum()}")
    print("\nGrocery counts summary:")
    print(master["grocery_count"].describe().round(2).to_string())
    print("\nFirst few rows:")
    print(master.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
