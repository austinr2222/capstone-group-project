"""
AHRF 2025 Data Exploration
==========================
Explores shape, data types, missing values, and basic summary statistics
across all AHRF 2025 CSV files.
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "AHRF_data")

# ── 1. Load all AHRF files ──────────────────────────────────────────────────
files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
datasets = {}
for f in files:
    name = f.replace(".csv", "")
    datasets[name] = pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False)

# ── 2. Shape overview ───────────────────────────────────────────────────────
print("=" * 70)
print("SHAPE OVERVIEW")
print("=" * 70)
for name, df in datasets.items():
    print(f"  {name:20s} → {df.shape[0]:,} rows × {df.shape[1]:,} columns")
print()

# ── 3. Data types per file ──────────────────────────────────────────────────
print("=" * 70)
print("DATA TYPE SUMMARY (per file)")
print("=" * 70)
for name, df in datasets.items():
    type_counts = df.dtypes.value_counts()
    type_str = ", ".join(f"{dtype}: {cnt}" for dtype, cnt in type_counts.items())
    print(f"  {name:20s} → {type_str}")
print()

# ── 4. Missing values ──────────────────────────────────────────────────────
print("=" * 70)
print("MISSING VALUE SUMMARY")
print("=" * 70)
for name, df in datasets.items():
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    cols_with_missing = (df.isna().sum() > 0).sum()
    print(f"  {name:20s} → {missing_cells:>10,} missing cells "
          f"({pct:5.1f}%)  |  {cols_with_missing} / {df.shape[1]} columns affected")
print()

# ── 5. Top columns with missing data (main file) ───────────────────────────
print("=" * 70)
print("TOP 20 COLUMNS WITH MOST MISSING VALUES (AHRF2025 main file)")
print("=" * 70)
main = datasets["AHRF2025"]
missing_by_col = main.isna().sum().sort_values(ascending=False)
missing_by_col = missing_by_col[missing_by_col > 0].head(20)
for col, cnt in missing_by_col.items():
    pct = cnt / len(main) * 100
    print(f"  {col:50s}  {cnt:>5,}  ({pct:5.1f}%)")
print()

# ── 6. Detailed look at key sub-files ──────────────────────────────────────
key_files = {
    "AHRF2025pop": "Population",
    "AHRF2025hp":  "Health Professions",
    "AHRF2025hf":  "Health Facilities",
    "AHRF2025env": "Environment",
}

print("=" * 70)
print("DESCRIPTIVE STATISTICS FOR KEY NUMERIC COLUMNS")
print("=" * 70)
for name, label in key_files.items():
    df = datasets[name]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # pick first 5 numeric columns (skip fips)
    sample_cols = [c for c in numeric_cols if "fips" not in c][:5]
    if sample_cols:
        print(f"\n--- {label} ({name}) ---")
        print(df[sample_cols].describe().round(1).to_string())
        print()

# ── 7. Unique states & counties ─────────────────────────────────────────────
print("=" * 70)
print("GEOGRAPHIC COVERAGE")
print("=" * 70)
if "fips_st" in main.columns:
    print(f"  Unique state FIPS codes:  {main['fips_st'].nunique()}")
if "fips_cnty" in main.columns:
    print(f"  Unique county FIPS codes: {main['fips_st_cnty'].nunique()}")
if "st_name" in main.columns:
    print(f"  States represented:       {main['st_name'].nunique()}")
    print(f"  Example states:           {', '.join(main['st_name'].dropna().unique()[:10])}")
print()

# ── 8. Rural-Urban breakdown ────────────────────────────────────────────────
print("=" * 70)
print("RURAL-URBAN CONTINUUM CODE DISTRIBUTION")
print("=" * 70)
if "rural_urban_contnm_23" in main.columns:
    ruc = main["rural_urban_contnm_23"].value_counts().sort_index()
    for code, cnt in ruc.items():
        print(f"  Code {code}: {cnt:>5,} counties")
print()

print("Exploration complete.")
