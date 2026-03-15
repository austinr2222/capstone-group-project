"""
AHRF SN 2025 — State-Level Data Exploration
============================================
Explores shape, data types, missing values, and basic summary statistics
for the AHRF State/National (SN) 2025 health workforce file.

Data: 52 states/territories × 1,449 columns covering 30+ health
professions with demographics, work settings, wages, and education.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "AHRF_data")
CSV_FILE = os.path.join(DATA_DIR, "ahrfsn2025.csv")

# ── 1. Load ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE, low_memory=False)

print("=" * 70)
print("1. SHAPE")
print("=" * 70)
print(f"  Rows (states/territories):  {df.shape[0]}")
print(f"  Columns:                    {df.shape[1]:,}")
print()

# ── 2. Data types ────────────────────────────────────────────────────────────
print("=" * 70)
print("2. DATA TYPE SUMMARY")
print("=" * 70)
for dtype, cnt in df.dtypes.value_counts().items():
    print(f"  {str(dtype):12s}  {cnt:>6,} columns")
print()

# ── 3. Missing values overview ──────────────────────────────────────────────
print("=" * 70)
print("3. MISSING VALUE OVERVIEW")
print("=" * 70)
total_cells = df.shape[0] * df.shape[1]
missing_cells = df.isna().sum().sum()
cols_with_missing = (df.isna().sum() > 0).sum()
rows_complete = (df.isna().sum(axis=1) == 0).sum()
print(f"  Total cells:             {total_cells:>10,}")
print(f"  Missing cells:           {missing_cells:>10,}  ({missing_cells/total_cells*100:.1f}%)")
print(f"  Columns with any NaN:    {cols_with_missing:>10,} / {df.shape[1]:,}")
print(f"  Rows fully complete:     {rows_complete:>10,} / {df.shape[0]}")
print()

# ── 4. Top 20 columns with most missing values ─────────────────────────────
print("=" * 70)
print("4. TOP 20 COLUMNS WITH MOST MISSING VALUES")
print("=" * 70)
missing_by_col = df.isna().sum().sort_values(ascending=False)
missing_by_col = missing_by_col[missing_by_col > 0].head(20)
if len(missing_by_col) == 0:
    print("  No missing values found!")
else:
    for col, cnt in missing_by_col.items():
        pct = cnt / len(df) * 100
        print(f"  {col:45s}  {cnt:>3} / {len(df)}  ({pct:5.1f}%)")
print()

# ── 5. Identify profession groups ───────────────────────────────────────────
print("=" * 70)
print("5. HEALTH PROFESSION GROUPS IDENTIFIED")
print("=" * 70)
# Each profession has a total workforce column like "phys_wkforc_23" or "rn_23"
profession_map = {
    "phys":           "Physicians",
    "pa":             "Physician Assistants",
    "rn":             "Registered Nurses",
    "aprn":           "Advanced Practice RNs",
    "lpnlvn":         "Licensed Practical/Vocational Nurses",
    "dent":           "Dentists",
    "dent_hygn":      "Dental Hygienists",
    "dent_asst":      "Dental Assistants",
    "pharm":          "Pharmacists",
    "vetin":          "Veterinarians",
    "podtrst":        "Podiatrists",
    "chiro":          "Chiropractors",
    "opto":           "Optometrists",
    "opticn":         "Opticians",
    "psychol":        "Psychologists",
    "conslrs":        "Counselors",
    "socwk":          "Social Workers",
    "pt":             "Physical Therapists",
    "ot":             "Occupational Therapists",
    "resp_ther":      "Respiratory Therapists",
    "spech_path":     "Speech-Language Pathologists",
    "massg_ther":     "Massage Therapists",
    "dietn":          "Dietitians/Nutritionists",
    "medmgr":         "Medical/Health Services Managers",
    "med_secrtrs":    "Medical Secretaries/Admin Assistants",
    "clin_lab_techs": "Clinical Laboratory Technologists",
    "diag_reltd_techs": "Diagnostic Related Technologists",
    "emt_parmdcs":    "EMTs & Paramedics",
    "hlth_sup_techs": "Health Support Technicians",
    "med_rec_spec":   "Medical Records Specialists",
    "med_asst":       "Medical Assistants",
    "pers_care_aide": "Personal Care Aides",
    "nurse_aide":     "Nursing Aides/Orderlies",
}

# Count columns per profession group
for prefix, label in profession_map.items():
    cols = [c for c in df.columns if c.startswith(prefix + "_") or c == prefix + "_23"]
    # Get the total workforce column
    total_col = f"{prefix}_wkforc_23" if f"{prefix}_wkforc_23" in df.columns else f"{prefix}_23"
    total_val = df[total_col].sum() if total_col in df.columns else "N/A"
    total_str = f"{total_val:>12,.0f}" if isinstance(total_val, (int, float)) else "N/A"
    print(f"  {label:40s}  {len(cols):>3} cols  |  US total: {total_str}")
print()

# ── 6. Workforce totals by state (top 10) ───────────────────────────────────
print("=" * 70)
print("6. TOP 10 STATES BY TOTAL PHYSICIAN WORKFORCE")
print("=" * 70)
if "phys_wkforc_23" in df.columns:
    top_states = (df[["st_abbrev", "phys_wkforc_23"]]
                  .dropna()
                  .sort_values("phys_wkforc_23", ascending=False)
                  .head(10))
    for _, row in top_states.iterrows():
        print(f"  {row['st_abbrev']:5s}  {row['phys_wkforc_23']:>10,.0f}")
print()

# ── 7. Gender breakdown across professions ──────────────────────────────────
print("=" * 70)
print("7. GENDER COMPOSITION (% FEMALE) BY PROFESSION — US TOTAL")
print("=" * 70)
gender_rows = []
for prefix, label in profession_map.items():
    mal_col = f"{prefix}_mal_23"
    fem_col = f"{prefix}_fem_23"
    if mal_col in df.columns and fem_col in df.columns:
        total_m = df[mal_col].sum()
        total_f = df[fem_col].sum()
        total = total_m + total_f
        pct_f = (total_f / total * 100) if total > 0 else 0
        gender_rows.append((label, pct_f, total))

gender_rows.sort(key=lambda x: x[1], reverse=True)
for label, pct_f, total in gender_rows:
    bar = "#" * int(pct_f / 2)
    print(f"  {label:40s}  {pct_f:5.1f}%  {bar}")
print()

# ── 8. Median wages comparison ──────────────────────────────────────────────
print("=" * 70)
print("8. MEDIAN HOURLY WAGES (2024 BLS) — US AVERAGE ACROSS STATES")
print("=" * 70)
wage_rows = []
for prefix, label in profession_map.items():
    wage_col = f"{prefix}_medn_wage_24"
    if wage_col in df.columns:
        avg_wage = df[wage_col].mean()
        if not np.isnan(avg_wage):
            wage_rows.append((label, avg_wage))

wage_rows.sort(key=lambda x: x[1], reverse=True)
for label, wage in wage_rows:
    print(f"  {label:40s}  ${wage:>8,.2f}/hr")
print()

# ── 9. Descriptive statistics for key workforce columns ─────────────────────
print("=" * 70)
print("9. DESCRIPTIVE STATISTICS — KEY WORKFORCE COUNTS (PER STATE)")
print("=" * 70)
key_cols = [c for c in ["phys_wkforc_23", "rn_23", "pa_23", "aprn_23",
                         "lpnlvn_23", "dent_23", "pharm_23", "psychol_23",
                         "socwk_23", "emt_parmdcs_23"] if c in df.columns]
if key_cols:
    print(df[key_cols].describe().round(0).to_string())
print()

# ── 10. Geographic coverage ─────────────────────────────────────────────────
print("=" * 70)
print("10. GEOGRAPHIC COVERAGE")
print("=" * 70)
print(f"  States/territories:  {df['st_abbrev'].nunique()}")
print(f"  Entries:             {', '.join(df['st_abbrev'].dropna().tolist())}")
print()

# ── 11. Correlation peek ────────────────────────────────────────────────────
print("=" * 70)
print("11. TOP CORRELATIONS WITH PHYSICIAN WORKFORCE (WORKFORCE COLS)")
print("=" * 70)
wkforc_cols = [c for c in df.columns if c.endswith("_23") and
               ("wkforc" in c or c in [f"{p}_23" for p in profession_map.keys()])]
if "phys_wkforc_23" in df.columns and len(wkforc_cols) > 1:
    corrs = df[wkforc_cols].corrwith(df["phys_wkforc_23"]).drop("phys_wkforc_23", errors="ignore")
    corrs = corrs.dropna().sort_values(ascending=False).head(10)
    for col, r in corrs.items():
        print(f"  {col:40s}  r = {r:.3f}")
print()

print("Exploration complete.")
