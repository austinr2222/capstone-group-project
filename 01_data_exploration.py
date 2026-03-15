"""
AHRF 2025 — Combined State & County Data Exploration
=====================================================
Explores shape, data types, missing values, and basic summary statistics
for both the AHRF State/National (SN) 2025 health workforce file AND
the AHRF County-level 2025 files.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "AHRF_data")
STATE_CSV = os.path.join(DATA_DIR, "ahrfsn2025.csv")
COUNTY_DIR = os.path.join(DATA_DIR, "NCHWA-2024-2025+AHRF+COUNTY+CSV")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PART A — STATE-LEVEL DATA (ahrfsn2025.csv)                               ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("=" * 70)
print("PART A: STATE-LEVEL DATA (AHRF SN 2025)")
print("=" * 70)

df = pd.read_csv(STATE_CSV, low_memory=False)

# ── A1. Shape ────────────────────────────────────────────────────────────────
print(f"\nA1. SHAPE")
print(f"  Rows (states/territories):  {df.shape[0]}")
print(f"  Columns:                    {df.shape[1]:,}")

# ── A2. Data types ───────────────────────────────────────────────────────────
print(f"\nA2. DATA TYPE SUMMARY")
for dtype, cnt in df.dtypes.value_counts().items():
    print(f"  {str(dtype):12s}  {cnt:>6,} columns")

# ── A3. Missing values ──────────────────────────────────────────────────────
print(f"\nA3. MISSING VALUE OVERVIEW")
total_cells = df.shape[0] * df.shape[1]
missing_cells = df.isna().sum().sum()
cols_with_missing = (df.isna().sum() > 0).sum()
rows_complete = (df.isna().sum(axis=1) == 0).sum()
print(f"  Total cells:             {total_cells:>10,}")
print(f"  Missing cells:           {missing_cells:>10,}  ({missing_cells/total_cells*100:.1f}%)")
print(f"  Columns with any NaN:    {cols_with_missing:>10,} / {df.shape[1]:,}")
print(f"  Rows fully complete:     {rows_complete:>10,} / {df.shape[0]}")

# ── A4. Top 20 columns with most missing ─────────────────────────────────────
print(f"\nA4. TOP 20 COLUMNS WITH MOST MISSING VALUES")
missing_by_col = df.isna().sum().sort_values(ascending=False)
missing_by_col = missing_by_col[missing_by_col > 0].head(20)
if len(missing_by_col) == 0:
    print("  No missing values found!")
else:
    for col, cnt in missing_by_col.items():
        pct = cnt / len(df) * 100
        print(f"  {col:45s}  {cnt:>3} / {len(df)}  ({pct:5.1f}%)")

# ── A5. Profession groups ───────────────────────────────────────────────────
print(f"\nA5. HEALTH PROFESSION GROUPS IDENTIFIED")
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

for prefix, label in profession_map.items():
    cols = [c for c in df.columns if c.startswith(prefix + "_") or c == prefix + "_23"]
    total_col = f"{prefix}_wkforc_23" if f"{prefix}_wkforc_23" in df.columns else f"{prefix}_23"
    total_val = df[total_col].sum() if total_col in df.columns else "N/A"
    total_str = f"{total_val:>12,.0f}" if isinstance(total_val, (int, float)) else "N/A"
    print(f"  {label:40s}  {len(cols):>3} cols  |  US total: {total_str}")

# ── A6. Top states by physician workforce ────────────────────────────────────
print(f"\nA6. TOP 10 STATES BY TOTAL PHYSICIAN WORKFORCE")
if "phys_wkforc_23" in df.columns:
    top_states = (df[["st_abbrev", "phys_wkforc_23"]]
                  .dropna()
                  .sort_values("phys_wkforc_23", ascending=False)
                  .head(10))
    for _, row in top_states.iterrows():
        print(f"  {row['st_abbrev']:5s}  {row['phys_wkforc_23']:>10,.0f}")

# ── A7. Gender breakdown ────────────────────────────────────────────────────
print(f"\nA7. GENDER COMPOSITION (% FEMALE) BY PROFESSION — US TOTAL")
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

# ── A8. Median wages ────────────────────────────────────────────────────────
print(f"\nA8. MEDIAN HOURLY WAGES (2024 BLS) — US AVERAGE ACROSS STATES")
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

# ── A9. Descriptive stats ───────────────────────────────────────────────────
print(f"\nA9. DESCRIPTIVE STATISTICS — KEY WORKFORCE COUNTS (PER STATE)")
key_cols = [c for c in ["phys_wkforc_23", "rn_23", "pa_23", "aprn_23",
                         "lpnlvn_23", "dent_23", "pharm_23", "psychol_23",
                         "socwk_23", "emt_parmdcs_23"] if c in df.columns]
if key_cols:
    print(df[key_cols].describe().round(0).to_string())

# ── A10. Geographic coverage ────────────────────────────────────────────────
print(f"\nA10. GEOGRAPHIC COVERAGE")
print(f"  States/territories:  {df['st_abbrev'].nunique()}")
print(f"  Entries:             {', '.join(df['st_abbrev'].dropna().tolist())}")
print()


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PART B — COUNTY-LEVEL DATA (8 CSV files, 3,235 counties)                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("=" * 70)
print("PART B: COUNTY-LEVEL DATA (AHRF 2025)")
print("=" * 70)

# ── B1. Load all county files & shape overview ──────────────────────────────
county_files = sorted(f for f in os.listdir(COUNTY_DIR) if f.endswith(".csv"))
county_datasets = {}
for f in county_files:
    name = f.replace(".csv", "")
    county_datasets[name] = pd.read_csv(os.path.join(COUNTY_DIR, f), low_memory=False)

print(f"\nB1. SHAPE OVERVIEW — COUNTY FILES")
for name, cdf in county_datasets.items():
    print(f"  {name:20s} → {cdf.shape[0]:,} rows × {cdf.shape[1]:,} columns")

# ── B2. Data types per file ──────────────────────────────────────────────────
print(f"\nB2. DATA TYPE SUMMARY (per file)")
for name, cdf in county_datasets.items():
    type_counts = cdf.dtypes.value_counts()
    type_str = ", ".join(f"{dtype}: {cnt}" for dtype, cnt in type_counts.items())
    print(f"  {name:20s} → {type_str}")

# ── B3. Missing values per file ──────────────────────────────────────────────
print(f"\nB3. MISSING VALUE SUMMARY (per file)")
for name, cdf in county_datasets.items():
    total_cells = cdf.shape[0] * cdf.shape[1]
    missing_cells = cdf.isna().sum().sum()
    pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    cols_affected = (cdf.isna().sum() > 0).sum()
    print(f"  {name:20s} → {missing_cells:>10,} missing cells "
          f"({pct:5.1f}%)  |  {cols_affected} / {cdf.shape[1]} columns affected")

# ── B4. Top columns with missing data (main county file) ────────────────────
print(f"\nB4. TOP 20 COLUMNS WITH MOST MISSING (AHRF2025 main file)")
main = county_datasets["AHRF2025"]
missing_by_col = main.isna().sum().sort_values(ascending=False)
missing_by_col = missing_by_col[missing_by_col > 0].head(20)
for col, cnt in missing_by_col.items():
    pct = cnt / len(main) * 100
    print(f"  {col:50s}  {cnt:>5,}  ({pct:5.1f}%)")

# ── B5. Geographic coverage ─────────────────────────────────────────────────
geo = county_datasets["AHRF2025geo"]
print(f"\nB5. GEOGRAPHIC COVERAGE")
if "fips_st" in geo.columns:
    print(f"  Unique state FIPS codes:  {geo['fips_st'].nunique()}")
if "fips_st_cnty" in geo.columns:
    print(f"  Total counties:           {geo['fips_st_cnty'].nunique():,}")
if "st_name" in geo.columns:
    print(f"  States represented:       {geo['st_name'].nunique()}")
    print(f"  Example states:           {', '.join(geo['st_name'].dropna().unique()[:10])}")

# ── B6. Rural-Urban breakdown ────────────────────────────────────────────────
print(f"\nB6. RURAL-URBAN CONTINUUM CODE DISTRIBUTION")
ruc_labels = {
    1: "1-Metro >=1M", 2: "2-Metro 250K-1M", 3: "3-Metro <250K",
    4: "4-Nonmetro >=20K", 5: "5-Nonmetro 20K adj",
    6: "6-Nonmetro 2.5-20K", 7: "7-Nonmetro 2.5-20K adj",
    8: "8-Rural <2.5K", 9: "9-Rural <2.5K adj"
}
if "rural_urban_contnm_23" in geo.columns:
    ruc = geo["rural_urban_contnm_23"].value_counts().sort_index()
    for code, cnt in ruc.items():
        label = ruc_labels.get(int(code), f"Code {code}")
        print(f"  {label:30s}  {cnt:>5,} counties")

# ── B7. Descriptive stats for key county-level variables ─────────────────────
print(f"\nB7. DESCRIPTIVE STATISTICS — KEY COUNTY VARIABLES")
pop = county_datasets["AHRF2025pop"]
hp = county_datasets["AHRF2025hp"]
hf = county_datasets["AHRF2025hf"]
env = county_datasets["AHRF2025env"]

# Build a merged county-level summary frame
cnty = geo[["fips_st_cnty", "cnty_name_st_abbrev", "st_name_abbrev",
            "cens_regn_name", "rural_urban_contnm_23"]].copy()
cnty = cnty.merge(pop[["fips_st_cnty", "popn_est_23"]], on="fips_st_cnty", how="left")
cnty = cnty.merge(hp[["fips_st_cnty", "phys_nf_prim_care_pc_exc_rsdt_23",
                       "md_nf_activ_23"]], on="fips_st_cnty", how="left")
cnty = cnty.merge(hf[["fips_st_cnty", "hosp_23", "nurs_fac_23"]], on="fips_st_cnty", how="left")
cnty = cnty.merge(env[["fips_st_cnty", "popn_densty_per_squr_mi_20",
                        "good_air_qulty_dys_pct_24"]], on="fips_st_cnty", how="left")

# PCP per 100k
cnty["pcp_per_100k"] = (cnty["phys_nf_prim_care_pc_exc_rsdt_23"]
                         / cnty["popn_est_23"].replace(0, np.nan) * 100_000)
cnty["metro_status"] = cnty["rural_urban_contnm_23"].apply(
    lambda x: "Metro" if x in [1, 2, 3] else ("Nonmetro" if x in [4, 5, 6, 7] else "Rural")
)

summary_cols = ["popn_est_23", "pcp_per_100k", "md_nf_activ_23",
                "hosp_23", "nurs_fac_23", "popn_densty_per_squr_mi_20",
                "good_air_qulty_dys_pct_24"]
print(cnty[summary_cols].describe().round(1).to_string())

# ── B8. County stats by Metro Status ────────────────────────────────────────
print(f"\nB8. COUNTY AVERAGES BY METRO STATUS")
metro_summary = (cnty.groupby("metro_status")[summary_cols]
                 .mean().round(1))
print(metro_summary.to_string())

# ── B9. Counties with zero primary care physicians ──────────────────────────
print(f"\nB9. COUNTIES WITH ZERO PRIMARY CARE PHYSICIANS")
zero_pcp = cnty[cnty["phys_nf_prim_care_pc_exc_rsdt_23"] == 0]
total_zero = len(zero_pcp)
print(f"  Total counties with 0 PCPs:  {total_zero} / {len(cnty)} "
      f"({total_zero/len(cnty)*100:.1f}%)")
if total_zero > 0:
    by_metro = zero_pcp["metro_status"].value_counts()
    for status, cnt in by_metro.items():
        print(f"    {status:12s}  {cnt:>4}")

# ── B10. Counties with zero hospitals ────────────────────────────────────────
print(f"\nB10. COUNTIES WITH ZERO HOSPITALS")
zero_hosp = cnty[cnty["hosp_23"] == 0]
total_zero_h = len(zero_hosp)
print(f"  Total counties with 0 hospitals:  {total_zero_h} / {len(cnty)} "
      f"({total_zero_h/len(cnty)*100:.1f}%)")
if total_zero_h > 0:
    by_metro = zero_hosp["metro_status"].value_counts()
    for status, cnt in by_metro.items():
        print(f"    {status:12s}  {cnt:>4}")

# ── B11. Top/bottom 10 counties by PCP rate ──────────────────────────────────
print(f"\nB11. TOP 10 COUNTIES BY PCP RATE (per 100K)")
top_pcp = (cnty.dropna(subset=["pcp_per_100k"])
           .query("popn_est_23 >= 10000")  # avoid tiny counties with huge rates
           .nlargest(10, "pcp_per_100k"))
for _, row in top_pcp.iterrows():
    print(f"  {row['cnty_name_st_abbrev']:35s}  "
          f"Pop: {row['popn_est_23']:>10,.0f}  "
          f"PCP/100K: {row['pcp_per_100k']:>6.1f}")

print(f"\nB12. BOTTOM 10 COUNTIES BY PCP RATE (pop >= 10K, > 0 PCPs)")
bottom_pcp = (cnty.dropna(subset=["pcp_per_100k"])
              .query("popn_est_23 >= 10000 and pcp_per_100k > 0")
              .nsmallest(10, "pcp_per_100k"))
for _, row in bottom_pcp.iterrows():
    print(f"  {row['cnty_name_st_abbrev']:35s}  "
          f"Pop: {row['popn_est_23']:>10,.0f}  "
          f"PCP/100K: {row['pcp_per_100k']:>6.1f}")

print()
print("Exploration complete.")
