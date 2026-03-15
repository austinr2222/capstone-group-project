"""
AHRF SN 2025 — Trellis Plots (State-Level)
===========================================
Uses plotnine (ggplot2 for Python) to create faceted/trellis plots
illustrating distributions and relationships across US states for
the health workforce data in AHRF SN 2025.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from plotnine import (
    ggplot, aes, geom_histogram, geom_boxplot, geom_point, geom_bar,
    geom_col, geom_text, geom_segment,
    facet_wrap, facet_grid, labs, theme, theme_minimal, element_text,
    scale_x_log10, scale_y_log10, scale_fill_brewer, scale_fill_manual,
    scale_color_manual, coord_flip, position_dodge,
    ggsave, after_stat, element_blank
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "AHRF_data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, "ahrfsn2025.csv"), low_memory=False)

# Map states to Census regions
region_map = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "FL": "South", "GA": "South", "MD": "South",
    "NC": "South", "SC": "South", "VA": "South", "DC": "South",
    "WV": "South", "AL": "South", "KY": "South", "MS": "South",
    "TN": "South", "AR": "South", "LA": "South", "OK": "South", "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West",
    "NV": "West", "NM": "West", "UT": "West", "WY": "West",
    "AK": "West", "CA": "West", "HI": "West", "OR": "West", "WA": "West",
}
df["region"] = df["st_abbrev"].map(region_map)
df = df.dropna(subset=["region"])  # drop territories without region mapping

print(f"Working dataset: {df.shape[0]} states × {df.shape[1]} columns")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1: Workforce size across professions — faceted bar chart
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 1: Total US workforce by profession...")

prof_totals = {
    "Physicians":       df["phys_wkforc_23"].sum(),
    "RNs":              df["rn_23"].sum(),
    "LPN/LVNs":         df["lpnlvn_23"].sum(),
    "APRNs":            df["aprn_23"].sum(),
    "PAs":              df["pa_23"].sum(),
    "Pharmacists":      df["pharm_23"].sum(),
    "Dentists":         df["dent_23"].sum(),
    "Psychologists":    df["psychol_23"].sum(),
    "Social Workers":   df["socwk_23"].sum(),
    "Phys. Therapists": df["pt_23"].sum(),
    "EMTs/Paramedics":  df["emt_parmdcs_23"].sum(),
    "Counselors":       df["conslrs_23"].sum(),
}
prof_df = pd.DataFrame({"Profession": list(prof_totals.keys()),
                         "Total": list(prof_totals.values())})
prof_df = prof_df.sort_values("Total")
prof_df["Profession"] = pd.Categorical(prof_df["Profession"],
                                        categories=prof_df["Profession"].tolist())

p1 = (
    ggplot(prof_df, aes(x="Profession", y="Total"))
    + geom_col(fill="#4C72B0", alpha=0.85)
    + coord_flip()
    + labs(
        title="Total US Health Workforce by Profession (2023)",
        x="", y="Total Workers"
    )
    + theme_minimal()
    + theme(
        figure_size=(10, 6),
        plot_title=element_text(size=14, weight="bold"),
    )
)
ggsave(p1, os.path.join(PLOT_DIR, "01_workforce_by_profession.png"), dpi=150)
print("  Saved: plots/01_workforce_by_profession.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2: Gender composition — faceted by profession
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 2: Gender composition by profession...")

gender_data = []
prof_keys = {
    "phys": "Physicians", "rn": "RNs", "pa": "PAs", "aprn": "APRNs",
    "lpnlvn": "LPN/LVNs", "dent": "Dentists", "pharm": "Pharmacists",
    "psychol": "Psychologists", "socwk": "Social Workers",
    "pt": "Phys. Therapists", "emt_parmdcs": "EMTs/Paramedics",
    "conslrs": "Counselors",
}
for prefix, label in prof_keys.items():
    for _, row in df.iterrows():
        m = row.get(f"{prefix}_mal_23", np.nan)
        f = row.get(f"{prefix}_fem_23", np.nan)
        if pd.notna(m) and pd.notna(f) and (m + f) > 0:
            gender_data.append({"Profession": label, "State": row["st_abbrev"],
                                "Region": row["region"],
                                "pct_female": f / (m + f) * 100})

gender_df = pd.DataFrame(gender_data)

p2 = (
    ggplot(gender_df, aes(x="Region", y="pct_female", fill="Region"))
    + geom_boxplot(alpha=0.7, outlier_alpha=0.4)
    + facet_wrap("~Profession", ncol=4)
    + labs(
        title="% Female Workforce by Profession & Census Region",
        x="", y="% Female"
    )
    + theme_minimal()
    + theme(
        figure_size=(14, 9),
        plot_title=element_text(size=14, weight="bold"),
        strip_text=element_text(size=9, weight="bold"),
        axis_text_x=element_text(rotation=45, ha="right", size=7),
        legend_position="none"
    )
    + scale_fill_brewer(type="qual", palette="Set2")
)
ggsave(p2, os.path.join(PLOT_DIR, "02_gender_by_profession_region.png"), dpi=150)
print("  Saved: plots/02_gender_by_profession_region.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3: Age distribution across professions — stacked bar, faceted
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 3: Age distribution across professions...")

age_data = []
for prefix, label in prof_keys.items():
    for age_label, suffix in [("<30", "lt30_23"), ("30–39", "30_39_23"),
                               ("40–49", "40_49_23"), ("50–59", "50_59_23"),
                               ("60+", "ge60_23")]:
        col = f"{prefix}_{suffix}"
        if col in df.columns:
            total = df[col].sum()
            age_data.append({"Profession": label, "Age Group": age_label, "Count": total})

age_df = pd.DataFrame(age_data)
# Compute percentage within each profession
age_df["Total"] = age_df.groupby("Profession")["Count"].transform("sum")
age_df["Pct"] = age_df["Count"] / age_df["Total"] * 100
age_df["Age Group"] = pd.Categorical(age_df["Age Group"],
                                      categories=["<30", "30–39", "40–49", "50–59", "60+"])

p3 = (
    ggplot(age_df, aes(x="Profession", y="Pct", fill="Age Group"))
    + geom_col(position="stack", alpha=0.85)
    + coord_flip()
    + labs(
        title="Age Distribution of Health Workers by Profession (US Total, 2023)",
        x="", y="Percentage of Workforce", fill="Age Group"
    )
    + theme_minimal()
    + theme(
        figure_size=(10, 7),
        plot_title=element_text(size=14, weight="bold"),
    )
    + scale_fill_brewer(type="seq", palette="YlOrRd")
)
ggsave(p3, os.path.join(PLOT_DIR, "03_age_distribution.png"), dpi=150)
print("  Saved: plots/03_age_distribution.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 4: Physicians per capita — scatter faceted by region
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 4: Physicians per 100K by state, faceted by region...")

df["phys_per_100k"] = df["phys_wkforc_23"] / df["popn_pums_23"] * 100_000
df["rn_per_100k"] = df["rn_23"] / df["popn_pums_23"] * 100_000

per_cap = df.dropna(subset=["phys_per_100k", "rn_per_100k"]).copy()

p4 = (
    ggplot(per_cap, aes(x="phys_per_100k", y="rn_per_100k",
                         label="st_abbrev", color="region"))
    + geom_point(size=3, alpha=0.7)
    + geom_text(aes(label="st_abbrev"), size=7, nudge_y=30,
                va="bottom", ha="center")
    + facet_wrap("~region", ncol=2, scales="free")
    + labs(
        title="Physicians vs RNs per 100K Population by Region",
        x="Physicians per 100K",
        y="RNs per 100K",
        color="Region"
    )
    + theme_minimal()
    + theme(
        figure_size=(11, 8),
        plot_title=element_text(size=14, weight="bold"),
        strip_text=element_text(size=10, weight="bold"),
        legend_position="none"
    )
)
ggsave(p4, os.path.join(PLOT_DIR, "04_phys_vs_rn_per_capita.png"), dpi=150)
print("  Saved: plots/04_phys_vs_rn_per_capita.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 5: Racial/ethnic diversity — faceted by profession
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 5: Racial/ethnic composition by profession...")

race_data = []
race_labels = {"wh": "White", "bl": "Black", "hsp": "Hispanic",
               "asn": "Asian", "aian": "AI/AN", "nhpi": "NH/PI", "2race": "2+ Races"}
for prefix, label in prof_keys.items():
    for race_suffix, race_label in race_labels.items():
        col = f"{prefix}_{race_suffix}_23"
        if col in df.columns:
            total = df[col].sum()
            race_data.append({"Profession": label, "Race/Ethnicity": race_label, "Count": total})

race_df = pd.DataFrame(race_data)
race_df["Total"] = race_df.groupby("Profession")["Count"].transform("sum")
race_df["Pct"] = race_df["Count"] / race_df["Total"] * 100
race_df["Race/Ethnicity"] = pd.Categorical(
    race_df["Race/Ethnicity"],
    categories=["White", "Black", "Hispanic", "Asian", "AI/AN", "NH/PI", "2+ Races"]
)
# Focus on non-White for clarity in trellis
race_nonwh = race_df[race_df["Race/Ethnicity"] != "White"].copy()

p5 = (
    ggplot(race_nonwh, aes(x="Race/Ethnicity", y="Pct", fill="Race/Ethnicity"))
    + geom_col(alpha=0.85)
    + facet_wrap("~Profession", ncol=4)
    + labs(
        title="Racial/Ethnic Diversity in Health Professions (% of Workforce, excl. White)",
        x="", y="% of Profession"
    )
    + theme_minimal()
    + theme(
        figure_size=(14, 9),
        plot_title=element_text(size=13, weight="bold"),
        strip_text=element_text(size=9, weight="bold"),
        axis_text_x=element_text(rotation=50, ha="right", size=7),
        legend_position="none"
    )
    + scale_fill_brewer(type="qual", palette="Dark2")
)
ggsave(p5, os.path.join(PLOT_DIR, "05_racial_diversity_by_profession.png"), dpi=150)
print("  Saved: plots/05_racial_diversity_by_profession.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 6: Median wages by profession — faceted dot plot by region
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 6: Median wages by profession & region...")

wage_data = []
wage_keys = {
    "phys": "Physicians", "rn": "RNs", "pa": "PAs",
    "lpnlvn": "LPN/LVNs", "dent_gen": "Dentists (Gen)",
    "pharm": "Pharmacists", "pt": "Phys. Therapists",
    "ot": "Occup. Therapists", "resp_ther": "Resp. Therapists",
    "emt": "EMTs", "parmdcs": "Paramedics",
    "nursng_asst": "Nursing Assistants",
    "home_hlth_aide": "Home Health Aides",
}
for prefix, label in wage_keys.items():
    wage_col = f"{prefix}_medn_wage_24"
    if wage_col in df.columns:
        for _, row in df.iterrows():
            if pd.notna(row[wage_col]) and row[wage_col] > 0:
                wage_data.append({"Profession": label, "State": row["st_abbrev"],
                                   "Region": row["region"], "Median Wage": row[wage_col]})

wage_df = pd.DataFrame(wage_data)

if len(wage_df) > 0:
    p6 = (
        ggplot(wage_df, aes(x="Profession", y="Median Wage", fill="Region"))
        + geom_boxplot(alpha=0.7, outlier_alpha=0.3)
        + coord_flip()
        + labs(
            title="Median Hourly Wages (2024 BLS) by Profession & Census Region",
            x="", y="Median Hourly Wage ($)"
        )
        + theme_minimal()
        + theme(
            figure_size=(11, 7),
            plot_title=element_text(size=14, weight="bold"),
        )
        + scale_fill_brewer(type="qual", palette="Set2")
    )
    ggsave(p6, os.path.join(PLOT_DIR, "06_wages_by_profession_region.png"), dpi=150)
    print("  Saved: plots/06_wages_by_profession_region.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 7: Work setting distribution — faceted by profession
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 7: Work setting distribution...")

setting_data = []
settings_map = {
    "hosp": "Hospital", "ofcs_phys": "Physician Office",
    "outpt_care_ctr": "Outpatient Center", "nurs_fac": "Nursing Facility",
    "home_hlth_svc": "Home Health", "oth_hlth_svc": "Other Health Svc",
    "sch": "Schools",
}
setting_profs = {"rn": "RNs", "lpnlvn": "LPN/LVNs", "socwk": "Social Workers",
                 "pt": "Phys. Therapists", "ot": "Occup. Therapists"}

for prefix, label in setting_profs.items():
    for setting_key, setting_label in settings_map.items():
        col = f"{prefix}_{setting_key}_23"
        if col in df.columns:
            total = df[col].sum()
            if total > 0:
                setting_data.append({"Profession": label,
                                     "Setting": setting_label, "Count": total})

setting_df = pd.DataFrame(setting_data)
setting_df["Total"] = setting_df.groupby("Profession")["Count"].transform("sum")
setting_df["Pct"] = setting_df["Count"] / setting_df["Total"] * 100

p7 = (
    ggplot(setting_df, aes(x="Setting", y="Pct", fill="Setting"))
    + geom_col(alpha=0.85)
    + facet_wrap("~Profession", ncol=3, scales="free_x")
    + coord_flip()
    + labs(
        title="Work Setting Distribution by Profession (US Total, 2023)",
        x="", y="% of Workforce in Setting"
    )
    + theme_minimal()
    + theme(
        figure_size=(12, 8),
        plot_title=element_text(size=14, weight="bold"),
        strip_text=element_text(size=10, weight="bold"),
        legend_position="none"
    )
    + scale_fill_brewer(type="qual", palette="Set3")
)
ggsave(p7, os.path.join(PLOT_DIR, "07_work_settings_by_profession.png"), dpi=150)
print("  Saved: plots/07_work_settings_by_profession.png")

print("\nAll trellis plots saved to: group_project/plots/")
