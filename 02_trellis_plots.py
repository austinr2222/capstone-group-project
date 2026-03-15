"""
AHRF 2025 — Trellis Plots (State + County Level)
=================================================
Uses plotnine (ggplot2 for Python) to create faceted/trellis plots
illustrating distributions and relationships across US states (plots 1-7)
and counties (plots 8-13) using the AHRF 2025 data.
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

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  COUNTY-LEVEL PLOTS (Plots 8–13)                                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

COUNTY_DIR = os.path.join(DATA_DIR, "NCHWA-2024-2025+AHRF+COUNTY+CSV")
print("\n--- Loading county-level data ---")
geo_c  = pd.read_csv(os.path.join(COUNTY_DIR, "AHRF2025geo.csv"), low_memory=False)
pop_c  = pd.read_csv(os.path.join(COUNTY_DIR, "AHRF2025pop.csv"), low_memory=False)
hp_c   = pd.read_csv(os.path.join(COUNTY_DIR, "AHRF2025hp.csv"),  low_memory=False)
hf_c   = pd.read_csv(os.path.join(COUNTY_DIR, "AHRF2025hf.csv"),  low_memory=False)
env_c  = pd.read_csv(os.path.join(COUNTY_DIR, "AHRF2025env.csv"), low_memory=False)

# Build merged county frame
cnty = geo_c[["fips_st_cnty", "cnty_name_st_abbrev", "st_name_abbrev",
              "cens_regn_name", "rural_urban_contnm_23"]].copy()
cnty = cnty.merge(pop_c[["fips_st_cnty", "popn_est_23"]], on="fips_st_cnty", how="left")
cnty = cnty.merge(hp_c[["fips_st_cnty", "phys_nf_prim_care_pc_exc_rsdt_23",
                         "md_nf_activ_23"]], on="fips_st_cnty", how="left")
cnty = cnty.merge(hf_c[["fips_st_cnty", "hosp_23", "nurs_fac_23",
                         "rural_hlth_clincs_23"]], on="fips_st_cnty", how="left")
cnty = cnty.merge(env_c[["fips_st_cnty", "popn_densty_per_squr_mi_20",
                          "good_air_qulty_dys_pct_24"]], on="fips_st_cnty", how="left")

cnty["pcp_per_100k"] = (cnty["phys_nf_prim_care_pc_exc_rsdt_23"]
                         / cnty["popn_est_23"].replace(0, np.nan) * 100_000)
cnty["metro_status"] = cnty["rural_urban_contnm_23"].apply(
    lambda x: "Metro" if x in [1, 2, 3] else ("Nonmetro" if x in [4, 5, 6, 7] else "Rural")
)
cnty["ruc_label"] = cnty["rural_urban_contnm_23"].map({
    1: "1-Metro >=1M", 2: "2-Metro 250K-1M", 3: "3-Metro <250K",
    4: "4-Nonmetro >=20K", 5: "5-Nonmetro 20K adj",
    6: "6-Nonmetro 2.5-20K", 7: "7-Nonmetro 2.5-20K adj",
    8: "8-Rural <2.5K", 9: "9-Rural <2.5K adj"
})
cnty["log_pop"] = np.log10(cnty["popn_est_23"].replace(0, np.nan))
cnty = cnty.dropna(subset=["cens_regn_name", "metro_status"])

print(f"County dataset: {cnty.shape[0]} counties × {cnty.shape[1]} columns")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 8: County population distribution by Census Region
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 8: County population distributions by Census Region...")

p8 = (
    ggplot(cnty.dropna(subset=["log_pop"]), aes(x="log_pop"))
    + geom_histogram(aes(fill="cens_regn_name"), bins=30, alpha=0.7, color="white")
    + facet_wrap("~cens_regn_name", ncol=2)
    + labs(
        title="County Population Distribution by Census Region",
        x="Log10(Population Estimate 2023)",
        y="Count",
        fill="Region"
    )
    + theme_minimal()
    + theme(
        figure_size=(10, 7),
        plot_title=element_text(size=14, weight="bold"),
        strip_text=element_text(size=10, weight="bold"),
        legend_position="none"
    )
)
ggsave(p8, os.path.join(PLOT_DIR, "08_county_pop_by_region.png"), dpi=150)
print("  Saved: plots/08_county_pop_by_region.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 9: PCP rate boxplots by Metro Status & Region
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 9: County PCP rate by metro status & region...")

pcp_df = cnty.dropna(subset=["pcp_per_100k"]).copy()
pcp_df["pcp_per_100k_clipped"] = pcp_df["pcp_per_100k"].clip(upper=300)

p9 = (
    ggplot(pcp_df, aes(x="metro_status", y="pcp_per_100k_clipped", fill="metro_status"))
    + geom_boxplot(alpha=0.7, outlier_alpha=0.3)
    + facet_wrap("~cens_regn_name", ncol=2)
    + labs(
        title="Primary Care Physicians per 100K by Metro Status & Region (County Level)",
        x="",
        y="PCPs per 100,000 Population",
        fill="Status"
    )
    + theme_minimal()
    + theme(
        figure_size=(10, 7),
        plot_title=element_text(size=13, weight="bold"),
        strip_text=element_text(size=10, weight="bold"),
        axis_text_x=element_text(rotation=25, ha="right")
    )
    + scale_fill_brewer(type="qual", palette="Set2")
)
ggsave(p9, os.path.join(PLOT_DIR, "09_county_pcp_rate_by_metro_region.png"), dpi=150)
print("  Saved: plots/09_county_pcp_rate_by_metro_region.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 10: Hospitals per county faceted by region
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 10: Hospital count distribution by region...")

hosp_df = cnty.dropna(subset=["hosp_23"]).copy()
hosp_df["hosp_binned"] = pd.cut(
    hosp_df["hosp_23"],
    bins=[-1, 0, 1, 2, 3, 5, 200],
    labels=["0", "1", "2", "3", "4-5", "6+"]
)

p10 = (
    ggplot(hosp_df.dropna(subset=["hosp_binned"]),
           aes(x="hosp_binned", fill="metro_status"))
    + geom_bar(position="dodge", alpha=0.8)
    + facet_wrap("~cens_regn_name", ncol=2)
    + labs(
        title="Number of Hospitals per County by Region & Metro Status",
        x="Hospitals in County",
        y="Number of Counties",
        fill="Metro Status"
    )
    + theme_minimal()
    + theme(
        figure_size=(10, 7),
        plot_title=element_text(size=14, weight="bold"),
        strip_text=element_text(size=10, weight="bold"),
    )
    + scale_fill_brewer(type="qual", palette="Set2")
)
ggsave(p10, os.path.join(PLOT_DIR, "10_county_hospitals_by_region.png"), dpi=150)
print("  Saved: plots/10_county_hospitals_by_region.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 11: Population vs Active MDs scatter — faceted by region
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 11: County population vs Active MDs scatter...")

scatter_df = cnty.dropna(subset=["popn_est_23", "md_nf_activ_23"]).copy()
scatter_df = scatter_df[(scatter_df["popn_est_23"] > 0) & (scatter_df["md_nf_activ_23"] > 0)]

p11 = (
    ggplot(scatter_df, aes(x="popn_est_23", y="md_nf_activ_23", color="metro_status"))
    + geom_point(alpha=0.35, size=1.2)
    + facet_wrap("~cens_regn_name", ncol=2)
    + scale_x_log10()
    + scale_y_log10()
    + labs(
        title="County Population vs Active MDs (log-log) by Census Region",
        x="Population Estimate 2023 (log scale)",
        y="Active Non-Federal MDs 2023 (log scale)",
        color="Metro Status"
    )
    + theme_minimal()
    + theme(
        figure_size=(10, 7),
        plot_title=element_text(size=14, weight="bold"),
        strip_text=element_text(size=10, weight="bold"),
    )
)
ggsave(p11, os.path.join(PLOT_DIR, "11_county_pop_vs_mds.png"), dpi=150)
print("  Saved: plots/11_county_pop_vs_mds.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 12: Air quality by Rural-Urban Continuum Code
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 12: Air quality by Rural-Urban Continuum Code...")

air_df = cnty.dropna(subset=["good_air_qulty_dys_pct_24", "ruc_label"]).copy()

p12 = (
    ggplot(air_df, aes(x="ruc_label", y="good_air_qulty_dys_pct_24", fill="ruc_label"))
    + geom_boxplot(alpha=0.7, outlier_alpha=0.3)
    + coord_flip()
    + labs(
        title="% Good Air Quality Days by Rural-Urban Continuum Code",
        x="",
        y="Good Air Quality Days (%)",
        fill=""
    )
    + theme_minimal()
    + theme(
        figure_size=(10, 6),
        plot_title=element_text(size=14, weight="bold"),
        legend_position="none"
    )
)
ggsave(p12, os.path.join(PLOT_DIR, "12_county_air_quality_by_ruc.png"), dpi=150)
print("  Saved: plots/12_county_air_quality_by_ruc.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 13: Population density — faceted grid (Metro Status × Region)
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 13: Population density by metro status × region...")

dens_df = cnty.dropna(subset=["popn_densty_per_squr_mi_20"]).copy()
dens_df["log_density"] = np.log10(dens_df["popn_densty_per_squr_mi_20"].replace(0, np.nan))
dens_df = dens_df.dropna(subset=["log_density"])

p13 = (
    ggplot(dens_df, aes(x="log_density", fill="metro_status"))
    + geom_histogram(bins=30, alpha=0.7, color="white")
    + facet_grid("metro_status~cens_regn_name")
    + labs(
        title="Population Density Distribution: Metro Status x Census Region",
        x="Log10(Population Density per sq mi)",
        y="Count"
    )
    + theme_minimal()
    + theme(
        figure_size=(12, 7),
        plot_title=element_text(size=14, weight="bold"),
        strip_text=element_text(size=9, weight="bold"),
        legend_position="none"
    )
    + scale_fill_brewer(type="qual", palette="Set2")
)
ggsave(p13, os.path.join(PLOT_DIR, "13_county_density_by_metro_region.png"), dpi=150)
print("  Saved: plots/13_county_density_by_metro_region.png")

print("\nAll trellis plots saved to: group_project/plots/")
