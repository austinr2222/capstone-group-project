"""
AHRF 2025 Trellis Plots
=======================
Uses plotnine (ggplot2 for Python) to create faceted/trellis plots
illustrating distributions and relationships in the AHRF data.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from plotnine import (
    ggplot, aes, geom_histogram, geom_boxplot, geom_point, geom_bar,
    facet_wrap, facet_grid, labs, theme, theme_minimal, element_text,
    scale_x_log10, scale_y_log10, scale_fill_brewer, coord_flip,
    ggsave, after_stat
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "AHRF_data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load & merge key sub-files ──────────────────────────────────────────────
geo = pd.read_csv(os.path.join(DATA_DIR, "AHRF2025geo.csv"), low_memory=False)
pop = pd.read_csv(os.path.join(DATA_DIR, "AHRF2025pop.csv"), low_memory=False)
hp  = pd.read_csv(os.path.join(DATA_DIR, "AHRF2025hp.csv"),  low_memory=False)
hf  = pd.read_csv(os.path.join(DATA_DIR, "AHRF2025hf.csv"),  low_memory=False)
env = pd.read_csv(os.path.join(DATA_DIR, "AHRF2025env.csv"), low_memory=False)

# Merge on FIPS
df = geo[["fips_st_cnty", "cnty_name_st_abbrev", "st_name", "st_name_abbrev",
          "cens_regn_name", "rural_urban_contnm_23"]].copy()
df = df.merge(pop[["fips_st_cnty", "popn_est_23", "popn_est_24",
                    "popn_mal_23", "popn_fem_23"]], on="fips_st_cnty", how="left")
df = df.merge(hp[["fips_st_cnty",
                   "phys_nf_prim_care_pc_exc_rsdt_23",
                   "md_nf_activ_23",
                   "do_nf_prim_care_pc_excl_rsdnt_23"]], on="fips_st_cnty", how="left")
df = df.merge(hf[["fips_st_cnty", "hosp_23", "stgh_23",
                   "nurs_fac_23", "rural_hlth_clincs_23"]], on="fips_st_cnty", how="left")
df = df.merge(env[["fips_st_cnty", "land_area_mi2_20",
                    "popn_densty_per_squr_mi_20",
                    "good_air_qulty_dys_pct_24"]], on="fips_st_cnty", how="left")

# Derived columns
df["ruc_label"] = df["rural_urban_contnm_23"].map({
    1: "1-Metro ≥1M", 2: "2-Metro 250K–1M", 3: "3-Metro <250K",
    4: "4-Nonmetro ≥20K", 5: "5-Nonmetro ≥20K adj",
    6: "6-Nonmetro 2.5–20K", 7: "7-Nonmetro 2.5–20K adj",
    8: "8-Rural <2.5K", 9: "9-Rural <2.5K adj"
})
df["metro_status"] = df["rural_urban_contnm_23"].apply(
    lambda x: "Metro" if x in [1, 2, 3] else ("Nonmetro" if x in [4, 5, 6, 7] else "Rural")
)
df["pcp_per_100k"] = (df["phys_nf_prim_care_pc_exc_rsdt_23"] / df["popn_est_23"] * 100_000)
df["log_pop"] = np.log10(df["popn_est_23"].replace(0, np.nan))

df = df.dropna(subset=["cens_regn_name", "metro_status"])

print(f"Working dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1: Population distribution faceted by Census Region
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 1: Population distributions by Census Region...")

p1 = (
    ggplot(df.dropna(subset=["log_pop"]), aes(x="log_pop"))
    + geom_histogram(aes(fill="cens_regn_name"), bins=30, alpha=0.7, color="white")
    + facet_wrap("~cens_regn_name", ncol=2)
    + labs(
        title="County Population Distribution by Census Region",
        x="Log₁₀(Population Estimate 2023)",
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
ggsave(p1, os.path.join(PLOT_DIR, "01_pop_dist_by_region.png"), dpi=150)
print("  Saved: plots/01_pop_dist_by_region.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2: Primary care physicians per 100K faceted by Metro Status
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 2: PCP rate boxplots by metro status × region...")

pcp_df = df.dropna(subset=["pcp_per_100k"]).copy()
pcp_df["pcp_per_100k_clipped"] = pcp_df["pcp_per_100k"].clip(upper=300)

p2 = (
    ggplot(pcp_df, aes(x="metro_status", y="pcp_per_100k_clipped", fill="metro_status"))
    + geom_boxplot(alpha=0.7, outlier_alpha=0.3)
    + facet_wrap("~cens_regn_name", ncol=2)
    + labs(
        title="Primary Care Physicians per 100K by Metro Status & Region",
        x="",
        y="PCPs per 100,000 Population",
        fill="Status"
    )
    + theme_minimal()
    + theme(
        figure_size=(10, 7),
        plot_title=element_text(size=14, weight="bold"),
        strip_text=element_text(size=10, weight="bold"),
        axis_text_x=element_text(rotation=25, ha="right")
    )
    + scale_fill_brewer(type="qual", palette="Set2")
)
ggsave(p2, os.path.join(PLOT_DIR, "02_pcp_rate_by_metro_region.png"), dpi=150)
print("  Saved: plots/02_pcp_rate_by_metro_region.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3: Hospitals per county faceted by region
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 3: Hospital count distribution by region...")

hosp_df = df.dropna(subset=["hosp_23"]).copy()
hosp_df["hosp_binned"] = pd.cut(
    hosp_df["hosp_23"],
    bins=[-1, 0, 1, 2, 3, 5, 200],
    labels=["0", "1", "2", "3", "4–5", "6+"]
)

p3 = (
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
ggsave(p3, os.path.join(PLOT_DIR, "03_hospitals_by_region.png"), dpi=150)
print("  Saved: plots/03_hospitals_by_region.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 4: Population vs Active MDs — scatter faceted by region
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 4: Population vs Active MDs scatter by region...")

scatter_df = df.dropna(subset=["popn_est_23", "md_nf_activ_23"]).copy()
scatter_df = scatter_df[(scatter_df["popn_est_23"] > 0) & (scatter_df["md_nf_activ_23"] > 0)]

p4 = (
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
    + scale_fill_brewer(type="qual", palette="Set2")
)
ggsave(p4, os.path.join(PLOT_DIR, "04_pop_vs_mds_by_region.png"), dpi=150)
print("  Saved: plots/04_pop_vs_mds_by_region.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 5: Air quality by rural-urban continuum code
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 5: Air quality by Rural-Urban Continuum Code...")

air_df = df.dropna(subset=["good_air_qulty_dys_pct_24", "ruc_label"]).copy()

p5 = (
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
ggsave(p5, os.path.join(PLOT_DIR, "05_air_quality_by_ruc.png"), dpi=150)
print("  Saved: plots/05_air_quality_by_ruc.png")

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 6: Population density histograms faceted by metro status
# ═════════════════════════════════════════════════════════════════════════════
print("Creating Plot 6: Population density distributions by metro status...")

dens_df = df.dropna(subset=["popn_densty_per_squr_mi_20"]).copy()
dens_df["log_density"] = np.log10(dens_df["popn_densty_per_squr_mi_20"].replace(0, np.nan))
dens_df = dens_df.dropna(subset=["log_density"])

p6 = (
    ggplot(dens_df, aes(x="log_density", fill="metro_status"))
    + geom_histogram(bins=30, alpha=0.7, color="white")
    + facet_grid("metro_status~cens_regn_name")
    + labs(
        title="Population Density Distribution: Metro Status × Census Region",
        x="Log₁₀(Population Density per sq mi)",
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
ggsave(p6, os.path.join(PLOT_DIR, "06_density_by_metro_region.png"), dpi=150)
print("  Saved: plots/06_density_by_metro_region.png")

print("\nAll trellis plots saved to: group_project/plots/")
