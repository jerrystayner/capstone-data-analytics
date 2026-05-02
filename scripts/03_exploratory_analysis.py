# 03_exploratory_analysis.py
# Jerry Stayner / CIS 480
#
# EDA before fitting the regression. Generates summary stats, a correlation
# heatmap, distribution histograms, and scatter plots of obesity against
# each predictor. All charts get saved to /visualizations/.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "datasets" / "mesa_tract_master.csv"
VIZ_DIR   = ROOT / "visualizations"
OUT_DIR   = ROOT / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# Friendlier names for chart labels
LABELS = {
    "obesity_pct":     "Adult Obesity (%)",
    "median_income":   "Median Household Income ($)",
    "grocery_per_10k": "Grocery Stores per 10,000 Residents",
    "drove_pct":       "Workers Driving Alone (%)",
    "bachelors_pct":   "Adults with Bachelor's Degree (%)",
    "uninsured_pct":   "Uninsured (%)",
}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
})


def main():
    df = pd.read_csv(DATA_FILE, dtype={"fips": str})

    # Drop tracts with any missing values in the analysis columns. The 4
    # zero-pop tracts get caught here because grocery_per_10k is NaN for them.
    analysis_vars = list(LABELS.keys())
    df_clean = df.dropna(subset=analysis_vars).copy()
    print(f"Tracts kept for analysis: {len(df_clean)} of {len(df)}")

    # Summary stats
    summary = df_clean[analysis_vars].describe().T.round(2)
    summary.to_csv(OUT_DIR / "summary_statistics.csv")
    print("\nSummary stats:")
    print(summary.to_string())

    # Correlation matrix as both a CSV and a heatmap
    corr = df_clean[analysis_vars].corr().round(3)
    corr.to_csv(OUT_DIR / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        corr, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, cbar_kws={"shrink": 0.8},
        xticklabels=[LABELS[v] for v in analysis_vars],
        yticklabels=[LABELS[v] for v in analysis_vars],
        ax=ax,
    )
    ax.set_title(f"Correlation Matrix: Obesity & Predictors\nMesa, AZ Census Tracts (n={len(df_clean)})")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.savefig(VIZ_DIR / "01_correlation_heatmap.png")
    plt.close(fig)

    # Distribution histograms (one panel per variable)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, var in zip(axes.flat, analysis_vars):
        sns.histplot(df_clean[var], kde=True, ax=ax, color="steelblue")
        ax.set_title(LABELS[var], fontsize=11)
        ax.set_xlabel("")
    fig.suptitle("Distribution of Outcome and Predictors", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_distributions.png")
    plt.close(fig)

    # Scatter plots: obesity vs each predictor with a regression line
    predictors = [v for v in analysis_vars if v != "obesity_pct"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, var in zip(axes.flat, predictors):
        sns.regplot(
            data=df_clean, x=var, y="obesity_pct",
            scatter_kws={"alpha": 0.55, "s": 35, "color": "steelblue"},
            line_kws={"color": "crimson"},
            ax=ax,
        )
        r = df_clean[[var, "obesity_pct"]].corr().iloc[0, 1]
        ax.set_title(f"{LABELS[var]}\n(r = {r:+.3f})", fontsize=11)
        ax.set_xlabel(LABELS[var])
        ax.set_ylabel("Adult Obesity (%)")
    axes.flat[-1].set_visible(False)  # hide the empty 6th panel
    fig.suptitle("Obesity vs. Each Predictor (Mesa Census Tracts)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_scatter_predictors.png")
    plt.close(fig)

    print("\nDone. Charts saved to:", VIZ_DIR)
    print("CSVs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
