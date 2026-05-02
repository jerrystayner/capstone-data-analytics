# 04_regression_model.py
# Jerry Stayner / CIS 480
#
# Fits the multiple linear regression and runs the standard diagnostics.
# Model:
#   obesity_pct ~ median_income + grocery_per_10k + drove_pct
#                 + bachelors_pct + uninsured_pct
#
# Outputs:
#   - regression_summary.txt    (full statsmodels OLS summary)
#   - regression_coefficients.csv
#   - standardized_betas.csv
#   - vif.csv                   (multicollinearity check)
#   - train_test_metrics.csv    (held-out RMSE per the Ch 1 plan)
#   - hypothesis_decisions.csv
#   - 04_regression_diagnostics.png    (residuals, Q-Q, Cook's distance)
#   - 05_standardized_coefficients.png
#   - 06_predicted_vs_actual.png

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "datasets" / "mesa_tract_master.csv"
VIZ_DIR   = ROOT / "visualizations"
OUT_DIR   = ROOT / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# Order matches the hypotheses in Chapter 2
PREDICTORS = [
    "median_income",
    "grocery_per_10k",
    "drove_pct",
    "bachelors_pct",
    "uninsured_pct",
]
TARGET = "obesity_pct"

LABELS = {
    "median_income":   "Median household income ($)",
    "grocery_per_10k": "Grocery stores per 10k residents",
    "drove_pct":       "% drove to work",
    "bachelors_pct":   "% with bachelor's degree+",
    "uninsured_pct":   "% uninsured",
}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})


def fit_full_model(df):
    # Standard OLS with an intercept term added
    X = df[PREDICTORS]
    y = df[TARGET]
    model = sm.OLS(y, sm.add_constant(X)).fit()
    return model, X, y


def vif_table(X):
    # VIF for each predictor. Anything above 5 starts to look concerning.
    Xc = sm.add_constant(X)
    rows = []
    for i, name in enumerate(Xc.columns):
        if name == "const":
            continue
        rows.append({"variable": name, "VIF": variance_inflation_factor(Xc.values, i)})
    return pd.DataFrame(rows).round(3)


def standardized_betas(df):
    # Z-score everything and re-fit. Lets me compare predictors that are on
    # totally different scales (income in $, grocery as a count, % variables).
    z = df[[TARGET] + PREDICTORS].apply(stats.zscore, ddof=1)
    Xz = sm.add_constant(z[PREDICTORS])
    yz = z[TARGET]
    return sm.OLS(yz, Xz).fit().params.drop("const").rename("std_beta").round(3)


def run_train_test(df, seed=42):
    # 80/20 split. Chapter 1 said I'd report held-out RMSE, so this is
    # how I get that number.
    X = df[PREDICTORS].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    m = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    y_pred = m.predict(sm.add_constant(X_test))
    return {
        "train_n":   len(y_train),
        "test_n":    len(y_test),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
        "test_r2":   round(float(r2_score(y_test, y_pred)), 3),
    }


def plot_diagnostics(model, X, y, out_path):
    # 4-panel diagnostic plot: residuals vs fitted, Q-Q, residual hist,
    # and Cook's distance. This is the standard set of OLS checks.
    influence = OLSInfluence(model)
    fitted = model.fittedvalues
    resid  = model.resid
    stud   = influence.resid_studentized_internal
    cooks  = influence.cooks_distance[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.6, color="steelblue")
    ax.axhline(0, color="crimson", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Fitted")

    ax = axes[0, 1]
    sm.qqplot(stud, line="45", ax=ax,
              markerfacecolor="steelblue", markeredgecolor="steelblue", alpha=0.6)
    ax.set_title("Normal Q-Q Plot of Studentized Residuals")

    ax = axes[1, 0]
    sns.histplot(resid, kde=True, ax=ax, color="steelblue")
    ax.axvline(0, color="crimson", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual")
    ax.set_title("Distribution of Residuals")

    ax = axes[1, 1]
    n = len(cooks)
    ax.stem(range(n), cooks, basefmt=" ", markerfmt="o", linefmt="steelblue")
    threshold = 4 / n
    ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1,
               label=f"Cutoff = 4/n = {threshold:.3f}")
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Cook's distance")
    ax.set_title("Influential Observations (Cook's Distance)")
    ax.legend()

    fig.suptitle("Regression Diagnostics: OLS Model of Obesity Prevalence",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def hypothesis_decisions(model):
    # One-tailed test for each H per Chapter 2's directional alternatives.
    # Reject H0 if the sign matches AND one-tailed p < .05.
    expected_dir = {
        "median_income":   "negative",
        "grocery_per_10k": "negative",
        "drove_pct":       "positive",
        "bachelors_pct":   "negative",
        "uninsured_pct":   "positive",
    }
    rows = []
    for var in PREDICTORS:
        beta = model.params[var]
        p_two = model.pvalues[var]
        p_one = p_two / 2
        sign_ok = (
            (expected_dir[var] == "negative" and beta < 0) or
            (expected_dir[var] == "positive" and beta > 0)
        )
        decision = "Reject H0" if (sign_ok and p_one < 0.05) else "Fail to reject H0"
        rows.append({
            "variable": var,
            "expected_direction": expected_dir[var],
            "coefficient": round(beta, 4),
            "p_value_two_tailed": round(p_two, 4),
            "p_value_one_tailed": round(p_one, 4),
            "sign_matches": sign_ok,
            "decision_alpha_05": decision,
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_FILE, dtype={"fips": str})
    df_clean = df.dropna(subset=[TARGET] + PREDICTORS).copy()
    print(f"Analysis dataset: {len(df_clean)} census tracts")

    # Fit the model
    model, X, y = fit_full_model(df_clean)

    # Save the full statsmodels summary
    with open(OUT_DIR / "regression_summary.txt", "w") as f:
        f.write(str(model.summary()))
        f.write("\n\n")
    print("\n" + "=" * 70)
    print(model.summary())

    # Coefficient table
    coef_table = pd.DataFrame({
        "coefficient": model.params,
        "std_error":   model.bse,
        "t_stat":      model.tvalues,
        "p_value":     model.pvalues,
        "ci_lower":    model.conf_int()[0],
        "ci_upper":    model.conf_int()[1],
    }).round(4)
    coef_table.to_csv(OUT_DIR / "regression_coefficients.csv")

    # Standardized betas (so I can compare effect sizes)
    std_b = standardized_betas(df_clean)
    std_b.to_csv(OUT_DIR / "standardized_betas.csv")
    print("\nStandardized coefficients:")
    print(std_b.to_string())

    # VIF
    vifs = vif_table(X)
    vifs.to_csv(OUT_DIR / "vif.csv", index=False)
    print("\nVIFs:")
    print(vifs.to_string(index=False))

    # Train/test split
    tt = run_train_test(df_clean)
    print("\nTrain/test:", tt)
    pd.DataFrame([tt]).to_csv(OUT_DIR / "train_test_metrics.csv", index=False)

    # Per-hypothesis decisions
    decisions = hypothesis_decisions(model)
    decisions.to_csv(OUT_DIR / "hypothesis_decisions.csv", index=False)
    print("\nHypothesis decisions (one-tailed, alpha = .05):")
    print(decisions.to_string(index=False))

    # Diagnostic plot
    plot_diagnostics(model, X, y, VIZ_DIR / "04_regression_diagnostics.png")

    # Bar chart of standardized coefficients
    fig, ax = plt.subplots(figsize=(9, 5.5))
    std_b_sorted = std_b.reindex(PREDICTORS)
    colors = ["crimson" if v < 0 else "steelblue" for v in std_b_sorted.values]
    bars = ax.barh([LABELS[v] for v in std_b_sorted.index], std_b_sorted.values,
                   color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized coefficient (β)")
    ax.set_title(f"Standardized Predictor Effects on Adult Obesity\n"
                 f"Mesa Census Tracts (n = {len(df_clean)})", fontweight="bold")
    # Pad x-axis so labels never overlap with predictor names
    xmin, xmax = std_b_sorted.min(), std_b_sorted.max()
    pad = max(abs(xmin), abs(xmax)) * 0.20
    ax.set_xlim(xmin - pad, xmax + pad)
    for bar, val in zip(bars, std_b_sorted.values):
        # Always place label OUTSIDE the bar so it never overlaps the y-axis name
        offset = 0.012
        if val >= 0:
            ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha="left", fontsize=10)
        else:
            ax.text(val - offset, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha="right", fontsize=10)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "05_standardized_coefficients.png")
    plt.close(fig)

    # Predicted vs actual
    fig, ax = plt.subplots(figsize=(7, 7))
    fitted = model.fittedvalues
    ax.scatter(y, fitted, alpha=0.6, color="steelblue", s=40)
    lims = [min(y.min(), fitted.min()) - 0.5, max(y.max(), fitted.max()) + 0.5]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual obesity (%)")
    ax.set_ylabel("Predicted obesity (%)")
    ax.set_title(f"Predicted vs. Actual\n"
                 f"R² = {model.rsquared:.3f}, Adj R² = {model.rsquared_adj:.3f}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "06_predicted_vs_actual.png")
    plt.close(fig)

    print("\nAll done. CSVs in:", OUT_DIR)
    print("Plots in:", VIZ_DIR)


if __name__ == "__main__":
    main()
