from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "world_happiness_matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "world_happiness_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from world_happiness_core import (
    CORE_FEATURES,
    FEATURE_LABELS,
    FIG_DIR,
    PROJECT_DIR,
    compute_custom_index,
    load_happiness_data,
    run_index_replication_experiment,
    run_lagged_forecasting_experiment,
)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 110


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def plot_central_tendency(happy: pd.DataFrame) -> pd.DataFrame:
    central = happy.groupby("year")["score"].agg(["mean", "median", "std", "min", "max"]).round(3)
    print(central)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(
        central.index,
        central["mean"],
        yerr=central["std"],
        marker="o",
        capsize=4,
        label="mean +/- 1 std",
    )
    ax.plot(central.index, central["median"], "--s", color="tab:red", label="median")
    ax.set_title("Global happiness score over time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Happiness score")
    ax.legend()
    save_figure(fig, "central_tendency.png")
    return central


def plot_ranking_movers(happy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rank_wide = happy.pivot_table(index="country", columns="year", values="rank")
    rank_full = rank_wide.dropna()
    rank_summary = pd.DataFrame(
        {
            "rank_std": rank_full.std(axis=1).round(2),
            "rank_2015": rank_full[2015].astype(int),
            "rank_2019": rank_full[2019].astype(int),
            "improved_by": (rank_full[2015] - rank_full[2019]).astype(int),
        }
    )

    most_stable = rank_summary.sort_values("rank_std").head(10)
    most_improved = rank_summary.sort_values("improved_by", ascending=False).head(10)
    biggest_fallers = rank_summary.sort_values("improved_by").head(10)

    print("Most stable rankings:")
    print(most_stable)
    print("\nMost improved 2015 to 2019:")
    print(most_improved)
    print("\nBiggest fallers 2015 to 2019:")
    print(biggest_fallers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(
        data=most_stable.reset_index(),
        y="country",
        x="rank_std",
        ax=axes[0],
        color="tab:blue",
    )
    axes[0].set_title("Most stable rankings")
    axes[0].set_xlabel("Std of rank")
    axes[0].set_ylabel("")

    mover_data = pd.concat(
        [
            most_improved.head(8).assign(kind="improved"),
            biggest_fallers.head(8).assign(kind="fell"),
        ]
    ).reset_index()
    sns.barplot(
        data=mover_data,
        y="country",
        x="improved_by",
        hue="kind",
        palette={"improved": "tab:green", "fell": "tab:red"},
        ax=axes[1],
    )
    axes[1].set_title("Largest rank moves, 2015 to 2019")
    axes[1].set_xlabel("Rank improvement")
    axes[1].set_ylabel("")
    save_figure(fig, "ranking_stability.png")
    return most_stable, most_improved, biggest_fallers


def plot_feature_relationships(happy: pd.DataFrame) -> pd.DataFrame:
    train = happy.loc[happy["year"] <= 2018].copy()
    corr = train[["score", *CORE_FEATURES]].corr()
    print(corr["score"].drop("score").sort_values(ascending=False).round(3))

    label_corr = corr.rename(index=FEATURE_LABELS, columns=FEATURE_LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(label_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation matrix, 2015-2018")
    save_figure(fig, "correlation_heatmap.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, feature in zip(axes.flat, CORE_FEATURES):
        sns.regplot(data=train, x=feature, y="score", scatter_kws={"alpha": 0.4, "s": 20}, ax=ax)
        r = train[feature].corr(train["score"])
        ax.set_title(f"{FEATURE_LABELS[feature]} (r = {r:+.2f})")
        ax.set_xlabel(FEATURE_LABELS[feature])
        ax.set_ylabel("Score")
    fig.suptitle("WHR component variables vs official score", y=1.02)
    save_figure(fig, "feature_scatter.png")
    return corr


def plot_index_replication(replication: dict[str, object]) -> None:
    results = replication["results"]
    coefficients = replication["coefficients"]

    print("Question: Can machine learning recover the WHR implicit weighting scheme?")
    print("Important: this is an index-replication experiment, not true happiness forecasting.")
    print(results.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.barplot(data=results, x="model", y="RMSE", ax=axes[0], color="tab:blue")
    axes[0].set_title("2019 score RMSE")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=results, x="model", y="Spearman_rank", ax=axes[1], color="tab:green")
    axes[1].set_title("Rank agreement")
    axes[1].tick_params(axis="x", rotation=20)

    sns.barplot(data=results, x="model", y="Top10Overlap", ax=axes[2], color="tab:purple")
    axes[2].set_title("Top-10 overlap")
    axes[2].tick_params(axis="x", rotation=20)
    save_figure(fig, "model_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labeled_coefficients = coefficients.rename(index=FEATURE_LABELS)
    colors = ["tab:green" if value >= 0 else "tab:red" for value in labeled_coefficients.values]
    ax.barh(labeled_coefficients.index[::-1], labeled_coefficients.values[::-1], color=colors[::-1])
    ax.set_title("Recovered WHR-style component weights")
    ax.set_xlabel("Standardized linear coefficient")
    save_figure(fig, "contributing_factors.png")


def plot_custom_formula(happy: pd.DataFrame, weights: pd.Series) -> tuple[pd.DataFrame, float]:
    custom = compute_custom_index(happy, weights, year=2019)
    rho = float(stats.spearmanr(custom["rank"], custom["custom_rank"]).statistic)

    print("Custom weighted index, 2019:")
    print(weights.rename(index=FEATURE_LABELS).round(3))
    print(f"Spearman rank agreement with official 2019 rank: {rho:.3f}")
    print(custom[["country", "score", "rank", "custom_index", "custom_rank", "rank_delta"]].head(15))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    labeled_weights = weights.rename(index=FEATURE_LABELS).sort_values()
    axes[0].barh(labeled_weights.index, labeled_weights.values, color="teal")
    axes[0].set_title("Custom-index weights")
    axes[0].set_xlabel("Normalized weight")

    axes[1].scatter(custom["rank"], custom["custom_rank"], alpha=0.6, s=35, color="teal")
    lim = [0, custom["rank"].max() + 2]
    axes[1].plot(lim, lim, "k--", lw=1)
    axes[1].invert_yaxis()
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Official 2019 rank")
    axes[1].set_ylabel("Custom-index rank")
    axes[1].set_title(f"Rank agreement, Spearman = {rho:.2f}")
    save_figure(fig, "custom_formula.png")
    return custom, rho


def plot_lagged_forecast(forecast: dict[str, object]) -> None:
    results = forecast["results"]
    print("No-leakage forecasting experiment:")
    print("Models predict 2019 using only information available before each target year.")
    print(results.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=results, x="model", y="RMSE", ax=axes[0], color="tab:blue")
    axes[0].set_title("Forecast RMSE")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(data=results, x="model", y="Spearman_rank", ax=axes[1], color="tab:green")
    axes[1].set_title("Forecast rank agreement")
    axes[1].tick_params(axis="x", rotation=20)
    save_figure(fig, "forecasting_experiment.png")


def write_results(
    central: pd.DataFrame,
    most_stable: pd.DataFrame,
    most_improved: pd.DataFrame,
    replication: dict[str, object],
    custom_spearman: float,
    forecast: dict[str, object],
) -> None:
    replication_results = replication["results"].round(3)
    forecast_results = forecast["results"].round(3)

    artifacts = {
        "central": central.reset_index().to_dict(orient="records"),
        "modeling_note": (
            "The main ML section is an index-replication experiment because WHR "
            "component variables are part of the official score construction."
        ),
        "model_results": replication_results.to_dict(orient="records"),
        "index_replication_results": replication_results.to_dict(orient="records"),
        "best_replication_model": str(replication_results.iloc[0]["model"]),
        "replication_weights": replication["weights"].round(3).to_dict(),
        "custom_spearman": custom_spearman,
        "forecasting_note": (
            "The lagged forecast uses only information available before the target year."
        ),
        "forecast_results": forecast_results.to_dict(orient="records"),
        "best_forecast_model": str(forecast_results.iloc[0]["model"]),
        "top_stable": most_stable.reset_index().to_dict(orient="records"),
        "top_improved": most_improved.reset_index().to_dict(orient="records"),
    }

    with open(PROJECT_DIR / "project_results.json", "w") as file:
        json.dump(artifacts, file, indent=2, default=str)
    print(f"Wrote {PROJECT_DIR / 'project_results.json'}")


def main() -> None:
    happy = load_happiness_data()
    print(f"Loaded {len(happy)} country-year rows from {happy['year'].min()}-{happy['year'].max()}.")
    print(f"Countries represented: {happy['country'].nunique()}")

    print_section("Score Trends")
    central = plot_central_tendency(happy)

    print_section("Ranking Stability")
    most_stable, most_improved, _ = plot_ranking_movers(happy)

    print_section("Feature Relationships")
    plot_feature_relationships(happy)

    print_section("WHR Index Replication")
    replication = run_index_replication_experiment(happy)
    plot_index_replication(replication)

    print_section("Custom Weighted Index")
    _, custom_spearman = plot_custom_formula(happy, replication["weights"])

    print_section("No-Leakage Forecasting")
    forecast = run_lagged_forecasting_experiment(happy)
    plot_lagged_forecast(forecast)

    write_results(central, most_stable, most_improved, replication, custom_spearman, forecast)


if __name__ == "__main__":
    main()
