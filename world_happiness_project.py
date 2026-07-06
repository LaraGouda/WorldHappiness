#!/usr/bin/env python
# coding: utf-8

# # CSE 351 — Project #3: What Makes People in a Country Happy?
# 
# **Course:** CSE 351 — Introduction to Data Science (Spring 2026)
# **Team:** Alan John, Lara Gouda
# **Dataset:** World Happiness Report, 2015–2019
# 
# ---
# 
# ## Project goal
# 
# The World Happiness Report ranks countries by how happy their citizens perceive themselves to be.
# In this script we (1) explore happiness data from 2015–2019, (2) train three machine-learning
# models on 2015–2018 to predict the 2019 happiness ranking, and (3) propose our own
# formula for a happiness score.
# 
# Both team members contributed equally to every stage of the project.
# 

# Imports and global config
import json
import math
import os
import warnings
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = PROJECT_DIR / ".matplotlib"
CACHE_DIR = PROJECT_DIR / ".cache"
MPLCONFIG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 110

RNG = 351
np.random.seed(RNG)

DATA_DIR = PROJECT_DIR / "world_happiness_data"
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

print("Setup complete. Looking for data in:", DATA_DIR.resolve())


# ## 1. Loading and cleaning the data
# 
# The five CSV files (2015–2019) **do not share a schema**: column names changed almost every year,
# the *Region* column was dropped after 2016, and `Family` was renamed to `Social support` in 2018.
# Before any analysis we unify the schema, back-fill missing regions from earlier years, and
# normalize country names so that the same country in different years lines up.
# 


# Schema map for each year. We rename to a single canonical set of column names.
RENAME_MAPS = {
    2015: {
        "Country": "country", "Region": "region",
        "Happiness Rank": "rank", "Happiness Score": "score",
        "Economy (GDP per Capita)": "gdp", "Family": "social_support",
        "Health (Life Expectancy)": "health", "Freedom": "freedom",
        "Trust (Government Corruption)": "corruption", "Generosity": "generosity",
    },
    2016: {
        "Country": "country", "Region": "region",
        "Happiness Rank": "rank", "Happiness Score": "score",
        "Economy (GDP per Capita)": "gdp", "Family": "social_support",
        "Health (Life Expectancy)": "health", "Freedom": "freedom",
        "Trust (Government Corruption)": "corruption", "Generosity": "generosity",
    },
    2017: {
        "Country": "country",
        "Happiness.Rank": "rank", "Happiness.Score": "score",
        "Economy..GDP.per.Capita.": "gdp", "Family": "social_support",
        "Health..Life.Expectancy.": "health", "Freedom": "freedom",
        "Trust..Government.Corruption.": "corruption", "Generosity": "generosity",
    },
    2018: {
        "Country or region": "country",
        "Overall rank": "rank", "Score": "score",
        "GDP per capita": "gdp", "Social support": "social_support",
        "Healthy life expectancy": "health",
        "Freedom to make life choices": "freedom",
        "Perceptions of corruption": "corruption", "Generosity": "generosity",
    },
    2019: {
        "Country or region": "country",
        "Overall rank": "rank", "Score": "score",
        "GDP per capita": "gdp", "Social support": "social_support",
        "Healthy life expectancy": "health",
        "Freedom to make life choices": "freedom",
        "Perceptions of corruption": "corruption", "Generosity": "generosity",
    },
}

CORE_FEATURES = ["gdp", "social_support", "health", "freedom", "generosity", "corruption"]

# A few country-name variants observed across the five files. Mapping them to a single form
# lets the same country line up across years.
COUNTRY_FIXES = {
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Hong Kong S.A.R., China": "Hong Kong",
    "Taiwan Province of China": "Taiwan",
    "North Cyprus": "Northern Cyprus",
    "Somaliland Region": "Somaliland region",
}

def load_year(year: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{year}.csv")
    df = df.rename(columns=RENAME_MAPS[year])
    keep = [c for c in ["country", "region", "rank", "score", *CORE_FEATURES] if c in df.columns]
    df = df[keep].copy()
    df["country"] = df["country"].replace(COUNTRY_FIXES)
    df["year"] = year
    # Coerce numerics; corruption in 2018 had one stray non-numeric (UAE).
    for c in ["rank", "score", *CORE_FEATURES]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

raw = {y: load_year(y) for y in range(2015, 2020)}
for y, df in raw.items():
    print(f"{y}: {df.shape[0]:>3} rows, {df.shape[1]} cols")



# Build a country->region map from 2015 + 2016 (the only years with Region) and back-fill
region_map = (
    pd.concat([raw[2015][["country", "region"]], raw[2016][["country", "region"]]])
      .dropna()
      .drop_duplicates(subset="country")
      .set_index("country")["region"]
)
print(f"Region map covers {len(region_map)} countries.")

# Stack everything into one long DataFrame
happy = pd.concat(raw.values(), ignore_index=True)
happy["region"] = happy["region"].fillna(happy["country"].map(region_map))
print("After back-fill, rows missing region:", happy["region"].isna().sum())

# Show which countries still lack a region (newcomers in 2017+ that never appeared earlier)
still_missing = happy.loc[happy["region"].isna(), "country"].unique()
print("Countries without a region after back-fill:", list(still_missing))



# Manually assign regions for the few countries that appeared only after 2017
manual_regions = {
    "Taiwan": "Eastern Asia",
    "Hong Kong": "Eastern Asia",
    "Trinidad and Tobago": "Latin America and Caribbean",
    "Northern Cyprus": "Western Europe",
    "North Macedonia": "Central and Eastern Europe",
    "Mozambique": "Sub-Saharan Africa",
    "Lesotho": "Sub-Saharan Africa",
    "Central African Republic": "Sub-Saharan Africa",
    "Gambia": "Sub-Saharan Africa",
    "Namibia": "Sub-Saharan Africa",
    "South Sudan": "Sub-Saharan Africa",
    "Belize": "Latin America and Caribbean",
    "Somalia": "Sub-Saharan Africa",
    "Somaliland region": "Sub-Saharan Africa",
}
mask = happy["region"].isna()
happy.loc[mask, "region"] = happy.loc[mask, "country"].map(manual_regions)
print("Rows still missing region:", happy["region"].isna().sum())

# Missing values in numeric columns
print("\nMissing-value counts per numeric column:")
print(happy[["score", *CORE_FEATURES]].isna().sum())

# Fill the very few missing numeric values with the country's own median across years; if that
# is also NaN, fall back to the global median for that feature.
for c in CORE_FEATURES + ["score"]:
    happy[c] = happy.groupby("country")[c].transform(lambda s: s.fillna(s.median()))
    happy[c] = happy[c].fillna(happy[c].median())

print("\nAfter imputation, missing numeric values:", happy[["score", *CORE_FEATURES]].isna().sum().sum())
print("\nFinal merged dataset:")
print(happy.head())



# Quick outlier scan via IQR. We log them but do not drop — every country is meaningful.
def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return s[(s < lo) | (s > hi)]

print("IQR outliers per feature (count of country-years):")
for c in CORE_FEATURES + ["score"]:
    print(f"  {c:<14} {len(iqr_outliers(happy[c]))}")


# ## 2. Central tendencies of the happiness score
# 
# Did global happiness drift up or down between 2015 and 2019?
# We compute mean / median / std for each year and visualize the trend.
# 

central = happy.groupby("year")["score"].agg(["mean", "median", "std", "min", "max"]).round(3)
print(central)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(central.index, central["mean"], yerr=central["std"],
            marker="o", capsize=4, label="mean ± 1 std")
ax.plot(central.index, central["median"], "--s", color="tab:red", label="median")
ax.set_title("Global happiness score over time")
ax.set_xlabel("Year"); ax.set_ylabel("Happiness score")
ax.legend()
plt.tight_layout(); plt.savefig(FIG_DIR/"central_tendency.png", bbox_inches="tight")
plt.close(fig)


# **Reading the trend.** The mean and median both stay close to 5.4, a fraction of a point.
# The world average happiness score has been **essentially flat** across 2015–2019, with a slight
# upward drift of ~0.05 points. The spread (std ≈ 1.1) is much larger than the year-to-year
# movement, so what changes is *who* is happy, not the global average.
# 

# ## 3. Ranking stability and big movers
# 
# Some countries are perennially near the top or bottom; others rise or fall sharply.
# We compute each country's rank standard deviation and the rank delta from 2015 → 2019.
# 


rank_wide = happy.pivot_table(index="country", columns="year", values="rank")

# Keep only countries present in every year so std/delta are meaningful
rank_full = rank_wide.dropna()
rank_summary = pd.DataFrame({
    "rank_std":   rank_full.std(axis=1).round(2),
    "rank_2015":  rank_full[2015].astype(int),
    "rank_2019":  rank_full[2019].astype(int),
    "improved_by": (rank_full[2015] - rank_full[2019]).astype(int),
})

# Most stable (lowest std)
most_stable = rank_summary.sort_values("rank_std").head(10)
print("=== 10 most stable rankings ===")
print(most_stable)

# Most improved (largest positive improved_by)
most_improved = rank_summary.sort_values("improved_by", ascending=False).head(10)
print("\n=== 10 most improved 2015 -> 2019 ===")
print(most_improved)

# Biggest fallers (largest negative improved_by)
biggest_fallers = rank_summary.sort_values("improved_by").head(10)
print("\n=== 10 biggest fallers 2015 -> 2019 ===")
print(biggest_fallers)



fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sns.barplot(data=most_stable.reset_index(), y="country", x="rank_std", ax=axes[0], color="tab:blue")
axes[0].set_title("Most stable rankings\n(lowest std across 2015–2019)")
axes[0].set_xlabel("Std of rank"); axes[0].set_ylabel("")

mover_data = pd.concat([most_improved.head(8).assign(kind="improved"),
                        biggest_fallers.head(8).assign(kind="fell")]).reset_index()
sns.barplot(data=mover_data, y="country", x="improved_by", hue="kind",
            palette={"improved": "tab:green", "fell": "tab:red"}, ax=axes[1])
axes[1].set_title("Largest rank moves 2015 → 2019")
axes[1].set_xlabel("Rank improvement (positive = climbed)")
axes[1].set_ylabel("")
plt.tight_layout(); plt.savefig(FIG_DIR/"ranking_stability.png", bbox_inches="tight")
plt.close(fig)


# **Observations.**
# - *Most stable*: Nordic and Western-European countries (Switzerland, Norway, Denmark, Finland,
#   Iceland, Sweden, Netherlands, Austria) plus Australia and Canada — all consistently in the top
#   ~12.
# - *Biggest climbers*: Benin, Honduras, Ivory Coast, Togo and Guinea each climbed a large number
#   of ranks. These are countries that started near the bottom and made measurable gains in
#   GDP / social-support indicators.
# - *Biggest fallers*: Venezuela's collapse (multiple decades of economic crisis) is the single
#   largest drop, alongside India, Zambia and Botswana.
# 

# ## 4. How does each feature relate to happiness?
# 
# We pool 2015–2018 (the training years) and look at:
# 1. The Pearson correlation between every feature and the happiness score.
# 2. Scatter plots with regression lines for each of the six core features.
# 

train = happy[happy["year"] <= 2018].copy()
test  = happy[happy["year"] == 2019].copy()
print(f"Train: {train.shape[0]} country-years (2015-2018)   Test: {test.shape[0]} country-years (2019)")

corr = train[["score", *CORE_FEATURES]].corr()
print("\nCorrelation with happiness score:")
print(corr["score"].drop("score").sort_values(ascending=False).round(3))

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Pearson correlation matrix (training years)")
plt.tight_layout(); plt.savefig(FIG_DIR/"correlation_heatmap.png", bbox_inches="tight")
plt.close(fig)


fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, feat in zip(axes.flat, CORE_FEATURES):
    sns.regplot(data=train, x=feat, y="score", scatter_kws={"alpha": 0.4, "s": 20}, ax=ax)
    r = train[feat].corr(train["score"])
    ax.set_title(f"{feat}   (r = {r:+.2f})")
fig.suptitle("Score vs each feature, 2015–2018 pooled", y=1.02)
plt.tight_layout(); plt.savefig(FIG_DIR/"feature_scatter.png", bbox_inches="tight")
plt.close(fig)


# **Strongest drivers** (highest correlation with score): **GDP per capita**, **social support**,
# and **healthy life expectancy** — each above r ≈ 0.7. These three economic / structural factors
# together explain the bulk of cross-country variation in reported happiness.
# 
# **Weakest drivers**: **generosity** (r ≈ 0.15) and **corruption** (r ≈ 0.4 — but the sign is
# positive because the dataset measures *perceptions of corruption* in a way where higher = better).
# Generosity, in particular, varies almost independently of happiness rank.
# 

# ## 5. What contributes to happiness? — "If you were president…"
# 
# Correlation tells us how a feature *associates* with happiness in isolation. To compare features
# on the same footing while controlling for one another, we standardize all six features on the
# training set and fit an ordinary-least-squares regression. The standardized coefficients give a
# clean ranking of how much an extra standard-deviation of each feature adds to the predicted score.
# 


X_train = train[CORE_FEATURES].values
y_train = train["score"].values

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)

ols = LinearRegression().fit(X_train_s, y_train)
coef = pd.Series(ols.coef_, index=CORE_FEATURES).sort_values(ascending=False)
print("Standardized OLS coefficients (effect of +1 std of each feature on score):")
print(coef.round(3))

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["tab:green" if v >= 0 else "tab:red" for v in coef.values]
ax.barh(coef.index[::-1], coef.values[::-1], color=colors[::-1])
ax.set_title("What contributes most to happiness\n(standardized regression coefficients)")
ax.set_xlabel("Effect on happiness score (per +1 std)")
plt.tight_layout(); plt.savefig(FIG_DIR/"contributing_factors.png", bbox_inches="tight")
plt.close(fig)


# **If we were president of a country**, the standardized coefficients tell us where to invest:
# 
# 1. **GDP per capita** and **social support** dominate — together they account for the largest
#    share of the predicted score increase. Policies that grow real income and that strengthen
#    community / family safety nets produce the biggest happiness gains per unit effort.
# 2. **Healthy life expectancy** is next: invest in public health, primary care and clean
#    environments.
# 3. **Freedom to make life choices** has a smaller but still positive effect — open civic
#    institutions matter.
# 4. **Corruption perceptions** contributes a positive effect (because higher values in this dataset
#    mean *less* perceived corruption). Anti-corruption reform helps.
# 5. **Generosity** has the smallest effect. It is more an output of an already-happy society than
#    a lever to pull.
# 

# ## 6. Modeling — predicting the 2019 happiness ranking
# 
# **Setup.** Train on pooled 2015–2018 country-years; test on 2019. We drop the `score` and `rank`
# columns from the 2019 test set as required, predict scores from the six features, then derive a
# predicted ranking by sorting predicted scores in descending order.
# 
# **Three models** (each is briefly described in its own subsection):
# 1. **Linear Regression (OLS)** — closed-form fit; gives interpretable coefficients.
# 2. **Random Forest Regressor** — bagged decision trees; captures non-linearities and
#    interactions; robust to feature scaling.
# 3. **Gradient Boosting Regressor** — additive trees fit sequentially to the residuals; usually
#    the strongest baseline on small tabular data.
# 
# **Evaluation metrics on 2019**:
# - RMSE / MAE on predicted score (regression accuracy)
# - **Spearman rank correlation** between predicted and actual ranks (the question of interest)
# - Mean absolute rank error
# - Top-10 overlap (how many of the true top-10 each model places in its top-10)
# 


X_test  = test[CORE_FEATURES].values
y_test  = test["score"].values
true_rank_2019 = test.set_index("country")["rank"].astype(int)

def evaluate(name, y_pred):
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    pred_rank = (
        pd.Series(y_pred, index=test["country"].values)
          .rank(ascending=False, method="min")
          .astype(int)
    )
    common = pred_rank.index.intersection(true_rank_2019.index)
    rho, _ = stats.spearmanr(true_rank_2019.loc[common], pred_rank.loc[common])
    rank_err = float(np.abs(true_rank_2019.loc[common] - pred_rank.loc[common]).mean())
    top10_true = set(true_rank_2019[true_rank_2019 <= 10].index)
    top10_pred = set(pred_rank[pred_rank <= 10].index)
    top10_overlap = len(top10_true & top10_pred)
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2,
            "Spearman_rho": rho, "MeanRankErr": rank_err,
            "Top10Overlap": top10_overlap, "pred_rank": pred_rank, "y_pred": y_pred}

models = {
    "Linear Regression": Pipeline([("scale", StandardScaler()),
                                   ("lin", LinearRegression())]),
    "Random Forest":     RandomForestRegressor(n_estimators=400, max_depth=None,
                                               random_state=RNG, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                                   learning_rate=0.05, random_state=RNG),
}

results = []
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    res = evaluate(name, mdl.predict(X_test))
    results.append(res)

results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("pred_rank", "y_pred")}
                           for r in results])
print(results_df.round(3).to_string(index=False))



# 5-fold cross-validation on the training set (2015–2018) to compare CV RMSE vs held-out RMSE
kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
cv_rows = []
for name, mdl in models.items():
    rmse_scores = -cross_val_score(mdl, X_train, y_train,
                                    cv=kf, scoring="neg_root_mean_squared_error")
    cv_rows.append({"model": name,
                    "CV_RMSE_mean": rmse_scores.mean(),
                    "CV_RMSE_std":  rmse_scores.std()})
cv_df = pd.DataFrame(cv_rows)

merged = results_df.merge(cv_df, on="model")[["model", "CV_RMSE_mean", "CV_RMSE_std",
                                              "RMSE", "MAE", "Spearman_rho",
                                              "MeanRankErr", "Top10Overlap"]]
print("Combined results (CV on train, held-out metrics on 2019):")
print(merged.round(3).to_string(index=False))



# Visualize the model comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
sns.barplot(data=merged, x="model", y="RMSE", ax=axes[0], color="tab:blue")
axes[0].set_title("Test RMSE on 2019"); axes[0].tick_params(axis='x', rotation=20)
sns.barplot(data=merged, x="model", y="Spearman_rho", ax=axes[1], color="tab:green")
axes[1].set_title("Spearman ρ (predicted vs true 2019 rank)"); axes[1].tick_params(axis='x', rotation=20)
sns.barplot(data=merged, x="model", y="Top10Overlap", ax=axes[2], color="tab:purple")
axes[2].set_title("Top-10 overlap (out of 10)"); axes[2].tick_params(axis='x', rotation=20)
plt.tight_layout(); plt.savefig(FIG_DIR/"model_comparison.png", bbox_inches="tight")
plt.close(fig)



# Show the predicted vs true 2019 top 15 for the best model (highest Spearman)
best_idx = int(np.argmax([r["Spearman_rho"] for r in results]))
best = results[best_idx]
print(f"Best model by Spearman: {best['model']}  (rho = {best['Spearman_rho']:.3f})")

pred_rank = best["pred_rank"]
side = pd.DataFrame({
    "true_rank": true_rank_2019,
    "pred_rank": pred_rank,
}).dropna().sort_values("true_rank").head(15).astype(int)
print("\nTrue top 15 in 2019 vs predicted ranks:")
print(side)



# Predicted-vs-actual score scatter for the best model
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.scatter(y_test, best["y_pred"], alpha=0.6, s=40)
lims = [min(y_test.min(), best["y_pred"].min()) - 0.2,
        max(y_test.max(), best["y_pred"].max()) + 0.2]
ax.plot(lims, lims, "k--", lw=1, label="perfect prediction")
ax.set_xlabel("Actual 2019 happiness score")
ax.set_ylabel(f"Predicted score ({best['model']})")
ax.set_title(f"{best['model']}: predicted vs actual score, 2019")
ax.legend()
plt.tight_layout(); plt.savefig(FIG_DIR/"pred_vs_actual.png", bbox_inches="tight")
plt.close(fig)


# ### How each model works
# 
# **Linear Regression (OLS).** Fits a hyperplane ŷ = β₀ + β·x by closed-form least squares.
# The standardized coefficients we already saw above *are* this model. It assumes the relationship
# between each feature and the score is linear and that the features act additively. It is the most
# interpretable model and serves as our baseline.
# 
# **Random Forest.** Trains many decision trees, each on a random bootstrap sample of the rows
# and a random subset of the features. The forest's prediction is the average of the trees'
# predictions. This averaging reduces variance, captures non-linear effects (a feature can matter
# more in one regime than another), and tolerates correlated features without exploding.
# 
# **Gradient Boosting.** Builds shallow trees one at a time, each new tree fit to the residual
# errors of the running ensemble. Together with a small learning rate, this gives a flexible
# non-linear function that often outperforms random forests on small tabular datasets.
# 
# ### Discussion of results
# 
# - The Linear Regression model performed best on the 2019 holdout set, with Random Forest next
#   and Gradient Boosting trailing on this small dataset.
# - The Spearman rank correlations are high overall, confirming that the *ordering* of countries
#   is recovered fairly well even when the exact score is off by a few tenths.
# - **Why we still miss.** Several factors limit accuracy:
#   1. The Family/Social-support definition changed between 2017 and 2018, so the training years
#      mix two slightly different feature definitions.
#   2. The Region column was dropped after 2016, so we can not condition on macro-region effects
#      in 2019.
#   3. The dataset is small (~150 countries × 4 train years = ~600 rows).
# - **Did cross-validation help?** The CV RMSE on 2015–2018 is similar to (and a touch higher than)
#   the 2019 held-out RMSE. The training years are stable, so models that are well-tuned on CV
#   generalize well to 2019 — there was no obvious overfitting to fix.
# 

# ## 7. Inventing our own happiness-score formula
# 
# The official happiness score is a constructed index. We define our own using the standardized
# linear-regression coefficients (Section 5) as weights, normalized so the positive-direction
# weights sum to 1. Each feature is first re-scaled to a 0-1 range so the formula is interpretable
# on a familiar scale, then weighted and summed.
# 
# $$
# \text{my\_score} = \sum_i w_i \cdot \tilde x_i,
# \qquad \tilde x_i = \frac{x_i - x_i^{\min}}{x_i^{\max} - x_i^{\min}}.
# $$
# 


# Use the (positive) standardized OLS coefficients as weights
weights = ols.coef_.copy()
weights = np.clip(weights, 0, None)        # keep direction = "more is better"
weights = weights / weights.sum()           # normalize to sum to 1
weight_series = pd.Series(weights, index=CORE_FEATURES).round(3)
print("Custom-formula weights (sum to 1):")
print(weight_series)

# Min-max scale each feature on the 2019 test set, then apply weights
scaled_2019 = test[CORE_FEATURES].copy()
for c in CORE_FEATURES:
    lo, hi = scaled_2019[c].min(), scaled_2019[c].max()
    scaled_2019[c] = (scaled_2019[c] - lo) / (hi - lo + 1e-9)

my_score = (scaled_2019 * weights).sum(axis=1)
my_rank  = my_score.rank(ascending=False, method="min").astype(int)

custom = pd.DataFrame({
    "country":    test["country"].values,
    "true_score": test["score"].values,
    "true_rank":  test["rank"].values.astype(int),
    "my_score":   my_score.values.round(3),
    "my_rank":    my_rank.values,
}).sort_values("true_rank").reset_index(drop=True)

rho_custom, _ = stats.spearmanr(custom["true_rank"], custom["my_rank"])
print(f"\nSpearman correlation (custom rank vs true 2019 rank): {rho_custom:.3f}")
print("\nTop 15 countries by our custom formula vs the official ranking:")
print(custom.head(15).to_string(index=False))



# Visualize the weights and the agreement of our ranking with the official one
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

axes[0].barh(weight_series.sort_values().index, weight_series.sort_values().values, color="teal")
axes[0].set_title("Custom-formula weights")
axes[0].set_xlabel("Weight (sum = 1)")

axes[1].scatter(custom["true_rank"], custom["my_rank"], alpha=0.55, s=30, color="teal")
lim = [0, custom["true_rank"].max() + 2]
axes[1].plot(lim, lim, "k--", lw=1)
axes[1].set_xlabel("Official 2019 rank")
axes[1].set_ylabel("Our custom-formula rank")
axes[1].set_title(f"Custom rank vs official (Spearman ρ = {rho_custom:.2f})")
plt.tight_layout(); plt.savefig(FIG_DIR/"custom_formula.png", bbox_inches="tight")
plt.close(fig)


# **Why these weights look the way they do.** GDP per capita, social support and healthy life
# expectancy together carry most of the weight — exactly the same three features that came out
# on top in the correlation analysis and in the standardized OLS regression. Generosity gets a
# near-zero weight, consistent with its weak link to score in the data. The custom ranking lines
# up with the official ranking with high Spearman correlation, despite using a much simpler
# formula.
# 

# ## 8. Conclusions
# 
# 1. **Global happiness was effectively flat 2015–2019.** Means hover near 5.4 with std ≈ 1.1.
#    What matters is *who* is happy, not the global average.
# 2. **Nordic and Western-European countries dominate the top of the ranking** every single year.
#    The top of the table is remarkably stable. The largest movers are Venezuela (down sharply)
#    and a handful of African and Latin-American countries that climbed.
# 3. **GDP, social support and healthy life expectancy are the dominant drivers** of cross-country
#    happiness — both in pairwise correlation and in standardized regression. Generosity has the
#    weakest link.
# 4. **The models capture the 2019 ranking reasonably well**, with Linear Regression performing
#    best in this run. The biggest sources of error are schema drift in the feature definitions
#    across years and a small (~600-row) training set.
# 5. **Our custom score** is a simple weighted sum of min-max-scaled features. It reproduces the
#    official ranking with high Spearman correlation, demonstrating that the official index is well
#    approximated by a transparent, interpretable formula.
# 
# ### Limitations
# - The Family → Social-support rename in 2018 may reflect a definition change, not a pure rename.
# - Some countries that appeared after 2016 needed manual region labels.
# - 5 years × ~150 countries is a small data set; conclusions about *trends* should be taken
#   cautiously.
# 


# Persist key numbers so they can be reused without re-running every plot.
artifacts = {
    "central": central.reset_index().to_dict(orient="records"),
    "model_results": merged.round(3).to_dict(orient="records"),
    "best_model": best["model"],
    "spearman_best": float(best["Spearman_rho"]),
    "custom_weights": weight_series.to_dict(),
    "custom_spearman": float(rho_custom),
    "top_stable":   most_stable.reset_index().to_dict(orient="records"),
    "top_improved": most_improved.reset_index().to_dict(orient="records"),
}
with open(PROJECT_DIR / "project_results.json", "w") as f:
    json.dump(artifacts, f, indent=2, default=str)
print("Wrote project_results.json")
