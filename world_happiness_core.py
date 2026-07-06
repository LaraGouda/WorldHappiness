from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "world_happiness_data"
FIG_DIR = PROJECT_DIR / "figures"
RNG = 351

CORE_FEATURES = ["gdp", "social_support", "health", "freedom", "generosity", "corruption"]

FEATURE_LABELS = {
    "gdp": "GDP per capita",
    "social_support": "Social support",
    "health": "Healthy life expectancy",
    "freedom": "Freedom",
    "generosity": "Generosity",
    "corruption": "Low corruption perception",
}

RENAME_MAPS = {
    2015: {
        "Country": "country",
        "Region": "region",
        "Happiness Rank": "rank",
        "Happiness Score": "score",
        "Economy (GDP per Capita)": "gdp",
        "Family": "social_support",
        "Health (Life Expectancy)": "health",
        "Freedom": "freedom",
        "Trust (Government Corruption)": "corruption",
        "Generosity": "generosity",
    },
    2016: {
        "Country": "country",
        "Region": "region",
        "Happiness Rank": "rank",
        "Happiness Score": "score",
        "Economy (GDP per Capita)": "gdp",
        "Family": "social_support",
        "Health (Life Expectancy)": "health",
        "Freedom": "freedom",
        "Trust (Government Corruption)": "corruption",
        "Generosity": "generosity",
    },
    2017: {
        "Country": "country",
        "Happiness.Rank": "rank",
        "Happiness.Score": "score",
        "Economy..GDP.per.Capita.": "gdp",
        "Family": "social_support",
        "Health..Life.Expectancy.": "health",
        "Freedom": "freedom",
        "Trust..Government.Corruption.": "corruption",
        "Generosity": "generosity",
    },
    2018: {
        "Country or region": "country",
        "Overall rank": "rank",
        "Score": "score",
        "GDP per capita": "gdp",
        "Social support": "social_support",
        "Healthy life expectancy": "health",
        "Freedom to make life choices": "freedom",
        "Perceptions of corruption": "corruption",
        "Generosity": "generosity",
    },
    2019: {
        "Country or region": "country",
        "Overall rank": "rank",
        "Score": "score",
        "GDP per capita": "gdp",
        "Social support": "social_support",
        "Healthy life expectancy": "health",
        "Freedom to make life choices": "freedom",
        "Perceptions of corruption": "corruption",
        "Generosity": "generosity",
    },
}

COUNTRY_FIXES = {
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Hong Kong S.A.R., China": "Hong Kong",
    "Taiwan Province of China": "Taiwan",
    "North Cyprus": "Northern Cyprus",
    "Somaliland Region": "Somaliland region",
}

MANUAL_REGIONS = {
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


def load_year(year: int, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{year}.csv")
    df = df.rename(columns=RENAME_MAPS[year])
    keep = [c for c in ["country", "region", "rank", "score", *CORE_FEATURES] if c in df.columns]
    df = df[keep].copy()
    df["country"] = df["country"].replace(COUNTRY_FIXES)
    df["year"] = year

    for col in ["rank", "score", *CORE_FEATURES]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_happiness_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    raw = {year: load_year(year, data_dir) for year in range(2015, 2020)}

    region_map = (
        pd.concat([raw[2015][["country", "region"]], raw[2016][["country", "region"]]])
        .dropna()
        .drop_duplicates(subset="country")
        .set_index("country")["region"]
    )

    happy = pd.concat(raw.values(), ignore_index=True)
    happy["region"] = happy["region"].fillna(happy["country"].map(region_map))
    missing_region = happy["region"].isna()
    happy.loc[missing_region, "region"] = happy.loc[missing_region, "country"].map(MANUAL_REGIONS)

    for col in [*CORE_FEATURES, "score"]:
        happy[col] = happy.groupby("country")[col].transform(lambda s: s.fillna(s.median()))
        happy[col] = happy[col].fillna(happy[col].median())

    happy["rank"] = happy["rank"].astype(int)
    happy["year"] = happy["year"].astype(int)
    return happy.sort_values(["year", "rank"]).reset_index(drop=True)


def normalize_weights(weights: dict[str, float] | pd.Series) -> pd.Series:
    series = pd.Series(weights, dtype=float).reindex(CORE_FEATURES).fillna(0.0)
    series = series.clip(lower=0)
    total = float(series.sum())
    if total == 0:
        return pd.Series(1 / len(CORE_FEATURES), index=CORE_FEATURES)
    return series / total


def minmax_scale(frame: pd.DataFrame, features: list[str] = CORE_FEATURES) -> pd.DataFrame:
    scaled = frame[features].copy()
    for col in features:
        lo = scaled[col].min()
        hi = scaled[col].max()
        scaled[col] = (scaled[col] - lo) / (hi - lo + 1e-9)
    return scaled


def compute_custom_index(
    happy: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    year: int | None = None,
) -> pd.DataFrame:
    frame = happy.copy() if year is None else happy.loc[happy["year"] == year].copy()
    norm_weights = normalize_weights(weights)
    scaled = minmax_scale(frame, CORE_FEATURES)
    frame["custom_index"] = scaled.dot(norm_weights)
    frame["custom_rank"] = frame["custom_index"].rank(ascending=False, method="min").astype(int)
    frame["rank_delta"] = frame["rank"] - frame["custom_rank"]
    return frame.sort_values(["year", "custom_rank"]).reset_index(drop=True)


def evaluate_predictions(
    test_df: pd.DataFrame,
    y_pred: np.ndarray | pd.Series,
    model_name: str,
    score_col: str = "score",
    rank_col: str = "rank",
) -> dict[str, float | int | str]:
    y_true = test_df[score_col].to_numpy()
    pred = pd.Series(y_pred, index=test_df.index, dtype=float)
    pred_rank = pred.rank(ascending=False, method="min").astype(int)
    true_rank = test_df[rank_col].astype(int)
    rho = stats.spearmanr(true_rank, pred_rank).statistic

    top10_true = set(test_df.loc[true_rank <= 10, "country"])
    top10_pred = set(test_df.loc[pred_rank <= 10, "country"])

    return {
        "model": model_name,
        "RMSE": math.sqrt(mean_squared_error(y_true, pred)),
        "MAE": mean_absolute_error(y_true, pred),
        "R2": r2_score(y_true, pred),
        "Spearman_rank": float(rho),
        "MeanRankErr": float(np.abs(true_rank - pred_rank).mean()),
        "Top10Overlap": len(top10_true & top10_pred),
    }


def model_suite() -> dict[str, object]:
    return {
        "Linear Regression": Pipeline([("scale", StandardScaler()), ("lin", LinearRegression())]),
        "Random Forest": RandomForestRegressor(n_estimators=400, random_state=RNG, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            random_state=RNG,
        ),
    }


def run_index_replication_experiment(
    happy: pd.DataFrame,
    train_end_year: int = 2018,
    test_year: int = 2019,
) -> dict[str, object]:
    """Use same-year WHR components to recover the official score formula.

    This is an interpretability experiment. It intentionally uses WHR component
    variables, so it should not be described as true happiness forecasting.
    """

    train = happy.loc[happy["year"] <= train_end_year].copy()
    test = happy.loc[happy["year"] == test_year].copy()
    x_train = train[CORE_FEATURES]
    y_train = train["score"]
    x_test = test[CORE_FEATURES]

    rows = []
    predictions = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    for name, model in model_suite().items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics = evaluate_predictions(test, y_pred, name)
        cv_rmse = -cross_val_score(
            model,
            x_train,
            y_train,
            cv=kf,
            scoring="neg_root_mean_squared_error",
        )
        metrics["CV_RMSE_mean"] = float(cv_rmse.mean())
        metrics["CV_RMSE_std"] = float(cv_rmse.std())
        rows.append(metrics)
        predictions[name] = pd.Series(y_pred, index=test.index, name="predicted_score")

    results = pd.DataFrame(rows).sort_values("Spearman_rank", ascending=False).reset_index(drop=True)

    scaler = StandardScaler().fit(x_train)
    linear = LinearRegression().fit(scaler.transform(x_train), y_train)
    coefficients = pd.Series(linear.coef_, index=CORE_FEATURES).sort_values(ascending=False)
    positive_weights = normalize_weights(coefficients.clip(lower=0))

    return {
        "train": train,
        "test": test,
        "results": results,
        "predictions": predictions,
        "coefficients": coefficients,
        "weights": positive_weights,
    }


def _slope(years: pd.Series, values: pd.Series) -> float:
    if len(values) < 2 or values.nunique() <= 1:
        return 0.0
    return float(np.polyfit(years.to_numpy(dtype=float), values.to_numpy(dtype=float), 1)[0])


def build_lagged_forecast_frame(happy: pd.DataFrame, min_history: int = 2) -> pd.DataFrame:
    rows = []
    for target_year in sorted(happy["year"].unique()):
        current_year = happy.loc[happy["year"] == target_year]
        for _, target in current_year.iterrows():
            history = happy.loc[
                (happy["country"] == target["country"]) & (happy["year"] < target_year)
            ].sort_values("year")
            if len(history) < min_history:
                continue

            latest = history.iloc[-1]
            row = {
                "country": target["country"],
                "region": target["region"],
                "target_year": int(target_year),
                "score": float(target["score"]),
                "rank": int(target["rank"]),
                "latest_year": int(latest["year"]),
                "score_lag1": float(latest["score"]),
                "rank_lag1": float(latest["rank"]),
                "score_mean": float(history["score"].mean()),
                "score_trend": _slope(history["year"], history["score"]),
            }
            for feature in CORE_FEATURES:
                row[f"{feature}_lag1"] = float(latest[feature])
                row[f"{feature}_mean"] = float(history[feature].mean())
                row[f"{feature}_trend"] = _slope(history["year"], history[feature])
            rows.append(row)
    return pd.DataFrame(rows)


def lagged_feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude = {"country", "region", "target_year", "score", "rank", "latest_year"}
    return [col for col in frame.columns if col not in exclude]


def run_lagged_forecasting_experiment(
    happy: pd.DataFrame,
    test_year: int = 2019,
) -> dict[str, object]:
    """Forecast a later WHR score using only information from earlier years."""

    lagged = build_lagged_forecast_frame(happy)
    train = lagged.loc[lagged["target_year"] < test_year].copy()
    test = lagged.loc[lagged["target_year"] == test_year].copy()
    features = lagged_feature_columns(lagged)

    rows = [evaluate_predictions(test, test["score_lag1"], "Last observed score")]
    predictions = {"Last observed score": test["score_lag1"].copy()}

    models = {
        "Lagged Linear Regression": Pipeline(
            [("scale", StandardScaler()), ("lin", LinearRegression())]
        ),
        "Lagged Random Forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            random_state=RNG,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        model.fit(train[features], train["score"])
        y_pred = model.predict(test[features])
        rows.append(evaluate_predictions(test, y_pred, name))
        predictions[name] = pd.Series(y_pred, index=test.index, name="predicted_score")

    results = pd.DataFrame(rows).sort_values("Spearman_rank", ascending=False).reset_index(drop=True)
    return {
        "lagged": lagged,
        "train": train,
        "test": test,
        "features": features,
        "results": results,
        "predictions": predictions,
    }
