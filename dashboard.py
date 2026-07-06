from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from world_happiness_core import (
    CORE_FEATURES,
    FEATURE_LABELS,
    compute_custom_index,
    load_happiness_data,
    normalize_weights,
    run_index_replication_experiment,
    run_lagged_forecasting_experiment,
)

st.set_page_config(page_title="World Happiness Dashboard", page_icon=":bar_chart:", layout="wide")


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_happiness_data()


@st.cache_data(show_spinner=False)
def get_replication_results() -> dict[str, object]:
    return run_index_replication_experiment(get_data())


@st.cache_data(show_spinner=False)
def get_forecast_results() -> dict[str, object]:
    return run_lagged_forecasting_experiment(get_data())


def feature_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


happy = get_data()
years = sorted(happy["year"].unique())
countries = sorted(happy["country"].unique())

st.title("World Happiness Dashboard")
st.caption(
    "World Happiness Report data, 2015-2019. The model section separates WHR index "
    "replication from real forecasting."
)

tab_country, tab_year, tab_custom, tab_models = st.tabs(
    ["Country Trends", "Year Compare", "Custom Index", "Modeling"]
)

with tab_country:
    country = st.selectbox("Country", countries, index=countries.index("United States") if "United States" in countries else 0)
    country_data = happy.loc[happy["country"] == country].sort_values("year")
    first = country_data.iloc[0]
    latest = country_data.iloc[-1]

    score_change = latest["score"] - first["score"]
    rank_change = int(first["rank"] - latest["rank"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest score", f"{latest['score']:.3f}", f"{score_change:+.3f}")
    c2.metric("Latest rank", f"{int(latest['rank'])}", f"{rank_change:+d} places")
    c3.metric("Best rank", f"{int(country_data['rank'].min())}")
    c4.metric("Years shown", f"{country_data['year'].min()}-{country_data['year'].max()}")

    left, right = st.columns(2)
    score_fig = px.line(
        country_data,
        x="year",
        y="score",
        markers=True,
        title=f"{country}: happiness score",
        labels={"year": "Year", "score": "Official WHR score"},
    )
    score_fig.update_traces(line_color="#0f766e", marker_size=9)
    score_fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=390)
    left.plotly_chart(score_fig, width="stretch")

    rank_fig = px.line(
        country_data,
        x="year",
        y="rank",
        markers=True,
        title=f"{country}: rank",
        labels={"year": "Year", "rank": "Official WHR rank"},
    )
    rank_fig.update_yaxes(autorange="reversed")
    rank_fig.update_traces(line_color="#b45309", marker_size=9)
    rank_fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=390)
    right.plotly_chart(rank_fig, width="stretch")

    trend_table = country_data[
        ["year", "rank", "score", "gdp", "social_support", "health", "freedom", "generosity", "corruption"]
    ].rename(columns=FEATURE_LABELS)
    st.dataframe(trend_table, width="stretch", hide_index=True)

with tab_year:
    year = st.selectbox("Year", years, index=len(years) - 1)
    year_data = happy.loc[happy["year"] == year].copy()

    c1, c2, c3 = st.columns([1, 1, 2])
    top_n = c1.slider("Countries shown", min_value=10, max_value=50, value=20, step=5)
    compare_mode = c2.radio("Compare by", ["Official rank", "Score"], horizontal=True)
    feature = c3.selectbox("Feature scatter", CORE_FEATURES, format_func=feature_name)

    if compare_mode == "Score":
        bar_data = year_data.sort_values("score", ascending=False).head(top_n)
    else:
        bar_data = year_data.sort_values("rank").head(top_n)

    bar_fig = px.bar(
        bar_data.sort_values("score"),
        x="score",
        y="country",
        color="region",
        orientation="h",
        hover_data=["rank", "score"],
        title=f"Top {top_n} countries in {year}",
        labels={"score": "Official WHR score", "country": ""},
    )
    bar_fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=560, legend_title_text="Region")
    st.plotly_chart(bar_fig, width="stretch")

    scatter_fig = px.scatter(
        year_data,
        x=feature,
        y="score",
        color="region",
        hover_name="country",
        hover_data=["rank"],
        title=f"{feature_name(feature)} vs official score, {year}",
        labels={feature: feature_name(feature), "score": "Official WHR score"},
    )
    scatter_fig.update_traces(marker=dict(size=9, opacity=0.8))
    scatter_fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=460, legend_title_text="Region")
    st.plotly_chart(scatter_fig, width="stretch")

with tab_custom:
    replication = get_replication_results()
    default_weights = replication["weights"]

    year_custom = st.selectbox("Custom-index year", years, index=len(years) - 1)
    st.subheader("Feature weights")

    sliders = {}
    slider_cols = st.columns(3)
    for idx, feature in enumerate(CORE_FEATURES):
        with slider_cols[idx % 3]:
            sliders[feature] = st.slider(
                feature_name(feature),
                min_value=0,
                max_value=100,
                value=int(round(float(default_weights[feature]) * 100)),
                step=1,
            )

    weights = normalize_weights(sliders)
    custom = compute_custom_index(happy, weights, year=year_custom)
    rho = float(stats.spearmanr(custom["rank"], custom["custom_rank"]).statistic)
    top10_official = set(custom.loc[custom["rank"] <= 10, "country"])
    top10_custom = set(custom.loc[custom["custom_rank"] <= 10, "country"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Rank agreement", f"{rho:.3f}")
    m2.metric("Top-10 overlap", f"{len(top10_official & top10_custom)} / 10")
    m3.metric("Countries", f"{len(custom)}")

    weight_fig = px.bar(
        weights.rename(index=FEATURE_LABELS).sort_values(),
        orientation="h",
        title="Normalized custom weights",
        labels={"value": "Weight", "index": ""},
    )
    weight_fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10), height=350)

    scatter = px.scatter(
        custom,
        x="score",
        y="custom_index",
        color="region",
        hover_name="country",
        hover_data=["rank", "custom_rank", "rank_delta"],
        title=f"Custom index vs official score, {year_custom}",
        labels={"score": "Official WHR score", "custom_index": "Custom weighted index"},
    )
    scatter.update_traces(marker=dict(size=9, opacity=0.8))
    scatter.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=350, legend_title_text="Region")

    left, right = st.columns([1, 2])
    left.plotly_chart(weight_fig, width="stretch")
    right.plotly_chart(scatter, width="stretch")

    rank_fig = go.Figure()
    rank_fig.add_trace(
        go.Scatter(
            x=custom["rank"],
            y=custom["custom_rank"],
            mode="markers",
            text=custom["country"],
            marker=dict(size=8, color=custom["score"], colorscale="Teal", showscale=True),
            hovertemplate="%{text}<br>Official rank: %{x}<br>Custom rank: %{y}<extra></extra>",
        )
    )
    limit = int(custom[["rank", "custom_rank"]].max().max()) + 2
    rank_fig.add_trace(
        go.Scatter(x=[1, limit], y=[1, limit], mode="lines", line=dict(color="black", dash="dash"), showlegend=False)
    )
    rank_fig.update_xaxes(title="Official rank", autorange="reversed")
    rank_fig.update_yaxes(title="Custom rank", autorange="reversed")
    rank_fig.update_layout(title="Official rank vs custom rank", height=430, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(rank_fig, width="stretch")

    table = custom[
        ["country", "region", "rank", "custom_rank", "rank_delta", "score", "custom_index"]
    ].sort_values("custom_rank")
    st.dataframe(table, width="stretch", hide_index=True)

with tab_models:
    replication = get_replication_results()
    forecast = get_forecast_results()

    st.subheader("WHR index replication")
    st.caption(
        "These models use same-year WHR component variables, so the result should be read as "
        "recovering the official index structure rather than forecasting happiness."
    )
    st.dataframe(replication["results"].round(3), width="stretch", hide_index=True)

    coeffs = replication["coefficients"].rename(index=FEATURE_LABELS).sort_values()
    coeff_fig = px.bar(
        coeffs,
        orientation="h",
        title="Standardized linear coefficients",
        labels={"value": "Coefficient", "index": ""},
    )
    coeff_fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10), height=390)
    st.plotly_chart(coeff_fig, width="stretch")

    st.subheader("No-leakage lagged forecast")
    st.caption(
        "This experiment predicts a target year using only data from earlier years for each country. "
        "The 2019 row's WHR components are not used to predict 2019."
    )
    st.dataframe(forecast["results"].round(3), width="stretch", hide_index=True)

    forecast_fig = px.bar(
        forecast["results"],
        x="model",
        y="RMSE",
        color="Spearman_rank",
        title="Forecasting experiment results",
        labels={"RMSE": "RMSE", "model": ""},
    )
    forecast_fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=390)
    st.plotly_chart(forecast_fig, width="stretch")

st.markdown(
    """
    <div style="
        margin-top: 3rem;
        padding: 1rem 0 0.25rem;
        border-top: 1px solid rgba(49, 51, 63, 0.18);
        color: rgba(49, 51, 63, 0.72);
        font-size: 0.9rem;
        text-align: center;
    ">
        Made by Lara
    </div>
    """,
    unsafe_allow_html=True,
)
