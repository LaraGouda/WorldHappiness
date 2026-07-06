# World Happiness Report — Exploratory Analysis & Modeling (2015–2019)

An end-to-end analysis of the UN World Happiness Report, unifying five years of inconsistent survey data, identifying the strongest drivers of self-reported happiness across ~150 countries, and testing whether the official ranking can be recovered with machine learning and with a simple transparent formula.

## Why this analysis matters

The World Happiness Report is widely cited in policy and media discussions, but its ranking is a constructed index, and its underlying survey schema changed almost every year between 2015 and 2019 (renamed columns, a dropped region field, inconsistent country names). Before any of the interesting questions — *what actually drives happiness, and how stable are these rankings?* — can be answered, the five yearly files first have to be reconciled into one consistent dataset. This project does that reconciliation and then uses the cleaned data to answer three questions:

1. Which factors are most strongly associated with a country's happiness score, and how does that compare to what the official methodology implies?
2. How stable are country rankings year over year, and which countries moved the most?
3. How closely can standard ML and a simple custom formula recover the WHR scoring pattern, and what changes when we run a stricter no-leakage forecasting check?

## Data

- **Source:** [World Happiness Report](https://worldhappiness.report/), yearly CSVs for 2015–2019 (originally distributed via Kaggle). Not original data — used here for analysis only.
- **Coverage:** ~150 countries per year, 6 core features per country: GDP per capita, social support, healthy life expectancy, freedom to make life choices, generosity, and perceptions of corruption.
- **Schema challenges handled:**
  - Column names differ by year (e.g. `Happiness.Score` vs `Score`, `Family` vs `Social support`).
  - The `Region` column exists only in 2015–2016; back-filled for later years, with 14 newly-appearing countries manually labeled.
  - Inconsistent country naming (e.g. `Hong Kong S.A.R., China` vs `Hong Kong`) normalized across years.
  - A small number of missing/non-numeric values (e.g. a stray corruption value for the UAE in 2018) imputed using the country's own median across years, falling back to the global median.

## Methods

| Step | Approach |
|---|---|
| Data cleaning | Custom schema-mapping per year, country name normalization, region back-fill, median imputation, IQR outlier scan (logged, not removed) |
| Trend analysis | Year-over-year mean/median/std of happiness score, 2015–2019 |
| Ranking stability | Rank standard deviation and 2015→2019 rank delta per country (restricted to countries present in all 5 years) |
| Feature relationships | Pearson correlation matrix; standardized OLS regression coefficients (2015–2018 pooled) |
| Index replication | Linear Regression, Random Forest, and Gradient Boosting tested on whether they can recover the WHR score from the same WHR component variables |
| Forecasting check | A separate no-leakage experiment that predicts a later year using only information from earlier years |
| Custom formula | A transparent weighted-sum score built from the OLS coefficients, using min-max scaled features, benchmarked against the official 2019 ranking via Spearman correlation |
| Dashboard | Streamlit + Plotly app for exploring countries, years, model results, and custom index weights interactively |

## Key results

- **Global happiness was essentially flat 2015–2019** — the mean score moved from 5.38 to 5.41 (std ≈ 1.1 each year), a much smaller shift than the year-to-year spread between countries. What changes over time is *who* is happy, not the world average.
- **Nordic and Western European countries were the most stable top performers**: New Zealand, Australia, the Netherlands, Denmark, and Iceland all had a rank standard deviation under 1 point across five years.
- **Benin, Ivory Coast, and Honduras were the biggest climbers**, rising 40–53 ranking positions between 2015 and 2019.
- **GDP per capita, healthy life expectancy, and freedom to make life choices carried the largest standardized regression weights**; generosity had the smallest and weakest association with score.
- **Linear Regression outperformed the more flexible models on the 2019 holdout** (RMSE 0.55 vs 0.63 for Random Forest and 0.69 for Gradient Boosting), and reached a Spearman rank correlation of 0.89 against the true 2019 ranking — a case where the added flexibility of tree ensembles did not pay off on a small, mostly-linear dataset.
- **A simple transparent formula matched the ML models closely** (Spearman ρ = 0.89 vs the official ranking) using weighted, min-max scaled component features — suggesting the official index is well approximated by an easily explainable linear combination.

## An important methodological caveat

The official happiness score is itself constructed by the report's authors as roughly a weighted combination of these same six features (relative to a hypothetical "dystopia" baseline). That means the modeling step in this project is better understood as **testing whether standard ML can recover a known, mostly-linear scoring formula**, not as true out-of-sample prediction of an independent outcome. This is worth stating plainly rather than presenting the high Spearman correlations as evidence of strong "predictive power" in the usual sense — the strong result is expected given how the target was constructed, and Linear Regression's edge over the tree ensembles is consistent with that.

## Limitations

- The `Family` → `Social support` rename between 2017 and 2018 may reflect a genuine definition change in the underlying survey, not a pure relabeling, which slightly muddies the pooled 2015–2018 training data.
- Countries that only appear in the data from 2017 onward required manually assigned regions.
- Five years and ~150 countries per year is a small dataset (~600 training rows); year-over-year *trend* claims should be read cautiously rather than as strong statistical evidence.
- As noted above, the "prediction" task has an inherent ceiling because the target is a near-linear function of the inputs by construction.

## Repository structure

```
.
├── dashboard.py                   # Interactive Streamlit dashboard
├── world_happiness_core.py        # Shared cleaning, scoring, and modeling helpers
├── world_happiness_project.py     # Full analysis pipeline and static figures
├── world_happiness_data/          # Source CSVs (2015–2019)
├── figures/                       # Generated plots (created on run)
├── project_results.json           # Generated summary metrics (created on run)
└── requirements.txt
```

## Requirements

This project uses:

- pandas, numpy, scipy
- matplotlib, seaborn
- scikit-learn
- plotly
- streamlit

Install everything with:

```bash
pip install -r requirements.txt
```

## Running the analysis

```bash
python world_happiness_project.py
```

This regenerates all plots in `figures/` and writes summary metrics to `project_results.json`.

## Running the dashboard

After installing the requirements, start the interactive dashboard with:

```bash
streamlit run dashboard.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

The dashboard lets you:

- select a country and view its happiness score/rank trend over time
- select a year and compare countries visually
- adjust the custom index feature weights live
- compare the custom weighted index against the official WHR score/rank
- view the difference between the WHR index-replication experiment and the no-leakage forecasting experiment

## Possible extensions

- Test predictive power on a variable genuinely external to the official formula (e.g. internet penetration, urbanization rate) to move past the circularity noted above.
- Add screenshots or a short screen recording of the dashboard for a GitHub/LinkedIn post.
- Deploy the dashboard on Streamlit Community Cloud.

## Credits

Built by Lara Gouda and Alan John for CSE 351: Introduction to Data Science, Stony Brook University. Data © World Happiness Report / Gallup World Poll, used here for educational analysis only.
