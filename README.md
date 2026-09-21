# statistics-standards
First-year statistics project 
## Team Members

- Natetat Tubdoung - Project Lead & Data Curator
- Theerawat Rungroung - Statistical Analyst
- Theerathat Rakkiat - Visualization Specialist

## Project Overview
Analysis of the top 1,500 Steam games by revenue released between 1 Jan 2024 and 9 Sep 2024. This project examines how review sentiment, publisher class, pricing, and genre relate to commercial performance.

## Dataset
Source: Kaggle — Top 1500 games on steam by revenue 09-09-2024

## Setup Instructions
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Analysis notebooks

Run in order (`00` → `06`); each depends on outputs from the ones before it.

- `00_initial_inspection.ipynb` — first look at the raw data
- `01_data_cleaning.ipynb` — cleaning pipeline, plus an appended section that audits the data further and builds the log-scale/indicator features (disguised missing review scores, free-to-play flag, left-truncation) used everywhere below
- `02_statistical_summary.ipynb` — descriptive statistics, concentration (Gini/Lorenz), Spearman correlation, and regression models for revenue (OLS, quantile regression, Gamma GLM)
- `03_exploratory_visualizations.ipynb` — distribution plots, boxplots, correlation heatmap, interactive plots
- `04_hypothesis_testing.ipynb` — group comparisons (Kruskal–Wallis, Dunn/Holm post-hoc, Cliff's δ) and chi-square association test
- `05_distribution_fitting.ipynb` — truncated-likelihood distribution fits with AIC/BIC and bootstrap goodness-of-fit
- `06_confidence_intervals.ipynb` — BCa bootstrap confidence intervals with a coverage simulation

These notebooks use rank-based tests, log-scale/quantile regression, truncated likelihood fits, and BCa bootstrap rather than mean/normal-based methods (t-tests, ANOVA, Pearson correlation, untruncated distribution fits), because this dataset is heavily right-skewed, left-truncated (top-1,500 selection), and has disguised missing values (reviewScore == 0). See `reports/week3_report.md` for the full rationale and findings; helper code in `src/statistics.py`; figures in `reports/figures/`.
