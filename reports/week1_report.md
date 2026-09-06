# Week 1 Report: Data Acquisition & Exploration

## Team: Statistics Superstars
**Date:** September 6, 2026

---

## 1. Dataset Overview

**Selected Dataset:** Top 1,500 Steam Games by Revenue (2024)
**Source:** Kaggle — "Top 1500 games on Steam by revenue," snapshot dated 09-09-2024
**Description:** Covers the top 1,500 Steam games by revenue released between Jan 1 and Sep 9, 2024. Used to examine how review sentiment, publisher class, pricing, and genre relate to commercial performance.

**Shape:** 1,500 rows × 12 columns (11 original + 1 derived)

**Variables:**

| Variable | Type | Missing (%) |
|---|---|---|
| name | string | 0% |
| releaseDate | datetime | 0% |
| copiesSold | integer | 0% |
| price | float | 0% |
| revenue | float | 0% |
| avgPlaytime | float | 0% |
| reviewScore | integer | 0% |
| publisherClass | category | 0% |
| publishers | string | 0.1% |
| developers | string | 0.1% |
| steamId | integer | 0% |
| revenue_per_copy *(derived)* | float | 0% |

---

## 2. Data Cleaning Summary

**Steps performed:**
1. **Missing values:** 1 missing value in `publishers` filled with the mode ("Kagura Games"); 2 missing values in `developers` filled with the mode ("Lust Desires").
2. **Duplicates:** None found.
3. **Derived column:** Added `revenue_per_copy` (revenue ÷ copiesSold) to normalize revenue by unit sales.

Final cleaned shape: **1,500 rows × 12 columns**.

See `reports/cleaning_log.txt` for the raw log.

---

## 3. Statistical Findings

**Key summary statistics:**

| Variable | Mean | Median | Std | Skew |
|---|---|---|---|---|
| price | $17.52 | $14.99 | 12.65 | 1.58 |
| revenue | $2.63M | $109K | $27.8M | 22.9 |
| copiesSold | 141,483 | 11,929 | 1,132,757 | 18.6 |
| reviewScore | 76.2 | 83.0 | 24.3 | -2.05 |
| revenue_per_copy | $14.37 | $12.15 | 11.65 | 5.05 |

Revenue and copies sold are extremely right-skewed — a small number of hit games drive most of the market's revenue, while review scores skew left (most games are reviewed favorably).

**Normality tests (Shapiro-Wilk):** Every numeric variable failed the normality test (p < 0.001 in all cases), confirming none of these variables are normally distributed — expected given the heavy skew.

**Strongest correlations:**
1. **price ↔ revenue_per_copy: 0.75** (strong positive) — pricier games earn more per copy sold, unsurprisingly.
2. **copiesSold ↔ revenue: 0.63** (moderate-strong positive) — sales volume is the main revenue driver.
3. **reviewScore ↔ revenue: 0.01** (essentially none) — review score alone has almost no linear relationship with total revenue.

---

## 4. Key Visualizations

- `reports/figures/distribution_plots.png` — histograms of all numeric variables
- `reports/figures/correlation_heatmap.png` — Pearson correlation matrix
- `reports/figures/boxplots.png` — numeric variables by `publisherClass`
- `reports/figures/qq_plots.png` — normality check per variable
- `reports/figures/3d_scatter.html` — interactive 3D scatter (avgPlaytime, copiesSold, price)

---

## 5. Initial Insights

1. Revenue is driven far more by **units sold** than by price or review score — a "hit-driven" market.
2. Review score is a weak/negligible predictor of revenue on its own, despite being the most visible quality signal to consumers.
3. Pricing strategy matters more for **per-copy** revenue than for total revenue — worth digging into by genre/publisher class in Week 2.

---

## 6. Data Quality Issues

- Very few missing values overall (< 0.1%), so cleaning was minimal.
- Heavy right-skew and outliers in `revenue` and `copiesSold` mean mean-based statistics are less informative than medians for this dataset — worth using robust/non-parametric methods going forward.

**Recommendations for Week 2:** since nothing here is normally distributed, plan on non-parametric hypothesis tests and consider log-transforming `revenue`/`copiesSold` before further modeling.

---

## 7. Next Steps (Week 2)

- [ ] Hypothesis testing (e.g., does `publisherClass` affect revenue?)
- [ ] Confidence intervals for mean revenue by publisher class
- [ ] Distribution fitting (log-normal likely candidate for revenue)
- [ ] Bootstrap methods for skewed variables

---

## Appendix — Files Generated

- `data/processed/cleaned_data.csv`
- `reports/data_dictionary.csv`
- `reports/summary_statistics.csv`
- `reports/normality_tests.csv`
- `reports/pearson_correlation.csv` / `reports/spearman_correlation.csv`
- `reports/cleaning_log.txt`
- `reports/figures/*`

**Notebooks**
- `notebooks/00_initial_inspection.ipynb`
- `notebooks/01_data_cleaning.ipynb`
- `notebooks/02_statistical_summary.ipynb`
- `notebooks/03_exploratory_visualizations.ipynb`
