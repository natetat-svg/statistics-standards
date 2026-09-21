# dashboard/app.py
"""
Steam 2024 revenue dashboard -- answers five questions about the top-1,500 Steam
games by revenue, using the robust (Week 3) analysis in this repo.

    1. How much does a typical Steam game in this dataset earn?
    2. What share of revenue do the biggest games take?
    3. Which type of publisher makes the most money?
    4. Do higher-priced games earn more?
    5. Do better-reviewed games earn more?

and three technical pages that cover the rest of the analysis: data exploration
(scale, distributions, correlations), distribution fitting (truncated MLE, AIC/BIC,
bootstrap KS) and hypothesis tests (Kruskal-Wallis, chi-square, Holm correction).

Run from the project root:
    streamlit run dashboard/app.py

Numbers are computed live from data/processed/cleaned_data.csv with the helpers in
src/statistics.py. Model tables (regressions, post-hoc tests, CIs by class, ...)
are read from the CSVs the notebooks wrote to reports/, so the dashboard always
agrees with the report. Run notebooks 01-06 first if any of them are missing.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # lets `streamlit run dashboard/app.py` find src/
    sys.path.insert(0, str(ROOT))

from src import statistics as rs  # noqa: E402
from src import visualizations as viz  # noqa: E402

LAYOUT = viz.dashboard_layout()
st.set_page_config(**LAYOUT["page_config"])

st.markdown(
    """
    <style>
    /* Wide layout is nice for the surrounding text/metrics, but letting every
       chart stretch to the full browser width makes bars, boxes and heatmap
       cells look distorted. Cap chart width and center it instead. */
    div[data-testid="stPlotlyChart"] {
        max-width: 1000px;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ORDER = LAYOUT["class_order"]  # ["Indie", "AA", "AAA"]
COLORS = LAYOUT["class_colors"]
BLUE, GREY, RED = LAYOUT["accent_colors"]["blue"], LAYOUT["accent_colors"]["grey"], LAYOUT["accent_colors"]["red"]
TEMPLATE = LAYOUT["plotly_template"]

PRICE_BINS = [-0.01, 0, 9.99, 19.99, 29.99, 59.99, np.inf]
PRICE_LABELS = ["Free", "Under $10", "$10–19.99", "$20–29.99", "$30–59.99", "$60+"]
REVIEW_BINS = [0, 59, 69, 79, 89, 100]
REVIEW_LABELS = ["Under 60", "60–69", "70–79", "80–89", "90–100"]

REQUIRED_REPORTS = [
    "ci_robust.csv",
    "ci_median_by_class.csv",
    "robust_kruskal_wallis.csv",
    "robust_pairwise_posthoc.csv",
    "robust_spearman_pairs.csv",
    "regression_ols_log_revenue.csv",
    "regression_ols_vs_glm.csv",
    "regression_quantile_coefficients.csv",
    "robust_data_decisions.csv",
    "distribution_fits_robust.csv",
    "robust_chi_square_sensitivity.csv",
    "hypothesis_tests_robust.csv",
]


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────
def money(x: float) -> str:
    """$109K, $2.63M, $1.59B ..."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    ax = abs(x)
    if ax >= 1e9:
        return f"${x / 1e9:.2f}B"
    if ax >= 1e8:
        return f"${x / 1e6:.0f}M"
    if ax >= 1e6:
        return f"${x / 1e6:.2f}M"
    if ax >= 1e5:
        return f"${x / 1e3:.0f}K"
    if ax >= 1e3:
        return f"${x / 1e3:.1f}K"
    return f"${x:,.0f}"


def tex(text: str) -> str:
    """Escape $ so Streamlit's markdown does not read '$5 ... $10' as LaTeX."""
    return text.replace("$", "\\$")


def fmt_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "–"
    if p <= 0:
        return "< 1e-300"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def style(fig: go.Figure, height: int = 420, **layout) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=10, r=10, t=55, b=10), **layout)
    return fig


def log_revenue_axis(fig: go.Figure, axis: str = "y") -> go.Figure:
    ticks = dict(
        type="log",
        tickvals=[1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
        ticktext=["$10K", "$100K", "$1M", "$10M", "$100M", "$1B"],
    )
    (fig.update_yaxes if axis == "y" else fig.update_xaxes)(**ticks)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Data + cached statistics
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def get_data() -> pd.DataFrame:
    return rs.load_robust()


@st.cache_data(show_spinner=False)
def report(name: str) -> pd.DataFrame | None:
    path = rs.REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner="Bootstrapping a confidence interval…")
def bca(values: np.ndarray, stat: str) -> tuple[float, float, float]:
    fn = {"median": np.median, "geomean": rs.geometric_mean}[stat]
    return rs.bca_ci(values, statistic=fn)  # (estimate, low, high), 95% BCa


def spearman_vs_revenue(df: pd.DataFrame, col: str) -> dict:
    """Spearman rho of revenue vs `col`; CI and Holm p come from the report table."""
    sub = df[["revenue", col]].dropna()
    rho, p = stats.spearmanr(sub["revenue"], sub[col])
    out = dict(rho=float(rho), p=float(p), n=len(sub), ci_low=np.nan, ci_high=np.nan, p_holm=np.nan)
    key = {"price": "price", "reviewScore_clean": "review"}[col]
    tbl = report("robust_spearman_pairs.csv")
    row = tbl[(tbl["X"] == "revenue") & (tbl["Y"] == key)] if tbl is not None else []
    if len(row):
        nums = re.findall(r"-?\d+\.\d+", str(row.iloc[0]["CI95%"]))
        if len(nums) >= 2:
            out["ci_low"], out["ci_high"] = float(nums[0]), float(nums[1])
        out["p_holm"] = float(row.iloc[0]["p_holm"])
    if np.isnan(out["ci_low"]):  # fallback: Bonett-Wright Fisher-z interval
        se = np.sqrt((1 + rho**2 / 2) / (len(sub) - 3))
        out["ci_low"], out["ci_high"] = float(np.tanh(np.arctanh(rho) - 1.96 * se)), float(np.tanh(np.arctanh(rho) + 1.96 * se))
    return out


def ols_term(prefix: str) -> dict:
    t = report("regression_ols_log_revenue.csv")
    r = t[t["term"].str.startswith(prefix)].iloc[0]
    return dict(mult=r["multiplier on revenue"], lo=r["mult. CI low"], hi=r["mult. CI high"], p=r["p (HC3)"])


def glm_term(prefix: str) -> dict:
    t = report("regression_ols_vs_glm.csv")
    r = t[t["term"].str.startswith(prefix)].iloc[0]
    return dict(mult=r["GLM multiplier"], p=r["GLM p (HC3)"])


@st.cache_data(show_spinner="Fitting the regression…")
def driver_importance(_df: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    """Drop-one change in R² for the log10-revenue OLS of notebook 02 (same design)."""
    d = _df[_df["has_review"] & _df["log_playtime"].notna()].copy()
    lp = d["log_price"]
    X = pd.DataFrame(
        {
            "AA": (d["publisherClass"] == "AA").astype(float),
            "AAA": (d["publisherClass"] == "AAA").astype(float),
            "free": d["is_free"].astype(float),
            "price": (lp - lp.median()).where(~d["is_free"], 0.0),
            "review": (d["reviewScore_clean"] - d["reviewScore_clean"].median()) / 10,
            "playtime": d["log_playtime"] - d["log_playtime"].median(),
        }
    )
    y = d["log_revenue"].to_numpy(float)

    def r2(cols: list[str]) -> float:
        A = np.column_stack([np.ones(len(X))] + [X[c].to_numpy(float) for c in cols])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    all_cols = list(X.columns)
    full = r2(all_cols)
    blocks = {
        "Playtime": ["playtime"],
        "Publisher class": ["AA", "AAA"],
        "Price": ["price"],
        "Review score": ["review"],
        "Free-to-play flag": ["free"],
    }
    rows = [{"driver": k, "delta_r2": full - r2([c for c in all_cols if c not in v])} for k, v in blocks.items()]
    return pd.DataFrame(rows), float(full), len(d)


@st.cache_data(show_spinner="Computing headline numbers…")
def facts(_df: pd.DataFrame) -> dict:
    """Every headline number used on more than one page, computed once."""
    rev = _df["revenue"].to_numpy(float)
    med, med_lo, med_hi = bca(rev, "median")
    geo, geo_lo, geo_hi = bca(rev, "geomean")
    p, lorenz = rs.lorenz_curve(rev)
    cls = _df.groupby("publisherClass", observed=True)["revenue"].agg(games="size", total="sum", median="median").reindex(ORDER)
    pw = report("robust_pairwise_posthoc.csv")
    pw = pw[(pw["outcome"] == "revenue") & pw["contrast"].isin(["AA vs Indie", "AAA vs Indie"])].set_index("contrast")["median_ratio"]
    cls["pct_games"] = cls["games"] / cls["games"].sum() * 100
    cls["pct_rev"] = cls["total"] / cls["total"].sum() * 100
    imp, r2_all, n_reg = driver_importance(_df)
    price_ols, review_ols = ols_term("price"), ols_term("review")
    return dict(
        n=len(_df),
        median=med, median_lo=med_lo, median_hi=med_hi,
        geo=geo, geo_lo=geo_lo, geo_hi=geo_hi,
        mean=float(rev.mean()),
        q25=float(np.quantile(rev, 0.25)), q75=float(np.quantile(rev, 0.75)),
        copies_median=float(_df["copiesSold"].median()),
        gini=rs.gini(rev),
        top1=rs.top_share(rev, k=1), top10=rs.top_share(rev, k=10), top15=rs.top_share(rev, k=15),
        top5pct=rs.top_share(rev, frac=0.05), top10pct=rs.top_share(rev, frac=0.10),
        bottom_half=float(np.interp(0.5, p, lorenz)),
        cls=cls, ratio_aa=float(pw["AA vs Indie"]), ratio_aaa=float(pw["AAA vs Indie"]),
        sp_price=spearman_vs_revenue(_df, "price"),
        sp_review=spearman_vs_revenue(_df, "reviewScore_clean"),
        price_ols=price_ols,
        price_per_doubling=dict(
            mult=price_ols["mult"] ** np.log10(2), lo=price_ols["lo"] ** np.log10(2), hi=price_ols["hi"] ** np.log10(2)
        ),
        review_ols=review_ols,
        importance=imp, r2_all=r2_all, n_reg=n_reg,
    )


def require_inputs() -> None:
    missing = [n for n in REQUIRED_REPORTS if not (rs.REPORTS_DIR / n).exists()]
    if not (rs.PROCESSED_DIR / "cleaned_data.csv").exists():
        missing.append("data/processed/cleaned_data.csv")
    if missing:
        st.error(
            "Some analysis outputs are missing. Run notebooks `01`–`06` (in order) first, "
            "then reload this page.\n\nMissing: " + ", ".join(f"`{m}`" for m in missing)
        )
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Shared chart / UI pieces
# ─────────────────────────────────────────────────────────────────────────────
def short_answer(text: str) -> None:
    st.info(tex("**Short answer:** " + text))


def quantile_effect_chart(term_prefix: str, title: str, unit: str) -> go.Figure:
    """Revenue multiplier across the revenue distribution (quantile regression)."""
    q = report("regression_quantile_coefficients.csv")
    sub = q[q["term"].str.startswith(term_prefix)].sort_values("quantile")
    mult, lo, hi = 10 ** sub["coef"], 10 ** sub["lo"], 10 ** sub["hi"]
    labels = [f"{int(round(v * 100))}th" for v in sub["quantile"]]
    fig = go.Figure(
        go.Scatter(
            x=labels, y=mult, mode="lines+markers", line=dict(color=BLUE, width=3), marker=dict(size=9),
            error_y=dict(type="data", symmetric=False, array=hi - mult, arrayminus=mult - lo, color=BLUE),
            hovertemplate="%{x} percentile<br>×%{y:.2f}<extra></extra>", name=title,
        )
    )
    fig.add_hline(y=1, line_dash="dash", line_color=GREY, annotation_text="no effect", annotation_position="bottom right")
    style(
        fig, 380, title=title, showlegend=False,
        xaxis_title="Where the game sits in the revenue distribution (percentile)",
        yaxis_title=f"Revenue multiplier {unit}",
    )
    return fig


def driver_chart(imp: pd.DataFrame, highlight: str) -> go.Figure:
    imp = imp.sort_values("delta_r2")
    fig = go.Figure(
        go.Bar(
            x=imp["delta_r2"] * 100, y=imp["driver"], orientation="h",
            marker_color=[RED if d == highlight else "#B8B8B8" for d in imp["driver"]],
            text=[f"{v * 100:.1f}%" for v in imp["delta_r2"]], textposition="outside",
        )
    )
    style(fig, 330, title="How much of the variation in revenue each factor uniquely explains", xaxis_title="Percentage points of R² lost when the factor is dropped")
    fig.update_xaxes(range=[0, max(imp["delta_r2"].max() * 100 * 1.25, 1)])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────
ORIGINAL_COLUMNS = [
    "name", "releaseDate", "copiesSold", "price", "revenue", "avgPlaytime",
    "reviewScore", "publisherClass", "publishers", "developers", "steamId",
]
# column -> (type, description)
COLUMN_INFO = {
    "name": ("Text", "Game title"),
    "releaseDate": ("Date", "Release date (2024)"),
    "release_month": ("Integer", "Month number of the release date"),
    "publisherClass": ("Category", "Publisher type: Indie, AA or AAA (one 'Hobbyist' game was merged into Indie)"),
    "copiesSold": ("Integer", "Estimated copies sold"),
    "price": ("Decimal", "Price in US dollars (0 = free-to-play)"),
    "revenue": ("Decimal", "Estimated revenue in US dollars"),
    "avgPlaytime": ("Decimal", "Average playtime in hours"),
    "reviewScore": ("Integer", "Review score, 0–100 (0 = no score recorded)"),
    "reviewScore_clean": ("Decimal", "Review score with 0 recoded to missing"),
    "has_review": ("True/False", "True if the game has a review score"),
    "is_free": ("True/False", "True if price is 0"),
    "log_revenue": ("Decimal", "log10 of revenue"),
    "log_copies": ("Decimal", "log10 of copies sold"),
    "log_price": ("Decimal", "log10 of price (paid games only)"),
    "log_playtime": ("Decimal", "log10 of average playtime (playtime > 0 only)"),
    "revenue_per_copy": ("Decimal", "Revenue ÷ copies sold (US dollars)"),
    "steamId": ("Integer", "Steam app ID"),
    "publishers": ("Text", "Publisher name(s)"),
    "developers": ("Text", "Developer name(s)"),
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def page_overview(df: pd.DataFrame) -> None:
    n_rows, n_cols = df.shape
    n_orig = len(ORIGINAL_COLUMNS)
    st.title("🎮 Dataset overview")
    st.caption(
        f"Kaggle – *Top 1500 games on Steam by revenue* (snapshot 09-09-2024). Each row is one Steam game released between "
        f"{df['releaseDate'].min():%d %b} and {df['releaseDate'].max():%d %b %Y}."
    )

    m = st.columns(4)
    m[0].metric("Rows (games)", f"{n_rows:,}")
    m[1].metric("Columns", n_cols, help=f"{n_orig} columns come from the source file; {n_cols - n_orig} were derived during cleaning.")
    m[2].metric("Original / derived columns", f"{n_orig} / {n_cols - n_orig}")
    m[3].metric("Missing values (original columns)", int(df[ORIGINAL_COLUMNS].isna().sum().sum()))
    m = st.columns(4)
    m[0].metric("Unique publishers", f"{df['publishers'].nunique():,}")
    m[1].metric("Unique developers", f"{df['developers'].nunique():,}")
    m[2].metric("Revenue range", f"{money(df['revenue'].min())} – {money(df['revenue'].max())}")
    m[3].metric("Duplicate games", int(df["steamId"].duplicated().sum()), help="Rows sharing a Steam ID.")

    left, right = st.columns(2)
    counts = df["publisherClass"].value_counts().reindex(ORDER)
    fig = go.Figure(go.Bar(
        x=counts.index.astype(str), y=counts.values, marker_color=[COLORS[c_] for c_ in counts.index],
        text=[f"{v:,} ({v / n_rows:.1%})" for v in counts.values], textposition="outside",
    ))
    fig.update_layout(title="Games by publisher class", yaxis_title="Number of games", xaxis_title="", yaxis_range=[0, counts.max() * 1.15])
    left.plotly_chart(style(fig, height=340))
    months = df["release_month"].value_counts().sort_index()
    fig = go.Figure(go.Bar(x=[MONTHS[i - 1] for i in months.index], y=months.values, marker_color=BLUE,
                           text=[f"{v:,}" for v in months.values], textposition="outside"))
    fig.update_layout(title="Games by release month (2024)", yaxis_title="Number of games", xaxis_title="", yaxis_range=[0, months.max() * 1.15])
    right.plotly_chart(style(fig, height=340))
    st.caption(f"The data end on {df['releaseDate'].max():%d %b %Y}, so September is only partly covered.")

    tab_cols, tab_quality, tab_stats, tab_browse = st.tabs(["Columns", "Data quality", "Summary statistics", "Browse the data"])

    with tab_cols:
        rows = []
        for c_ in df.columns:
            typ, desc = COLUMN_INFO.get(c_, ("", ""))
            rows.append({
                "Column": c_, "Description": desc, "Type": typ,
                "Origin": "Original" if c_ in ORIGINAL_COLUMNS else "Derived",
                "Non-missing": int(df[c_].notna().sum()), "Missing": int(df[c_].isna().sum()), "Unique values": int(df[c_].nunique()),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=35 * (len(rows) + 1) + 3)
        st.caption(
            "Derived columns are added in the cleaning step. Their missing values are intentional: log10 is undefined for 0, "
            "and a review score of 0 means no score was recorded."
        )

    with tab_quality:
        n_free, n_noscore, n_noplay = int(df["is_free"].sum()), int((~df["has_review"]).sum()), int((df["avgPlaytime"] == 0).sum())
        quality = pd.DataFrame([
            ("Duplicate rows / Steam IDs", f"{int(df.duplicated().sum())} / {int(df['steamId'].duplicated().sum())}", "None to remove."),
            ("Missing values in original columns", str(int(df[ORIGINAL_COLUMNS].isna().sum().sum())),
             "Three blank publisher/developer names in the raw file were filled in during cleaning."),
            ("Review score recorded as 0", f"{n_noscore} ({n_noscore / n_rows:.1%})",
             "Treated as 'no score' (missing in reviewScore_clean); the has_review flag is kept."),
            ("Price = 0 (free-to-play)", f"{n_free} ({n_free / n_rows:.1%})",
             "Flagged with is_free and left out of price effects (log10 of 0 is undefined)."),
            ("Average playtime = 0", f"{n_noplay} ({n_noplay / n_rows:.1%})", "Left out of log10 playtime."),
            ("Hobbyist publisher class", "1 game", "Merged into Indie (group too small to analyse)."),
            ("Selection on revenue", f"minimum revenue {money(df['revenue'].min())}",
             "Only the top 1,500 games are included, so results describe already-successful games."),
        ], columns=["Check", "Result", "How it is handled"])
        st.dataframe(quality, hide_index=True)

    with tab_stats:
        specs = [("Revenue", "revenue", "revenue"), ("Copies sold", "copiesSold", "copiesSold"), ("Price", "price", "price"),
                 ("Average playtime", "avgPlaytime", "avgPlaytime"), ("Review score (scored games)", "reviewScore_clean", "reviewScore")]
        out = []
        for label, col, fmt_col in specs:
            s = df[col].astype(float).dropna()
            f_ = lambda v, c_=fmt_col: fmt_outcome(c_, v)
            out.append({
                "Variable": label, "Count": f"{len(s):,}", "Mean": f_(s.mean()), "Std. dev.": f_(s.std()), "Min": f_(s.min()),
                "25%": f_(s.quantile(0.25)), "Median": f_(s.median()), "75%": f_(s.quantile(0.75)), "Max": f_(s.max()),
                "Mean ÷ median": f"{s.mean() / s.median():.1f}×", "Skewness": f"{stats.skew(s):.1f}",
            })
        st.dataframe(pd.DataFrame(out), hide_index=True)
        st.caption(tex(
            "Revenue and copies sold are extremely right-skewed: the mean is many times the median, and a few blockbusters pull everything. "
            "That is why the analysis uses medians, ranks and log scales. Price counts all games (free ones included); playtime counts games with playtime recorded."
        ))

    with tab_browse:
        c1, c2, c3 = st.columns([2, 2, 2])
        classes = c1.multiselect("Publisher class", ORDER, default=ORDER, key="ov_classes")
        query = c2.text_input("Search game or publisher", key="ov_query")
        sort_col = c3.selectbox("Sort by", ["revenue", "copiesSold", "price", "avgPlaytime", "reviewScore", "releaseDate"], key="ov_sort")
        ascending = st.radio("Order", ["Highest first", "Lowest first"], horizontal=True, key="ov_order") == "Lowest first"
        view = df[df["publisherClass"].isin(classes)]
        if query.strip():
            q = query.strip()
            view = view[view["name"].str.contains(q, case=False, regex=False, na=False) | view["publishers"].str.contains(q, case=False, regex=False, na=False)]
        view = view.sort_values(sort_col, ascending=ascending)
        st.caption(f"{len(view):,} of {n_rows:,} games match.")
        show = view[ORIGINAL_COLUMNS].copy()
        show["releaseDate"] = show["releaseDate"].dt.strftime("%Y-%m-%d")
        st.dataframe(show, hide_index=True, height=420)
        st.download_button("Download these rows as CSV", show.to_csv(index=False).encode("utf-8"), file_name="steam_games_filtered.csv", mime="text/csv")


def page_q1(df: pd.DataFrame) -> None:
    f = facts(df)
    st.header("Q1 · How much does a typical Steam game in this dataset earn?")
    short_answer(
        f"about **{money(f['median'])}** in revenue – that is the median game (95% CI {money(f['median_lo'])}–{money(f['median_hi'])}). "
        f"The middle half of games earn between {money(f['q25'])} and {money(f['q75'])}. The *average* game earns {money(f['mean'])}, "
        f"but that is {f['mean'] / f['median']:.0f}× the median because a few blockbusters drag it up, so it does not describe a typical game."
    )

    c = st.columns(4)
    c[0].metric("Median revenue", money(f["median"]))
    c[1].metric("Geometric mean", money(f["geo"]))
    c[2].metric("Arithmetic mean (misleading)", money(f["mean"]))
    c[3].metric("Median copies sold", f"{np.floor(f['copies_median'] + 0.5):,.0f}")
    st.caption(tex(f"95% BCa bootstrap CIs – median {money(f['median_lo'])}–{money(f['median_hi'])}; "
                   f"geometric mean {money(f['geo_lo'])}–{money(f['geo_hi'])}."))

    # Histogram of log10 revenue with the three "averages"
    step = 0.125
    edges = np.arange(np.floor(df["log_revenue"].min() * 8) / 8, df["log_revenue"].max() + step, step)
    counts, edges = np.histogram(df["log_revenue"], bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    ranges = [f"{money(10 ** a)} – {money(10 ** b)}" for a, b in zip(edges[:-1], edges[1:])]
    fig = go.Figure(go.Bar(x=centers, y=counts, width=step * 0.95, marker_color=BLUE, name="Games",
                           customdata=ranges, hovertemplate="%{customdata}<br>%{y} games<extra></extra>"))
    top = counts.max() * 1.08
    for name, val, color in [("Median", f["median"], RED), ("Geometric mean", f["geo"], "#F58518"), ("Mean", f["mean"], "#54A24B")]:
        fig.add_trace(go.Scatter(x=[np.log10(val)] * 2, y=[0, top], mode="lines", line=dict(color=color, width=3, dash="dash"),
                                 name=f"{name}: {money(val)}"))
    fig.update_xaxes(tickvals=[4, 5, 6, 7, 8, 9], ticktext=["$10K", "$100K", "$1M", "$10M", "$100M", "$1B"])
    style(fig, 430, title="Number of games by revenue (log scale)", xaxis_title="Revenue", yaxis_title="Number of games",
          legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(tex(f"The distribution is cut off on the left at {money(df['revenue'].min())} because the dataset only contains the top 1,500 games – "
                   "'typical' here means typical among already-successful games, not among all games on Steam."))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Where would a game land?")
        x = st.number_input("Revenue (USD)", min_value=0, value=250_000, step=25_000, key="q1_revenue")
        if x < df["revenue"].min():
            st.write(tex(f"{money(x)} is below this dataset's cutoff ({money(df['revenue'].min())}) – a game earning that little would not have made the top 1,500."))
        else:
            below = (df["revenue"] < x).mean()
            rank = int((df["revenue"] > x).sum() + 1)
            st.write(tex(f"A game earning **{money(x)}** out-earned **{below:.0%}** of the games here and would rank about **#{rank:,}** of {len(df):,}."))
    with right:
        st.subheader("Percentiles of revenue")
        qs = [0.05, 0.25, 0.5, 0.75, 0.95]
        tbl = pd.DataFrame({"Percentile": [f"{int(q * 100)}th" + (" (median)" if q == 0.5 else "") for q in qs],
                            "Revenue": [money(df["revenue"].quantile(q)) for q in qs]})
        st.dataframe(tbl, hide_index=True)

    with st.expander("Why the median (or geometric mean) and not the average?"):
        sim = report("ci_coverage_simulation.csv")
        cov = ""
        if sim is not None:
            c_ = sim.set_index("procedure")["coverage"]
            cov = (f" In a simulation of samples like this one, the usual 95% t-interval for the mean covered the true mean only "
                   f"**{c_['mean: t-interval']:.1%}** of the time, versus {c_['median: rank-based']:.1%} for the median.")
        st.markdown(tex(
            f"Revenue has a skewness of {df['revenue'].skew():.0f} and the mean is {f['mean'] / f['median']:.0f}× the median." + cov +
            " The median and geometric mean are stable; the mean depends on a few unobservable extreme values."))
        ci = report("ci_robust.csv")
        ci = ci[ci["variable"] == "revenue"]
        st.dataframe(pd.DataFrame({
            "Estimator": ci["statistic"], "Method": ci["method"], "Estimate": ci["estimate"].map(money),
            "95% CI": [f"{money(a)} – {money(b)}" for a, b in zip(ci["ci_low"], ci["ci_high"])],
            "CI width / estimate": ci["relative_width"].round(2)}), hide_index=True)


def page_q2(df: pd.DataFrame) -> None:
    f = facts(df)
    st.header("Q2 · What share of revenue do the biggest games take?")
    top_name = df.loc[df["revenue"].idxmax(), "name"]
    short_answer(
        f"a very large one. The **top 15 games (1%) take {f['top15']:.1%}** of all revenue, and {top_name} alone takes {f['top1']:.1%}. "
        f"The top 10% of games take {f['top10pct']:.1%}, while the bottom half of games share only {f['bottom_half']:.1%}. "
        f"The Gini coefficient is {f['gini']:.2f} (0 = perfectly equal, 1 = one game gets everything)."
    )

    c = st.columns(4)
    c[0].metric("Biggest game", f"{f['top1']:.1%}")
    c[1].metric("Top 15 games (1%)", f"{f['top15']:.1%}")
    c[2].metric("Top 10% of games", f"{f['top10pct']:.1%}")
    c[3].metric("Gini coefficient", f"{f['gini']:.3f}")

    n_games = len(df)
    pct = st.slider("Top N% of games", 1, 100, 10, key="q2_pct", help="Drag to see what share of revenue the top N% of games take.")
    n_top = int(np.ceil(pct / 100 * n_games))
    share = rs.top_share(df["revenue"], frac=pct / 100)
    st.markdown(tex(f"The top **{pct}%** of games (**{n_top:,}** games) earn **{share:.1%}** of revenue; the other {n_games - n_top:,} share the remaining {1 - share:.1%}."))

    left, right = st.columns(2)
    with left:
        p, lz = rs.lorenz_curve(df["revenue"])
        x_m = 1 - n_top / n_games
        y_m = float(np.interp(x_m, p, lz))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dash"), name="Perfect equality"))
        fig.add_trace(go.Scatter(x=p, y=lz, mode="lines", fill="tonexty", fillcolor="rgba(76,120,168,0.15)",
                                 line=dict(color=BLUE, width=3), name="Actual revenue"))
        fig.add_trace(go.Scatter(x=[x_m], y=[y_m], mode="markers+text", marker=dict(size=12, color=RED),
                                 text=[f"Top {pct}% take {share:.0%}"], textposition="middle left", name=f"Top {pct}%"))
        style(fig, 430, title="Lorenz curve of revenue", xaxis_title="Cumulative share of games (smallest → biggest)",
              yaxis_title="Cumulative share of revenue", legend=dict(orientation="h", y=-0.2))
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        k = st.slider("Show the top … games", 5, 25, 10, key="q2_k")
        topk = df.nlargest(k, "revenue").copy()
        topk["share"] = topk["revenue"] / df["revenue"].sum()
        fig = px.bar(topk.iloc[::-1], x="revenue", y="name", orientation="h", color="publisherClass",
                     color_discrete_map=COLORS, category_orders={"publisherClass": ORDER},
                     text=[f"{s:.1%}" for s in topk["share"].iloc[::-1]], labels={"revenue": "Revenue", "name": "", "publisherClass": "Publisher class"})
        fig.update_traces(textposition="outside")
        fig.update_yaxes(categoryorder="array", categoryarray=list(topk["name"].iloc[::-1]))
        style(fig, 430, title=f"The top {k} games – together {topk['share'].sum():.1%} of revenue", legend=dict(orientation="h", y=-0.15))
        fig.update_xaxes(tickvals=[0, 2e8, 4e8, 6e8, 8e8], ticktext=["$0", "$200M", "$400M", "$600M", "$800M"], range=[0, topk["revenue"].max() * 1.15])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Concentration at a glance")
    tbl = pd.DataFrame({
        "Slice of games": ["Top 1 game", "Top 10 games", "Top 15 games (1%)", "Top 5%", "Top 10%", "Top 20%", "Bottom 50%"],
        "Share of revenue": [f["top1"], f["top10"], f["top15"], f["top5pct"], f["top10pct"], rs.top_share(df["revenue"], frac=0.2), f["bottom_half"]],
    })
    tbl["Share of revenue"] = tbl["Share of revenue"].map(lambda v: f"{v:.1%}")
    st.dataframe(tbl, hide_index=True)
    st.caption("These shares describe the top 1,500 games only. The dataset cannot tell us how concentrated revenue is across all games on Steam.")


def class_table(df: pd.DataFrame, drop_top: bool = False) -> pd.DataFrame:
    d = df.drop(df["revenue"].idxmax()) if drop_top else df
    g = d.groupby("publisherClass", observed=True)["revenue"].agg(games="size", total="sum", median="median", mean="mean").reindex(ORDER)
    g["pct_games"] = g["games"] / g["games"].sum() * 100
    g["pct_rev"] = g["total"] / g["total"].sum() * 100
    return g


def page_q3(df: pd.DataFrame) -> None:
    f = facts(df)
    cls = f["cls"]
    st.header("Q3 · Which type of publisher makes the most money?")
    short_answer(
        f"**AAA publishers earn the most in total** – {money(cls.loc['AAA', 'total'])} ({cls.loc['AAA', 'pct_rev']:.1f}% of revenue) from only "
        f"{int(cls.loc['AAA', 'games'])} games – narrowly ahead of AA ({money(cls.loc['AA', 'total'])}, {cls.loc['AA', 'pct_rev']:.1f}%). "
        f"Indie games together earn {money(cls.loc['Indie', 'total'])} ({cls.loc['Indie', 'pct_rev']:.1f}%) across {int(cls.loc['Indie', 'games']):,} games. "
        f"Per game, AAA and AA titles typically earn {f['ratio_aa']:.0f}–{f['ratio_aaa']:.0f}× an Indie title, and the data cannot separate AAA from AA."
    )

    tab_total, tab_typical, tab_tests = st.tabs(["Total revenue", "Typical game", "Statistical tests"])

    with tab_total:
        top_name = df.loc[df["revenue"].idxmax(), "name"]
        top_cls = df.loc[df["revenue"].idxmax(), "publisherClass"]
        drop = st.checkbox(f"Sensitivity check: remove the single biggest game ({top_name}, {top_cls})", key="q3_drop")
        g = class_table(df, drop_top=drop)
        leader = g["total"].idxmax()
        st.markdown(tex(f"Largest total revenue{' without ' + top_name if drop else ''}: **{leader}** – {g.loc[leader, 'pct_rev']:.1f}% of revenue from {g.loc[leader, 'pct_games']:.1f}% of games."))
        if drop:
            st.caption(f"{top_name} alone is {f['top1']:.1%} of all revenue, so one title decides whether AAA or AA comes first in total revenue.")

        left, right = st.columns(2)
        with left:
            fig = go.Figure(go.Bar(x=list(g.index.astype(str)), y=g["total"], marker_color=[COLORS[c] for c in g.index],
                                   text=[money(v) for v in g["total"]], textposition="outside"))
            style(fig, 400, title="Total revenue by publisher class", yaxis_title="Total revenue", showlegend=False)
            fig.update_yaxes(range=[0, g["total"].max() * 1.15])
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="% of games", x=list(g.index.astype(str)), y=g["pct_games"], marker_color=GREY, text=[f"{v:.1f}%" for v in g["pct_games"]], textposition="outside"))
            fig.add_trace(go.Bar(name="% of revenue", x=list(g.index.astype(str)), y=g["pct_rev"], marker_color=BLUE, text=[f"{v:.1f}%" for v in g["pct_rev"]], textposition="outside"))
            style(fig, 400, title="Share of games vs share of revenue", yaxis_title="Percent", barmode="group", legend=dict(orientation="h", y=-0.15))
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pd.DataFrame({
            "Publisher class": g.index.astype(str), "Games": g["games"].map("{:,}".format), "% of games": g["pct_games"].map("{:.1f}%".format),
            "Total revenue": g["total"].map(money), "% of revenue": g["pct_rev"].map("{:.1f}%".format),
            "Median per game": g["median"].map(money), "Mean per game (hit-driven)": g["mean"].map(money)}), hide_index=True)

    with tab_typical:
        cm = report("ci_median_by_class.csv")
        cm = cm[cm["variable"] == "revenue"].set_index("publisherClass").reindex(ORDER)
        left, right = st.columns(2)
        with left:
            fig = go.Figure(go.Bar(
                x=ORDER, y=cm["median"], marker_color=[COLORS[c] for c in ORDER], text=[money(v) for v in cm["median"]], textposition="outside",
                error_y=dict(type="data", symmetric=False, array=cm["ci_high"] - cm["median"], arrayminus=cm["median"] - cm["ci_low"]),
            ))
            style(fig, 420, title="Median revenue per game (95% CI)", yaxis_title="Median revenue", showlegend=False)
            fig.update_yaxes(range=[0, cm["ci_high"].max() * 1.15])
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.box(df, x="publisherClass", y="revenue", color="publisherClass", color_discrete_map=COLORS,
                         category_orders={"publisherClass": ORDER}, labels={"publisherClass": "", "revenue": "Revenue"})
            log_revenue_axis(fig)
            style(fig, 420, title="Revenue distribution by class (log scale)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("The boxes overlap a lot: many Indie games out-earn many AAA games. The class effect is about the typical game, not a guarantee.")

    with tab_tests:
        kw = report("robust_kruskal_wallis.csv")
        kw = kw[kw["outcome"] == "revenue"].iloc[0]
        st.markdown(tex(
            f"**Kruskal–Wallis test (Indie vs AA vs AAA, revenue):** H = {kw['H']:.1f}, Holm-adjusted p = {fmt_p(kw['p_holm'])}, "
            f"effect size ε² = {kw['epsilon2']:.3f}. Revenue differs across publisher classes."))
        pw = report("robust_pairwise_posthoc.csv")
        pw = pw[pw["outcome"] == "revenue"]
        st.dataframe(pd.DataFrame({
            "Contrast": pw["contrast"],
            "Median ratio (95% CI)": [f"{r.median_ratio:.1f}× ({r.ratio_lo:.1f}–{r.ratio_hi:.1f})" for r in pw.itertuples()],
            "Cliff's δ [95% CI]": [f"{r.cliffs_delta:.2f} [{r.delta_lo:.2f}, {r.delta_hi:.2f}]" for r in pw.itertuples()],
            "Effect size": pw["magnitude"], "Dunn p (Holm)": pw["p_dunn_holm"].map(fmt_p)}), hide_index=True)
        aaa_aa = pw[pw["contrast"] == "AAA vs AA"].iloc[0]
        st.markdown(tex(
            "**Reading it:** AA and AAA games earn far more than Indie games (large effects). AAA vs AA is negligible "
            f"(δ = {aaa_aa['cliffs_delta']:.2f}, p = {fmt_p(aaa_aa['p_dunn_holm'])}), so the data give no reason to say AAA beats AA per game. "
            "Median ratios are used because a t-test on means is dominated by a few blockbusters – it wrongly reported no AAA vs Indie difference."))
        st.caption("Class labels come from the dataset's publisherClass column as provided (one 'Hobbyist' row was merged into Indie).")


def page_q4(df: pd.DataFrame) -> None:
    f = facts(df)
    sp, dbl, po = f["sp_price"], f["price_per_doubling"], f["price_ols"]
    d = df.copy()
    d["band"] = pd.cut(d["price"], PRICE_BINS, labels=PRICE_LABELS)
    bands = d.groupby("band", observed=True)["revenue"].agg(
        games="size", median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75), total="sum")
    bands["share"] = bands["total"] / bands["total"].sum()
    st.header("Q4 · Do higher-priced games earn more?")
    short_answer(
        f"**yes, on average – but that does not show that price drives revenue.** Revenue rises with price (Spearman ρ = {sp['rho']:.2f}, 95% CI {sp['ci_low']:.2f}–{sp['ci_high']:.2f}). "
        f"The median game priced $30–59.99 earned {money(bands.loc['$30–59.99', 'median'])}, versus {money(bands.loc['Under $10', 'median'])} for games under $10. "
        f"Even after accounting for publisher class, playtime and review score, doubling the price goes with about ×{dbl['mult']:.1f} revenue "
        f"(95% CI ×{dbl['lo']:.1f}–{dbl['hi']:.1f}). The data cannot show that raising a price would raise revenue; bigger-budget games simply charge more."
    )

    c = st.columns(3)
    c[0].metric("Spearman ρ (revenue vs price)", f"{sp['rho']:.2f}")
    c[1].metric("Revenue per doubling of price†", f"×{dbl['mult']:.2f}")
    imp = f["importance"].set_index("driver")["delta_r2"]
    c[2].metric("Variation uniquely explained by price", f"{imp['Price']:.1%}")
    st.caption("†Adjusted for publisher class, playtime, review score and free-to-play status (OLS on log revenue, notebook 02).")

    tab_band, tab_all, tab_adj = st.tabs(["By price band", "Every game", "Adjusted effect"])

    with tab_band:
        fig = go.Figure(go.Bar(
            x=list(bands.index.astype(str)), y=bands["median"], marker_color=[GREY if b == "Free" else BLUE for b in bands.index],
            text=[f"{money(m)}<br>n={n}" for m, n in zip(bands["median"], bands["games"])], textposition="outside"))
        style(fig, 430, title="Median revenue by price band", yaxis_title="Median revenue (log scale)", showlegend=False)
        log_revenue_axis(fig)
        fig.update_yaxes(range=[4, np.log10(bands["median"].max() * 4)])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame({
            "Price band": bands.index.astype(str), "Games": bands["games"], "Median revenue": bands["median"].map(money),
            "Middle 50% of games": [f"{money(a)} – {money(b)}" for a, b in zip(bands["q1"], bands["q3"])],
            "Share of all revenue": bands["share"].map("{:.1%}".format),
            "Note": ["Small sample" if n < 30 else "" for n in bands["games"]]}), hide_index=True)
        st.caption("Free-to-play games (grey) run on a different business model – they earn from in-game purchases – so they are kept separate from paid games.")

    with tab_all:
        paid = df[df["price"] > 0]
        by_class = st.checkbox("Colour by publisher class", value=True, key="q4_colour")
        fig = viz.create_interactive_scatter(paid, x_col="price", y_col="revenue", color_col="publisherClass" if by_class else None,
                                             log_x=True, log_y=True, hover_name="name", opacity=0.45)
        fig.update_layout(xaxis_title="Price (USD)", yaxis_title="Revenue", legend_title_text="Publisher class")
        slope, intercept, *_ = stats.linregress(paid["log_price"], paid["log_revenue"])
        xs = np.linspace(paid["log_price"].min(), paid["log_price"].max(), 50)
        fig.add_trace(go.Scatter(x=10 ** xs, y=10 ** (intercept + slope * xs), mode="lines", line=dict(color="black", width=3),
                                 name="Log-log trend (unadjusted)"))
        log_revenue_axis(fig)
        fig.update_xaxes(type="log", tickvals=[1, 2, 5, 10, 20, 40, 80], ticktext=["$1", "$2", "$5", "$10", "$20", "$40", "$80"])
        style(fig, 470, title=f"Price vs revenue, {len(paid):,} paid games (both axes log)", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(tex(f"Unadjusted trend: doubling the price goes with about ×{2 ** slope:.1f} revenue. It is larger than the adjusted ×{dbl['mult']:.1f} "
                       "because publisher class, playtime and review score travel with price. Hover a dot to see the game."))

    with tab_adj:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(quantile_effect_chart("price", "Effect of a 10× higher price, by revenue percentile", "(×10 price)"), use_container_width=True)
        with right:
            st.plotly_chart(driver_chart(f["importance"], "Price"), use_container_width=True)
        glm = glm_term("price")
        qp = report("regression_quantile_coefficients.csv")
        qp = qp[qp["term"].str.startswith("price")].set_index("quantile")["multiplier"]
        st.markdown(tex(
            f"**Model results.** Median-scale (OLS on log revenue): a 10× higher price goes with ×{po['mult']:.1f} revenue (95% CI {po['lo']:.1f}–{po['hi']:.1f}, p = {fmt_p(po['p'])}). "
            f"Quantile regression shows the price effect **fans out**: it is about ×{qp[0.1]:.1f} at the 10th revenue percentile and ×{qp[0.9]:.1f} at the 90th. "
            f"On the mean scale (Gamma GLM) the estimate is ×{glm['mult']:.1f} with p = {fmt_p(glm['p'])}, i.e. not clearly different from zero, because means are dominated by a few hits."))


def page_q5(df: pd.DataFrame) -> None:
    f = facts(df)
    sp, ro = f["sp_review"], f["review_ols"]
    imp = f["importance"].set_index("driver")["delta_r2"]
    qtab = report("regression_quantile_coefficients.csv")
    qr = qtab[qtab["term"].str.startswith("review")].set_index("quantile")["multiplier"]
    st.header("Q5 · Do better-reviewed games earn more?")
    short_answer(
        f"**barely.** For a typical game, review score has almost no relationship with revenue (Spearman ρ = {sp['rho']:.2f}, 95% CI {sp['ci_low']:.2f}–{sp['ci_high']:.2f}, "
        f"Holm-adjusted p = {fmt_p(sp['p_holm'])}). Holding price, playtime and publisher class fixed, +10 review points goes with about ×{ro['mult']:.2f} revenue "
        f"(95% CI ×{ro['lo']:.2f}–{ro['hi']:.2f}) – detectable but small, explaining only {imp['Review score']:.1%} of the variation. "
        f"The effect is larger among hits (×{qr[0.9]:.2f} at the 90th revenue percentile vs ×{qr[0.1]:.2f} at the 10th)."
    )

    scored = df[df["has_review"]].copy()
    c = st.columns(3)
    c[0].metric("Spearman ρ (revenue vs review)", f"{sp['rho']:.2f}", help=f"p (Holm-adjusted) = {fmt_p(sp['p_holm'])}; n = {sp['n']:,} scored games")
    c[1].metric("Revenue per +10 review points†", f"×{ro['mult']:.2f}")
    c[2].metric("Variation uniquely explained", f"{imp['Review score']:.1%}")
    st.caption("†Adjusted for publisher class, price, playtime and free-to-play status. Review score = % positive reviews; 0 is treated as 'no score' (see the Unscored tab).")

    tab_rel, tab_unscored, tab_adj = st.tabs(["Reviews vs revenue", "Unscored games", "Adjusted effect"])

    with tab_rel:
        by_class = st.checkbox("Colour by publisher class", value=False, key="q5_colour")
        fig = viz.create_interactive_scatter(scored, x_col="reviewScore_clean", y_col="revenue", color_col="publisherClass" if by_class else None,
                                             log_y=True, hover_name="name", opacity=0.35)
        fig.update_layout(xaxis_title="Review score (% positive)", yaxis_title="Revenue", legend_title_text="Publisher class")
        scored["band"] = pd.cut(scored["reviewScore_clean"], REVIEW_BINS, labels=REVIEW_LABELS)
        bands = scored.groupby("band", observed=True).agg(
            n=("revenue", "size"), median=("revenue", "median"), q1=("revenue", lambda s: s.quantile(0.25)),
            q3=("revenue", lambda s: s.quantile(0.75)), x=("reviewScore_clean", "mean"))
        fig.add_trace(go.Scatter(x=bands["x"], y=bands["median"], mode="lines+markers", line=dict(color=RED, width=4), marker=dict(size=10),
                                 name="Median by review band", customdata=[[b, n] for b, n in zip(bands.index.astype(str), bands["n"])],
                                 hovertemplate="%{customdata[0]}: n=%{customdata[1]}<br>median %{y:$,.0f}<extra></extra>"))
        log_revenue_axis(fig)
        style(fig, 470, title=f"Review score vs revenue, {len(scored):,} scored games", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame({
            "Review band": bands.index.astype(str), "Games": bands["n"], "Median revenue": bands["median"].map(money),
            "Middle 50% of games": [f"{money(a)} – {money(b)}" for a, b in zip(bands["q1"], bands["q3"])]}), hide_index=True)
        st.caption("The red line is nearly flat: games with 60% and 95% positive reviews earn similar amounts at the median. Note the wide spread in every band.")

    with tab_unscored:
        un, sc = df[~df["has_review"]], df[df["has_review"]]
        u_stat = stats.mannwhitneyu(sc["revenue"], un["revenue"], alternative="two-sided")
        left, right = st.columns([1, 1])
        with left:
            fig = go.Figure(go.Bar(x=[f"Scored (n={len(sc):,})", f"No score (n={len(un)})"], y=[sc["revenue"].median(), un["revenue"].median()],
                                   marker_color=[BLUE, GREY], text=[money(sc["revenue"].median()), money(un["revenue"].median())], textposition="outside"))
            style(fig, 380, title="Median revenue: scored vs unscored games", yaxis_title="Median revenue", showlegend=False)
            fig.update_yaxes(range=[0, un["revenue"].median() * 1.2])
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown(tex(
                f"**{len(un)} games have a review score of 0.** A true 0% positive rating is implausible for games with this much revenue, so the analysis treats 0 as *no score recorded*. "
                f"These games earn **more** (median {money(un['revenue'].median())} vs {money(sc['revenue'].median())}; Mann–Whitney p = {fmt_p(u_stat.pvalue)}), "
                "so they are not missing at random – counting them as 'terrible reviews' would make reviews look less related to revenue than they are."))
            st.markdown("**Biggest unscored games**")
            st.dataframe(un.nlargest(5, "revenue")[["name", "publisherClass", "revenue"]].assign(revenue=lambda t: t["revenue"].map(money))
                         .rename(columns={"name": "Game", "publisherClass": "Class", "revenue": "Revenue"}), hide_index=True)

    with tab_adj:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(quantile_effect_chart("review", "Effect of +10 review points, by revenue percentile", "(+10 points)"), use_container_width=True)
        with right:
            st.plotly_chart(driver_chart(f["importance"], "Review score"), use_container_width=True)
        glm = glm_term("review")
        st.markdown(tex(
            f"**Model results.** OLS on log revenue: +10 review points goes with ×{ro['mult']:.2f} revenue (95% CI {ro['lo']:.2f}–{ro['hi']:.2f}, p = {fmt_p(ro['p'])}). "
            f"Quantile regression: ×{qr[0.1]:.2f} at the 10th percentile, ×{qr[0.5]:.2f} at the median, ×{qr[0.9]:.2f} at the 90th – good reviews matter a little more for hits. "
            f"On the mean scale (Gamma GLM) the estimate is ×{glm['mult']:.2f}. Together, all factors explain {f['r2_all']:.0%} of the variation in log revenue; "
            f"playtime, publisher class and price each explain more than review score does."))
        st.caption("Association, not causation: playtime and reviews may both reflect game quality.")


def page_methods(df: pd.DataFrame) -> None:
    f = facts(df)
    st.header("Data & methods")
    st.markdown(tex(
        f"**Dataset.** Kaggle – *Top 1500 games on Steam by revenue* (09-09-2024): {len(df):,} games released "
        f"{df['releaseDate'].min():%d %b} – {df['releaseDate'].max():%d %b %Y}. Columns: release date, copies sold, price, revenue, average playtime, "
        "review score, publisher class (Indie / AA / AAA), publisher and developer."))

    st.subheader("Data decisions")
    st.dataframe(report("robust_data_decisions.csv"), hide_index=True)

    st.subheader("Why medians, ranks and log scales?")
    st.markdown(tex(
        f"- **Extreme concentration** – Gini {f['gini']:.3f} and mean = {f['mean'] / f['median']:.0f}× median, so averages, Pearson correlations, t-tests and ANOVA are driven by a few games.\n"
        f"- **Selection on the outcome** – only the top {len(df):,} games are included (minimum revenue {money(df['revenue'].min())}), so every result is about already-successful games.\n"
        f"- **Disguised missing values** – {int((~df['has_review']).sum())} games with review score 0 are treated as unscored, not as 0% positive.\n"
        f"- **Two business models** – {int(df['is_free'].sum())} free-to-play games are flagged and kept out of price effects.\n\n"
        "So the dashboard reports medians and geometric means (BCa bootstrap CIs), Spearman correlation, Kruskal–Wallis with Dunn–Holm post-hoc tests and Cliff's δ, "
        "OLS / quantile regression on log10 revenue with robust (HC3) errors, left-truncated distribution fits ranked by AIC/BIC with bootstrap goodness-of-fit, "
        "and chi-square tests with permutation p-values and Holm correction across the family of tests."
    ))

    st.subheader("Limitations")
    st.markdown(
        "- Results describe the top 1,500 games; they cannot say what makes a game *become* a top seller.\n"
        "- Everything here is association, not causation (for example, bigger budgets raise both price and revenue).\n"
        "- Anything about the *mean* revenue depends on the unobservable extreme tail.\n"
        "- Genre, release timing and developer effects are not yet modelled."
    )

    st.subheader("Summary statistics")
    summary = report("robust_summary_statistics.csv")
    if summary is not None:
        st.dataframe(summary.round(2), hide_index=True)
    with st.expander("Preview the cleaned data"):
        cols = ["name", "releaseDate", "publisherClass", "price", "copiesSold", "revenue", "avgPlaytime", "reviewScore", "publishers"]
        st.dataframe(df[cols].sort_values("revenue", ascending=False).head(50), hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Technical pages: data exploration, distribution fitting, hypothesis tests
# ─────────────────────────────────────────────────────────────────────────────
VARIABLES = {  # label -> (column in the robust frame, log axis by default)
    "Revenue ($)": ("revenue", True),
    "Copies sold": ("copiesSold", True),
    "Price ($, paid games)": ("price", True),
    "Average playtime (hours)": ("avgPlaytime", True),
    "Review score (scored games)": ("reviewScore_clean", False),
}
FIT_VARIABLES = {k: v for k, v in VARIABLES.items() if v[0] != "reviewScore_clean"}
OUTCOMES = {  # outcome column in robust_kruskal_wallis.csv -> (label, column in df, log axis)
    "revenue": ("Revenue", "revenue", True),
    "copiesSold": ("Copies sold", "copiesSold", True),
    "price": ("Price", "price", True),
    "avgPlaytime": ("Average playtime", "avgPlaytime", True),
    "reviewScore": ("Review score (scored games)", "reviewScore_clean", False),
}
FAMILY_COLORS = {
    "Lognormal": "#E45756", "Weibull": "#4C78A8", "Log-logistic": "#54A24B",
    "Lomax (Pareto II)": "#B279A2", "Inverse Gaussian": "#F58518", "Gamma": "#7F7F7F",
}
GREY_BAR = "#D0D0D0"


def var_values(df: pd.DataFrame, col: str) -> pd.Series:
    """Positive, non-missing values of one variable (free games / unscored games drop out)."""
    s = df[col].astype(float)
    return s.where(s > 0).dropna()


def class_frame(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = pd.DataFrame({"publisherClass": df["publisherClass"], "value": df[col].astype(float).where(df[col] > 0)})
    return d.dropna(subset=["value"])


def fmt_outcome(col: str, x: float) -> str:
    if col == "revenue":
        return money(x)
    if col == "price":
        return f"${x:,.2f}"
    if col == "avgPlaytime":
        return f"{x:,.1f} h"
    if col == "copiesSold":
        return f"{x:,.0f}"
    return f"{x:,.1f}"


# ── Explore ─────────────────────────────────────────────────────────────────
def page_explore(df: pd.DataFrame) -> None:
    st.header("Explore the data")
    short_answer(
        "Revenue, copies sold, price and playtime are all **heavily right-skewed** – a few games sit far to the right, so averages mislead. "
        "On a log axis their shape becomes readable, and rank-based (Spearman) correlations show revenue moving with copies sold, playtime and price – "
        "but not with review score."
    )
    tab_scale, tab_class, tab_corr = st.tabs(["Linear vs log scale", "Distribution by publisher class", "Correlations"])

    with tab_scale:
        label = st.selectbox("Variable", list(VARIABLES), key="ex_scale_var")
        col, _ = VARIABLES[label]
        x = var_values(df, col)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Games", f"{len(x):,}")
        c2.metric("Mean ÷ median", f"{x.mean() / x.median():.1f}×")
        c3.metric("Skewness (raw)", f"{stats.skew(x):.1f}")
        c4.metric("Skewness (log10)", f"{stats.skew(np.log10(x)):.1f}")
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Linear scale", "log10 scale"))
        fig.add_trace(go.Histogram(x=x, nbinsx=40, marker_color=BLUE, showlegend=False), row=1, col=1)
        fig.add_trace(go.Histogram(x=np.log10(x), nbinsx=40, marker_color=BLUE, showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text=label, row=1, col=1)
        fig.update_xaxes(title_text=f"log10 of {label.split(' (')[0].lower()}", row=1, col=2)
        fig.update_yaxes(title_text="Number of games", row=1, col=1)
        st.plotly_chart(style(fig, height=380), use_container_width=True)
        st.caption(
            "On the linear axis nearly every game falls into the first bar, so the chart says almost nothing. "
            "Taking log10 spreads the games out. Skewness near 0 means symmetric; large positive values mean a long right tail. "
            "The log scale is not needed for review score, which is already roughly symmetric."
        )

    with tab_class:
        label = st.selectbox("Variable", list(VARIABLES), key="ex_class_var")
        col, log_default = VARIABLES[label]
        c1, c2 = st.columns(2)
        kind = c1.radio("Chart", ["Box plot", "Histogram"], horizontal=True, key="ex_kind")
        logscale = c2.checkbox("Log scale", value=log_default, key=f"ex_log_{col}")
        d = class_frame(df, col)
        if kind == "Box plot":
            fig = px.box(d, x="publisherClass", y="value", color="publisherClass", points="outliers", log_y=logscale,
                         category_orders={"publisherClass": ORDER}, color_discrete_map=COLORS,
                         labels={"value": label, "publisherClass": "Publisher class"})
            fig.update_layout(showlegend=False)
            if col == "revenue" and logscale:
                log_revenue_axis(fig)
        else:
            d = d.assign(x=np.log10(d["value"]) if logscale else d["value"])
            fig = px.histogram(d, x="x", color="publisherClass", barmode="overlay", opacity=0.6, nbins=40, histnorm="percent",
                               category_orders={"publisherClass": ORDER}, color_discrete_map=COLORS,
                               labels={"x": f"log10 of {label}" if logscale else label, "publisherClass": "Publisher class"})
            fig.update_yaxes(title_text="% of games in the class")
        st.plotly_chart(style(fig, height=430), use_container_width=True)
        g = d.groupby("publisherClass", observed=True)["value"]
        tbl = pd.DataFrame({"Games": g.size(), "Q1": g.quantile(0.25), "Median": g.median(), "Q3": g.quantile(0.75)}).reindex(ORDER)
        for c_ in ["Q1", "Median", "Q3"]:
            tbl[c_] = tbl[c_].map(lambda v: fmt_outcome(col if col != "reviewScore_clean" else "reviewScore", v))
        st.dataframe(tbl.reset_index().rename(columns={"publisherClass": "Publisher class"}), hide_index=True)
        st.caption("Boxes show the middle 50% of games; the line is the median. The overlap between classes is large for every variable.")

    with tab_corr:
        method = st.radio("Correlation type", ["Spearman (ranks)", "Pearson (raw values)"], horizontal=True, key="ex_corr")
        names = {"Revenue": "revenue", "Copies sold": "copiesSold", "Price": "price", "Playtime": "avgPlaytime", "Review score": "reviewScore_clean"}
        m = df[list(names.values())].astype(float).corr(method="spearman" if method.startswith("Spearman") else "pearson")
        m.index = m.columns = list(names)
        fig = px.imshow(m, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="equal")
        fig.update_layout(coloraxis_colorbar=dict(title="r" if method.startswith("Pearson") else "ρ"))
        st.plotly_chart(style(fig, height=430), use_container_width=False)
        st.caption(
            "Pearson on raw values is driven by the few blockbusters, so it can overstate or hide relationships; Spearman uses ranks and is the measure the analysis relies on. "
            "Review scores are pairwise complete (unscored games excluded)."
        )
        pairs = report("robust_spearman_pairs.csv").copy()
        st.subheader("Spearman correlations with Holm-adjusted p-values")
        st.dataframe(pd.DataFrame({
            "Variable 1": pairs["X"], "Variable 2": pairs["Y"], "n": pairs["n"],
            "ρ": pairs["spearman_rho"].round(3), "95% CI": pairs["CI95%"],
            "p (raw)": pairs["p_raw"].map(fmt_p), "p (Holm)": pairs["p_holm"].map(fmt_p),
            "Significant (Holm, 5%)": np.where(pairs["p_holm"] < 0.05, "Yes", "No")}), hide_index=True)


# ── Distribution fitting ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fitting candidate distributions…")
def fit_variable(_df: pd.DataFrame, col: str, truncate: bool) -> pd.DataFrame:
    x = var_values(_df, col).values
    return rs.fit_candidates(x, truncate_at=float(x.min()) if truncate else None)


@st.cache_data(show_spinner="Simulating and re-fitting (this takes a few seconds)…")
def boot_ks(_df: pd.DataFrame, col: str, family: str, truncate: bool, n_sim: int) -> tuple[float, float]:
    x = var_values(_df, col).values
    return rs.parametric_bootstrap_ks(x, rs.POSITIVE_CANDIDATES[family], truncate_at=float(x.min()) if truncate else None, n_sim=n_sim)


def pdf_log10(dist, params, y: np.ndarray, trunc: float | None = None) -> np.ndarray:
    """Density of log10(X) implied by a fitted (optionally left-truncated) model."""
    xv = 10 ** y
    f = dist.pdf(xv, *params) * xv * np.log(10)
    if trunc is not None:
        f = np.where(xv >= trunc, f / dist.sf(trunc * (1 - 1e-9), *params), 0.0)
    return f


def ppf_trunc(dist, params, u: np.ndarray, trunc: float | None = None) -> np.ndarray:
    if trunc is None:
        return dist.ppf(u, *params)
    f0 = dist.cdf(trunc * (1 - 1e-9), *params)
    return dist.ppf(f0 + u * (1 - f0), *params)


def page_distributions(df: pd.DataFrame) -> None:
    fits = report("distribution_fits_robust.csv")
    rev = fits[fits["variable"].str.startswith("revenue")].iloc[0]
    rejected = (fits.loc[fits["bootstrap p"] < 0.05, "variable"].str.replace(r" \(.*\)", "", regex=True)
                .replace({"copiesSold": "copies sold", "avgPlaytime": "average playtime"}).tolist())
    st.header("Distribution fitting")
    short_answer(
        f"**Revenue is best described by a left-truncated {rev['best family (AIC)'].lower()} distribution.** The runner-up ({rev['runner-up']}) is only "
        f"ΔAIC = {rev['dAIC runner-up']:.1f} behind, so the two are hard to tell apart, and the bootstrap goodness-of-fit test does not reject the model "
        f"(p = {rev['bootstrap p']:.2f}). For {' and '.join(rejected) if rejected else 'no variable'} even the best-fitting family is rejected, "
        f"so no standard distribution describes {'them' if len(rejected) != 1 else 'it'} well."
    )
    tab_sum, tab_fit = st.tabs(["Summary for every variable", "Fit explorer"])

    with tab_sum:
        tbl = pd.DataFrame({
            "Variable": fits["variable"], "n": fits["n"], "Best family (AIC)": fits["best family (AIC)"],
            "Runner-up": fits["runner-up"], "ΔAIC of runner-up": fits["dAIC runner-up"].round(1),
            "KS D": fits["KS_D"].round(3),
            "Bootstrap KS p": fits["bootstrap p"].map(lambda p: "not run" if pd.isna(p) else fmt_p(p)),
        })
        st.dataframe(tbl, hide_index=True)
        st.markdown(tex(
            "- **Left-truncation.** Only the top 1,500 games by revenue are in the data, so revenue never falls below "
            f"{money(df['revenue'].min())}. Fitting an ordinary (untruncated) distribution to it is misspecified; the truncated fit conditions the likelihood on that cut-off.\n"
            "- **AIC / BIC** rank the candidate families (lower is better); ΔAIC below about 2 means the models are practically tied.\n"
            "- **Bootstrap KS.** The textbook Kolmogorov–Smirnov p-value is invalid when parameters are estimated from the same data, so the model is simulated and re-fitted "
            "many times to get a valid p-value. Small p means the model is rejected.\n"
            "- Review score is fitted on the proportion scale (Beta vs logit-normal vs normal), price on paid games only. "
            "The power-law tail analysis is in notebook `05`."
        ))

    with tab_fit:
        label = st.selectbox("Variable", list(FIT_VARIABLES), key="fit_var")
        col, _ = FIT_VARIABLES[label]
        truncate = False
        if col == "revenue":
            truncate = st.checkbox("Account for left-truncation (the data are only the top 1,500 games)", value=True, key="fit_trunc",
                                   help="Turn this off to see what an ordinary fit does: it puts the peak in the wrong place.")
        fit = fit_variable(df, col, truncate)
        x = var_values(df, col).values
        trunc = float(x.min()) if truncate else None
        tag = f"{col}_{truncate}"

        st.dataframe(pd.DataFrame({
            "Family": fit["distribution"], "Parameters": fit["n_params"], "Log-likelihood": fit["loglik"].round(1),
            "AIC": fit["AIC"].round(1), "ΔAIC": fit["dAIC"].round(1), "BIC": fit["BIC"].round(1), "KS D": fit["KS_D"].round(3),
        }), hide_index=True)
        st.caption("Sorted by AIC. KS D is a descriptive distance only; use the bootstrap test below for a valid p-value.")

        picks = st.multiselect("Families to draw", fit["distribution"].tolist(), default=fit["distribution"].head(3).tolist(), key=f"fit_pick_{tag}")
        qq_family = st.selectbox("Q-Q plot against", fit["distribution"].tolist(), key=f"fit_qq_{tag}")
        pars = {r.distribution: r.params for r in fit.itertuples()}

        left, right = st.columns(2)
        y_grid = np.linspace(np.log10(x.min()), np.log10(x.max()), 400)
        fig = go.Figure(go.Histogram(x=np.log10(x), nbinsx=40, histnorm="probability density", marker_color=GREY_BAR, name="observed"))
        for nm in picks:
            fig.add_trace(go.Scatter(x=y_grid, y=pdf_log10(rs.POSITIVE_CANDIDATES[nm], pars[nm], y_grid, trunc), mode="lines",
                                     name=nm, line=dict(color=FAMILY_COLORS.get(nm), width=2.5)))
        fig.update_layout(title="Observed vs fitted density (log10 axis)", xaxis_title=f"log10 of {label.split(' (')[0].lower()}", yaxis_title="Density", legend=dict(orientation="h", y=-0.2))
        left.plotly_chart(style(fig, height=420))

        xs = np.sort(x)
        u = (np.arange(1, len(xs) + 1) - 0.5) / len(xs)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q = ppf_trunc(rs.POSITIVE_CANDIDATES[qq_family], pars[qq_family], u, trunc)
        ok = np.isfinite(q) & (q > 0)
        lim = [float(xs.min()), float(xs.max())]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q[ok], y=xs[ok], mode="markers", marker=dict(size=4, color=FAMILY_COLORS.get(qq_family, BLUE), opacity=0.6), name="games"))
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="black", dash="dash"), name="perfect fit"))
        fig.update_xaxes(type="log", title=f"Theoretical quantile ({qq_family}{', truncated' if truncate else ''})")
        fig.update_yaxes(type="log", title="Observed")
        fig.update_layout(title="Q-Q plot (log–log axes)", legend=dict(orientation="h", y=-0.2))
        right.plotly_chart(style(fig, height=420))
        st.caption("Points on the dashed line mean the fitted family reproduces the data. Departures in the upper-right corner mean the tail of the real data is heavier or lighter than the model's.")

        with st.expander("Goodness-of-fit: parametric-bootstrap KS test"):
            st.markdown("Simulates data from the fitted family, re-fits every simulated sample, and asks how often the simulated KS distance is at least as large as the observed one.")
            c1, c2 = st.columns(2)
            fam = c1.selectbox("Family", fit["distribution"].tolist(), key=f"fit_boot_{tag}")
            n_sim = c2.select_slider("Simulations", options=[100, 200, 500], value=200, key="fit_nsim")
            if st.button("Run test", key=f"fit_run_{tag}"):
                d_obs, p_boot = boot_ks(df, col, fam, truncate, n_sim)
                m1, m2 = st.columns(2)
                m1.metric("Observed KS distance", f"{d_obs:.3f}")
                m2.metric("Bootstrap p-value", fmt_p(p_boot))
                st.caption(f"Smallest possible p with {n_sim} simulations is {1 / (n_sim + 1):.3f}. "
                           + ("The model is **rejected** at the 5% level." if p_boot < 0.05 else "The model is **not rejected** at the 5% level.")
                           + " The p-value shifts a little with the number of simulations; the summary tab uses 500.")


# ── Hypothesis tests ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running chi-square and permutation test…")
def chi_square_tiers(_df: pd.DataFrame, cut_lo: float, cut_hi: float, include_unscored: bool) -> dict | None:
    d = _df if include_unscored else _df.dropna(subset=["reviewScore_clean"])
    scores = d["reviewScore"] if include_unscored else d["reviewScore_clean"]
    tiers = pd.cut(scores, [-np.inf, cut_lo, cut_hi, np.inf], right=False, labels=["Low", "Mid", "High"])
    tab = pd.crosstab(d["publisherClass"], tiers)
    tab = tab.loc[:, tab.sum() > 0]
    if tab.shape[1] < 2:
        return None
    chi2, p, dof, exp = stats.chi2_contingency(tab, correction=False)
    _, p_perm = rs.permutation_chi2(d["publisherClass"], tiers, n_perm=5000)
    return {"tab": tab, "chi2": float(chi2), "p": float(p), "dof": int(dof), "p_perm": float(p_perm),
            "v": rs.cramers_v(tab), "min_exp": float(exp.min()), "n": int(tab.values.sum())}


def page_tests(df: pd.DataFrame) -> None:
    fam = report("hypothesis_tests_robust.csv")
    kw_all = report("robust_kruskal_wallis.csv")
    post = report("robust_pairwise_posthoc.csv")
    sig = fam[fam["significant_holm_.05"]]
    st.header("Hypothesis tests")
    short_answer(
        f"**{len(sig)} of the {len(fam)} tests are significant after Holm correction** – publisher class differs in "
        + ", ".join(t.replace("Kruskal-Wallis: ", "").replace("copiesSold", "copies sold").replace("avgPlaytime", "playtime") for t in sig["test"])
        + ", but not in review score, and review tier shows no detectable association with publisher class."
    )
    tab_cls, tab_chi, tab_holm, tab_cmp = st.tabs(["Class comparison", "Chi-square: review tier", "Holm-corrected family", "Classical vs robust"])

    with tab_cls:
        key = st.radio("Outcome", list(OUTCOMES), format_func=lambda k: OUTCOMES[k][0], horizontal=True, key="ht_outcome")
        label, col, logscale = OUTCOMES[key]
        kw = kw_all[kw_all["outcome"] == key].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kruskal–Wallis H", f"{kw['H']:.1f}")
        c2.metric("Effect size ε²", f"{kw['epsilon2']:.3f}")
        c3.metric("p (Holm)", fmt_p(kw["p_holm"]))
        c4.metric("Significant (5%)", "Yes" if kw["reject_holm_.05"] else "No")
        d = class_frame(df, col)
        fig = px.box(d, x="publisherClass", y="value", color="publisherClass", points="outliers", log_y=logscale,
                     category_orders={"publisherClass": ORDER}, color_discrete_map=COLORS, labels={"value": label, "publisherClass": "Publisher class"})
        fig.update_layout(showlegend=False)
        if col == "revenue":
            log_revenue_axis(fig)
        st.plotly_chart(style(fig, height=400), use_container_width=True)
        if logscale:
            st.caption("Zeros (free games, or games with no recorded playtime) cannot be drawn on a log axis, but they are included in the test and the medians below.")
        oc = "reviewScore" if key == "reviewScore" else col
        st.dataframe(pd.DataFrame({"Publisher class": ORDER, "Median": [fmt_outcome(oc, kw[f"median_{c_}"]) for c_ in ORDER]}), hide_index=True)
        pw = post[post["outcome"] == key]
        if len(pw):
            st.markdown("**Pairwise comparisons** (Dunn test with Holm adjustment)")
            st.dataframe(pd.DataFrame({
                "Contrast": pw["contrast"],
                "Median ratio (95% CI)": [f"{r.median_ratio:.1f}× ({r.ratio_lo:.1f}–{r.ratio_hi:.1f})" for r in pw.itertuples()],
                "Cliff's δ [95% CI]": [f"{r.cliffs_delta:.2f} [{r.delta_lo:.2f}, {r.delta_hi:.2f}]" for r in pw.itertuples()],
                "Effect size": pw["magnitude"], "Dunn p (Holm)": pw["p_dunn_holm"].map(fmt_p)}), hide_index=True)
        else:
            st.caption("No post-hoc comparisons: the overall Kruskal–Wallis test is not significant, so there is nothing to locate.")
        st.caption("Kruskal–Wallis compares whole distributions by rank, so a few blockbusters cannot dominate it. ε² is the share of rank variation explained by class (about 0.01 small, 0.06 medium, 0.14 large).")

    with tab_chi:
        st.markdown("Is review tier associated with publisher class? Move the cut-points to see how much the answer depends on the choice.")
        c1, c2 = st.columns([3, 2])
        lo, hi = c1.slider("Cut-points: Low | Mid | High", 50, 95, (70, 90), key="chi_cuts")
        include = c2.checkbox("Count unscored games (0) as the lowest tier", value=False, key="chi_zero",
                              help="This was the Week 2 approach. Score 0 means 'no score recorded', not a real 0% rating.")
        if lo == hi:
            st.warning("Choose two different cut-points.")
        else:
            res = chi_square_tiers(df, float(lo), float(hi), include)
            if res is None:
                st.warning("These cut-points leave fewer than two non-empty tiers.")
            else:
                m = st.columns(5)
                m[0].metric("χ²", f"{res['chi2']:.2f}")
                m[1].metric("df", res["dof"])
                m[2].metric("p (asymptotic)", fmt_p(res["p"]))
                m[3].metric("p (permutation)", fmt_p(res["p_perm"]))
                m[4].metric("Cramér's V", f"{res['v']:.3f}")
                if res["min_exp"] < 5:
                    st.warning(f"Smallest expected count is {res['min_exp']:.1f} (< 5), so trust the permutation p-value over the asymptotic one.")
                verdict = "significant" if res["p_perm"] < 0.05 else "not significant"
                st.markdown(f"**Result:** n = {res['n']:,}; the association is **{verdict}** at 5% (permutation p = {fmt_p(res['p_perm'])}), "
                            f"and Cramér's V = {res['v']:.3f} means the association is tiny even where it is detectable (0 = none, 1 = perfect).")
                tab = res["tab"]
                pct = tab.div(tab.sum(axis=1), axis=0) * 100
                t1, t2 = st.columns(2)
                t1.markdown("Counts")
                t1.dataframe(tab.reset_index().rename(columns={"publisherClass": "Publisher class"}), hide_index=True)
                t2.markdown("Row % (share of each class in each tier)")
                t2.dataframe(pct.round(1).reset_index().rename(columns={"publisherClass": "Publisher class"}), hide_index=True)
                long = pct.reset_index().melt(id_vars="publisherClass", var_name="Review tier", value_name="pct")
                fig = px.bar(long, x="pct", y="publisherClass", color="Review tier", orientation="h", barmode="stack",
                             category_orders={"publisherClass": ORDER[::-1], "Review tier": ["Low", "Mid", "High"]},
                             labels={"pct": "% of games in class", "publisherClass": ""},
                             color_discrete_map={"Low": "#E45756", "Mid": "#F2CF5B", "High": "#54A24B"})
                st.plotly_chart(style(fig, height=280), use_container_width=True)

        st.subheader("Sensitivity to the cut-points (from the analysis notebooks)")
        cs = report("robust_chi_square_sensitivity.csv")
        st.dataframe(pd.DataFrame({
            "Scheme": cs["scheme"], "n": cs["n"], "χ² (df = 4)": cs["chi2"].round(2), "p (asymptotic)": cs["p_asymptotic"].map(fmt_p),
            "p (permutation)": cs["p_permutation"].map(fmt_p), "Cramér's V": cs["cramers_v"].round(3)}), hide_index=True)
        tab2 = pd.crosstab(df["publisherClass"], df["has_review"])
        chi2_2, p2, dof2, _ = stats.chi2_contingency(tab2, correction=False)
        st.caption(tex(f"Separate question – is having a score at all related to class? χ² = {chi2_2:.2f}, df = {dof2}, p = {fmt_p(p2)}, Cramér's V = {rs.cramers_v(tab2):.3f}."))

    with tab_holm:
        st.markdown("Running several tests raises the chance of a false positive. The Holm step-down correction controls the family-wise error rate across all "
                    f"{len(fam)} tests below at 5%.")
        st.dataframe(pd.DataFrame({
            "Test": fam["test"], "p (raw)": fam["p_raw"].map(fmt_p), "p (Holm)": fam["p_holm"].map(fmt_p),
            "Significant (Holm, 5%)": np.where(fam["significant_holm_.05"], "Yes", "No")}), hide_index=True)
        score = -np.log10(fam["p_holm"].clip(lower=1e-300))
        fig = go.Figure(go.Bar(x=score, y=fam["test"], orientation="h", marker_color=[RED if s_ else GREY for s_ in fam["significant_holm_.05"]]))
        fig.add_vline(x=-np.log10(0.05), line_dash="dash", annotation_text="p = 0.05", annotation_position="top")
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Evidence against 'no difference' (−log10 of Holm p; longer = stronger)", xaxis_title="−log10 (Holm p)")
        st.plotly_chart(style(fig, height=380), use_container_width=True)

    with tab_cmp:
        st.markdown("The Week 2 analysis used tests that compare means. Here are those tests next to the rank-based ones used in the final analysis, computed live on the same data.")
        rv = {c_: df.loc[df["publisherClass"] == c_, "revenue"] for c_ in ORDER}
        a, b = rv["AAA"], rv["Indie"]
        welch, welch_log, mw = stats.ttest_ind(a, b, equal_var=False), stats.ttest_ind(np.log10(a), np.log10(b), equal_var=False), stats.mannwhitneyu(a, b)
        anova, kw3 = stats.f_oneway(*rv.values()), stats.kruskal(*rv.values())
        rows = [
            ("AAA vs Indie", "Welch t-test on mean revenue", "Week 2", f"t = {welch.statistic:.2f}", welch.pvalue),
            ("AAA vs Indie", "Mann–Whitney U (ranks)", "Final", f"U = {mw.statistic:,.0f}", mw.pvalue),
            ("AAA vs Indie", "Welch t-test on log10 revenue", "Cross-check", f"t = {welch_log.statistic:.2f}", welch_log.pvalue),
            ("All three classes", "One-way ANOVA on mean revenue", "Week 2", f"F = {anova.statistic:.1f}", anova.pvalue),
            ("All three classes", "Kruskal–Wallis (ranks)", "Final", f"H = {kw3.statistic:.1f}", kw3.pvalue),
        ]
        st.dataframe(pd.DataFrame({
            "Comparison": [r[0] for r in rows], "Test": [r[1] for r in rows], "Used in": [r[2] for r in rows], "Statistic": [r[3] for r in rows],
            "p": [fmt_p(r[4]) for r in rows], "Verdict (5%)": ["Difference" if r[4] < 0.05 else "No difference detected" for r in rows]}), hide_index=True)
        st.markdown(tex(
            f"**The AAA vs Indie result flips.** Comparing means gives p = {fmt_p(welch.pvalue)}, which looks like no difference, because the AAA mean "
            f"({money(a.mean())}) is pulled around by a handful of blockbusters (only {len(a)} AAA games). Every method that is not driven by the extreme tail "
            f"finds a large difference (Mann–Whitney p = {fmt_p(mw.pvalue)}; Welch on log revenue p = {fmt_p(welch_log.pvalue)})."))
        g = df.groupby("publisherClass", observed=True)["revenue"]
        st.dataframe(pd.DataFrame({"Mean": g.mean().map(money), "Median": g.median().map(money), "Mean ÷ median": (g.mean() / g.median()).round(1)}).reindex(ORDER).reset_index()
                     .rename(columns={"publisherClass": "Publisher class"}), hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# App shell
# ─────────────────────────────────────────────────────────────────────────────
require_inputs()
data = get_data()

PAGES = {
    "Overview": page_overview,
    "Q1 · Typical game earnings": page_q1,
    "Q2 · Share taken by the biggest games": page_q2,
    "Q3 · Publisher type": page_q3,
    "Q4 · Price": page_q4,
    "Q5 · Review score": page_q5,
    "Explore the data": page_explore,
    "Distribution fitting": page_distributions,
    "Hypothesis tests": page_tests,
    "Data & methods": page_methods,
}

st.sidebar.title("🎮 Steam 2024 Revenue")
choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.markdown("---")
st.sidebar.caption(tex(
    f"{len(data):,} games · released {data['releaseDate'].min():%d %b} – {data['releaseDate'].max():%d %b %Y}\n\n"
    f"Top-1,500 sample (minimum revenue {money(data['revenue'].min())}), so results describe already-successful games."))

PAGES[choice](data)
