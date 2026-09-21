"""
src/visualizations.py
Reusable plotting helpers shared by the analysis notebooks and the Streamlit
dashboard (dashboard/app.py).

Static (matplotlib/seaborn) figures are used in the notebooks and saved to
reports/figures/; interactive (Plotly) figures are used in the dashboard for
tooltips and zoom. This module has no Streamlit import so it works in both
places without pulling in a Streamlit runtime inside a plain notebook.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

try:  # package-relative when imported as `from src import visualizations`
    from src.statistics import CLASS_ORDER, CLASS_COLORS
except ImportError:  # pragma: no cover - fallback so the module still loads standalone
    CLASS_ORDER = ["Indie", "AA", "AAA"]
    CLASS_COLORS = {"Indie": "#4C78A8", "AA": "#F58518", "AAA": "#54A24B"}

sns.set_theme(style="whitegrid")

BLUE, GREY, RED = "#4C78A8", "#9D9D9D", "#E45756"

# scipy.stats distributions available to plot_histogram_with_distribution / fitting
DIST_MAP = {
    "normal": stats.norm,
    "lognormal": stats.lognorm,
    "weibull": stats.weibull_min,
    "gamma": stats.gamma,
    "loglogistic": stats.fisk,
}


# ---------------------------------------------------------------------------
# Static (matplotlib / seaborn) figures — used in notebooks & saved reports
# ---------------------------------------------------------------------------
def plot_histogram_with_distribution(
    data: Sequence[float],
    distribution: str = "normal",
    bins: int = 40,
    log_scale: bool = False,
    label: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Histogram with a fitted distribution overlay.

    Parameters
    ----------
    data : array-like of floats.
    distribution : one of DIST_MAP ("normal", "lognormal", "weibull", "gamma",
        "loglogistic").
    log_scale : fit and plot log10(data) instead of the raw values —
        appropriate for the heavily right-skewed variables in this dataset
        (revenue, copiesSold, price, avgPlaytime); the fitted curve is still
        drawn correctly transformed onto the log10 axis.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if log_scale or distribution != "normal":
        x = x[x > 0]
    values = np.log10(x) if log_scale else x

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=bins, density=True, color=BLUE, alpha=0.75, edgecolor="white")

    dist = DIST_MAP.get(distribution, stats.norm)
    grid = np.linspace(values.min(), values.max(), 400)
    try:
        if distribution == "normal":
            params = stats.norm.fit(values)
            pdf = stats.norm.pdf(grid, *params)
        elif log_scale:
            # fit on the raw (untransformed) values, then map the density onto log10(x)
            params = dist.fit(x, floc=0)
            xv = 10 ** grid
            pdf = dist.pdf(xv, *params) * xv * np.log(10)
        else:
            params = dist.fit(x, floc=0)
            pdf = dist.pdf(grid, *params)
        ax.plot(grid, pdf, color=RED, lw=2, label=f"fitted {distribution}")
        ax.legend()
    except Exception as exc:  # pragma: no cover - a family can fail to converge
        ax.set_title(f"(fit failed: {exc})")

    xlabel = f"log10({label})" if log_scale and label else (label or "value")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {label or 'variable'}")
    fig.tight_layout()
    return fig


def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    method: str = "spearman",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Correlation matrix heatmap.

    Defaults to Spearman (rank) correlation — the measure used throughout
    this analysis, since the underlying variables are heavily right-skewed
    (see src/statistics.py). Pass method="pearson" for the raw-value version.
    """
    cols = list(columns) if columns is not None else df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[cols].corr(method=method)

    size = max(4.5, 0.85 * len(cols))
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(size, size))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True,
        cbar_kws={"label": "ρ" if method == "spearman" else "r"}, ax=ax,
    )
    ax.set_title(f"{method.title()} correlation")
    fig.tight_layout()
    return fig


def plot_boxplots_by_category(
    df: pd.DataFrame,
    numeric_col: str,
    category_col: str,
    log_scale: bool = False,
    order: Sequence[str] | None = None,
    palette: dict | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Boxplots of `numeric_col` grouped by `category_col`."""
    if order is None:
        order = CLASS_ORDER if category_col == "publisherClass" else sorted(df[category_col].dropna().unique())
    if palette is None:
        palette = CLASS_COLORS if category_col == "publisherClass" else None

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.5))
    sns.boxplot(
        data=df, x=category_col, y=numeric_col, order=order, palette=palette,
        hue=category_col, legend=False, ax=ax,
    )
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(category_col)
    ax.set_ylabel(numeric_col)
    ax.set_title(f"{numeric_col} by {category_col}")
    fig.tight_layout()
    return fig


def plot_qq_comparison(
    data: Sequence[float],
    distribution: str = "norm",
    dist_params: tuple | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Q-Q plot of `data` against `distribution` (any scipy.stats.<name>, e.g.
    "norm", "lognorm"). Pass `dist_params` (e.g. from `dist.fit(data)`) to
    compare against a fitted model instead of the family's default parameters.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    dist = getattr(stats, distribution)

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5.5, 5.5))
    if dist_params is not None:
        osm, osr = stats.probplot(x, dist=dist, sparams=dist_params, fit=False)
        ax.scatter(osm, osr, s=14, color=BLUE, alpha=0.7)
        lo, hi = min(osm.min(), osr.min()), max(osm.max(), osr.max())
        ax.plot([lo, hi], [lo, hi], color=RED, lw=1.5, ls="--")
    else:
        stats.probplot(x, dist=dist, plot=ax)
        ax.get_lines()[0].set(markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=4, alpha=0.7)
        ax.get_lines()[1].set(color=RED, lw=1.5)
    ax.set_title(f"Q-Q plot vs {distribution}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interactive (Plotly) figure — used by the dashboard
# ---------------------------------------------------------------------------
def create_interactive_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    hover_name: str | None = None,
    hover_data: Sequence[str] | None = None,
    opacity: float = 0.7,
    title: str | None = None,
) -> go.Figure:
    """Interactive Plotly scatter plot (used on the dashboard's price/review pages)."""
    is_class_color = color_col == "publisherClass"
    if hover_name is None and hover_data is None and "name" in df.columns:
        hover_name = "name"
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col,
        color_discrete_map=CLASS_COLORS if is_class_color else None,
        category_orders={"publisherClass": CLASS_ORDER} if is_class_color else None,
        log_x=log_x, log_y=log_y,
        hover_name=hover_name,
        hover_data=hover_data,
        opacity=opacity,
        title=title or f"{y_col} vs {x_col}",
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
    return fig


# ---------------------------------------------------------------------------
# Shared dashboard configuration
# ---------------------------------------------------------------------------
def dashboard_layout() -> dict:
    """Shared Streamlit page-config / theming constants, so dashboard/app.py
    (and any future pages) configure Streamlit and Plotly identically."""
    return {
        "page_config": dict(page_title="Steam 2024 Revenue", page_icon="🎮", layout="wide"),
        "class_order": CLASS_ORDER,
        "class_colors": CLASS_COLORS,
        "plotly_template": "plotly_white",
        "accent_colors": {"blue": BLUE, "grey": GREY, "red": RED},
    }
