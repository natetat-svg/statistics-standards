"""
src/statistics.py
Week 3 -- Robust / non-parametric toolkit for heavy-tailed commercial data.

Why this module exists
----------------------
Week 2 applied mean-based, parametric procedures (t-test, ANOVA, Pearson, KS with
estimated parameters) to variables with skewness up to ~23 and kurtosis up to ~610.
The helpers here provide rank-based, log-scale, and resampling alternatives plus
the effect-size and concentration measures those procedures need.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

CLASS_ORDER = ["Indie", "AA", "AAA"]
CLASS_COLORS = {"Indie": "#4C78A8", "AA": "#F58518", "AAA": "#54A24B"}
RNG_SEED = 20260920


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_clean() -> pd.DataFrame:
    """Load the Week 1 cleaned dataset (Hobbyist already collapsed into Indie)."""
    df = pd.read_csv(PROCESSED_DIR / "cleaned_data.csv", parse_dates=["releaseDate"])
    df["publisherClass"] = pd.Categorical(df["publisherClass"], CLASS_ORDER, ordered=True)
    return df


def add_robust_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recode disguised missing values and add log-scale features.

    * reviewScore == 0  -> NaN in ``reviewScore_clean`` (no score recorded; a genuine
      0 % positive rating is implausible for 99 games with median revenue > $250K).
    * price == 0        -> ``is_free`` flag; ``log_price`` is defined for paid games only.
    * avgPlaytime == 0  -> NaN in the log feature.
    """
    out = df.copy()
    out["has_review"] = out["reviewScore"] > 0
    out["reviewScore_clean"] = out["reviewScore"].where(out["reviewScore"] > 0)
    out["is_free"] = out["price"] == 0
    out["log_revenue"] = np.log10(out["revenue"])
    out["log_copies"] = np.log10(out["copiesSold"].astype(float))
    out["log_price"] = np.log10(out["price"].where(out["price"] > 0))
    out["log_playtime"] = np.log10(out["avgPlaytime"].where(out["avgPlaytime"] > 0))
    out["release_month"] = out["releaseDate"].dt.month
    return out


def load_robust() -> pd.DataFrame:
    return add_robust_features(load_clean())


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def lorenz_curve(x) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x, dtype=float))
    cum = np.cumsum(x) / x.sum()
    p = np.arange(1, len(x) + 1) / len(x)
    return np.insert(p, 0, 0.0), np.insert(cum, 0, 0.0)


def gini(x) -> float:
    p, l = lorenz_curve(x)
    return float(1 - 2 * _trapz(l, p))


def top_share(x, frac: float | None = None, k: int | None = None) -> float:
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    n = k if k is not None else int(np.ceil(frac * len(x)))
    return float(x[:n].sum() / x.sum())


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------
def cliffs_delta(a, b) -> float:
    """Cliff's delta = P(a > b) - P(a < b), computed from the Mann-Whitney U."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def cliffs_label(d: float) -> str:
    d = abs(d)
    return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def epsilon_squared(H: float, n: int) -> float:
    """Epsilon-squared effect size for the Kruskal-Wallis H statistic."""
    return float(H / (n - 1)) if n > 1 else np.nan


def cramers_v(table: pd.DataFrame, bias_correct: bool = True) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    phi2 = chi2 / n
    if not bias_correct:
        return float(np.sqrt(phi2 / min(r - 1, k - 1)))
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2c / min(kc - 1, rc - 1)))


def permutation_chi2(x, y, n_perm: int = 5000, seed: int = RNG_SEED) -> tuple[float, float]:
    """Permutation p-value for the chi-square independence statistic (valid when
    expected counts are small, where the asymptotic chi-square approximation is not)."""
    rng = np.random.default_rng(seed)
    xi = pd.factorize(np.asarray(x))[0]
    yi = pd.factorize(np.asarray(y))[0]
    r, k = xi.max() + 1, yi.max() + 1

    def chi2(yv):
        obs = np.bincount(xi * k + yv, minlength=r * k).reshape(r, k).astype(float)
        exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
        return float(((obs - exp) ** 2 / exp).sum())

    obs_stat = chi2(yi)
    ge = sum(chi2(rng.permutation(yi)) >= obs_stat - 1e-12 for _ in range(n_perm))
    return obs_stat, (ge + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def bca_ci(x, statistic=np.median, n_resamples: int = 9999, conf: float = 0.95, seed: int = RNG_SEED):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    res = stats.bootstrap(
        (x,), statistic, n_resamples=n_resamples, confidence_level=conf,
        method="BCa", random_state=np.random.default_rng(seed),
    )
    return float(statistic(x)), float(res.confidence_interval.low), float(res.confidence_interval.high)


def geometric_mean(x, axis=-1):
    x = np.asarray(x, float)
    return np.exp(np.mean(np.log(x), axis=axis))


def t_ci_on_log(x, conf: float = 0.95):
    """t-interval on log scale, back-transformed => CI for the geometric mean."""
    lx = np.log(np.asarray(x, float))
    m, se = lx.mean(), stats.sem(lx)
    lo, hi = stats.t.interval(conf, len(lx) - 1, loc=m, scale=se)
    return float(np.exp(m)), float(np.exp(lo)), float(np.exp(hi))


# ---------------------------------------------------------------------------
# Distribution fitting (likelihood-based, optionally left-truncated)
# ---------------------------------------------------------------------------
POSITIVE_CANDIDATES = {
    "Lognormal": stats.lognorm,
    "Weibull": stats.weibull_min,
    "Log-logistic": stats.fisk,
    "Inverse Gaussian": stats.invgauss,
    "Gamma": stats.gamma,
    "Lomax (Pareto II)": stats.lomax,
}


def _fit_one(dist, x: np.ndarray, truncate_at: float | None):
    """MLE with location fixed at 0. If ``truncate_at`` is given, the likelihood is
    f(x) / S(truncate_at), i.e. left-truncated -- appropriate when a sample only
    contains units above a cut-off (here: the top 1,500 games by revenue)."""
    p0 = dist.fit(x, floc=0)
    if truncate_at is None:
        return p0, float(np.sum(dist.logpdf(x, *p0))), len(p0) - 1
    q0 = np.log(np.array([p0[i] for i in range(len(p0) - 2)] + [p0[-1]], dtype=float))

    def unpack(t):
        e = np.exp(t)
        return tuple(e[:-1]) + (0.0, e[-1])

    def nll(t):
        p = unpack(t)
        with np.errstate(all="ignore"):
            ll = dist.logpdf(x, *p).sum() - len(x) * dist.logsf(truncate_at * (1 - 1e-9), *p)
        return -ll if np.isfinite(ll) else 1e12

    res = optimize.minimize(nll, q0, method="Nelder-Mead",
                            options=dict(maxiter=6000, xatol=1e-7, fatol=1e-7))
    return unpack(res.x), float(-res.fun), len(q0)


def fit_candidates(x, candidates: dict | None = None, truncate_at: float | None = None) -> pd.DataFrame:
    """Fit candidate families by MLE and rank by AIC/BIC.

    KS ``D`` is a descriptive distance only -- its textbook p-value is invalid when
    parameters are estimated from the same sample (use ``parametric_bootstrap_ks``).
    """
    candidates = candidates or POSITIVE_CANDIDATES
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    rows = []
    for name, dist in candidates.items():
        try:
            params, ll, k = _fit_one(dist, x, truncate_at)
            if truncate_at is None:
                cdf = lambda v, d=dist, p=params: d.cdf(v, *p)
            else:
                f0 = dist.cdf(truncate_at * (1 - 1e-9), *params)
                cdf = lambda v, d=dist, p=params, f0=f0: (d.cdf(v, *p) - f0) / (1 - f0)
            rows.append({
                "distribution": name, "n_params": k, "loglik": ll,
                "AIC": 2 * k - 2 * ll, "BIC": k * np.log(len(x)) - 2 * ll,
                "KS_D": stats.kstest(x, cdf).statistic, "params": params,
                "truncated": truncate_at is not None,
            })
        except Exception:  # pragma: no cover - a family may fail to converge
            continue
    out = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    out["dAIC"] = out["AIC"] - out["AIC"].min()
    out["dBIC"] = out["BIC"] - out["BIC"].min()
    return out


def parametric_bootstrap_ks(x, dist, truncate_at: float | None = None,
                            n_sim: int = 200, seed: int = RNG_SEED):
    """Lilliefors-style p-value: simulate from the fitted model, RE-FIT on every
    simulated sample, and compare KS distances. Accounts for parameter estimation."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[x > 0]
    params, _, _ = _fit_one(dist, x, truncate_at)

    def ks_d(sample, p):
        if truncate_at is None:
            return stats.kstest(sample, dist.cdf, args=p).statistic
        f0 = dist.cdf(truncate_at * (1 - 1e-9), *p)
        return stats.kstest(sample, lambda v: (dist.cdf(v, *p) - f0) / (1 - f0)).statistic

    d_obs = ks_d(x, params)
    f0 = 0.0 if truncate_at is None else dist.cdf(truncate_at * (1 - 1e-9), *params)
    ge = 0
    for _ in range(n_sim):
        u = rng.uniform(f0, 1.0, size=len(x))
        xs = dist.ppf(u, *params)
        p_s, _, _ = _fit_one(dist, xs, truncate_at)
        if ks_d(xs, p_s) >= d_obs:
            ge += 1
    return float(d_obs), (ge + 1) / (n_sim + 1)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def save_fig(fig, name: str, dpi: int = 200) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / name
    df.to_csv(path, index=index)
    return path
