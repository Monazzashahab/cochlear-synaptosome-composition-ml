"""
Vectorized empirical Bayes moderated t-statistics (Smyth 2004, limma eBayes)
for a group-means design, with NaN-aware per-protein residual degrees of freedom.

Validated against inmoose.limma (Python port of limma) in validate_limma.py.
"""
import numpy as np
from scipy import special, stats


def trigamma_inverse(x):
    """Solve trigamma(y) = x for y (limma's Newton iteration)."""
    x = np.atleast_1d(np.asarray(x, float))
    y = np.empty_like(x)
    big, small = x > 1e7, x < 1e-6
    mid = ~(big | small)
    y[big] = 1 / np.sqrt(x[big])
    y[small] = 1 / x[small]
    ym = 0.5 + 1 / x[mid]
    for _ in range(50):
        tri = special.polygamma(1, ym)
        dif = tri * (1 - tri / x[mid]) / special.polygamma(2, ym)
        ym = ym + dif
        if np.max(-dif / ym) < 1e-8:
            break
    y[mid] = ym
    return y


def fit_f_dist(s2, df):
    """Moment estimation of the prior (d0, s0^2) for sample variances (limma fitFDist)."""
    ok = np.isfinite(s2) & (s2 > 0) & np.isfinite(df) & (df > 0)
    s2, df = s2[ok], df[ok]
    z = np.log(s2)
    e = z - special.digamma(df / 2) + np.log(df / 2)
    emean = e.mean()
    evar = np.sum((e - emean) ** 2) / (len(e) - 1) - np.mean(special.polygamma(1, df / 2))
    if evar > 0:
        d0 = 2 * trigamma_inverse(evar)[0]
        s02 = np.exp(emean + special.digamma(d0 / 2) - np.log(d0 / 2))
    else:
        d0 = np.inf
        s02 = np.exp(emean)
    return d0, s02


def group_fit(L, groups, levels):
    """Per-group NaN-aware means and counts, pooled residual variance and df."""
    m, n = {}, {}
    ss = np.zeros(L.shape[0]); N = np.zeros(L.shape[0]); G = np.zeros(L.shape[0])
    for g in levels:
        X = L[:, groups == g]
        cnt = np.sum(~np.isnan(X), 1).astype(float)
        mu = np.where(cnt > 0, np.nanmean(np.where(np.isnan(X), np.nan, X), 1), np.nan)
        m[g], n[g] = mu, cnt
        ss += np.nansum((X - mu[:, None]) ** 2, 1)
        N += cnt; G += (cnt > 0)
    df = N - G
    with np.errstate(invalid='ignore', divide='ignore'):
        s2 = np.where(df > 0, ss / df, np.nan)
    return m, n, s2, df


def moderated_t(L, groups, contrasts, levels=None):
    """L: proteins x samples (log2, NaN allowed). groups: array of labels per sample.
    contrasts: list of (g1, g0) -> statistic for mean(g1) - mean(g0).
    Returns dict contrast -> dict(logFC, t, p, df_total) and prior (d0, s02)."""
    groups = np.asarray(groups)
    levels = levels or list(dict.fromkeys(groups))
    m, n, s2, df = group_fit(L, groups, levels)
    d0, s02 = fit_f_dist(s2, df)
    if np.isinf(d0):
        s2post = np.full_like(s2, s02); dft = np.full_like(df, np.inf)
    else:
        s2post = (d0 * s02 + df * np.nan_to_num(s2)) / (d0 + df)
        dft = df + d0
        dfp = np.nansum(df)               # limma caps df.total at df.pooled
        dft = np.minimum(dft, dfp)
    out = {}
    for g1, g0 in contrasts:
        with np.errstate(invalid='ignore', divide='ignore'):
            lfc = m[g1] - m[g0]
            se = np.sqrt(s2post * (1 / n[g1] + 1 / n[g0]))
            t = lfc / se
        p = 2 * stats.t.sf(np.abs(t), dft)
        bad = (n[g1] < 1) | (n[g0] < 1) | ~np.isfinite(t)
        t[bad] = np.nan; p[bad] = np.nan
        out[(g1, g0)] = {'logFC': lfc, 't': t, 'p': p, 'df_total': dft}
    return out, {'d0': float(d0), 's02': float(s02)}


def bh(p):
    """Benjamini-Hochberg adjusted p values, NaN preserved."""
    p = np.asarray(p, float); q = np.full_like(p, np.nan)
    ok = ~np.isnan(p); pv = p[ok]; nn = len(pv)
    o = np.argsort(pv); ranked = pv[o] * nn / np.arange(1, nn + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(nn); qq[o] = np.minimum(ranked, 1)
    q[ok] = qq
    return q


def moderated_t_design(Y, X, contrasts):
    """Moderated t for a general design on complete data.
    Y: proteins x samples (no NaN). X: samples x p design matrix. contrasts: dict name -> vector (p,).
    Returns dict name -> dict(logFC, t, p) and prior."""
    Y = np.asarray(Y, float); X = np.asarray(X, float)
    n, pdim = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    B = Y @ X @ XtX_inv                       # proteins x p
    resid = Y - B @ X.T
    df = np.full(Y.shape[0], n - pdim, float)
    s2 = np.sum(resid ** 2, 1) / df
    d0, s02 = fit_f_dist(s2, df)
    if np.isinf(d0):
        s2post = np.full_like(s2, s02); dft = np.full_like(df, np.inf)
    else:
        s2post = (d0 * s02 + df * s2) / (d0 + df); dft = np.minimum(df + d0, df.sum())
    out = {}
    for name, c in contrasts.items():
        c = np.asarray(c, float)
        est = B @ c
        se = np.sqrt(s2post * (c @ XtX_inv @ c))
        t = est / se
        out[name] = {'logFC': est, 't': t, 'p': 2 * stats.t.sf(np.abs(t), dft)}
    return out, {'d0': float(d0), 's02': float(s02)}
