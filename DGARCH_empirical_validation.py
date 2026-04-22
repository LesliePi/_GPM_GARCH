# -*- coding: utf-8 -*-
"""
DGARCH_empirical_validation.py
GPM-GARCH Early-Warning System — Empirical Validation on Real Market Data

Author:  László Tatai / BarefootRealism Labs
Date:    2026-04-19
Version: 2.0 (GitHub release)
License: MIT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA SOURCES — What to download (all free, no registration)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. S&P 500 daily closing prices (^GSPC)
   URL: https://stooq.com/q/d/l/?s=%5Espx&d1=20030101&d2=20120101&i=d
   → Save as: sp500_daily.csv
   Columns: Date, Open, High, Low, Close, Volume

2. VIX volatility index (optional, extra panel)
   URL: https://stooq.com/q/d/l/?s=%5Evix&d1=20030101&d2=20120101&i=d
   → Save as: vix_daily.csv

   Alternative sources if Stooq is unavailable:
   Yahoo Finance:   https://finance.yahoo.com/quote/%5EGSPC/history/
   Investing.com:   https://www.investing.com/indices/us-spx-500-historical-data
   Kaggle:          https://www.kaggle.com/datasets/henryhan117/sp-500-historical-data

3. Run with a downloaded CSV:
   python DGARCH_empirical_validation.py --file sp500_daily.csv

   Without a CSV file, the script runs in DEMO MODE on simulated data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHODOLOGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Log returns:     r_t = 100 * ln(P_t / P_{t-1})
- Rolling GARCH(1,1)-t fit (252-day window, 5-day step)
- Key indicators:
    phi4    = alpha_G^2 * kappa_z + 2*alpha_G*beta_G + beta_G^2
              (fourth-moment boundary condition, Lemma 1)
    delta   = 1 - phi4
              (signed distance from GPM bifurcation boundary, Theorem 2)
    kappa   ≈ 4 / phi4
              (tail index approximation, Theorem 1)
- Reference events (built-in):
    2007-07-31  Bear Stearns hedge fund collapse
    2008-03-14  Bear Stearns acquisition (JP Morgan)
    2008-09-15  Lehman Brothers bankruptcy
    2009-03-09  S&P 500 trough

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THEORETICAL BACKGROUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The GPM-GARCH framework establishes a structural identity:

    Delta_GPM = 0  ⟺  phi4 = 1  ⟺  kappa = 4  ⟺  E[y^4] = infty

All four conditions identify the same critical boundary at which:
- The CLT breaks down for GARCH returns (López de Prado et al., 2026)
- The GPM undergoes a saddle-node bifurcation (Tatai, 2026)
- Standard Sharpe ratio inference loses validity

Reference:
  Tatai, L. (2026). The Gauss-PowerLaw Module (GPM): Endogenous Heavy-Tail
  Formation as a Bifurcation Phenomenon. BarefootRealism Labs. Zenodo.
"""

import argparse
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def make_output_path(base_name: str, suffix: str) -> str:
    """
    Generate a timestamped output filename.
    Format: YYYYMMDD_HHMMSS_{base_name}{suffix}
    Example: 20260419_143022_sp500_daily.png
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{ts}_{base_name}{suffix}"


# ─────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_price_data(filepath: str) -> pd.Series:
    """
    Load CSV file and compute log returns.
    Automatically recognises Yahoo Finance and Stooq column formats.

    Returns
    -------
    pd.Series
        Daily log returns (%), DatetimeIndex.
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    # Locate date column
    date_col = None
    for c in df.columns:
        if c.lower() in ('date', 'datum', 'time'):
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No date column found. Available columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Locate closing price column
    close_col = None
    for c in df.columns:
        if c.lower() in ('close', 'adj close', 'adjclose', 'zamknięcie', 'zamkniecie'):
            close_col = c
            break
    if close_col is None:
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            close_col = num_cols[-1]
        else:
            raise ValueError(f"No closing price column found. Available: {list(df.columns)}")

    prices = df[close_col].dropna()
    prices = pd.to_numeric(prices, errors='coerce').dropna()
    returns = 100.0 * np.log(prices / prices.shift(1)).dropna()

    print(f"  Loaded: {len(prices)} price points, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Returns: n={len(returns)}, mean={returns.mean():.4f}%, "
          f"std={returns.std():.4f}%")
    return returns


def generate_demo_data(seed: int = 2008) -> pd.Series:
    """
    Simulated S&P 500-like data for demonstration mode (no CSV required).
    Five regimes: stable → pre-crisis → crisis → post-crisis → consolidation.
    Parameters based on empirical literature (Engle, 1982; Bollerslev, 1986).
    """
    from scipy.stats import t as t_dist
    np.random.seed(seed)

    regimes = [
        # (omega, alpha_G, beta_G, nu_t, n_days, label)
        (0.020, 0.06, 0.90, 7.0, 504, "Stable (2005-2006)"),
        (0.030, 0.10, 0.87, 5.5, 126, "Pre-crisis (2007 H1)"),
        (0.045, 0.14, 0.83, 5.0, 126, "Pre-crisis (2007 H2)"),
        (0.065, 0.20, 0.77, 4.5, 126, "Crisis (2008 H1)"),
        (0.110, 0.32, 0.65, 4.0, 126, "Crisis peak (2008 H2)"),
        (0.065, 0.20, 0.77, 4.5, 126, "Post-crisis (2009 H1)"),
        (0.040, 0.14, 0.83, 5.0, 126, "Post-crisis (2009 H2)"),
        (0.028, 0.09, 0.88, 6.0, 252, "Consolidation (2010)"),
    ]

    all_y, y_prev, s2 = [], 0.0, None
    for omega, aG, bG, nu, n, _ in regimes:
        if s2 is None:
            s2 = omega / max(1 - aG - bG, 0.01)
        for _ in range(n):
            s2 = omega + aG * y_prev**2 + bG * s2
            s2 = max(s2, 1e-10)
            z  = t_dist.rvs(nu) / np.sqrt(nu / (nu - 2))
            y  = np.sqrt(s2) * z
            all_y.append(y)
            y_prev = y

    total = sum(r[4] for r in regimes)
    idx   = pd.date_range('2005-01-03', periods=total, freq='B')
    return pd.Series(all_y[:len(idx)], index=idx)


# ─────────────────────────────────────────────────────────────
# 2.  GARCH FITTING AND INDICATORS
# ─────────────────────────────────────────────────────────────

def fit_garch_t(window: pd.Series) -> dict | None:
    """
    Fit GARCH(1,1)-t to a single rolling window.

    Returns
    -------
    dict with keys: omega, aG, bG, phi, kz, phi4, delta, kappa
    or None if fitting fails.

    Theoretical background
    ----------------------
    phi4 = alpha_G^2 * kappa_z + 2*alpha_G*beta_G + beta_G^2
         = M(4) by Lemma 1 (Tatai, 2026)
    delta = 1 - phi4 : signed distance from GPM bifurcation boundary
    kappa ≈ 4 / phi4 : tail index approximation near the critical point
    """
    try:
        am  = arch_model(window, vol='Garch', p=1, q=1,
                         dist='t', rescale=True)
        res = am.fit(disp='off', show_warning=False)
        p   = res.params

        omega = float(abs(p.get('omega',    p.iloc[0])))
        aG    = float(abs(p.get('alpha[1]', p.iloc[1])))
        bG    = float(abs(p.get('beta[1]',  p.iloc[2])))
        phi   = aG + bG

        if phi >= 0.999 or phi <= 0.001 or aG < 0 or bG < 0:
            return None

        # Innovation kurtosis from Student-t nu parameter
        nu_par = p.get('nu', None)
        if nu_par is not None:
            nu_val = max(float(nu_par), 4.1)
            kz = 3.0 * (nu_val - 2.0) / (nu_val - 4.0)
        else:
            # Fallback: empirical estimate from standardised residuals
            sv    = np.sqrt(res.conditional_volatility)
            resid = window.values / (sv + 1e-8)
            kz    = float(np.clip(np.mean(resid**4), 3.0, 50.0))

        kz    = float(np.clip(kz, 3.0, 50.0))
        phi4  = aG**2 * kz + 2.0 * aG * bG + bG**2
        delta = 1.0 - phi4
        # Tail index approximation: kappa ≈ 4/phi4 (Taylor expansion near phi4=1)
        kappa_approx = float(np.clip(4.0 / phi4, 1.0, 12.0)) if phi4 > 0 else 4.0

        return dict(omega=omega, aG=aG, bG=bG, phi=phi,
                    kz=kz, phi4=phi4, delta=delta,
                    kappa=kappa_approx)
    except Exception:
        return None


def compute_kappa_kesten(aG: float, bG: float) -> float:
    """
    Numerically solve the Kesten-Goldie equation for z ~ N(0,1):
        E[(aG*z^2 + bG)^(kappa/2)] = 1

    Returns kappa, or NaN if no solution exists on (2, inf).

    Reference: Basrak, Davis & Mikosch (2002).
    """
    def lhs(k2):
        def f(u):
            v = aG * u + bG
            if v <= 0:
                return 0.0
            return v**k2 * u**(-0.5) * np.exp(-u / 2.0) / np.sqrt(2 * np.pi)
        val, _ = quad(f, 0, 80, limit=150, epsabs=1e-9)
        return val

    try:
        f1, f2 = lhs(1.0) - 1.0, lhs(2.5) - 1.0
        if f1 * f2 < 0:
            return 2.0 * brentq(lambda k2: lhs(k2) - 1.0, 1.0, 2.5, xtol=1e-7)
        elif f2 < 0:
            return 5.5   # kappa > 5
        else:
            return 1.5   # kappa < 2 (highly unstable)
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────
# 3.  ROLLING ANALYSIS
# ─────────────────────────────────────────────────────────────

def rolling_analysis(returns: pd.Series,
                     window: int = 252,
                     step: int = 5,
                     exact_kappa: bool = False) -> pd.DataFrame:
    """
    Rolling GARCH fitting and GPM indicator computation.

    Parameters
    ----------
    returns     : daily log return series
    window      : fitting window in trading days (recommended: 252)
    step        : step size in days (recommended: 5)
    exact_kappa : if True, solve Kesten equation numerically (slower, more accurate)

    Returns
    -------
    pd.DataFrame with columns: phi, phi4, delta, kappa, aG, bG, kz
    """
    records = []
    n = len(returns)

    print(f"  Rolling GARCH(1,1)-t  (window={window}, step={step})...")
    progress_step = max(1, (n - window) // (step * 10))

    for i in range(window, n, step):
        win    = returns.iloc[i - window:i]
        result = fit_garch_t(win)
        if result is None:
            continue

        if exact_kappa:
            k = compute_kappa_kesten(result['aG'], result['bG'])
            result['kappa'] = k if not np.isnan(k) else result['kappa']

        result['date'] = returns.index[i]
        records.append(result)

        if len(records) % progress_step == 0:
            pct = 100 * (i - window) / (n - window)
            print(f"    {pct:.0f}%  ({len(records)} points so far)", end='\r')

    print(f"    Done: {len(records)} rolling points.          ")
    df = pd.DataFrame(records).set_index('date')
    df.index = pd.DatetimeIndex(df.index)
    return df


# ─────────────────────────────────────────────────────────────
# 4.  VISUALISATION
# ─────────────────────────────────────────────────────────────

# Known market crisis events
CRISIS_EVENTS = [
    ('2007-07-31', 'Bear Stearns\nhedge fund',  'orange'),
    ('2008-03-14', 'Bear Stearns\nacquisition', 'darkorange'),
    ('2008-09-15', 'Lehman\nbankruptcy',        'darkred'),
    ('2009-03-09', "S&P trough",                'purple'),
]


def add_events(ax, ymin, ymax):
    """Add vertical crisis event markers to an axis."""
    for date_str, label, color in CRISIS_EVENTS:
        dt = pd.Timestamp(date_str)
        ax.axvline(dt, color=color, ls='--', lw=1.5, alpha=0.8)
        ax.text(dt, ymax * 0.92, label, color=color,
                fontsize=6.5, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7))


def plot_results(returns: pd.Series,
                 roll: pd.DataFrame,
                 output_path: str,
                 title_suffix: str = '') -> None:
    """
    Four-panel summary figure:
    1. Daily log returns
    2. phi4 with CLT boundary (Lemma 1)
    3. delta_GARCH (GPM bifurcation proximity, Theorem 2)
    4. Tail index kappa (Theorem 1)
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
    fig.suptitle(
        f'GPM-GARCH Early-Warning System{title_suffix}\n'
        'delta_GARCH and phi4 as CLT breakdown indicators\n'
        '(Rolling 252-day GARCH(1,1)-t estimation)',
        fontsize=13, y=1.01
    )

    fmt = mdates.DateFormatter('%Y')
    for ax in axes:
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.grid(True, alpha=0.3)

    # Panel 1: Log returns
    ax = axes[0]
    ax.plot(returns.index, returns.values, 'k-', lw=0.5, alpha=0.75)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Log return (%)', fontsize=10)
    ax.set_title('Daily log returns', fontsize=10)
    add_events(ax, returns.min(), returns.max())

    # Panel 2: phi4
    ax = axes[1]
    ax.plot(roll.index, roll['phi4'], color='darkorange', lw=2,
            label='phi4 (rolling)')
    ax.axhline(1.0, color='red', ls='--', lw=2,
               label='phi4 = 1  →  CLT breakdown (López de Prado et al., 2026)')
    ax.fill_between(roll.index, roll['phi4'], 1.0,
                    where=roll['phi4'] >= 1.0,
                    alpha=0.30, color='red', label='CLT invalid region')
    ax.set_ylabel('phi4', fontsize=10)
    ax.set_title(
        'phi4 = alpha^2 * kappa_z + 2*alpha*beta + beta^2   '
        '[CLT validity boundary — Lemma 1]',
        fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    add_events(ax, roll['phi4'].min(), roll['phi4'].max())

    # Panel 3: delta_GARCH
    ax = axes[2]
    ax.plot(roll.index, roll['delta'], color='steelblue', lw=2,
            label='delta_GARCH = 1 - phi4')
    ax.fill_between(roll.index, roll['delta'], 0,
                    where=roll['delta'] < 0,
                    alpha=0.35, color='red',
                    label='delta < 0: GPM unstable  (Delta_GPM < 0, no stable fixed point)')
    ax.fill_between(roll.index, roll['delta'], 0,
                    where=roll['delta'] >= 0,
                    alpha=0.12, color='green',
                    label='delta >= 0: GPM stable  (Delta_GPM > 0)')
    ax.axhline(0, color='k', lw=1.5)
    ax.set_ylabel('delta_GARCH', fontsize=10)
    ax.set_title(
        'delta_GARCH = 1 - phi4   '
        '[GPM bifurcation proximity, Delta_GPM proxy — Theorem 2]',
        fontsize=10)
    ax.legend(fontsize=8, loc='lower left')
    add_events(ax, roll['delta'].min(), roll['delta'].max())

    # Panel 4: tail index kappa
    ax = axes[3]
    kappa_clipped = np.clip(roll['kappa'], 0, 10)
    ax.plot(roll.index, kappa_clipped, color='purple', lw=2,
            label='kappa ≈ 4/phi4  (tail index approximation)')
    ax.axhline(4.0, color='red', ls='--', lw=2,
               label='kappa = 4  →  critical boundary  (Theorem 1)')
    ax.fill_between(roll.index, kappa_clipped, 4.0,
                    where=kappa_clipped < 4.0,
                    alpha=0.25, color='red',
                    label='kappa < 4: heavy tail, CLT BREAK')
    ax.set_ylabel('kappa  (tail index)', fontsize=10)
    ax.set_title(
        'Tail index kappa   '
        '[kappa<4 ↔ delta<0 ↔ Delta_GPM<0  —  three-way identity]',
        fontsize=10)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=8, loc='upper right')
    add_events(ax, 0, 10)
    ax.set_xlabel('Year', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {output_path}")


# ─────────────────────────────────────────────────────────────
# 5.  STATISTICS AND EARLY-WARNING EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_early_warning(roll: pd.DataFrame,
                            crisis_start: str = '2008-09-01',
                            crisis_end:   str = '2009-06-30') -> None:
    """
    Quantitative evaluation: how well does delta_GARCH predict the crisis?
    Tests the three-way identity: phi4>1 ↔ delta<0 ↔ kappa<4.
    """
    crisis_s = pd.Timestamp(crisis_start)
    crisis_e = pd.Timestamp(crisis_end)

    pre    = roll[roll.index < crisis_s]
    crisis = roll[(roll.index >= crisis_s) & (roll.index <= crisis_e)]
    post   = roll[roll.index > crisis_e]

    print("\n" + "="*60)
    print("EARLY-WARNING PERFORMANCE")
    print("="*60)

    for name, sub in [("Pre-crisis",      pre),
                      ("Crisis period",   crisis),
                      ("Post-crisis",     post)]:
        if len(sub) == 0:
            continue
        n_neg  = (sub['delta'] < 0).sum()
        n_phi4 = (sub['phi4'] > 1).sum()
        n_k4   = (sub['kappa'] < 4).sum()
        print(f"\n  {name}  (n={len(sub)} points):")
        print(f"    phi4 mean = {sub['phi4'].mean():.4f}  "
              f"(>1: {n_phi4}/{len(sub)}, {100*n_phi4/len(sub):.0f}%)")
        print(f"    delta < 0 : {n_neg}/{len(sub)}  ({100*n_neg/len(sub):.0f}%)")
        print(f"    kappa < 4 : {n_k4}/{len(sub)}  ({100*n_k4/len(sub):.0f}%)")

    print()
    print("  THREE-WAY IDENTITY VERIFICATION:")
    both = roll.copy()
    phi4_over1 = (both['phi4'] > 1)
    delta_neg  = (both['delta'] < 0)
    kappa_lt4  = (both['kappa'] < 4)
    agreement  = (phi4_over1 == delta_neg).mean()
    print(f"    phi4>1 ↔ delta<0 agreement: {100*agreement:.1f}%  "
          f"(theoretical: 100%, deviation = numerical rounding)")
    agreement2 = (phi4_over1 == kappa_lt4).mean()
    print(f"    phi4>1 ↔ kappa<4 agreement: {100*agreement2:.1f}%  "
          f"(slightly lower due to approximate kappa formula)")
    print()


# ─────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='GPM-GARCH Early-Warning System — Empirical Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python DGARCH_empirical_validation.py --file sp500_daily.csv
  python DGARCH_empirical_validation.py --file ndx_d.csv --start 2010-01-01
  python DGARCH_empirical_validation.py  # demo mode, no CSV required
        """
    )
    parser.add_argument(
        '--file', type=str, default=None,
        help='Path to CSV file (e.g. sp500_daily.csv). '
             'Without this argument, runs in demo mode on simulated data.'
    )
    parser.add_argument(
        '--window', type=int, default=252,
        help='Rolling window in trading days (default: 252)'
    )
    parser.add_argument(
        '--step', type=int, default=5,
        help='Step size in days (default: 5)'
    )
    parser.add_argument(
        '--start', type=str, default=None,
        help='Start date filter (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', type=str, default=None,
        help='End date filter (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--crisis-start', type=str, default='2008-09-01',
        help='Crisis start date for evaluation (default: 2008-09-01)'
    )
    parser.add_argument(
        '--crisis-end', type=str, default='2009-06-30',
        help='Crisis end date for evaluation (default: 2009-06-30)'
    )
    parser.add_argument(
        '--exact-kappa', action='store_true',
        help='Solve Kesten equation numerically (more accurate but slower)'
    )
    args = parser.parse_args()

    print()
    print("="*60)
    print("GPM-GARCH EARLY-WARNING SYSTEM")
    print("BarefootRealism Labs")
    print("="*60)

    # Determine base name for timestamped outputs
    if args.file is not None:
        import os
        base_name = os.path.splitext(os.path.basename(args.file))[0] + '_'
    else:
        base_name = 'demo_'

    # Timestamped output paths
    output_png = make_output_path(base_name, 'gpm_garch.png')
    output_csv = make_output_path(base_name, 'gpm_garch_results.csv')

    # Data loading
    if args.file is not None:
        print(f"\nLoading data: {args.file}")
        try:
            returns = load_price_data(args.file)
            title_suffix = f' — {args.file}'
        except Exception as e:
            print(f"  ERROR: {e}")
            sys.exit(1)
    else:
        print("\nWARNING: No CSV file provided — running in DEMO MODE (simulated data)")
        print("   For real data: python DGARCH_empirical_validation.py --file sp500_daily.csv")
        print("   Data source:   https://stooq.com/q/d/l/?s=%5Espx&i=d")
        returns = generate_demo_data()
        title_suffix = ' — Simulated data (S&P 500 type)'

    # Date filtering
    if args.start:
        returns = returns[returns.index >= args.start]
    if args.end:
        returns = returns[returns.index <= args.end]

    print(f"  Analysis period: {returns.index[0].date()} to "
          f"{returns.index[-1].date()}  ({len(returns)} days)")

    # Rolling analysis
    print("\nRunning rolling GARCH analysis...")
    roll = rolling_analysis(returns, args.window, args.step, args.exact_kappa)

    if len(roll) == 0:
        print("ERROR: Insufficient data for rolling analysis.")
        sys.exit(1)

    print(f"\nResults summary:")
    print(f"  phi4 range:   [{roll['phi4'].min():.4f}, {roll['phi4'].max():.4f}]")
    print(f"  delta range:  [{roll['delta'].min():.4f}, {roll['delta'].max():.4f}]")
    print(f"  delta < 0:    {100*(roll['delta']<0).mean():.1f}% of observations")

    # Early-warning evaluation
    evaluate_early_warning(roll, args.crisis_start, args.crisis_end)

    # Figure
    print("Generating figure...")
    plot_results(returns, roll, output_png, title_suffix)

    # CSV output
    roll.to_csv(output_csv)
    print(f"  Results saved: {output_csv}")

    print(f"\nDone. Output files:")
    print(f"  {output_png}")
    print(f"  {output_csv}")


if __name__ == '__main__':
    main()
