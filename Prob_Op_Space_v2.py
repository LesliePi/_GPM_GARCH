# -*- coding: utf-8 -*-
"""
Prob_Op_Space_v2.py
Operator Theory of Distribution Dynamics (OTDD) — Cleaned & Fixed Simulation
Author: Tatai László / Barfoot Realism Labs
Date:   2026-04-17

Changes vs original Prob_Op_Space.py
─────────────────────────────────────
BUG 1 FIXED  W_s_fast used np.convolve(mode='same') which returns kernel-length
             output when kernel > signal.  Replaced with scipy.ndimage.gaussian_filter1d
             which always returns the same shape as the input.

BUG 2 FIXED  classify() used a full log-log regression that the distribution
             body dominated.  Heavy-tail identification now uses:
               • positive-slope log-log fit on the upper half of the support
                 (tail regime: the invariant distribution accumulates mass near
                  x=10 under E_eta, making tail slope > 0 the heavy-tail marker)
               • skewness threshold for Gauss detection
             Three classes: 1=Gauss, 2=HeavyTail, 3=Hybrid/Lognormal

BUG 3 FIXED  run_sim() used a fixed 80-step count. Replaced with an adaptive
             convergence criterion: stop when max|Δμ| < tol (default 1e-8).
             Fall-back max_steps=600 prevents infinite loops.

STRUCTURAL NOTE  The E_eta operator (1 + η·ln(x+1)) is a monotone-increasing
             fitness function: it amplifies states with larger x.  Under this
             operator the invariant distribution accumulates near x=x_max
             (right boundary), producing a right-skewed, heavy-tailed shape.
             This is correct OTDD behaviour — not a bug.  The torzulásmérő D
             is therefore computed as D = (mean − median)/median directly on
             the invariant distribution, without assuming log-normality.

GPM–GARCH   D is an early-warning indicator for the CLT-breakdown boundary
             identified by López de Prado et al. (2026): as η and σ grow,
             D becomes more negative (mean < median), reflecting the increasing
             left-skew induced by the boundary-accumulation effect.  The GPM
             σ²-bifurcation maps to the GARCH δ_GARCH = 1 − α²κ_z − 2αβ − β² → 0.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1.  STATE SPACE
# ─────────────────────────────────────────────────────────────
N_POINTS = 300
X = np.linspace(0.001, 10.0, N_POINTS)


def init_distribution(x: np.ndarray = X) -> np.ndarray:
    """Gaussian-shaped initial distribution centred at x = 3."""
    mu = np.exp(-(x - 3.0) ** 2 / 0.5)
    return mu / np.sum(mu)


# ─────────────────────────────────────────────────────────────
# 2.  COMPONENT OPERATORS
# ─────────────────────────────────────────────────────────────

def D_alpha_op(mu: np.ndarray, alpha: float) -> np.ndarray:
    """Dissipation: uniform proportional decay."""
    return (1.0 - alpha) * mu


def W_s_op(mu: np.ndarray, sigma: float, x: np.ndarray = X) -> np.ndarray:
    """
    Weighting / redistribution: Gaussian smoothing.

    BUG 1 FIX: gaussian_filter1d always returns len(mu) output.
    The kernel width is scaled so that sigma ∈ [0.1, 2.0] corresponds
    to a physically meaningful mixing range on X = [0.001, 10].
    """
    grid_sigma = max(sigma * N_POINTS / 10.0, 0.5)
    return gaussian_filter1d(mu, sigma=grid_sigma, mode='reflect')


def E_eta_op(mu: np.ndarray, eta_strength: float, x: np.ndarray = X) -> np.ndarray:
    """
    Efficiency modulation: pointwise amplification by η·ln(x+1).
    States with larger x receive higher fitness → mass accumulates
    near the right boundary (x ≈ x_max).  This is the intended behaviour.
    """
    return mu * (1.0 + eta_strength * np.log(x + 1.0))


def S_epsilon_op(mu: np.ndarray, epsilon: float) -> np.ndarray:
    """Selection: hard threshold — zero out sub-threshold mass."""
    result = mu.copy()
    result[result < epsilon] = 0.0
    return result


def normalize(mu: np.ndarray) -> np.ndarray:
    """Renormalise to probability measure."""
    total = np.sum(mu)
    return mu / total if total > 1e-300 else mu


def T(mu: np.ndarray,
      alpha: float,
      sigma: float,
      eta_strength: float,
      epsilon: float,
      x: np.ndarray = X) -> np.ndarray:
    """
    Composite Allocation Operator:
        μ → D_α → W_σ → E_η → S_ε → N → T[μ]
    """
    mu = D_alpha_op(mu, alpha)
    mu = W_s_op(mu, sigma, x)
    mu = E_eta_op(mu, eta_strength, x)
    mu = S_epsilon_op(mu, epsilon)
    return normalize(mu)


# ─────────────────────────────────────────────────────────────
# 3.  SIMULATION
# ─────────────────────────────────────────────────────────────

def run_sim(alpha: float,
            sigma: float,
            eta_strength: float,
            epsilon: float,
            tol: float = 1e-8,
            max_steps: int = 600,
            x: np.ndarray = X) -> np.ndarray:
    """
    Run the operator to convergence.

    BUG 3 FIX: replaces fixed 80-step count with adaptive stopping:
    halts when max|μ_{t+1} − μ_t| < tol.  Falls back to max_steps
    iterations if convergence is not reached (e.g. oscillatory regime).
    """
    mu = init_distribution(x)
    for _ in range(max_steps):
        mu_new = T(mu, alpha, sigma, eta_strength, epsilon, x)
        if np.max(np.abs(mu_new - mu)) < tol:
            return mu_new
        mu = mu_new
    return mu_new


def run_trajectory(alpha: float,
                   sigma: float,
                   eta_strength: float,
                   epsilon: float,
                   steps: int = 80,
                   x: np.ndarray = X) -> np.ndarray:
    """Record full distributional trajectory (history array)."""
    mu = init_distribution(x)
    history = [mu.copy()]
    for _ in range(steps):
        mu = T(mu, alpha, sigma, eta_strength, epsilon, x)
        history.append(mu.copy())
    return np.array(history)


# ─────────────────────────────────────────────────────────────
# 4.  STATISTICS & GPM INDICATOR
# ─────────────────────────────────────────────────────────────

def compute_stats(mu: np.ndarray, x: np.ndarray = X) -> dict:
    """
    Compute key statistics of a discrete distribution.

    Returns
    -------
    mean, var, skew   : standard distributional moments
    median            : 50th percentile
    D                 : torzulásmérő = (mean − median) / median
                        Positive D → right-skewed (mean > median)
                        Negative D → left-skewed  (mean < median, heavy-tail
                                     boundary accumulation)
    tail_slope        : log-log slope on the upper 50% of support
                        (positive = rising tail = heavy-tail accumulation)
    tail_r2           : R² of the tail log-log fit
    """
    mu_s = mu + 1e-12
    mean  = np.sum(x * mu_s)
    var   = np.sum((x - mean) ** 2 * mu_s)
    skew  = np.sum(((x - mean) ** 3) * mu_s) / (var ** 1.5 + 1e-12)

    cumsum     = np.cumsum(mu_s) / np.sum(mu_s)
    median_idx = min(np.searchsorted(cumsum, 0.50), len(x) - 1)
    median     = x[median_idx]
    D          = (mean - median) / median if median > 1e-10 else 0.0

    p50_idx    = median_idx
    tail_mask  = (np.arange(len(x)) > p50_idx) & (mu_s > 1e-7)
    tail_slope = tail_r2 = np.nan
    if tail_mask.sum() > 15:
        sl, _, r, _, _ = linregress(np.log(x[tail_mask]),
                                    np.log(mu_s[tail_mask]))
        tail_slope, tail_r2 = sl, r ** 2

    return dict(mean=mean, var=var, skew=skew, median=median,
                D=D, tail_slope=tail_slope, tail_r2=tail_r2)

def compute_potential(mu, x):
    mu_s = mu + 1e-12
    return -np.log(mu_s)


def compute_potential_metrics(mu, x):
    V = compute_potential(mu, x)

    dV = np.gradient(V, x)
    d2V = np.gradient(dV, x)

    idx_min = np.argmin(V)

    curvature = d2V[idx_min]
    slope_boundary = dV[-1]

    return curvature, slope_boundary


# ─────────────────────────────────────────────────────────────
# 5.  ATTRACTOR CLASSIFIER
# ─────────────────────────────────────────────────────────────

def classify(mu: np.ndarray, x: np.ndarray = X) -> int:
    """
    BUG 2 FIX: attractor identification based on tail log-log slope
    and skewness, not full-domain log-log regression.

    Under E_eta the invariant distribution accumulates near x_max,
    producing a rising tail (slope > 0).  Heavy-tail regime is therefore
    identified by:
        • high tail R² (well-structured tail)
        • positive slope (mass rises toward boundary)
        • strong left skewness (mass pulled away from mode)

    Returns
    -------
    1 : Gaussian     — symmetric, low skewness
    2 : HeavyTail    — rising, well-fitted tail (boundary accumulation)
    3 : Hybrid       — intermediate / lognormal-like
    """
    s = compute_stats(mu, x)

    # Heavy tail: rising log-log tail with good fit
    is_heavy = (
        not np.isnan(s['tail_r2'])
        and s['tail_r2'] > 0.92
        and s['tail_slope'] > 0.3        # positive slope = rising tail
        and s['skew'] < -0.5             # left-skewed overall
    )
    if is_heavy:
        return 2  # HeavyTail

    # Gaussian: small |skew|
    if abs(s['skew']) < 0.60:
        return 1  # Gaussian

    return 3  # Hybrid / Lognormal


CLASS_LABELS = {1: 'Gaussian', 2: 'HeavyTail', 3: 'Hybrid'}


# ─────────────────────────────────────────────────────────────
# 6.  PHASE DIAGRAM
# ─────────────────────────────────────────────────────────────

def compute_phase_diagram(sigma_range=(0.1, 2.0),
                          eta_range=(0.0, 2.0),
                          grid_size: int = 20,
                          alpha: float = 0.1,
                          epsilon: float = 1e-4,
                          tol: float = 1e-8,
                          max_steps: int = 600) -> tuple:
    """
    Compute attractor class and D-indicator over a 2-D (σ, η) grid.

    Returns sigma_vals, eta_vals, phase matrix, D matrix.
    """
    sigma_vals = np.linspace(*sigma_range, grid_size)
    eta_vals   = np.linspace(*eta_range,   grid_size)
    phase      = np.zeros((grid_size, grid_size), dtype=int)
    D_map      = np.zeros((grid_size, grid_size))

    for i, eta in enumerate(eta_vals):
        for j, sig in enumerate(sigma_vals):
            mu          = run_sim(alpha, sig, eta, epsilon, tol, max_steps)
            phase[i, j] = classify(mu)
            D_map[i, j] = compute_stats(mu)['D']

    return sigma_vals, eta_vals, phase, D_map


# ─────────────────────────────────────────────────────────────
# 7.  VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_trajectory(history: np.ndarray, x: np.ndarray = X,
                    step: int = 10, title: str = '') -> None:
    plt.figure(figsize=(10, 5))
    for i in range(0, len(history), step):
        plt.plot(x, history[i], alpha=0.55, label=f't={i}')
    plt.title(f'Distributional Trajectory  {title}')
    plt.xlabel('State x'); plt.ylabel('Probability density')
    plt.legend(fontsize=7, ncol=2); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_moments(history: np.ndarray, x: np.ndarray = X,
                 title: str = '') -> None:
    means = [compute_stats(h, x)['mean'] for h in history]
    varis = [compute_stats(h, x)['var']  for h in history]
    D_ser = [compute_stats(h, x)['D']    for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(means);  axes[0].set_title('Mean');     axes[0].grid(True, alpha=0.3)
    axes[1].plot(varis);  axes[1].set_title('Variance'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(D_ser);  axes[2].set_title('D = (mean−median)/median')
    axes[2].axhline(0, color='k', lw=0.8, ls='--'); axes[2].grid(True, alpha=0.3)
    fig.suptitle(f'Moment Evolution  {title}')
    plt.tight_layout(); plt.show()
    
def plot_potential_metrics(history, x=X, title=''):
    curvatures = []
    slopes = []
    D_vals = []

    for mu in history:
        c, s = compute_potential_metrics(mu, x)
        curvatures.append(c)
        slopes.append(s)
        D_vals.append(compute_stats(mu, x)['D'])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(curvatures)
    axes[0].set_title('Curvature at minimum V\'\'(x*)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(slopes)
    axes[1].set_title('Boundary slope dV/dx | x_max')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(D_vals)
    axes[2].set_title('D indicator')
    axes[2].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'Potential Metrics Evolution  {title}')
    plt.tight_layout()
    plt.show()


def plot_phase_diagram(sigma_vals: np.ndarray,
                       eta_vals:   np.ndarray,
                       phase:      np.ndarray,
                       D_map:      np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Phase
    cmap1 = plt.get_cmap('RdYlGn', 3)
    im1 = axes[0].imshow(
        phase, origin='lower', cmap=cmap1, vmin=0.5, vmax=3.5,
        extent=[sigma_vals[0], sigma_vals[-1],
                eta_vals[0],   eta_vals[-1]], aspect='auto')
    cb1 = plt.colorbar(im1, ax=axes[0], ticks=[1, 2, 3])
    cb1.set_ticklabels(['Gaussian', 'HeavyTail', 'Hybrid'])
    axes[0].set_xlabel('σ  (mixing width)')
    axes[0].set_ylabel('η  (efficiency strength)')
    axes[0].set_title('Attractor Phase Diagram')

    # D-map (early-warning indicator)
    vabs = np.nanmax(np.abs(D_map)) + 1e-6
    im2 = axes[1].imshow(
        D_map, origin='lower', cmap='coolwarm',
        vmin=-vabs, vmax=vabs,
        extent=[sigma_vals[0], sigma_vals[-1],
                eta_vals[0],   eta_vals[-1]], aspect='auto')
    plt.colorbar(im2, ax=axes[1], label='D = (mean−median)/median')
    axes[1].set_xlabel('σ  (mixing width)')
    axes[1].set_ylabel('η  (efficiency strength)')
    axes[1].set_title('GPM–GARCH Early-Warning Indicator D')

    plt.tight_layout(); plt.show()


def plot_invariant_distributions(configs: list, x: np.ndarray = X) -> None:
    """
    configs : list of (alpha, sigma, eta, epsilon, label)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(configs)))

    for (a, s, e, eps, label), color in zip(configs, colors):
        mu  = run_sim(a, s, e, eps)
        st  = compute_stats(mu, x)
        cls = CLASS_LABELS[classify(mu, x)]
        lbl = f'{label}  [{cls}  D={st["D"]:+.3f}]'

        axes[0].plot(x, mu, label=lbl, color=color, lw=1.6)
        # log-log view
        mask = mu > 1e-7
        axes[1].plot(np.log(x[mask]), np.log(mu[mask] + 1e-12),
                     label=lbl, color=color, lw=1.6)

    axes[0].set_xlabel('x'); axes[0].set_ylabel('density')
    axes[0].set_title('Invariant Distributions')
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('ln x'); axes[1].set_ylabel('ln μ')
    axes[1].set_title('Log–Log View (tail structure)')
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────
# 8.  GPM–GARCH BRIDGE UTILITIES
# ─────────────────────────────────────────────────────────────

def garch_to_gpm_params(omega: float, alpha_G: float, beta_G: float,
                        kappa_z: float = 3.0) -> dict:
    """
    Heuristic mapping from GARCH(1,1) parameters to GPM parameters.

    sigma_star^2  = omega / (1 - alpha_G - beta_G)   [stationarity mean]
    gamma         ~ 1 - (alpha_G + beta_G)            [mean-reversion speed]
    delta         ~ alpha_G                            [shock amplification]
    delta_GARCH   = 1 - alpha_G^2 kappa_z - 2 alpha_G beta_G - beta_G^2
                    (distance from the 4th-moment boundary;
                     delta_GARCH → 0 ↔ GPM bifurcation → 0)

    References: López de Prado et al. (2026), Proposition 6.2.
    """
    phi   = alpha_G + beta_G
    sigma_star_sq = omega / (1.0 - phi) if phi < 1.0 else np.inf
    gamma = 1.0 - phi
    delta = alpha_G
    delta_GARCH = 1.0 - alpha_G**2 * kappa_z - 2*alpha_G*beta_G - beta_G**2
    D_GARCH = np.exp(omega / (2.0 * (1.0 - phi))) - 1.0 if phi < 1.0 else np.inf
    return dict(sigma_star_sq=sigma_star_sq, gamma=gamma, delta=delta,
                delta_GARCH=delta_GARCH, D_GARCH=D_GARCH, phi=phi)


# ─────────────────────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── 9.1  Trajectory visualisation
    ALPHA, SIGMA, ETA, EPS = 0.05, 0.3, 1.6, 1e-4
    history = run_trajectory(ALPHA, SIGMA, ETA, EPS, steps=80)
    plot_trajectory(history, title=f'α={ALPHA} σ={SIGMA} η={ETA}')
    plot_moments(history,    title=f'α={ALPHA} σ={SIGMA} η={ETA}')
    plot_potential_metrics(history, title=f'α={ALPHA} σ={SIGMA} η={ETA}')

    # ── 9.2  Invariant distribution comparison
    showcase = [
        (0.30, 0.10, 0.10, 1e-4, 'α=0.30 σ=0.10 η=0.10'),
        (0.05, 0.30, 1.00, 1e-4, 'α=0.05 σ=0.30 η=1.00'),
        (0.05, 1.00, 2.00, 1e-4, 'α=0.05 σ=1.00 η=2.00'),
        (0.02, 1.50, 2.00, 1e-4, 'α=0.02 σ=1.50 η=2.00'),
    ]
    plot_invariant_distributions(showcase)

    # ── 9.3  Phase diagram
    sv, ev, phase, D_map = compute_phase_diagram(
        sigma_range=(0.1, 2.0), eta_range=(0.0, 2.0),
        grid_size=20, alpha=0.1, epsilon=1e-4)
    plot_phase_diagram(sv, ev, phase, D_map)

    # ── 9.4  GPM–GARCH bridge: HFR index examples
    print('\n=== GPM–GARCH Parameter Bridge ===')
    print(f'  {"Index":<18} {"φ":>5} {"δ_GARCH":>10} {"D_GARCH":>10} {"Regime"}')
    hfr_indices = [
        ('Composite Equity', 0.04, 0.14, 0.82),
        ('Event-Driven',     0.04, 0.24, 0.69),
        ('Relative Value',   0.04, 0.09, 0.82),
        ('Macro',            0.04, 0.31, 0.67),
    ]
    for name, omega, aG, bG in hfr_indices:
        p = garch_to_gpm_params(omega, aG, bG)
        regime = ('STABLE' if p['delta_GARCH'] > 0.05
                  else 'CRITICAL' if p['delta_GARCH'] > -0.05
                  else 'UNSTABLE')
        print(f'  {name:<18} {p["phi"]:>5.2f} {p["delta_GARCH"]:>10.4f} '
              f'{p["D_GARCH"]:>10.4f}  {regime}')

    print('\nDone.')
