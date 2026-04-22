# GPM-GARCH: Endogenous Heavy-Tail Formation as a Bifurcation Phenomenon

**Author:** László Tatai / BarefootRealism Labs  
**ORCID:** 0009-0007-5153-6306  
**License:** MIT  
**Zenodo DOI:** *(to be assigned on upload)*

---

## Overview

This repository implements the **Gauss–PowerLaw Module (GPM)** and its structural connection to GARCH(1,1) volatility dynamics, as described in:

> Tatai, L. (2026). *The Gauss–PowerLaw Module (GPM): Endogenous Heavy-Tail Formation as a Bifurcation Phenomenon.* BarefootRealism Labs. Zenodo.

The central theoretical result is a four-way structural identity:

```
Delta_GPM = 0  ⟺  phi4 = 1  ⟺  kappa = 4  ⟺  E[y^4] = infinity
```

All four conditions identify the same critical boundary at which:
- The CLT breaks down for GARCH returns (López de Prado et al., 2026, Theorem 4.1)
- The GPM undergoes a saddle-node bifurcation
- Standard Sharpe ratio inference loses validity

**In one sentence:** GARCH captures the symptom of heavy-tail emergence; the GPM captures the mechanism.

---

## Repository Structure

```
gpm-garch/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE
│
├── DGARCH_empirical_validation.py     # Empirical validation on real market data
├── Prob_Op_Space_v2.py                # OTDD operator simulation (GPM phase diagram)
│
├── paper/
│   └── GPM_GARCH_Paper.md             # Full theoretical paper (preprint)
│
├── data/                              # Place downloaded CSV files here
│   └── README_data.md                 # Data source instructions
│
└── outputs/                           # Timestamped results (auto-generated)
    └── YYYYMMDD_HHMMSS_*.png/csv
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download data (free, no registration)

S&P 500:
```
https://stooq.com/q/d/l/?s=%5Espx&i=d
```
Save as `data/sp500_daily.csv`.

Other supported markets: NASDAQ (`^ndx`), DAX (`^dax`), BTC/USD (`btc.v`), USD/EUR (`usdeur`).

### 3. Run empirical validation

```bash
# With real data
python DGARCH_empirical_validation.py --file data/sp500_daily.csv

# Demo mode (no data required)
python DGARCH_empirical_validation.py

# Full date range with exact Kesten equation
python DGARCH_empirical_validation.py --file data/ndx_d.csv --exact-kappa
```

Output files are automatically timestamped:
```
YYYYMMDD_HHMMSS_sp500_daily_gpm_garch.png
YYYYMMDD_HHMMSS_sp500_daily_gpm_garch_results.csv
```

### 4. Run OTDD operator simulation

```bash
python Prob_Op_Space_v2.py
```

Generates phase diagrams, distributional trajectories, and potential metric evolution.

---

## Key Indicators

| Indicator | Formula | Meaning |
|-----------|---------|---------|
| `phi4` | `alpha_G^2 * kappa_z + 2*alpha_G*beta_G + beta_G^2` | Fourth-moment proximity to CLT boundary (Lemma 1) |
| `delta_GARCH` | `1 - phi4` | Signed distance from GPM bifurcation (Theorem 2) |
| `kappa` | `≈ 4 / phi4` | Tail index approximation (Theorem 1) |

**Interpretation:**
- `delta_GARCH > 0`: system is in stable Gaussian regime — CLT valid
- `delta_GARCH = 0`: system is at the critical boundary — bifurcation point
- `delta_GARCH < 0`: system has crossed into heavy-tail regime — CLT invalid

---

## Empirical Results Summary

Seven markets validated (2002–2026):

| Market | Type | Crisis signal | Notes |
|--------|------|---------------|-------|
| S&P 500 | Equity index | 2008, 2018 | Clean pre-crisis warning |
| NASDAQ | Equity index | 2008, 2020, 2022 | Multiple regimes |
| DAX | Equity index | 2008, 2020, 2022 | European confirmation |
| OTP Bank | Single stock | 2008, 2015–16 | Emerging market character |
| Mercedes-Benz | Single stock | 2023–2026 | Currently in heavy-tail zone |
| BTC/USD | Cryptocurrency | Persistent | phi4 up to 40 — extreme regime |
| USD/EUR | FX | 2008 (brief) | Reference: "normal world" |

---

## Theoretical Background

### The GPM Dynamical System

```
d(sigma^2)/dt = gamma * (sigma_star^2 - sigma^2) + delta/alpha
alpha = alpha_0 - k * sigma^2
```

**Fixed points:** exist when discriminant `Delta > 0`  
**Bifurcation:** `Delta = 0` — saddle-node, two fixed points coalesce  
**Unstable:** `Delta < 0` — no real fixed point, sigma^2 diverges

### The GARCH-GPM Correspondence

| GPM parameter | GARCH correspondence |
|--------------|---------------------|
| `sigma_star^2` | `omega / (1 - phi)` |
| `gamma` | `1 - phi` (persistence complement) |
| `delta` | `alpha_G` (shock amplification) |
| `kappa` | Kesten-Goldie solution |

### The Algebraic Identity (Lemma 1)

For `X = alpha_G * z_0^2 + beta_G` with `z_0 ~ N(0,1)`:

```
M(4) = E[X^2] = alpha_G^2 * kappa_z + 2*alpha_G*beta_G + beta_G^2 = phi4
```

Verified numerically to 10^-12 precision.

---

## Citation

```bibtex
@techreport{tatai2026gpm,
  author      = {Tatai, László},
  title       = {The {Gauss--PowerLaw} Module ({GPM}): Endogenous Heavy-Tail
                 Formation as a Bifurcation Phenomenon},
  institution = {BarefootRealism Labs},
  year        = {2026},
  note        = {Preprint. Zenodo. doi: TBD}
}
```

---

## Related Work

- López de Prado, M., Porcu, E., Zoonekynd, V., & Engle, R. F. (2026). *A closed-form solution for Sharpe ratio inference under GARCH returns.* SSRN. https://ssrn.com/abstract=6568702
- Basrak, B., Davis, R. A., & Mikosch, T. (2002). Regular variation of GARCH processes. *Stochastic Processes and their Applications*, 99(1), 95–115.
- Tatai, L. (2026). *Operator Theory of Distribution Dynamics (OTDD).* BarefootRealism Labs. Zenodo.

---

*Integrity verified with [MDL (Markdown Logged)](https://github.com/BarefootRealismLabs/mdl) — BarefootRealism Labs*
