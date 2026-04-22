# Data Sources

This directory contains the market data used for empirical validation.

**The CSV files are NOT included in the repository** (copyright of data providers).
Download them using the links below — all sources are free, no registration required.

Place downloaded files directly in this `data/` folder.

---

## Markets and Download Links

### 1. S&P 500 Index (sp500_daily.csv)
**Source:** Stooq.com  
**URL:** https://stooq.com/q/d/l/?s=%5Espx&i=d  
**Save as:** `data/sp500_daily.csv`  
**Coverage:** 2003–present  
**Columns:** Date, Open, High, Low, Close, Volume

### 2. NASDAQ-100 Index (ndx_d.csv)
**Source:** Stooq.com  
**URL:** https://stooq.com/q/d/l/?s=%5Endx&i=d  
**Save as:** `data/ndx_d.csv`  
**Coverage:** 2002–present

### 3. DAX Index (dax_d.csv)
**Source:** Stooq.com  
**URL:** https://stooq.com/q/d/l/?s=%5Edax&i=d  
**Save as:** `data/dax_d.csv`  
**Coverage:** 2002–present

### 4. OTP Bank (otp_hu_d.csv)
**Source:** Stooq.com  
**URL:** https://stooq.com/q/d/l/?s=otp.hu&i=d  
**Save as:** `data/otp_hu_d.csv`  
**Coverage:** 2002–present  
**Note:** Hungarian emerging market stock — structurally different bifurcation profile

### 5. Mercedes-Benz Group (mbg_de_d.csv)
**Source:** Stooq.com  
**URL:** https://stooq.com/q/d/l/?s=mbg.de&i=d  
**Save as:** `data/mbg_de_d.csv`  
**Coverage:** 2023–present  
**Note:** Short series, currently operating in heavy-tail regime (kappa < 4)

### 6. Bitcoin / USD (btc_v_d.csv)
**Source:** Stooq.com  
**URL:** https://stooq.com/q/d/l/?s=btc.v&i=d  
**Save as:** `data/btc_v_d.csv`  
**Coverage:** 2013–present  
**Note:** Extreme regime — phi4 up to 40, qualitatively different scale

### 7. USD/EUR Exchange Rate (usdeur_d.csv)
**Source:** Stooq.com  
**URL:** https://stooq.com/q/d/l/?s=usdeur&i=d  
**Save as:** `data/usdeur_d.csv`  
**Coverage:** 2002–present  
**Note:** Reference case — most stable asset, kappa rarely below 4

---

## Reproduce All Results

After downloading all files, run:

```bash
# S&P 500 — full period
python DGARCH_empirical_validation.py --file data/sp500_daily.csv

# S&P 500 — Lehman crisis zoom
python DGARCH_empirical_validation.py --file data/sp500_daily.csv \
    --start 2003-01-01 --end 2013-01-01

# NASDAQ — full period
python DGARCH_empirical_validation.py --file data/ndx_d.csv

# DAX — full period
python DGARCH_empirical_validation.py --file data/dax_d.csv

# OTP Bank — full period
python DGARCH_empirical_validation.py --file data/otp_hu_d.csv

# Mercedes-Benz — full period
python DGARCH_empirical_validation.py --file data/mbg_de_d.csv

# Bitcoin
python DGARCH_empirical_validation.py --file data/btc_v_d.csv

# USD/EUR
python DGARCH_empirical_validation.py --file data/usdeur_d.csv
```

Each run produces two timestamped output files:
```
YYYYMMDD_HHMMSS_{market}_gpm_garch.png
YYYYMMDD_HHMMSS_{market}_gpm_garch_results.csv
```

---

## Alternative Data Sources

If Stooq is unavailable:

| Market | Alternative |
|--------|------------|
| S&P 500, NASDAQ | Yahoo Finance: https://finance.yahoo.com |
| DAX | Investing.com: https://www.investing.com |
| BTC/USD | CoinGecko: https://www.coingecko.com |
| OTP Bank | Budapest Stock Exchange: https://bse.hu |

**Column mapping:** Yahoo Finance uses `Adj Close` instead of `Close` —
the script automatically detects both formats.

---

## Data Format

All CSV files follow this structure:

```
Date,Open,High,Low,Close,Volume
2003-01-02,879.82,890.78,870.92,879.82,1234567890
2003-01-03,879.82,888.40,875.44,882.53,987654321
...
```

The script auto-detects date and closing price columns for both
Stooq and Yahoo Finance formats.
