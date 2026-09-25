# Equity Research Report Generator

One command produces a full equity research note (markdown + charts) from **live Yahoo Finance data** – no synthetic data.

**What it does:** 5y price/risk stats · historical financials · CAPM/WACC · fading-growth DCF with sensitivity grid · peer multiples · football-field chart · rules-based rating.

## Run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python report.py --ticker MSFT --peers GOOGL,AAPL,ORCL,CRM,ADBE
```
Works for other tickers too, e.g. `--ticker RELIANCE.NS --peers ONGC.NS,BPCL.NS,IOC.NS` (Indian fundamentals on Yahoo can have gaps).

Outputs land in `outputs/` (report + PNGs); the raw data used is saved to `data/`.

## Method notes
- Beta: 3y weekly returns vs S&P 500. Risk-free: latest ^TNX. ERP 5% and terminal growth 2.5% are stated assumptions (top of `report.py`).
- DCF: latest FCF grown at historical FCF CAGR (capped 0–15%) fading linearly to terminal growth over 5y.
- Rating is mechanical; treat it as a template for your own judgment.
