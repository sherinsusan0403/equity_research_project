
"""Equity research report generator.
 
Pulls REAL market and fundamental data from Yahoo Finance (via yfinance) and builds:
  * 5y price chart with 50/200d moving averages + risk stats (vol, beta, max drawdown)
  * Historical financials (revenue, net income, FCF, margins)
  * CAPM/WACC + 5-year fading-growth DCF with a WACC x terminal-growth sensitivity grid
  * Peer multiples table and comps-implied value
  * Football-field valuation chart and a rules-based rating
Output: outputs/<TICKER>_report.md plus PNG charts.
 
Usage:  python report.py --ticker MSFT --peers GOOGL,AAPL,ORCL,CRM,ADBE
"""
import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
ERP = 0.05          # equity risk premium (assumption)
TERMINAL_G = 0.025  # terminal growth (assumption)
TAX_RATE = 0.21     # marginal tax used for the debt shield (assumption)
YEARS = 5
OUT = "outputs"
 
 
def get_row(df, names):
    """Return the first matching statement row as an ascending-date float Series."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce").dropna()
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    return pd.Series(dtype=float)
 
 
def cagr(s):
    s = s.dropna()
    s = s[s > 0]
    if len(s) < 2:
        return np.nan
    return (s.iloc[-1] / s.iloc[0]) ** (1 / (len(s) - 1)) - 1
 
 
def dcf_value(fcf0, g_start, g_term, wacc, years=YEARS):
    wacc = max(wacc, g_term + 0.02)
    growth = np.linspace(g_start, g_term, years)
    flows, f = [], fcf0
    for g in growth:
        f *= 1 + g
        flows.append(f)
    pv = sum(cf / (1 + wacc) ** (i + 1) for i, cf in enumerate(flows))
    tv = flows[-1] * (1 + g_term) / (wacc - g_term)
    return pv + tv / (1 + wacc) ** years, flows
 
 
def fetch_beta_rf(ticker):
    px = yf.download([ticker, "^GSPC"], period="3y", auto_adjust=True, progress=False)["Close"]
    wk = px.resample("W-FRI").last().pct_change().dropna()
    beta = np.cov(wk[ticker], wk["^GSPC"])[0, 1] / wk["^GSPC"].var()
    tnx = yf.download("^TNX", period="5d", auto_adjust=True, progress=False)["Close"].dropna()
    rf = float(np.squeeze(tnx.iloc[-1])) / 100
    return float(beta), rf
 
 
def max_drawdown(px):
    return float((px / px.cummax() - 1).min())
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="MSFT")
    ap.add_argument("--peers", default="GOOGL,AAPL,ORCL,CRM,ADBE")
    a = ap.parse_args()
    T = a.ticker.upper()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("data", exist_ok=True)
 
    tk = yf.Ticker(T)
    info = tk.info
    name = info.get("longName", T)
    hist = tk.history(period="5y", auto_adjust=True)["Close"]
    price = float(hist.iloc[-1])
    hist.to_csv(f"data/{T}_prices.csv")
 
    inc, cf, bs = tk.income_stmt, tk.cashflow, tk.balance_sheet
    rev = get_row(inc, ["Total Revenue", "Operating Revenue"])
    ni = get_row(inc, ["Net Income", "Net Income Common Stockholders"])
    opinc = get_row(inc, ["Operating Income", "EBIT"])
    fcf = get_row(cf, ["Free Cash Flow"])
    if fcf.empty:
        ocf = get_row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
        capex = get_row(cf, ["Capital Expenditure"])
        fcf = (ocf + capex).dropna()
    fin = pd.DataFrame({"Revenue": rev, "Operating income": opinc, "Net income": ni, "FCF": fcf})
    fin.index = fin.index.year
    fin.to_csv(f"data/{T}_financials.csv")
 
    # ---- Charts ----
    fig, ax = plt.subplots(figsize=(9, 4))
    hist.plot(ax=ax, label="Close", lw=1.2)
    hist.rolling(50).mean().plot(ax=ax, label="50d")
    hist.rolling(200).mean().plot(ax=ax, label="200d")
    ax.set_title(f"{T} – 5y price"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/{T}_price.png", dpi=140); plt.close(fig)
 
    fig, ax = plt.subplots(figsize=(9, 4))
    (fin[["Revenue", "Net income", "FCF"]] / 1e9).plot.bar(ax=ax)
    ax.set_ylabel("USD bn (reporting currency)"); ax.set_title(f"{T} – annual financials"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/{T}_financials.png", dpi=140); plt.close(fig)
 
    # ---- Cost of capital ----
    beta, rf = fetch_beta_rf(T)
    ke = rf + beta * ERP
    debt = info.get("totalDebt") or 0.0
    cash = info.get("totalCash") or 0.0
    mcap = info.get("marketCap") or price * (info.get("sharesOutstanding") or np.nan)
    intexp = get_row(inc, ["Interest Expense"])
    kd = float(np.clip(abs(intexp.iloc[-1]) / debt, rf, 0.10)) if len(intexp) and debt else rf + 0.02
    wd = debt / (debt + mcap) if (debt + mcap) else 0
    wacc = (1 - wd) * ke + wd * kd * (1 - TAX_RATE)
    shares = info.get("sharesOutstanding") or mcap / price
 
    # ---- DCF ----
    dcf_txt, dcf_lo, dcf_hi, dcf_ps, sens = "", np.nan, np.nan, np.nan, None
    if len(fcf) and fcf.iloc[-1] > 0:
        g_hist = cagr(fcf) if len(fcf) >= 3 else cagr(rev)
        g0 = float(np.clip(g_hist if pd.notna(g_hist) else 0.05, 0.0, 0.15))
        ev, flows = dcf_value(float(fcf.iloc[-1]), g0, TERMINAL_G, wacc)
        dcf_ps = (ev - debt + cash) / shares
        waccs = [wacc - .01, wacc - .005, wacc, wacc + .005, wacc + .01]
        gts = [.015, .02, .025, .03, .035]
        sens = pd.DataFrame(
            [[(dcf_value(float(fcf.iloc[-1]), g0, g, w)[0] - debt + cash) / shares for g in gts] for w in waccs],
            index=[f"{w:.1%}" for w in waccs], columns=[f"g={g:.1%}" for g in gts])
        dcf_lo, dcf_hi = float(sens.values.min()), float(sens.values.max())
        dcf_txt = (f"Starting FCF growth {g0:.1%} fading to {TERMINAL_G:.1%} over {YEARS}y; WACC {wacc:.2%}. "
                   f"Enterprise value {ev/1e9:,.1f}bn; equity value per share **{dcf_ps:,.2f}**.")
    else:
        dcf_txt = "Latest free cash flow is not positive, so a simple FCF DCF is not meaningful for this company."
 
    # ---- Comps ----
    rows = {}
    for p in [T] + [x.strip().upper() for x in a.peers.split(",") if x.strip()]:
        try:
            i = yf.Ticker(p).info
            rows[p] = {"MktCap (bn)": (i.get("marketCap") or np.nan) / 1e9, "Trailing P/E": i.get("trailingPE"),
                       "Forward P/E": i.get("forwardPE"), "P/B": i.get("priceToBook"),
                       "EV/EBITDA": i.get("enterpriseToEbitda"), "EV/Revenue": i.get("enterpriseToRevenue"),
                       "Profit margin": i.get("profitMargins")}
        except Exception as e:
            print(f"peer {p} failed: {e}")
    comps = pd.DataFrame(rows).T.apply(pd.to_numeric, errors="coerce")
    comps.to_csv(f"data/{T}_comps.csv")
    peer_med = comps.drop(index=T, errors="ignore").median()
    eps = info.get("trailingEps")
    comps_ps = float(peer_med["Trailing P/E"] * eps) if pd.notna(peer_med.get("Trailing P/E")) and eps else np.nan
    fwd_eps = info.get("forwardEps")
    comps_fwd = float(peer_med["Forward P/E"] * fwd_eps) if pd.notna(peer_med.get("Forward P/E")) and fwd_eps else np.nan
 
    # ---- Football field ----
    bars = {"52-week range": (info.get("fiftyTwoWeekLow"), info.get("fiftyTwoWeekHigh"))}
    if pd.notna(dcf_lo):
        bars["DCF (WACC/g grid)"] = (dcf_lo, dcf_hi)
    cm = [v for v in (comps_ps, comps_fwd) if pd.notna(v)]
    if cm:
        bars["Peer P/E (trailing–forward)"] = (min(cm), max(cm))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for k, (label, bound) in enumerate(bars.items()):
        lo, hi = bound
        if lo and hi:
            ax.barh(k, hi - lo, left=lo, color="#4c78a8")
    ax.set_yticks(range(len(bars))); ax.set_yticklabels(list(bars))
    ax.axvline(price, color="crimson", ls="--", label=f"Price {price:,.2f}"); ax.legend()
    ax.set_title(f"{T} – valuation football field"); ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(f"{OUT}/{T}_football_field.png", dpi=140); plt.close(fig)
 
    # ---- Rating (mechanical, based on DCF and comps midpoint) ----
    vals = [v for v in (dcf_ps, comps_ps, comps_fwd) if pd.notna(v)]
    target = float(np.mean(vals)) if vals else np.nan
    upside = target / price - 1 if pd.notna(target) else np.nan
    rating = "n/a" if pd.isna(upside) else "BUY" if upside > .15 else "SELL" if upside < -.15 else "HOLD"
 
    ret = hist.pct_change().dropna()
    stats = pd.Series({"Price": price, "1y return": hist.iloc[-1] / hist.iloc[-252] - 1 if len(hist) > 252 else np.nan,
                       "Ann. volatility": ret.std() * np.sqrt(252), "Beta (3y weekly vs S&P 500)": beta,
                       "Max drawdown (5y)": max_drawdown(hist), "Cost of equity": ke, "WACC": wacc})
 
    md = f"""# {name} ({T}) – Equity Research Report
 
**Rating: {rating}** | Blended target: {target:,.2f} | Price: {price:,.2f} | Implied upside: {upside:.1%}
 
> Data: Yahoo Finance via yfinance, pulled at run time. Rating is rules-based (BUY >+15%, SELL <-15% vs blended DCF/comps value). Not investment advice.
 
## 1. Business snapshot
{info.get('longBusinessSummary', 'n/a')}
 
Sector: {info.get('sector', 'n/a')} | Industry: {info.get('industry', 'n/a')}
 
## 2. Price & risk
![price](outputs/{T}_price.png)
 
{stats.to_frame('Value').to_markdown(floatfmt='.3f')}
 
## 3. Financials
![fin](outputs/{T}_financials.png)
 
{(fin / 1e9).round(2).to_markdown()}
 
*(USD bn, reporting currency)*  Revenue CAGR: {cagr(rev):.1%} | FCF CAGR: {cagr(fcf):.1%}
 
## 4. DCF valuation
{dcf_txt}
 
{sens.round(2).to_markdown() if sens is not None else ''}
 
Assumptions: risk-free = latest 10y Treasury yield ({rf:.2%}), ERP {ERP:.0%}, tax shield {TAX_RATE:.0%}, terminal growth {TERMINAL_G:.1%}.
 
## 5. Relative valuation
{comps.round(2).to_markdown()}
 
Peer-median trailing P/E × EPS → **{comps_ps:,.2f}**; peer-median forward P/E × forward EPS → **{comps_fwd:,.2f}**.
 
![ff](outputs/{T}_football_field.png)
 
## 6. Risks & limitations
- DCF is highly sensitive to WACC and terminal growth (see grid); one-stage FCF growth extrapolates history.
- Yahoo fundamentals can be restated or incomplete; verify against filings before relying on any figure.
- Peer set is user-chosen; multiples are point-in-time.
"""
    path = f"{OUT}/{T}_report.md"
    open(path, "w", encoding="utf-8").write(md)
    print(f"Done -> {path}  | Rating {rating}, target {target:,.2f} vs price {price:,.2f}")
 
 
if __name__ == "__main__":
    main()
 
