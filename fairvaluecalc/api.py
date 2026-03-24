"""
Fair Value Pro  -  FastAPI Backend v3.1
========================================
Starten:  uvicorn api:app --reload --port 8000
Oeffnen:  http://localhost:8000

Neu in v3.1:
  - Alle Bewertungsparameter als optionale Override-Parameter
  - Engine-Komponenten direkt aufgerufen (kein run_valuation Wrapper)
  - Volle Kontrolle ueber Annahmen per URL-Parameter
"""

import math
import os
from typing import Optional
import requests as req_lib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import numpy as np
import pandas as pd
import yfinance as yf

from fair_value_calculator import (
    StockData, WACCCalculator, Assumptions,
    ValuationEngine, QualityChecker,
)

app = FastAPI(title="Fair Value Pro API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Frontend ausliefern --------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def root():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index):
        return FileResponse(index, media_type="text/html")
    return {"message": "frontend/index.html nicht gefunden"}


# -------------------------------------------------------------------------
#  HELPERS
# -------------------------------------------------------------------------

def _safe(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(float(v), 6)
    if isinstance(v, np.ndarray):
        return [_safe(x) for x in v.tolist()]
    if isinstance(v, pd.Series):
        return [_safe(x) for x in v.tolist()]
    return v


def _fx(target: str, source: str = "USD") -> float:
    if target == source:
        return 1.0
    try:
        tk = yf.Ticker("EURUSD=X")
        h  = tk.history(period="2d")
        if not h.empty:
            rate = float(h["Close"].iloc[-1])
            return round(1 / rate, 6) if target == "EUR" else round(rate, 6)
    except Exception:
        pass
    return 1.0



def _get_identifiers(ticker: str) -> dict:
    """
    Fetch WKN and ISIN for a given ticker.

    WKN (Wertpapierkennnummer) is a 6-character German security identifier.
    It is ONLY natively available for securities with a German ISIN (DE...).
    For international securities, WKN cannot be derived from ISIN — they are
    assigned separately by Deutsche Boerse and not available in yfinance.

    Returns dict with keys: isin, wkn (wkn may be None for non-DE securities).
    """
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}

        # ISIN from yfinance
        isin = None
        try:
            raw_isin = tk.isin
            if raw_isin and isinstance(raw_isin, str) and len(raw_isin) >= 10:
                isin = raw_isin.strip()
        except Exception:
            pass

        # Fallback from info dict
        if not isin:
            isin = info.get("isin")

        # WKN: only extractable from German ISINs
        # German ISIN format: DE + 9 alphanumeric chars = 12 total
        # Positions 2-7 (0-indexed) = the 6-digit WKN
        wkn = None
        if isin and isinstance(isin, str):
            isin_up = isin.upper()
            if isin_up.startswith("DE") and len(isin_up) == 12:
                candidate = isin_up[2:8]
                # Valid WKN: 6 alphanumeric chars, not all zeros
                if candidate.isalnum() and candidate != "000000":
                    wkn = candidate

            # Swiss securities (CH...) also have a Swiss Valorennummer — skip for now
            # Austrian (AT...) use ISIN directly — skip

        # Check info dict for explicit WKN field (rare but possible)
        if not wkn:
            wkn = info.get("wkn") or info.get("WKN") or None

        return {"isin": isin, "wkn": wkn}
    except Exception:
        return {"isin": None, "wkn": None}


def _find_row(df, candidates):
    if df is None or df.empty:
        return None
    idx = {str(i).lower(): i for i in df.index}
    for c in candidates:
        m = idx.get(c.lower())
        if m is not None:
            return df.loc[m]
    return None


def _to_daily(pts: pd.Series, target_dates: pd.DatetimeIndex):
    if pts is None or pts.empty:
        return None
    pts = pts.sort_index()
    pts = pts[~pts.index.duplicated(keep="last")]
    rng = pd.date_range(
        start=min(pts.index.min(), target_dates.min()),
        end=max(pts.index.max(), target_dates.max()),
        freq="D",
    )
    daily  = pts.reindex(rng).ffill()
    result = daily.reindex(target_dates, method="nearest").ffill().bfill()
    return result if not result.isna().all() else None


def _fv_history(ticker, price_dates, target_pe, target_ev_ebitda,
                shares, debt, cash, fx):
    tk    = yf.Ticker(ticker)
    dates = pd.DatetimeIndex(price_dates)

    try:
        qi = tk.quarterly_income_stmt
        ni = _find_row(qi, ["Net Income", "Net Income Common Stockholders"])
        if ni is not None and shares > 0:
            t4  = ni.sort_index().rolling(4, min_periods=4).sum().dropna()
            eps = (t4 / shares)[t4 > 0]
            if len(eps) >= 2:
                fv = (eps * target_pe * fx).rename("fv")
                fv.index = pd.to_datetime(fv.index)
                d = _to_daily(fv, dates)
                if d is not None:
                    return d.tolist(), f"Trailing 4Q EPS x {target_pe:.0f}x KGV"
    except Exception:
        pass

    try:
        qcf   = tk.quarterly_cashflow
        opcf  = _find_row(qcf, ["Operating Cash Flow"])
        capex = _find_row(qcf, ["Capital Expenditure", "Capital Expenditures"])
        if opcf is not None and shares > 0:
            fcf = opcf.sort_index()
            if capex is not None:
                fcf = fcf.add(capex.sort_index(), fill_value=0)
            t4  = fcf.rolling(4, min_periods=4).sum().dropna()
            fps = (t4 / shares)[t4 > 0]
            if len(fps) >= 2:
                mult = target_pe * 0.85
                fv   = (fps * mult * fx).rename("fv")
                fv.index = pd.to_datetime(fv.index)
                d = _to_daily(fv, dates)
                if d is not None:
                    return d.tolist(), f"Trailing 4Q FCF/Aktie x {mult:.0f}x"
    except Exception:
        pass

    return None, "Keine Quartalsdaten verfuegbar"


def _build_response(d, dcf, rel, scenarios, roic, drivers, sens_df,
                    quality, fx_rate, currency, overrides_active, identifiers=None):
    """Gemeinsame Response-Formatierung fuer beide Valuation-Pfade."""

    def fv(v):
        return _safe((v or 0) * fx_rate)

    sens_table = None
    if sens_df is not None:
        sens_table = {
            "growth_rates": [_safe(g) for g in sens_df.index.tolist()],
            "wacc_rates":   [_safe(c) for c in sens_df.columns.tolist()],
            "values":       [[_safe((val or 0) * fx_rate)
                              for val in row] for _, row in sens_df.iterrows()],
        }

    growth_display = [
        {"year": t, "stage": s, "growth": _safe(g),
         "fcf": _safe(fcf_v), "pv": _safe(pv_v)}
        for t, s, g, fcf_v, pv_v in (dcf.get("growth_display") or [])
    ]

    return {
        "ticker":              d.ticker,
        "company":             d.company_name,
        "sector":              d.sector,
        "currency":            currency,
        "fx_rate":             fx_rate,
        "overrides_active":    overrides_active,
        "isin":                (identifiers or {}).get("isin"),
        "wkn":                 (identifiers or {}).get("wkn"),

        "price":               fv(d.current_price),
        "market_cap":          fv((d.current_price or 0) * (d.shares_outstanding or 0)),
        "shares_outstanding":  _safe(d.shares_outstanding),
        "total_debt":          _safe(d.total_debt),
        "cash":                _safe(d.cash),

        "dcf": {
            "fair_value":      fv(dcf.get("fair_value_per_share")),
            "enterprise_value":fv(dcf.get("enterprise_value")),
            "equity_value":    fv(dcf.get("equity_value")),
            "stage1_growth":   _safe(dcf.get("stage1_growth")),
            "terminal_growth": _safe(dcf.get("terminal_growth")),
            "discount_rate":   _safe(dcf.get("discount_rate")),
            "tv_pct":          _safe(dcf.get("pv_terminal_pct")),
            "fcf_0":           fv(dcf.get("fcf_0")),
            "growth_display":  growth_display,
        },

        "relative": {
            "pe_fair_value":        fv(rel.get("pe_fair_value")),
            "ev_ebitda_fair_value": fv(rel.get("ev_ebitda_fair_value")),
            "target_pe":            _safe(rel.get("target_pe")),
            "target_ev_ebitda":     _safe(rel.get("target_ev_ebitda")),
            "sector_source":        rel.get("sector_source", "default"),
            "eps_ttm":              fv(rel.get("eps_ttm")),
            "ebitda_ttm":           fv(rel.get("ebitda_ttm")),
        },

        "scenarios": {
            "bear": {
                "fair_value": fv(scenarios["bear"].get("fair_value_per_share")),
                "growth":     _safe(scenarios["bear"].get("growth_rate")),
                "wacc":       _safe(scenarios["bear"].get("discount_rate")),
            },
            "base": {
                "fair_value": fv(scenarios["base"].get("fair_value_per_share")),
                "growth":     _safe(scenarios["base"].get("growth_rate")),
                "wacc":       _safe(scenarios["base"].get("discount_rate")),
            },
            "bull": {
                "fair_value": fv(scenarios["bull"].get("fair_value_per_share")),
                "growth":     _safe(scenarios["bull"].get("growth_rate")),
                "wacc":       _safe(scenarios["bull"].get("discount_rate")),
            },
            "weighted": fv(scenarios.get("weighted_fair_value")),
        },

        "wacc": {
            "value": _safe(d.wacc_computed),
            "rf":    _safe(d.risk_free_rate),
            "beta":  _safe(d.beta),
            "erp":   _safe(d.equity_risk_premium),
            "ke":    _safe(d.cost_of_equity),
            "kd":    _safe(d.cost_of_debt_after_tax),
            "we":    _safe(d.weight_equity),
            "wd":    _safe(d.weight_debt),
        },

        "roic": {
            "available":        roic.get("available", False),
            "roic":             _safe(roic.get("roic")),
            "wacc":             _safe(roic.get("wacc")),
            "spread":           _safe(roic.get("spread")),
            "signal":           roic.get("signal"),
            "nopat":            fv(roic.get("nopat")),
            "invested_capital": fv(roic.get("invested_capital")),
        },

        "sensitivity": sens_table,
        "drivers":     {k: _safe(v) for k, v in drivers.items()},

        "fundamentals": {
            "revenue_ttm": fv(d.revenue_ttm),
            "fcf_ttm":     fv(d.fcf_ttm),
            "fcf_norm":    fv(d.fcf_normalized),
            "fcf_method":  d.fcf_norm_method,
            "ebitda_ttm":  fv(d.ebitda_ttm),
            "eps_ttm":     fv(d.eps_ttm),
            "debt":        fv(d.total_debt),
            "cash":        fv(d.cash),
            "fcf_margin":  _safe(d.fcf_margin),
            "rev_cagr":    _safe(d.revenue_cagr),
            "fcf_yield":   _safe(getattr(d, "fcf_yield", None)),
            "hist_pe_median": _safe(getattr(d, "hist_pe_median", None)),
            "hist_pe_fv":     fv(getattr(d, "hist_pe_fv", None)),
        },

        "quality": {
            "warnings": quality.warnings if quality else [],
            "errors":   quality.errors   if quality else [],
        },
    }


# -------------------------------------------------------------------------
#  ENDPOINTS
# -------------------------------------------------------------------------


# ── Commodity tickers ────────────────────────────────────────────────
COMMODITY_TICKERS = {
    # Edelmetalle
    "GC=F":  "Gold (Futures)",        "GLD":  "Gold ETF (SPDR)",
    "SI=F":  "Silber (Futures)",      "SLV":  "Silber ETF (iShares)",
    "PL=F":  "Platin (Futures)",      "PA=F": "Palladium (Futures)",
    # Energie
    "CL=F":  "Rohöl WTI (Futures)",   "BZ=F": "Rohöl Brent (Futures)",
    "NG=F":  "Erdgas (Futures)",      "HO=F": "Heizöl (Futures)",
    "RB=F":  "Benzin RBOB (Futures)",
    # Agrar
    "ZC=F":  "Mais (Futures)",        "ZW=F": "Weizen (Futures)",
    "ZS=F":  "Sojabohnen (Futures)",  "ZO=F": "Hafer (Futures)",
    "KC=F":  "Kaffee (Futures)",      "CC=F": "Kakao (Futures)",
    "SB=F":  "Zucker (Futures)",      "CT=F": "Baumwolle (Futures)",
    # Metalle / Industrie
    "HG=F":  "Kupfer (Futures)",      "ALI=F":"Aluminium (Futures)",
    "ZN=F":  "Zink (Futures)",
    # ETFs
    "DJP":   "Rohstoff-ETF (DJP)",   "GSG":  "Rohstoff-ETF (GSG)",
    "PDBC":  "Rohstoff-ETF (PDBC)",
}

def _is_commodity(ticker: str, info: dict) -> bool:
    """Detect commodity futures and ETFs."""
    t = ticker.upper()
    if t in COMMODITY_TICKERS:
        return True
    # Futures always end in =F
    if t.endswith("=F"):
        return True
    qt = (info.get("quoteType") or "").upper()
    return qt in ("FUTURE", "PHYSICALCURRENCY")


def _commodity_valuation(ticker: str, currency: str) -> dict:
    """
    Commodity valuation using:
    1. 200-day MA (mean reversion fair value)
    2. 52-week range positioning
    3. Seasonal / historical percentile
    4. Backwardation / Contango signal for futures
    """
    tk   = yf.Ticker(ticker)
    info = tk.info or {}
    fx   = _fx(currency, info.get("currency", "USD"))

    price   = (info.get("currentPrice") or info.get("regularMarketPrice") or
               info.get("previousClose") or 0) * fx
    name    = COMMODITY_TICKERS.get(ticker.upper(),
              info.get("shortName") or info.get("longName") or ticker)
    w52_high = (info.get("fiftyTwoWeekHigh") or 0) * fx
    w52_low  = (info.get("fiftyTwoWeekLow")  or 0) * fx
    vol      = (info.get("regularMarketVolume") or 0)

    # ── Historical MA ────────────────────────────────────────────────
    try:
        hist  = tk.history(period="3y")["Close"]
        ma200 = float(hist.rolling(200, min_periods=60).mean().iloc[-1]) * fx if len(hist) >= 60 else None
        ma50  = float(hist.rolling(50,  min_periods=20).mean().iloc[-1]) * fx if len(hist) >= 20 else None
        # 5-year historical mean (if data available)
        hist5 = tk.history(period="5y")["Close"]
        mean5y = float(hist5.mean()) * fx if len(hist5) > 100 else None
        # 52-week percentile
        pct_52w = float((price/fx - hist.iloc[-252:].min()) /
                         (hist.iloc[-252:].max() - hist.iloc[-252:].min()) * 100) if len(hist) >= 252 else None
    except Exception:
        ma200, ma50, mean5y, pct_52w = None, None, None, None

    # ── Fair value scenarios ─────────────────────────────────────────
    # Bear: 200-day MA (mean reversion)
    bear_fv = ma200 if ma200 else price * 0.80
    # Base: 5-year mean or midpoint of 52w range
    if mean5y:
        base_fv = mean5y
    elif w52_high and w52_low:
        base_fv = (w52_high + w52_low) / 2
    else:
        base_fv = ma200 or price
    # Bull: 52-week high (momentum target)
    bull_fv = w52_high if w52_high else price * 1.20

    return {
        "ticker":           ticker,
        "company":          name,
        "sector":           "commodity",
        "currency":         currency,
        "fx_rate":          fx,
        "overrides_active": False,
        "is_crypto":        False,
        "is_commodity":     True,
        "isin":             None,
        "wkn":              None,

        "price":              price,
        "market_cap":         0,
        "shares_outstanding": 0,
        "total_debt":         0,
        "cash":               0,

        "dcf": {
            "fair_value":      base_fv,
            "enterprise_value":0,
            "equity_value":    0,
            "stage1_growth":   None,
            "terminal_growth": None,
            "discount_rate":   None,
            "tv_pct":          None,
            "fcf_0":           None,
            "growth_display":  [],
        },

        "relative": {
            "pe_fair_value":        ma200,
            "ev_ebitda_fair_value": mean5y,
            "target_pe":            None,
            "target_ev_ebitda":     None,
            "sector_source":        "commodity-model",
            "eps_ttm":              None,
            "ebitda_ttm":           vol,
        },

        "scenarios": {
            "bear": {"fair_value": bear_fv, "growth": None, "wacc": None},
            "base": {"fair_value": base_fv, "growth": None, "wacc": None},
            "bull": {"fair_value": bull_fv, "growth": None, "wacc": None},
            "weighted": bear_fv * 0.25 + base_fv * 0.50 + bull_fv * 0.25,
        },

        "wacc": {
            "value": None, "rf": None, "beta": info.get("beta"),
            "erp": None, "ke": None, "kd": None, "we": 1.0, "wd": 0.0,
        },

        "roic": {
            "available": False,
            "signal":    "N/A – Rohstoff",
        },

        "sensitivity": None,
        "drivers":     {},

        "fundamentals": {
            "revenue_ttm":  None,
            "fcf_ttm":      None,
            "fcf_norm":     None,
            "fcf_method":   ("Rohstoffe haben keinen Cashflow — DCF nicht anwendbar. "
                             "Szenarien basieren auf 200-Tage-MA (Bär), "
                             "5-Jahres-Mittel / 52W-Mitte (Basis), "
                             "52W-Hoch (Bulle)."),
            "ebitda_ttm":   None,
            "eps_ttm":      None,
            "debt":         0,
            "cash":         0,
            "fcf_margin":   None,
            "rev_cagr":     None,
        },

        "quality": {
            "warnings": [
                "Rohstoffe haben keine fundamentalen Kennzahlen (FCF, Gewinn, EBITDA).",
                f"Szenarien: Bär = 200-Tage-MA ({_safe(bear_fv):.2f} {currency}), "
                f"Basis = 5J-Mittel ({_safe(base_fv):.2f} {currency}), "
                f"Bulle = 52W-Hoch ({_safe(bull_fv):.2f} {currency}).",
                f"Positionierung im 52-Wochen-Band: "
                f"{pct_52w:.0f}% (0% = Tief, 100% = Hoch)." if pct_52w else
                "52-Wochen-Positionierung nicht verfügbar.",
            ],
            "errors": [],
        },

        "commodity_extra": {
            "ma200":     ma200,
            "ma50":      ma50,
            "mean5y":    mean5y,
            "w52_high":  w52_high,
            "w52_low":   w52_low,
            "pct_52w":   pct_52w,
            "vol":       vol,
        },
    }

# ── Crypto detection ─────────────────────────────────────────────────
CRYPTO_TICKERS = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "BNB-USD": "BNB",
    "SOL-USD": "Solana",  "XRP-USD": "XRP",      "ADA-USD": "Cardano",
    "DOGE-USD":"Dogecoin","DOT-USD": "Polkadot",  "AVAX-USD":"Avalanche",
    "MATIC-USD":"Polygon","LINK-USD":"Chainlink",  "LTC-USD": "Litecoin",
    "BCH-USD": "Bitcoin Cash","ATOM-USD":"Cosmos", "UNI-USD": "Uniswap",
}

def _is_crypto(ticker: str, info: dict) -> bool:
    """Detect crypto assets by ticker suffix or quoteType."""
    t = ticker.upper()
    if t in CRYPTO_TICKERS:
        return True
    if t.endswith("-USD") or t.endswith("-EUR") or t.endswith("-USDT"):
        return True
    qt = (info.get("quoteType") or "").upper()
    return qt == "CRYPTOCURRENCY"


def _crypto_valuation(ticker: str, currency: str) -> dict:
    """
    Crypto-specific valuation using:
    1. Stock-to-Flow model (scarcity proxy for BTC)
    2. Network Value to Metcalfe (NVM) ratio
    3. 200-day MA as technical fair value proxy
    4. Simple momentum model
    Returns structured dict matching standard valuation format.
    """
    import math as _math
    tk   = yf.Ticker(ticker)
    info = tk.info or {}
    fx   = _fx(currency, info.get("currency", "USD"))

    price    = (info.get("currentPrice") or info.get("regularMarketPrice") or
                info.get("previousClose") or 0) * fx
    mcap     = (info.get("marketCap") or 0) * fx
    circ     = info.get("circulatingSupply") or 0
    total    = info.get("totalSupply") or circ
    vol_24h  = (info.get("volume24Hr") or info.get("regularMarketVolume") or 0) * fx
    w52_high = (info.get("fiftyTwoWeekHigh") or 0) * fx
    w52_low  = (info.get("fiftyTwoWeekLow") or 0) * fx
    name     = info.get("shortName") or info.get("longName") or ticker

    # ── Historical price for MA calculation ──────────────────────────
    try:
        hist = tk.history(period="1y")["Close"]
        ma200 = float(hist.rolling(200, min_periods=30).mean().iloc[-1]) * fx if len(hist) >= 30 else None
        ma50  = float(hist.rolling(50,  min_periods=10).mean().iloc[-1]) * fx if len(hist) >= 10 else None
    except Exception:
        ma200, ma50 = None, None

    # ── Scenario fair values ──────────────────────────────────────────
    # Bear: 200-day MA (mean reversion)
    bear_fv = ma200 if ma200 else price * 0.55
    # Base: midpoint between 200-day MA and 52-week high
    base_fv = ((ma200 or price) + w52_high) / 2 if w52_high else price
    # Bull: 52-week high extended by avg bull run (1.5x)
    bull_fv = w52_high * 1.5 if w52_high else price * 2.0

    # ── MoS vs base ──────────────────────────────────────────────────
    mos = (base_fv - price) / base_fv if base_fv else 0

    # ── NVM-like ratio (simple: mcap / sqrt(active addresses proxy)) ─
    # We use volume as Metcalfe proxy (higher volume = more network usage)
    nvm_note = "N/A"
    if vol_24h and mcap and vol_24h > 0:
        nvm = mcap / (vol_24h ** 0.5)
        nvm_note = f"{nvm/1e6:.1f}M"

    # ── Stock-to-Flow (BTC only) ───────────────────────────────────────
    s2f_price = None
    if "BTC" in ticker.upper() and circ > 0:
        # Approximate: halvings reduce issuance ~50% every 4yr
        # Current issuance post-2024-halving: ~164,250 BTC/yr
        annual_new = 164250
        s2f = circ / annual_new
        # PlanB model: price ≈ exp(14.6) * s2f^3.3  (USD)
        s2f_price_usd = (_math.exp(14.6) * (s2f ** 3.3)) if s2f > 0 else None
        if s2f_price_usd:
            s2f_price = s2f_price_usd * fx

    return {
        "ticker":           ticker,
        "company":          name,
        "sector":           "cryptocurrency",
        "currency":         currency,
        "fx_rate":          fx,
        "overrides_active": False,
        "is_crypto":        True,
        "isin":             None,
        "wkn":              None,

        "price":              price,
        "market_cap":         mcap,
        "shares_outstanding": circ,   # = circulating supply
        "total_debt":         0,
        "cash":               0,

        "dcf": {
            "fair_value":      base_fv,
            "enterprise_value":mcap,
            "equity_value":    mcap,
            "stage1_growth":   None,
            "terminal_growth": None,
            "discount_rate":   None,
            "tv_pct":          None,
            "fcf_0":           None,
            "growth_display":  [],
        },

        "relative": {
            "pe_fair_value":        s2f_price,   # Stock-to-Flow as special metric
            "ev_ebitda_fair_value": ma200,        # 200-day MA as alternative
            "target_pe":            None,
            "target_ev_ebitda":     None,
            "sector_source":        "crypto-model",
            "eps_ttm":              None,
            "ebitda_ttm":           vol_24h,      # 24h volume as activity metric
        },

        "scenarios": {
            "bear": {"fair_value": bear_fv, "growth": None, "wacc": None},
            "base": {"fair_value": base_fv, "growth": None, "wacc": None},
            "bull": {"fair_value": bull_fv, "growth": None, "wacc": None},
            "weighted": bear_fv * 0.25 + base_fv * 0.50 + bull_fv * 0.25,
        },

        "wacc": {
            "value": None, "rf": None, "beta": info.get("beta"),
            "erp": None, "ke": None, "kd": None, "we": 1.0, "wd": 0.0,
        },

        "roic": {
            "available":        True,
            "roic":             None,
            "wacc":             None,
            "spread":           None,
            "signal":           "N/A – Kryptowährung",
            "nopat":            None,
            "invested_capital": None,
        },

        "sensitivity": None,
        "drivers":     {},

        "fundamentals": {
            "revenue_ttm":  vol_24h,
            "fcf_ttm":      None,
            "fcf_norm":     None,
            "fcf_method":   "Kryptowährungen haben keinen freien Cashflow – "
                            "DCF-Modell nicht anwendbar. Szenarien basieren auf "
                            "200-Tage-MA, 52-Wochen-Hoch und Stock-to-Flow.",
            "ebitda_ttm":   None,
            "eps_ttm":      None,
            "debt":         0,
            "cash":         0,
            "fcf_margin":   None,
            "rev_cagr":     None,
        },

        "quality": {
            "warnings": [
                "Kryptowährungen haben keine klassischen Fundamentaldaten (FCF, Gewinn, Cashflow).",
                "Die Szenarien basieren auf technischen Indikatoren: 200-Tage-MA (Bär), "
                "Mittelwert MA/52W-Hoch (Basis), 52W-Hoch × 1.5 (Bulle).",
                "Für Bitcoin wird zusätzlich das Stock-to-Flow-Modell (PlanB) angezeigt.",
                "Kryptowährungs-Bewertung ist hochspekulativ. Kein Modell ist verlässlich.",
            ],
            "errors": [],
        },

        "crypto_extra": {
            "ma200":         ma200,
            "ma50":          ma50,
            "s2f_price":     s2f_price,
            "w52_high":      w52_high,
            "w52_low":       w52_low,
            "vol_24h":       vol_24h,
            "circ_supply":   circ,
            "total_supply":  total,
        },
    }


@app.get("/search")
def search(q: str):
    try:
        url  = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}"
        resp = req_lib.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        qs   = resp.json().get("quotes", [])
        results = [
            {
              "symbol": x["symbol"],
              "name":   x.get("shortname") or x.get("longname", x["symbol"]),
              "type":   x.get("quoteType", ""),
              "exch":   x.get("exchDisp", ""),
              "is_commodity": x.get("quoteType","").upper() in ("FUTURE","PHYSICALCURRENCY")
                              or x["symbol"].upper().endswith("=F"),
              "is_crypto":    x.get("quoteType","").upper() == "CRYPTOCURRENCY",
            }
            for x in qs if "symbol" in x
        ]
        # Also inject matching commodity tickers from our list
        q_up = q.strip().upper()
        for sym, cname in COMMODITY_TICKERS.items():
            if q_up in sym or q_up in cname.upper():
                if not any(r["symbol"] == sym for r in results):
                    results.insert(0, {
                        "symbol": sym, "name": cname,
                        "type": "FUTURE", "exch": "Futures",
                        "is_commodity": True, "is_crypto": False,
                    })
        return results[:10]
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/valuation/{ticker}")
def valuation(
    ticker: str,
    currency: str = "USD",

    # -- DCF-Annahmen -------------------------------------------------------
    stage1_growth:    Optional[float] = None,   # z.B. 0.12 fuer 12%
    wacc_override:    Optional[float] = None,   # direkter WACC-Override
    terminal_growth:  Optional[float] = None,   # z.B. 0.025
    forecast_years:   Optional[int]   = None,   # 5, 7, 10, 15
    fcf_base_bn:      Optional[float] = None,   # FCF-Basis in Mrd. (z.B. 33.7)

    # -- WACC-Komponenten ---------------------------------------------------
    rf_override:      Optional[float] = None,   # Risikofreier Zins
    beta_override:    Optional[float] = None,   # Beta
    erp_override:     Optional[float] = None,   # Equity Risk Premium

    # -- Relative Bewertung -------------------------------------------------
    target_pe:        Optional[float] = None,   # Ziel-KGV
    target_ev_ebitda: Optional[float] = None,   # Ziel-EV/EBITDA

    # -- Szenario-Anpassungen (NEU) -----------------------------------------
    bear_factor:      Optional[float] = None,   # Bär g1-Faktor, z.B. 0.40
    bull_factor:      Optional[float] = None,   # Bulle g1-Faktor, z.B. 1.40
    bear_weight:      Optional[float] = None,   # Bär-Gewichtung, z.B. 0.25
    base_weight:      Optional[float] = None,   # Basis-Gewichtung, z.B. 0.50
    bull_weight:      Optional[float] = None,   # Bulle-Gewichtung, z.B. 0.25

    # -- Komposit-Gewichtung (NEU) ------------------------------------------
    dcf_weight:       Optional[float] = None,   # Gewicht DCF, z.B. 0.50
    pe_weight:        Optional[float] = None,   # Gewicht KGV, z.B. 0.25
    ev_weight:        Optional[float] = None,   # Gewicht EV/EBITDA, z.B. 0.25

    # -- Steuerrate & WACC-Struktur (NEU) ------------------------------------
    tax_rate:         Optional[float] = None,   # Steuerrate, z.B. 0.21
    kd_override:      Optional[float] = None,   # Fremdkapitalkosten (Kd) direkt
    stage2_type:      Optional[str]   = None,   # "linear" (default) oder "step"
):
    """
    Volle Bewertung mit optionalen Overrides.
    Alle Parameter sind optional - ohne Angabe werden Werte automatisch berechnet.
    """
    try:
        # 0. Detect crypto — route to separate handler
        _pre_info = {}
        try:
            _pre_tk = yf.Ticker(ticker.upper())
            _pre_info = _pre_tk.info or {}
        except Exception:
            pass
        if _is_commodity(ticker, _pre_info):
            return _commodity_valuation(ticker.upper(), currency)
        if _is_crypto(ticker, _pre_info):
            return _crypto_valuation(ticker.upper(), currency)

        # 1. Daten laden
        data = StockData(ticker.upper(), verbose=False)
        data.fetch()

        # 2. WACC-Komponenten overschreiben VOR WACC-Berechnung
        if rf_override    is not None: data.risk_free_rate      = rf_override
        if beta_override   is not None: data.beta               = beta_override
        if erp_override    is not None: data.equity_risk_premium = erp_override

        # 3. FCF normalisieren
        fcf_norm, fcf_method   = data.compute_normalized_fcf()
        data.fcf_normalized    = fcf_norm
        data.fcf_norm_method   = fcf_method

        # 3b. FCF-Yield berechnen
        try:
            _mcap = (data.current_price or 0) * (data.shares_outstanding or 0)
            _fcf  = data.fcf_ttm or 0
            data.fcf_yield = (_fcf / _mcap) if (_mcap > 0 and _fcf > 0) else None
        except Exception:
            data.fcf_yield = None

        # 3c. Historisches 5J-Median-KGV berechnen
        try:
            _tk2  = yf.Ticker(ticker.upper())
            _info2 = _tk2.info or {}
            _hist_prices = _tk2.history(period="5y")["Close"]
            _qi   = _tk2.quarterly_income_stmt
            _ni_r = _find_row(_qi, ["Net Income", "Net Income Common Stockholders"])
            _sh   = data.shares_outstanding or 1
            if _ni_r is not None and _sh > 0 and not _hist_prices.empty:
                _t4_ni = _ni_r.sort_index().rolling(4, min_periods=4).sum().dropna()
                _eps_q  = (_t4_ni / _sh)[_t4_ni > 0]
                _eps_q.index = pd.to_datetime(_eps_q.index)
                _dates_5y = pd.DatetimeIndex(_hist_prices.index).tz_localize(None)
                _eps_daily = _to_daily(_eps_q, _dates_5y)
                if _eps_daily is not None:
                    _prices_aligned = _hist_prices.values[:len(_eps_daily)]
                    _eps_arr = _eps_daily.values[:len(_prices_aligned)]
                    _valid   = (_eps_arr > 0) & (_prices_aligned > 0)
                    _pe_hist = _prices_aligned[_valid] / _eps_arr[_valid]
                    if len(_pe_hist) > 30:
                        import numpy as _np2
                        _pe_hist_trim = _pe_hist[(_pe_hist > 5) & (_pe_hist < 200)]
                        _med_pe = float(_np2.median(_pe_hist_trim)) if len(_pe_hist_trim) > 10 else None
                        data.hist_pe_median = _med_pe
                        data.hist_pe_fv = (_med_pe * data.eps_ttm) if (_med_pe and data.eps_ttm) else None
                    else:
                        data.hist_pe_median = None
                        data.hist_pe_fv     = None
                else:
                    data.hist_pe_median = None
                    data.hist_pe_fv     = None
            else:
                data.hist_pe_median = None
                data.hist_pe_fv     = None
        except Exception:
            data.hist_pe_median = None
            data.hist_pe_fv     = None

        # 4. WACC berechnen
        wacc_calc = WACCCalculator(data)
        wacc_calc.compute()

        # 5. WACC direkt ueberschreiben (nach CAPM-Berechnung)
        if wacc_override is not None:
            data.wacc_computed = wacc_override

        # 6. Annahmen erstellen
        assumptions = Assumptions(data)

        # 7. Annahmen-Overrides anwenden
        overrides_applied = []
        if stage1_growth    is not None:
            assumptions.stage1_growth        = stage1_growth
            overrides_applied.append("stage1_growth")
        if wacc_override     is not None:
            assumptions.base_discount_rate   = wacc_override
            overrides_applied.append("wacc")
        if terminal_growth   is not None:
            assumptions.base_terminal_growth = terminal_growth
            overrides_applied.append("terminal_growth")
        if forecast_years    is not None:
            assumptions.forecast_years       = forecast_years
            overrides_applied.append("forecast_years")
        if target_pe         is not None:
            assumptions.target_pe_ratio      = target_pe
            overrides_applied.append("target_pe")
        if target_ev_ebitda  is not None:
            assumptions.target_ev_ebitda     = target_ev_ebitda
            overrides_applied.append("target_ev_ebitda")
        if fcf_base_bn       is not None:
            assumptions.fcf_anchor           = fcf_base_bn * 1e9
            overrides_applied.append("fcf_base")

        # -- Scenario factor overrides: change bear/bull spread ---------------
        if bear_factor is not None or bull_factor is not None:
            bf  = bear_factor if bear_factor is not None else 0.40
            blf = bull_factor if bull_factor is not None else 1.40
            g1  = assumptions.stage1_growth
            r   = assumptions.base_discount_rate
            tg  = assumptions.base_terminal_growth
            assumptions.bear["stage1_growth"]  = max(g1 * bf,  0.005)
            assumptions.bear["discount_rate"]  = min(r + 0.020, 0.22)
            assumptions.bear["terminal_growth"]= max(tg - 0.010, 0.005)
            assumptions.bull["stage1_growth"]  = min(g1 * blf, 0.40)
            assumptions.bull["discount_rate"]  = max(r - 0.010, 0.06)
            assumptions.bull["terminal_growth"]= min(tg + 0.010, 0.04)
            overrides_applied.append("scenario_factors")

        # -- Scenario probability overrides -----------------------------------
        if any(x is not None for x in [bear_weight, base_weight, bull_weight]):
            bw  = bear_weight if bear_weight is not None else 0.25
            bsw = base_weight if base_weight is not None else 0.50
            blw = bull_weight if bull_weight is not None else 0.25
            total_w = bw + bsw + blw
            if total_w > 0:
                assumptions.bear["probability"] = bw  / total_w
                assumptions.base["probability"] = bsw / total_w
                assumptions.bull["probability"] = blw / total_w
            overrides_applied.append("scenario_weights")

        # -- Steuerrate --------------------------------------------------------
        _tax = tax_rate if tax_rate is not None else 0.21
        if tax_rate is not None:
            overrides_applied.append("tax_rate")
        # Store on data object for use in engine
        data._tax_rate_override = _tax

        # -- Kd override -------------------------------------------------------
        if kd_override is not None:
            data.cost_of_debt_after_tax = kd_override * (1 - _tax)
            # recompute WACC if not already overridden directly
            if wacc_override is None:
                we = data.weight_equity or 1.0
                wd = data.weight_debt   or 0.0
                ke = data.cost_of_equity or 0.10
                data.wacc_computed = ke * we + kd_override * (1 - _tax) * wd
                assumptions.base_discount_rate = data.wacc_computed
            overrides_applied.append("kd")

        # -- Stage-2 type -------------------------------------------------------
        if stage2_type is not None:
            assumptions.stage2_type = stage2_type
            overrides_applied.append("stage2_type")

        # -- Komposit-Gewichtung für Response ----------------------------------
        _dcf_w = dcf_weight if dcf_weight is not None else 0.50
        _pe_w  = pe_weight  if pe_weight  is not None else 0.25
        _ev_w  = ev_weight  if ev_weight  is not None else 0.25
        total_comp = _dcf_w + _pe_w + _ev_w
        if total_comp > 0:
            _dcf_w /= total_comp; _pe_w /= total_comp; _ev_w /= total_comp
        if any(x is not None for x in [dcf_weight, pe_weight, ev_weight]):
            overrides_applied.append("composite_weights")

        # Szenarien nach Overrides neu aufbauen
        if overrides_applied:
            assumptions._rebuild_scenarios()

        # 8. Qualitaetspruefung
        quality = QualityChecker(data, assumptions)
        quality.run_all()
        # Warnungen werden zurueckgegeben, aber blockieren nicht bei Override-Modus

        # 9. Engine laufen lassen
        engine  = ValuationEngine(data, assumptions)
        dcf_res = engine.two_stage_dcf()
        base_fv = dcf_res.get("fair_value_per_share")

        res = {
            "dcf":               dcf_res,
            "relative":          engine.relative_valuation(),
            "scenarios":         engine.scenario_analysis(),
            "sensitivity":       engine.sensitivity_table(),
            "sensitivity_impact":engine.sensitivity_impact(base_fv) if base_fv else {},
            "roic":              engine.roic_check(),
            "quality":           quality,
            "data":              data,
        }

    except Exception as e:
        raise HTTPException(500, f"Bewertung fehlgeschlagen: {e}")

    d           = res["data"]
    fx_rate     = _fx(currency, d.currency or "USD")
    identifiers = _get_identifiers(ticker.upper())

    # Collect all model assumptions for transparency panel
    transparency = {
        # DCF inputs
        "fcf_norm_method":   data.fcf_norm_method,
        "fcf_norm_value":    _safe((data.fcf_normalized or 0) * fx_rate),
        "stage1_growth_src": "override" if "stage1_growth" in overrides_applied else "auto (hist. CAGR)",
        "wacc_src":          "override" if "wacc" in overrides_applied else "auto (CAPM)",
        "terminal_src":      "override" if "terminal_growth" in overrides_applied else "auto",
        "forecast_years":    assumptions.forecast_years,
        "stage2_type":       getattr(assumptions, "stage2_type", "linear"),
        # WACC decomposition
        "rf_src":            "override" if "rf" in [x[:2] for x in overrides_applied] else "auto (10yr treasury)",
        "beta_src":          "override" if beta_override is not None else "auto (Yahoo Finance)",
        "erp_src":           "override" if erp_override  is not None else "auto (Damodaran)",
        "kd_src":            "override" if "kd" in overrides_applied else "auto (interest/debt)",
        "tax_rate":          _safe(tax_rate if tax_rate is not None else 0.21),
        "tax_src":           "override" if "tax_rate" in overrides_applied else "auto (21 % US-Standard)",
        # Scenario settings
        "bear_factor":       _safe(bear_factor if bear_factor is not None else 0.40),
        "bull_factor":       _safe(bull_factor if bull_factor is not None else 1.40),
        "bear_weight":       _safe(assumptions.bear.get("probability", 0.25)),
        "base_weight":       _safe(assumptions.base.get("probability", 0.50)),
        "bull_weight":       _safe(assumptions.bull.get("probability", 0.25)),
        "scenario_src":      "override" if "scenario" in " ".join(overrides_applied) else "auto",
        # Composite weights
        "dcf_weight":        _safe(_dcf_w),
        "pe_weight":         _safe(_pe_w),
        "ev_weight":         _safe(_ev_w),
        "composite_src":     "override" if "composite_weights" in overrides_applied else "auto (50/25/25)",
        # FCF normalization
        "fcf_years_used":    "3–5yr avg, outlier-removed",
    }

    resp = _build_response(
        d        = d,
        dcf      = res["dcf"],
        rel      = res["relative"],
        scenarios= res["scenarios"],
        roic     = res.get("roic", {}),
        drivers  = res.get("sensitivity_impact", {}),
        sens_df  = res.get("sensitivity"),
        quality  = res.get("quality"),
        fx_rate  = fx_rate,
        currency = currency,
        overrides_active = len(overrides_applied) > 0,
        identifiers      = identifiers,
    )
    resp["transparency"]       = transparency
    resp["composite_weights"]  = {"dcf": _safe(_dcf_w), "pe": _safe(_pe_w), "ev": _safe(_ev_w)}
    resp["overrides_list"]     = overrides_applied
    return resp


@app.get("/history/{ticker}")
def history(ticker: str, period: str = "10y", currency: str = "USD"):
    try:
        tk   = yf.Ticker(ticker.upper())
        hist = tk.history(period=period)
        if hist.empty:
            raise HTTPException(404, "No price data")
        hist         = hist.reset_index()
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
        fx_rate      = _fx(currency, tk.info.get("currency", "USD"))
        prices       = [round(float(c) * fx_rate, 4) for c in hist["Close"]]
        dates        = [str(d.date()) for d in hist["Date"]]
        return {"dates": dates, "prices": prices, "fx_rate": fx_rate}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/fvhistory/{ticker}")
def fvhistory(ticker: str, period: str = "10y", currency: str = "USD",
              target_pe: float = 22.0, target_ev_ebitda: float = 14.0,
              shares: float = 1e9, debt: float = 0, cash: float = 0,
              bear_fv: float = 0, bull_fv: float = 0):
    try:
        tk   = yf.Ticker(ticker.upper())
        hist = tk.history(period=period)
        if hist.empty:
            return {"dates": [], "fv": [], "method": "no data",
                    "bear_band": [], "bull_band": [], "median_pe_fv": []}
        hist         = hist.reset_index()
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
        src_currency = tk.info.get("currency", "USD")
        fx_rate      = _fx(currency, src_currency)
        dates_idx    = pd.DatetimeIndex(hist["Date"])
        fv_vals, method = _fv_history(
            ticker, dates_idx, target_pe, target_ev_ebitda,
            shares, debt, cash, fx_rate,
        )
        dates = [str(d.date()) for d in hist["Date"]]

        def _clean(vals):
            if vals is None:
                return [None] * len(dates)
            return [
                round(float(v), 4)
                if v is not None and not math.isnan(float(v)) else None
                for v in vals
            ]

        fv_clean = _clean(fv_vals)

        # Bear/Bull band: constant lines at scenario values
        bear_band = [round(bear_fv * fx_rate, 4)] * len(dates) if bear_fv else []
        bull_band = [round(bull_fv * fx_rate, 4)] * len(dates) if bull_fv else []

        # Median-PE historical fair value line
        median_pe_fv = []
        try:
            qi   = tk.quarterly_income_stmt
            ni_r = _find_row(qi, ["Net Income", "Net Income Common Stockholders"])
            if ni_r is not None and shares > 0:
                t4    = ni_r.sort_index().rolling(4, min_periods=4).sum().dropna()
                eps_q = (t4 / shares)[t4 > 0]
                eps_q.index = pd.to_datetime(eps_q.index)
                # compute rolling 5y median PE from price history
                price_s = hist.set_index("Date")["Close"]
                eps_daily = _to_daily(eps_q, dates_idx)
                if eps_daily is not None:
                    prices_arr = price_s.values
                    eps_arr    = eps_daily.values
                    pe_arr     = np.where(
                        (eps_arr > 0) & (prices_arr > 0),
                        prices_arr / eps_arr, np.nan
                    )
                    # rolling 252*5 day median PE (or all available)
                    pe_series = pd.Series(pe_arr)
                    roll_med  = pe_series.rolling(252 * 5, min_periods=252).median()
                    # fair value = rolling median PE × trailing EPS
                    med_pe_fv = roll_med * eps_daily.values
                    med_pe_fv = np.where(
                        (roll_med > 5) & (roll_med < 200) & (eps_daily.values > 0),
                        med_pe_fv * fx_rate, np.nan
                    )
                    median_pe_fv = [
                        round(float(v), 4) if not math.isnan(v) else None
                        for v in med_pe_fv
                    ]
        except Exception:
            median_pe_fv = []

        return {
            "dates":        dates,
            "fv":           fv_clean,
            "method":       method,
            "bear_band":    bear_band,
            "bull_band":    bull_band,
            "median_pe_fv": median_pe_fv,
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/financials/{ticker}")
def financials(ticker: str, currency: str = "USD"):
    """
    Fetch annual Umsatz (Revenue), EBIT and Nettogewinn (Net Income)
    for up to 10 years, returned as parallel arrays for charting.
    Values are returned in Billions of the requested currency.
    """
    try:
        tk   = yf.Ticker(ticker.upper())
        info = tk.info or {}
        fx   = _fx(currency, info.get("currency", "USD"))

        inc = tk.income_stmt   # annual, columns = dates newest→oldest
        if inc is None or inc.empty:
            return {"years": [], "revenue": [], "ebit": [], "net_income": [],
                    "currency": currency, "error": "Keine Daten verfügbar"}

        def _row(candidates):
            idx = {str(i).lower(): i for i in inc.index}
            for c in candidates:
                m = idx.get(c.lower())
                if m is not None:
                    return inc.loc[m]
            return None

        rev_row = _row(["Total Revenue", "Revenue"])
        ebit_row= _row(["EBIT", "Operating Income",
                         "Earnings Before Interest And Taxes"])
        ni_row  = _row(["Net Income", "Net Income Common Stockholders",
                         "Net Income Including Noncontrolling Interests"])

        # Columns = datetime index, sort oldest→newest, take up to 10
        cols = sorted(inc.columns)[-10:]

        def _extract(row):
            if row is None:
                return [None] * len(cols)
            vals = []
            for c in cols:
                try:
                    v = float(row.loc[c])
                    vals.append(round(v * fx / 1e9, 3) if not math.isnan(v) else None)
                except Exception:
                    vals.append(None)
            return vals

        years     = [str(c.year) if hasattr(c, 'year') else str(c)[:4] for c in cols]
        revenue   = _extract(rev_row)
        ebit      = _extract(ebit_row)
        net_income= _extract(ni_row)

        return {
            "years":      years,
            "revenue":    revenue,
            "ebit":       ebit,
            "net_income": net_income,
            "currency":   currency,
            "unit":       "Mrd.",
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Quality Check Endpoint ────────────────────────────────────────────
@app.get("/quality-check/{ticker}")
def quality_check(ticker: str, currency: str = "USD"):
    """
    Score a stock against Qualitätsaktien criteria.
    Returns per-criterion: value, passes (bool), score (0/1/2), label.
    """
    import math as _math
    try:
        tk   = yf.Ticker(ticker.upper())
        info = tk.info or {}
        fx   = _fx(currency, info.get("currency", "USD"))

        inc  = tk.income_stmt          # annual, newest→oldest
        bal  = tk.balance_sheet
        cf   = tk.cashflow

        results = {}

        # ── Helper: CAGR from series ──────────────────────────────────
        def cagr(series, years):
            """series sorted oldest→newest, take last (years+1) points"""
            s = [v for v in series if v and not _math.isnan(v) and v > 0]
            if len(s) < 2:
                return None
            s = s[-min(len(s), years + 1):]
            if len(s) < 2 or s[0] <= 0:
                return None
            n = len(s) - 1
            return (s[-1] / s[0]) ** (1 / n) - 1

        def get_row(df, candidates):
            if df is None or df.empty:
                return []
            idx = {str(i).lower(): i for i in df.index}
            for c in candidates:
                m = idx.get(c.lower())
                if m is not None:
                    row = df.loc[m]
                    cols = sorted(row.index)
                    return [_safe(row.loc[c2]) for c2 in cols]
            return []

        def criterion(key, label, value, threshold, unit="", higher_better=True,
                       fmt=None, description=""):
            if value is None:
                return {"key": key, "label": label, "value": None,
                        "display": "N/A", "threshold": threshold,
                        "passes": None, "score": 0,
                        "description": description, "unit": unit}
            passes = (value >= threshold) if higher_better else (value <= threshold)
            if fmt == "pct":
                display = f"{value*100:.1f} %"
                thr_display = f"{threshold*100:.0f} %"
            elif fmt == "x":
                display = f"{value:.1f}×"
                thr_display = f"{threshold:.0f}×"
            else:
                display = f"{value:.2f}"
                thr_display = str(threshold)
            return {"key": key, "label": label, "value": round(float(value), 4),
                    "display": display, "threshold": threshold,
                    "threshold_display": thr_display,
                    "passes": passes, "score": 2 if passes else 0,
                    "description": description, "unit": unit}

        # ── Revenue & EBIT series ─────────────────────────────────────
        rev_s  = get_row(inc, ["Total Revenue", "Revenue"])
        ebit_s = get_row(inc, ["EBIT", "Operating Income",
                                "Earnings Before Interest And Taxes"])
        ni_s   = get_row(inc, ["Net Income", "Net Income Common Stockholders"])

        rev_cagr_10  = cagr(rev_s,  10)
        rev_cagr_3   = cagr(rev_s,  3)
        ebit_cagr_10 = cagr(ebit_s, 10)
        ebit_cagr_3  = cagr(ebit_s, 3)

        # ── Net Debt / EBIT ───────────────────────────────────────────
        total_debt = info.get("totalDebt") or 0
        cash_equiv = info.get("totalCash") or info.get("cashAndCashEquivalents") or 0
        net_debt   = total_debt - cash_equiv
        ebit_ttm   = info.get("ebitda") or 0  # fallback
        # Try from income stmt
        if ebit_s:
            v = [x for x in ebit_s if x and not _math.isnan(x)]
            if v:
                ebit_ttm = v[-1]
        net_debt_ebit = (net_debt / ebit_ttm) if ebit_ttm and ebit_ttm > 0 else None

        # ── Profit continuity: no negative NI in last 10 years ────────
        ni_vals = [x for x in ni_s[-10:] if x is not None and not _math.isnan(x)]
        ni_continuous = (all(v > 0 for v in ni_vals) and len(ni_vals) >= 5) if ni_vals else None

        # ── EBIT Drawdown: max YoY decline in last 10 years ──────────
        ebit_vals = [x for x in ebit_s[-11:] if x is not None and not _math.isnan(x) and x != 0]
        max_drawdown = None
        if len(ebit_vals) >= 2:
            drops = []
            for i in range(1, len(ebit_vals)):
                if ebit_vals[i-1] > 0:
                    drops.append((ebit_vals[i-1] - ebit_vals[i]) / ebit_vals[i-1])
            if drops:
                max_drawdown = max(drops)  # worst single-year EBIT decline

        # ── ROE & ROCE ────────────────────────────────────────────────
        roe  = info.get("returnOnEquity")
        roic = info.get("returnOnAssets")  # fallback
        # Better ROIC from our own calc
        nopat = ebit_ttm * (1 - 0.21) if ebit_ttm else None
        equity = info.get("bookValue") or 0
        shares = info.get("sharesOutstanding") or 1
        equity_total = equity * shares
        debt_book    = total_debt
        invested_cap = equity_total + debt_book - cash_equiv
        roic_calc = (nopat / invested_cap) if (nopat and invested_cap > 0) else None

        # ── Expected return: FCF yield + revenue growth ───────────────
        price = (info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        mcap  = info.get("marketCap") or 0
        fcf_ttm = info.get("freeCashflow") or 0
        fcf_yield   = fcf_ttm / mcap if mcap > 0 and fcf_ttm > 0 else None
        exp_return  = (fcf_yield + rev_cagr_3) if (fcf_yield and rev_cagr_3) else None

        # ── Moat proxy: gross margin stability ───────────────────────
        gross_margin = info.get("grossMargins")
        op_margin    = info.get("operatingMargins")

        # ── Dividend continuity ───────────────────────────────────────
        div_yield    = info.get("dividendYield")
        div_rate     = info.get("dividendRate")
        div_5y       = info.get("fiveYearAvgDividendYield")

        # ── Build results ─────────────────────────────────────────────
        criteria = [
            criterion("rev_cagr_10",  "Umsatzwachstum (10J)",    rev_cagr_10,   0.05, fmt="pct",
                      description="Jährliches Umsatzwachstum über 10 Jahre. Zeigt langfristige Wachstumskraft."),
            criterion("rev_cagr_3",   "Umsatzwachstum (3J)",     rev_cagr_3,    0.05, fmt="pct",
                      description="Jährliches Umsatzwachstum über die letzten 3 Jahre. Aktuelle Dynamik."),
            criterion("ebit_cagr_10", "EBIT-Wachstum (10J)",     ebit_cagr_10,  0.05, fmt="pct",
                      description="Wachstum des operativen Gewinns über 10 Jahre. Kernprofitabilität."),
            criterion("ebit_cagr_3",  "EBIT-Wachstum (3J)",      ebit_cagr_3,   0.05, fmt="pct",
                      description="Wachstum des operativen Gewinns der letzten 3 Jahre."),
            criterion("net_debt_ebit","Nettoverschuldung/EBIT",   net_debt_ebit, 4.0,
                      fmt="x", higher_better=False,
                      description="Nettoverschuldung im Verhältnis zum EBIT. Unter 4× gilt als solide Bilanz."),
            {
              "key": "ni_continuous", "label": "Gewinnkontinuität (10J)",
              "value": 1 if ni_continuous else 0,
              "display": "✓ Durchgehend positiv" if ni_continuous else ("✗ Verlustjahre vorhanden" if ni_continuous is False else "N/A"),
              "threshold": 1, "threshold_display": "Kein Verlustjahr",
              "passes": ni_continuous, "score": 2 if ni_continuous else 0,
              "description": "Kein einziges Verlustjahr in den letzten 10 Jahren. Merkmal stabiler Geschäftsmodelle.",
            },
            criterion("ebit_drawdown","EBIT-Drawdown max. (10J)", max_drawdown,  0.50,
                      fmt="pct", higher_better=False,
                      description="Größter einmaliger EBIT-Rückgang in einem Jahr. Unter 50 % = widerstandsfähig."),
            criterion("roe",          "Eigenkapitalrendite (ROE)",roe,           0.15, fmt="pct",
                      description="Gewinn je eingesetztem Eigenkapital. Über 15 % = effizientes Management."),
            criterion("roic",         "Kapitalrendite (ROIC/ROCE)",roic_calc,    0.15, fmt="pct",
                      description="Rendite auf das gesamte eingesetzte Kapital. Über 15 % = starker Wettbewerbsvorteil."),
            criterion("exp_return",   "Renditeerwartung",         exp_return,    0.10, fmt="pct",
                      description="FCF-Rendite + Umsatzwachstum (3J). Schätzt die zu erwartende Gesamtrendite."),
        ]

        # ── Additional context (not scored) ──────────────────────────
        context = {
            "gross_margin":  _safe(gross_margin),
            "op_margin":     _safe(op_margin),
            "div_yield":     _safe(div_yield),
            "div_rate":      _safe(div_rate),
            "div_5y_avg":    _safe(div_5y),
            "fcf_yield":     _safe(fcf_yield),
            "net_debt_abs":  _safe(net_debt * fx / 1e9),
            "ebit_ttm_bn":   _safe(ebit_ttm * fx / 1e9),
        }

        passed = sum(1 for c in criteria if c.get("passes") is True)
        total  = sum(1 for c in criteria if c.get("passes") is not None)
        score  = round(passed / total * 100) if total > 0 else 0

        grade = ("A" if score >= 80 else "B" if score >= 60 else
                 "C" if score >= 40 else "D")

        return {
            "ticker":   ticker.upper(),
            "criteria": criteria,
            "context":  context,
            "summary": {
                "passed":  passed,
                "total":   total,
                "score":   score,
                "grade":   grade,
            }
        }
    except Exception as e:
        raise HTTPException(500, str(e))
