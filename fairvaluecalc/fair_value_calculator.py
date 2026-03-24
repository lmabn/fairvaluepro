"""
╔══════════════════════════════════════════════════════════════════════╗
║         PROFESSIONAL FAIR VALUE CALCULATOR  v2.0                    ║
║         Multi-Model Intrinsic Value Engine for Stocks               ║
║                                                                      ║
║  Models:  Two-Stage DCF · Dynamic WACC · Relative Valuation         ║
║           Scenario Analysis · ROIC Check · Sensitivity Analysis     ║
║  Data:    Yahoo Finance (yfinance) with full manual fallback        ║
║                                                                      ║
║  v2.0 Upgrades vs v1.0:                                              ║
║   • Two-stage DCF  (high growth stage → linear decay → terminal)   ║
║   • Dynamic WACC   (rf + β × ERP; leverage-adjusted cost of debt)  ║
║   • FCF Normalisation (3–5yr avg → margin-based → last FCF)        ║
║   • Quality checks & blocking-error engine                          ║
║   • Industry-aware P/E and EV/EBITDA defaults (Damodaran table)    ║
║   • Realistic Bear/Bull scenarios (absolute bps deltas)             ║
║   • ROIC vs WACC value-creation check                               ║
║   • Sensitivity driver identification                                ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
    pip install yfinance pandas numpy tabulate colorama
    python fair_value_calculator.py            # interactive
    python fair_value_calculator.py AAPL       # direct ticker
    python fair_value_calculator.py AAPL --no-interactive
    python fair_value_calculator.py --batch AAPL MSFT GOOGL

    Programmatic:
        from fair_value_calculator import run_valuation
        results = run_valuation("AAPL")
"""

# ──────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────
import sys
import math
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    print("ERROR: pandas and numpy are required.  Run: pip install pandas numpy")
    sys.exit(1)

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────
#  DISPLAY HELPERS  (backward-compatible with v1)
# ──────────────────────────────────────────────────────────────────

def _c(text, color=""):
    if not COLORAMA_AVAILABLE or not color:
        return str(text)
    colors = {
        "green":   Fore.GREEN,
        "red":     Fore.RED,
        "yellow":  Fore.YELLOW,
        "cyan":    Fore.CYAN,
        "white":   Fore.WHITE,
        "bold":    Style.BRIGHT,
        "magenta": Fore.MAGENTA,
    }
    return colors.get(color, "") + str(text) + Style.RESET_ALL


def _fmt_currency(value, currency="$", decimals=2):
    if value is None:
        return "N/A"
    try:
        if math.isnan(value) or math.isinf(value):
            return "N/A"
    except TypeError:
        return "N/A"
    sign  = "-" if value < 0 else ""
    abs_v = abs(value)
    if abs_v >= 1e12:
        return f"{sign}{currency}{abs_v/1e12:.{decimals}f}T"
    elif abs_v >= 1e9:
        return f"{sign}{currency}{abs_v/1e9:.{decimals}f}B"
    elif abs_v >= 1e6:
        return f"{sign}{currency}{abs_v/1e6:.{decimals}f}M"
    else:
        return f"{sign}{currency}{abs_v:,.{decimals}f}"


def _fmt_pct(value, decimals=1):
    if value is None:
        return "N/A"
    try:
        if math.isnan(value) or math.isinf(value):
            return "N/A"
    except TypeError:
        return "N/A"
    return f"{value*100:.{decimals}f}%"


def _fmt_x(value, decimals=1):
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}x"


def _print_header(text, width=72, char="="):
    print("\n" + _c(char * width, "cyan"))
    print(_c(f"  {text}", "bold"))
    print(_c(char * width, "cyan"))


def _print_subheader(text, width=72, char="-"):
    print("\n" + _c(f"  {text}", "yellow"))
    print(_c(char * width, "white"))


def _print_table(data, headers, fmt="simple"):
    if TABULATE_AVAILABLE:
        print(tabulate(data, headers=headers, tablefmt=fmt,
                       floatfmt=".2f", numalign="right"))
    else:
        col_w = [max(len(str(h)),
                     max((len(str(r[i])) for r in data), default=0))
                 for i, h in enumerate(headers)]
        row_fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
        print(row_fmt.format(*headers))
        print("  " + "-" * (sum(col_w) + 2 * len(col_w)))
        for row in data:
            print(row_fmt.format(*[str(x) for x in row]))


def _warn(msg):
    print(_c(f"  [WARN]  {msg}", "yellow"))


def _error(msg):
    print(_c(f"  [ERROR] {msg}", "red"))


def _ok(msg):
    print(_c(f"  [OK]    {msg}", "green"))


# ──────────────────────────────────────────────────────────────────
#  SECTOR MULTIPLES  (Damodaran-inspired median ranges)
# ──────────────────────────────────────────────────────────────────

SECTOR_MULTIPLES = {
    "technology":             {"pe": 28.0, "ev_ebitda": 20.0},
    "communication services": {"pe": 22.0, "ev_ebitda": 14.0},
    "consumer discretionary": {"pe": 22.0, "ev_ebitda": 13.0},
    "consumer staples":       {"pe": 20.0, "ev_ebitda": 13.0},
    "health care":            {"pe": 22.0, "ev_ebitda": 14.0},
    "financials":             {"pe": 14.0, "ev_ebitda":  9.0},
    "industrials":            {"pe": 18.0, "ev_ebitda": 12.0},
    "materials":              {"pe": 16.0, "ev_ebitda": 10.0},
    "energy":                 {"pe": 12.0, "ev_ebitda":  6.0},
    "utilities":              {"pe": 16.0, "ev_ebitda":  9.0},
    "real estate":            {"pe": 30.0, "ev_ebitda": 18.0},
    "default":                {"pe": 20.0, "ev_ebitda": 12.0},
}

# Equity Risk Premium by currency/region (Damodaran, approx.)
ERP_BY_CURRENCY = {
    "USD": 0.055, "EUR": 0.057, "GBP": 0.053, "JPY": 0.060,
    "CHF": 0.052, "CAD": 0.054, "AUD": 0.058, "CNY": 0.065,
    "INR": 0.075, "BRL": 0.085, "default": 0.055,
}

# 10-yr risk-free rates (approximate mid-2024)
RF_BY_CURRENCY = {
    "USD": 0.043, "EUR": 0.028, "GBP": 0.040, "JPY": 0.009,
    "CHF": 0.008, "CAD": 0.037, "AUD": 0.042, "CNY": 0.025,
    "INR": 0.070, "BRL": 0.105, "default": 0.040,
}

# Sector beta fallbacks (when yfinance returns no beta)
SECTOR_BETA = {
    "technology": 1.25, "communication services": 1.10,
    "consumer discretionary": 1.10, "consumer staples": 0.75,
    "health care": 0.80, "financials": 1.15, "industrials": 1.05,
    "materials": 1.10, "energy": 1.00, "utilities": 0.60,
    "real estate": 0.90, "default": 1.00,
}


# ──────────────────────────────────────────────────────────────────
#  1.  DATA LAYER
# ──────────────────────────────────────────────────────────────────

class StockData:
    """
    Fetches and normalises financial data.

    v2.0 additions
    --------------
    - beta, sector, currency stored for WACC
    - multi-year FCF history for normalisation
    - invested_capital + nopat for ROIC
    - wacc_computed written back after WACCCalculator runs
    """

    def __init__(self, ticker, verbose=True):
        self.ticker  = ticker.upper().strip()
        self.verbose = verbose
        self.source  = "manual"

        # Identity
        self.company_name       = ticker
        self.currency           = "USD"
        self.sector             = None
        self.industry           = None

        # Price & structure
        self.current_price      = None
        self.shares_outstanding = None
        self.total_debt         = None
        self.cash               = None
        self.market_cap         = None

        # P&L / CF
        self.revenue_ttm        = None
        self.ebitda_ttm         = None
        self.ebit_ttm           = None
        self.net_income_ttm     = None
        self.fcf_ttm            = None
        self.eps_ttm            = None

        # WACC inputs
        self.beta                = None
        self.risk_free_rate      = None
        self.equity_risk_premium = None
        self.tax_rate            = None

        # ROIC inputs
        self.invested_capital    = None
        self.nopat               = None

        # Historicals
        self.revenue_hist        = None
        self.fcf_hist            = None
        self.net_income_hist     = None

        # Derived
        self.revenue_cagr        = None
        self.fcf_cagr            = None
        self.fcf_margin          = None
        self.fcf_normalized      = None
        self.fcf_norm_method     = None

        # WACC output (written by WACCCalculator)
        self.wacc_computed           = None
        self.cost_of_equity          = None
        self.cost_of_debt_after_tax  = None
        self.weight_equity           = None
        self.weight_debt             = None

    # ── Entry point ─────────────────────────────────────────────

    def fetch(self):
        if not YFINANCE_AVAILABLE:
            if self.verbose:
                _warn("yfinance not installed — switching to manual input.")
            return self.manual_input()
        try:
            if self.verbose:
                print(f"  Fetching {_c(self.ticker, 'cyan')} from Yahoo Finance ...")
            self._fetch_yfinance()
            self.source = "yfinance"
            if self.verbose:
                _ok("Data fetched successfully.")
        except Exception as exc:
            if not self.verbose:
                raise ValueError(f"yfinance failed: {exc}")
            _warn(f"yfinance failed ({exc}). Switching to manual input.")
            return self.manual_input()
        return self

    # ── yfinance ────────────────────────────────────────────────

    def _fetch_yfinance(self):
        # Versions-erkennende Session: Browser-Header für Cloud-Server (Render),
        # automatischer Fallback für neue yfinance-Versionen die requests.Session ablehnen
        tk = None
        try:
            import requests as _req
            from requests.adapters import HTTPAdapter as _HA
            _s = _req.Session()
            _s.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://finance.yahoo.com/",
            })
            _s.mount("https://", _HA(max_retries=2))
            tk = yf.Ticker(self.ticker, session=_s)
        except Exception:
            tk = yf.Ticker(self.ticker)

        info = tk.info or {}

        self.company_name    = info.get("longName") or info.get("shortName", self.ticker)
        self.currency        = info.get("currency", "USD")
        self.sector          = (info.get("sector") or "").lower().strip() or None
        self.industry        = (info.get("industry") or "").lower().strip() or None
        self.current_price   = (info.get("currentPrice")
                                or info.get("regularMarketPrice")
                                or info.get("previousClose"))
        self.shares_outstanding = (info.get("sharesOutstanding")
                                   or info.get("impliedSharesOutstanding"))
        self.total_debt      = info.get("totalDebt", 0) or 0
        self.cash            = info.get("totalCash") or info.get("cashAndCashEquivalents", 0) or 0
        self.market_cap      = info.get("marketCap")
        self.ebitda_ttm      = info.get("ebitda")
        self.eps_ttm         = info.get("trailingEps") or info.get("forwardEps")
        self.beta            = info.get("beta")

        cur = self.currency.upper()
        self.risk_free_rate      = RF_BY_CURRENCY.get(cur,  RF_BY_CURRENCY["default"])
        self.equity_risk_premium = ERP_BY_CURRENCY.get(cur, ERP_BY_CURRENCY["default"])

        cf  = tk.cashflow
        inc = tk.income_stmt
        bal = tk.balance_sheet

        if inc is not None and not inc.empty:
            rev_row  = self._find_row(inc, ["Total Revenue", "Revenue"])
            ni_row   = self._find_row(inc, ["Net Income", "Net Income Common Stockholders"])
            ebit_row = self._find_row(inc, ["EBIT", "Operating Income",
                                             "Earnings Before Interest And Taxes"])
            tax_row  = self._find_row(inc, ["Tax Provision", "Income Tax Expense"])

            if rev_row is not None:
                self.revenue_hist = rev_row.sort_index()
                self.revenue_ttm  = float(self.revenue_hist.iloc[-1])
            if ni_row is not None:
                self.net_income_hist = ni_row.sort_index()
                self.net_income_ttm  = float(self.net_income_hist.iloc[-1])
            if ebit_row is not None:
                self.ebit_ttm = float(ebit_row.sort_index().iloc[-1])
            if ebit_row is not None and tax_row is not None:
                ebit_v = float(ebit_row.sort_index().iloc[-1])
                tax_v  = float(tax_row.sort_index().iloc[-1])
                if ebit_v > 0:
                    self.tax_rate = max(0.0, min(tax_v / ebit_v, 0.40))

        if cf is not None and not cf.empty:
            opcf_row  = self._find_row(cf, ["Operating Cash Flow",
                                             "Total Cash From Operating Activities"])
            capex_row = self._find_row(cf, ["Capital Expenditure",
                                             "Capital Expenditures"])
            if opcf_row is not None:
                if capex_row is not None:
                    fcf_s = opcf_row.add(capex_row, fill_value=0)
                else:
                    fcf_s = opcf_row.copy()
                self.fcf_hist = fcf_s.sort_index()
                self.fcf_ttm  = float(self.fcf_hist.iloc[-1])

        if bal is not None and not bal.empty:
            eq_row = self._find_row(bal, ["Stockholders Equity",
                                           "Total Stockholder Equity",
                                           "Common Stock Equity"])
            if eq_row is not None:
                eq = float(eq_row.sort_index().iloc[-1])
                self.invested_capital = eq + self.total_debt - self.cash

        if self.ebit_ttm:
            tax = self.tax_rate or 0.21
            self.nopat = self.ebit_ttm * (1 - tax)

        if not self.eps_ttm and self.net_income_ttm and self.shares_outstanding:
            self.eps_ttm = self.net_income_ttm / self.shares_outstanding

        if self.fcf_ttm and self.revenue_ttm:
            self.fcf_margin = self.fcf_ttm / self.revenue_ttm

        self.revenue_cagr = self._compute_cagr(self.revenue_hist)
        self.fcf_cagr     = self._compute_cagr(self.fcf_hist)

        missing = [f for f, v in [
            ("Current Price",      self.current_price),
            ("Shares Outstanding", self.shares_outstanding),
        ] if not v]
        if missing:
            raise ValueError(f"Missing critical data: {', '.join(missing)}")
        if not self.fcf_ttm and not self.revenue_ttm:
            raise ValueError("Both FCF and Revenue are missing.")

    # ── FCF Normalisation ────────────────────────────────────────

    def compute_normalized_fcf(self):
        """
        Three-level fallback hierarchy:
        1. 3-5 year average FCF (outlier-filtered if >= 4 years)
        2. Revenue × normalised FCF margin
        3. Last reported FCF (TTM)
        Returns (value, method_description_string).
        """
        # Method 1: multi-year average
        if self.fcf_hist is not None:
            clean = self.fcf_hist.dropna()
            if len(clean) >= 3:
                window = clean.iloc[-5:]
                if len(window) >= 4:
                    mean = window.mean()
                    std  = window.std()
                    if std > 0:
                        filtered = window[abs(window - mean) <= 2 * std]
                        if len(filtered) >= 2:
                            avg = float(filtered.mean())
                            return avg, f"{len(filtered)}-year avg FCF (outliers removed)"
                avg = float(window.mean())
                return avg, f"{len(window)}-year avg FCF"

        # Method 2: revenue × normalised margin
        if self.revenue_ttm and self.revenue_ttm > 0:
            margins = []
            if self.fcf_hist is not None and self.revenue_hist is not None:
                for period, fcf_v in self.fcf_hist.dropna().items():
                    if period in self.revenue_hist.index:
                        rev_v = float(self.revenue_hist.loc[period])
                        if rev_v > 0:
                            margins.append(fcf_v / rev_v)
            if margins:
                norm_m = float(np.mean(margins))
                return (self.revenue_ttm * norm_m,
                        f"Revenue x avg FCF margin ({norm_m*100:.1f}%)")
            if self.fcf_margin:
                return (self.revenue_ttm * self.fcf_margin,
                        f"Revenue x current FCF margin ({self.fcf_margin*100:.1f}%)")

        # Method 3: last reported
        if self.fcf_ttm:
            return self.fcf_ttm, "Last reported FCF (TTM)"

        return None, "unavailable"

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _find_row(df, candidates):
        idx_lower = {str(i).lower(): i for i in df.index}
        for c in candidates:
            match = idx_lower.get(c.lower())
            if match is not None:
                return df.loc[match]
        return None

    @staticmethod
    def _compute_cagr(series):
        if series is None or len(series) < 2:
            return None
        vals = series.dropna()
        if len(vals) < 2:
            return None
        start, end = float(vals.iloc[0]), float(vals.iloc[-1])
        n = len(vals) - 1
        if start <= 0 or end <= 0 or n < 1:
            return None
        return (end / start) ** (1 / n) - 1

    # ── Manual input ─────────────────────────────────────────────

    def manual_input(self):
        print(_c("\n  -- Manual Data Entry Mode --", "yellow"))
        print("  Fields marked * are optional (press Enter to skip).\n")

        def _ask(prompt, cast=float, required=True, default=None):
            while True:
                raw = input(f"    {prompt}: ").strip()
                if not raw:
                    if not required and default is not None:
                        return default
                    if not required:
                        return None
                    print("    [Required field]")
                    continue
                try:
                    return cast(raw)
                except ValueError:
                    print("    [Invalid – try again]")

        self.company_name       = input(f"    Company Name [{self.ticker}]: ").strip() or self.ticker
        self.currency           = input("    Currency [USD]: ").strip().upper() or "USD"
        self.sector             = input("    Sector (e.g. Technology) *: ").strip().lower() or None
        self.current_price      = _ask("Current Stock Price")
        self.shares_outstanding = _ask("Shares Outstanding (e.g. 1.5e10)")
        self.fcf_ttm            = _ask("Free Cash Flow TTM *", required=False)
        self.revenue_ttm        = _ask("Revenue TTM *", required=False)
        self.ebitda_ttm         = _ask("EBITDA TTM *", required=False)
        self.ebit_ttm           = _ask("EBIT / Operating Income TTM *", required=False)
        self.eps_ttm            = _ask("EPS TTM *", required=False)
        self.total_debt         = _ask("Total Debt *", required=False, default=0)
        self.cash               = _ask("Cash & Equivalents *", required=False, default=0)
        self.beta               = _ask("Beta (e.g. 1.2) *", required=False)
        self.tax_rate           = _ask("Effective Tax Rate (e.g. 0.21) *",
                                        required=False, default=0.21)
        cur = self.currency.upper()
        self.risk_free_rate      = _ask(
            f"Risk-Free Rate [{RF_BY_CURRENCY.get(cur, 0.040):.1%}] *",
            required=False, default=RF_BY_CURRENCY.get(cur, RF_BY_CURRENCY["default"]))
        self.equity_risk_premium = _ask(
            f"Equity Risk Premium [{ERP_BY_CURRENCY.get(cur, 0.055):.1%}] *",
            required=False, default=ERP_BY_CURRENCY.get(cur, ERP_BY_CURRENCY["default"]))
        self.revenue_cagr = _ask("5-yr Revenue CAGR (e.g. 0.10) *", required=False)
        self.fcf_cagr     = _ask("5-yr FCF CAGR (e.g. 0.12) *", required=False)

        if self.revenue_ttm and self.fcf_ttm:
            self.fcf_margin = self.fcf_ttm / self.revenue_ttm
        if self.ebit_ttm:
            self.nopat = self.ebit_ttm * (1 - (self.tax_rate or 0.21))
        if self.total_debt and self.cash is not None:
            self.invested_capital = self.total_debt - self.cash

        self.source = "manual"
        return self

    # ── Summary print ────────────────────────────────────────────

    def print_summary(self):
        _print_subheader(f"COMPANY DATA SUMMARY  [{self.source.upper()}]")
        rows = [
            ["Company",              self.company_name],
            ["Ticker",               self.ticker],
            ["Sector",               self.sector or "N/A"],
            ["Currency",             self.currency],
            ["Current Price",        _fmt_currency(self.current_price)],
            ["Shares Outstanding",   _fmt_currency(self.shares_outstanding, currency="")],
            ["Market Cap",           _fmt_currency((self.current_price or 0) *
                                                    (self.shares_outstanding or 0))],
            ["Revenue (TTM)",        _fmt_currency(self.revenue_ttm)],
            ["FCF (TTM, raw)",       _fmt_currency(self.fcf_ttm)],
            ["FCF (normalised)",     _fmt_currency(self.fcf_normalized)],
            ["FCF Norm. Method",     self.fcf_norm_method or "N/A"],
            ["EBITDA (TTM)",         _fmt_currency(self.ebitda_ttm)],
            ["EBIT (TTM)",           _fmt_currency(self.ebit_ttm)],
            ["EPS (TTM)",            _fmt_currency(self.eps_ttm)],
            ["Total Debt",           _fmt_currency(self.total_debt)],
            ["Cash",                 _fmt_currency(self.cash)],
            ["Beta",                 f"{self.beta:.2f}" if self.beta else "N/A"],
            ["FCF Margin",           _fmt_pct(self.fcf_margin)],
            ["Revenue CAGR (hist.)", _fmt_pct(self.revenue_cagr)],
            ["FCF CAGR (hist.)",     _fmt_pct(self.fcf_cagr)],
        ]
        _print_table(rows, ["Field", "Value"])


# ──────────────────────────────────────────────────────────────────
#  1b.  WACC CALCULATOR  (NEW)
# ──────────────────────────────────────────────────────────────────

class WACCCalculator:
    """
    CAPM-based WACC with leverage-adjusted cost of debt.

    Cost of Equity  = rf + beta x ERP
    Cost of Debt    = rf + credit_spread  (leverage-based proxy)
    WACC            = Ke x We + Kd(1-t) x Wd

    Results written back into data object for downstream use.
    """

    def __init__(self, data):
        self.data = data

    def compute(self):
        d   = self.data
        rf  = d.risk_free_rate  or RF_BY_CURRENCY.get(d.currency, 0.040)
        erp = d.equity_risk_premium or ERP_BY_CURRENCY.get(d.currency, 0.055)

        # Beta: use fetched, else sector fallback
        beta = d.beta
        if beta is None or (isinstance(beta, float) and (math.isnan(beta) or beta <= 0)):
            beta = SECTOR_BETA.get(d.sector or "default", SECTOR_BETA["default"])
            d.beta = beta

        # Cost of equity (CAPM), clipped to [6%, 25%]
        ke = rf + beta * erp
        ke = round(max(0.06, min(ke, 0.25)), 4)

        # Capital weights
        mc    = d.market_cap or ((d.current_price or 0) * (d.shares_outstanding or 0))
        debt  = d.total_debt or 0
        total = mc + debt

        if total <= 0:
            d.cost_of_equity          = ke
            d.cost_of_debt_after_tax  = 0.0
            d.weight_equity           = 1.0
            d.weight_debt             = 0.0
            d.wacc_computed           = ke
            return ke

        we = mc   / total
        wd = debt / total

        # Cost of debt: leverage-based credit spread
        de = debt / mc if mc > 0 else 0
        if de < 0.10:
            spread = 0.010
        elif de < 0.50:
            spread = 0.020
        elif de < 1.00:
            spread = 0.035
        else:
            spread = 0.060

        kd_pre   = rf + spread
        tax      = d.tax_rate or 0.21
        kd_after = kd_pre * (1 - tax)

        wacc = round(max(0.05, min(ke * we + kd_after * wd, 0.25)), 4)

        d.cost_of_equity          = ke
        d.cost_of_debt_after_tax  = kd_after
        d.weight_equity           = we
        d.weight_debt             = wd
        d.wacc_computed           = wacc
        return wacc

    def print_breakdown(self):
        d = self.data
        _print_subheader("WACC CALCULATION")
        rows = [
            ["Risk-Free Rate (rf)",          _fmt_pct(d.risk_free_rate)],
            ["Beta (beta)",                  f"{d.beta:.2f}" if d.beta else "N/A"],
            ["Equity Risk Premium (ERP)",    _fmt_pct(d.equity_risk_premium)],
            ["Cost of Equity  Ke = rf+b*ERP",_fmt_pct(d.cost_of_equity)],
            ["Cost of Debt (after-tax) Kd",  _fmt_pct(d.cost_of_debt_after_tax)],
            ["Weight Equity  We",            _fmt_pct(d.weight_equity)],
            ["Weight Debt    Wd",            _fmt_pct(d.weight_debt)],
            ["------------------",           "---------------"],
            ["WACC  (Ke*We + Kd*Wd)",        _c(_fmt_pct(d.wacc_computed), "cyan")],
        ]
        _print_table(rows, ["Component", "Value"])


# ──────────────────────────────────────────────────────────────────
#  1c.  QUALITY CHECKER  (NEW)
# ──────────────────────────────────────────────────────────────────

class QualityChecker:
    """
    Automatic diagnostics.  Accumulates warnings and fatal errors.
    run_all() returns True if safe to proceed with DCF.
    """

    def __init__(self, data, assumptions):
        self.data     = data
        self.asmp     = assumptions
        self.warnings = []
        self.errors   = []

    def run_all(self):
        self._check_fcf()
        self._check_growth()
        self._check_terminal_vs_wacc()
        self._check_fcf_margin()
        self._check_leverage()
        self._check_wacc_bounds()
        return len(self.errors) == 0

    def _check_fcf(self):
        fcf = self.data.fcf_normalized or self.data.fcf_ttm
        if fcf is None:
            self.errors.append("FCF is unavailable — DCF cannot be computed.")
        elif fcf < 0:
            self.warnings.append(
                f"Normalised FCF is negative ({_fmt_currency(fcf)}). "
                "DCF result will be negative — use scenario analysis with caution.")
        elif self.data.revenue_ttm and self.data.revenue_ttm > 0:
            margin = fcf / self.data.revenue_ttm
            if margin > 0.50:
                self.warnings.append(
                    f"FCF margin is very high ({_fmt_pct(margin)}). "
                    "Margins above 50% are unusual — verify the data.")

    def _check_growth(self):
        g = self.asmp.stage1_growth
        if g > 0.50:
            self.errors.append(
                f"Stage-1 growth ({_fmt_pct(g)}) exceeds 50%. "
                "Please enter a realistic rate.")
        elif g > 0.35:
            self.warnings.append(
                f"Stage-1 growth is {_fmt_pct(g)} — aggressive. "
                "Few companies sustain >35% for 5+ years.")

    def _check_terminal_vs_wacc(self):
        r  = self.data.wacc_computed or self.asmp.base_discount_rate
        tg = self.asmp.base_terminal_growth
        if tg >= r:
            self.errors.append(
                f"Terminal growth ({_fmt_pct(tg)}) >= WACC ({_fmt_pct(r)}). "
                "Gordon Growth formula breaks down.  Reduce terminal growth.")
        elif tg > 0.04:
            self.warnings.append(
                f"Terminal growth ({_fmt_pct(tg)}) > 4%. "
                "Long-run growth should stay below nominal GDP (~2.5-3.5%).")

    def _check_fcf_margin(self):
        if self.data.fcf_margin is not None and self.data.fcf_margin < -0.05:
            self.warnings.append(
                f"FCF margin significantly negative ({_fmt_pct(self.data.fcf_margin)}). "
                "Company burns cash — watch debt capacity carefully.")

    def _check_leverage(self):
        d = self.data
        if d.total_debt and d.ebitda_ttm and d.ebitda_ttm > 0:
            nd_ebitda = (d.total_debt - (d.cash or 0)) / d.ebitda_ttm
            if nd_ebitda > 5.0:
                self.warnings.append(
                    f"Net Debt / EBITDA = {nd_ebitda:.1f}x — high leverage. "
                    "Equity value is highly sensitive to EV changes.")

    def _check_wacc_bounds(self):
        w = self.data.wacc_computed
        if w is not None:
            if w < 0.05:
                self.warnings.append(f"WACC ({_fmt_pct(w)}) < 5%. Verify inputs.")
            elif w > 0.20:
                self.warnings.append(
                    f"WACC ({_fmt_pct(w)}) > 20%. Typical only for micro-caps / EM.")

    def print_report(self):
        if not self.warnings and not self.errors:
            _ok("All quality checks passed.")
            return
        for w in self.warnings:
            _warn(w)
        for e in self.errors:
            _error(e)


# ──────────────────────────────────────────────────────────────────
#  2.  ASSUMPTIONS  (v2.0 — two-stage, dynamic WACC seeded)
# ──────────────────────────────────────────────────────────────────

class Assumptions:
    """
    All model parameters in one place.

    v2.0 changes
    ------------
    - stage1_growth (years 1-5) replaces single growth_rate
    - Stage 2 fades linearly from stage1_growth to terminal_growth
    - base_discount_rate seeded from computed WACC
    - Scenarios use absolute bps shifts, not proportional ratios
    - Sector-aware multiples from SECTOR_MULTIPLES table
    """

    def __init__(self, data):
        d = data

        # Growth (Stage 1, years 1-5)
        hist_g = d.revenue_cagr or d.fcf_cagr or 0.08
        self.stage1_growth        = round(min(max(hist_g, 0.02), 0.40), 4)

        # Stage 2 fading behaviour
        self.stage2_fade          = True   # False = keep stage1 flat through yr 10

        # Terminal growth
        self.base_terminal_growth = 0.025

        # Discount rate seeded from WACC calculation
        self.base_discount_rate   = d.wacc_computed or 0.10

        # Forecast horizon (stage1 = yr1-5, stage2 = yr6-10)
        self.forecast_years       = 10

        # FCF anchor
        self.fcf_anchor           = d.fcf_normalized or d.fcf_ttm

        # Scenarios — realistic absolute deltas
        g1  = self.stage1_growth
        r   = self.base_discount_rate
        tg  = self.base_terminal_growth

        self.bear = {
            "stage1_growth":  max(g1 * 0.40, 0.01),       # shock: -60% of base
            "discount_rate":  min(r + 0.020, 0.22),         # +200 bps
            "terminal_growth":max(tg - 0.010, 0.005),       # -100 bps
            "probability":    0.25,
        }
        self.base = {
            "stage1_growth":  g1,
            "discount_rate":  r,
            "terminal_growth":tg,
            "probability":    0.50,
        }
        self.bull = {
            "stage1_growth":  min(g1 * 1.40, 0.40),        # +40%, hard cap 40%
            "discount_rate":  max(r - 0.010, 0.06),          # -100 bps, floor 6%
            "terminal_growth":min(tg + 0.010, 0.04),         # +100 bps, cap 4%
            "probability":    0.25,
        }

        # Relative valuation multiples — sector-aware
        sec_mults = SECTOR_MULTIPLES.get(d.sector or "default",
                                          SECTOR_MULTIPLES["default"])
        self.target_pe_ratio  = sec_mults["pe"]
        self.target_ev_ebitda = sec_mults["ev_ebitda"]
        self.sector_source    = d.sector or "default"

        # Sensitivity grid ranges
        self.sensitivity_growth_rates   = [0.02, 0.04, 0.06, 0.08,
                                            0.10, 0.12, 0.15, 0.20]
        self.sensitivity_discount_rates = [0.06, 0.07, 0.08, 0.09,
                                            0.10, 0.11, 0.12, 0.13]

    def interactive_override(self):
        print(_c("\n  Press Enter to keep each default.", "yellow"))

        def _ask(label, current, cast=float):
            raw = input(f"    {label} [{current}]: ").strip()
            return cast(raw) if raw else current

        self.stage1_growth        = _ask("Stage-1 Growth (yr 1-5), e.g. 0.10",
                                          _fmt_pct(self.stage1_growth))
        self.base_discount_rate   = _ask("Discount Rate / WACC, e.g. 0.10",
                                          _fmt_pct(self.base_discount_rate))
        self.base_terminal_growth = _ask("Terminal Growth, e.g. 0.025",
                                          _fmt_pct(self.base_terminal_growth))
        self.forecast_years       = _ask("Forecast Period (years)",
                                          self.forecast_years, cast=int)
        self.target_pe_ratio      = _ask("Target P/E Ratio", self.target_pe_ratio)
        self.target_ev_ebitda     = _ask("Target EV/EBITDA", self.target_ev_ebitda)

        for attr in ("stage1_growth", "base_discount_rate", "base_terminal_growth"):
            v = getattr(self, attr)
            if isinstance(v, str):
                v = float(v.replace("%", ""))
            if isinstance(v, float) and v > 1:
                v /= 100
            setattr(self, attr, v)

        self._rebuild_scenarios()

    def _rebuild_scenarios(self):
        g1 = self.stage1_growth
        r  = self.base_discount_rate
        tg = self.base_terminal_growth

        self.base["stage1_growth"]  = g1
        self.base["discount_rate"]  = r
        self.base["terminal_growth"]= tg

        self.bear["stage1_growth"]  = max(g1 * 0.40, 0.01)
        self.bear["discount_rate"]  = min(r + 0.020, 0.22)
        self.bear["terminal_growth"]= max(tg - 0.010, 0.005)

        self.bull["stage1_growth"]  = min(g1 * 1.40, 0.40)
        self.bull["discount_rate"]  = max(r - 0.010, 0.06)
        self.bull["terminal_growth"]= min(tg + 0.010, 0.04)

    def print_summary(self):
        _print_subheader("MODEL ASSUMPTIONS")
        rows = [
            ["Forecast Period",          f"{self.forecast_years} yrs  "
                                          "(Stage 1: yr 1-5  |  Stage 2: yr 6-10 fade)"],
            ["Stage 1 Growth (yr 1-5)", _fmt_pct(self.stage1_growth)],
            ["Stage 2 Growth (yr 6-10)","Linear decay  -->  terminal growth"],
            ["Terminal Growth Rate",    _fmt_pct(self.base_terminal_growth)],
            ["Discount Rate / WACC",    _fmt_pct(self.base_discount_rate)],
            ["FCF Anchor",              _fmt_currency(self.fcf_anchor)],
            ["Target P/E",              _fmt_x(self.target_pe_ratio)],
            ["Target EV/EBITDA",        _fmt_x(self.target_ev_ebitda)],
            ["Multiples Sector Ref.",   self.sector_source],
            ["Bear g1 / WACC",          f"{_fmt_pct(self.bear['stage1_growth'])} / "
                                         f"{_fmt_pct(self.bear['discount_rate'])}"],
            ["Bull g1 / WACC",          f"{_fmt_pct(self.bull['stage1_growth'])} / "
                                         f"{_fmt_pct(self.bull['discount_rate'])}"],
        ]
        _print_table(rows, ["Assumption", "Value"])


# ──────────────────────────────────────────────────────────────────
#  3.  CALCULATION ENGINE  (v2.0)
# ──────────────────────────────────────────────────────────────────

class ValuationEngine:
    """
    Pure calculation layer — no I/O.

    v2.0 additions
    --------------
    - two_stage_dcf()      two-stage model with linear Stage 2 fade
    - dcf()                alias for backward compatibility
    - roic_check()         ROIC vs WACC economic spread
    - sensitivity_impact() which variable moves fair value most
    - scenario_analysis()  updated for two-stage with realistic ranges
    """

    def __init__(self, data, assumptions):
        self.data = data
        self.asmp = assumptions

    # ── Two-Stage DCF ────────────────────────────────────────────

    def two_stage_dcf(self, stage1_growth=None, discount_rate=None,
                       terminal_growth=None, fcf_0=None, n=None):
        """
        Two-stage DCF.

        Stage 1 (yr 1 to n//2):  constant growth at stage1_growth
        Stage 2 (yr n//2+1 to n):linear decay from stage1_growth to terminal_growth
        Terminal Value: Gordon Growth Model on final-year FCF
        """
        d  = self.data
        a  = self.asmp
        g1 = stage1_growth  or a.stage1_growth
        r  = discount_rate  or a.base_discount_rate
        tg = terminal_growth or a.base_terminal_growth
        f0 = fcf_0          or a.fcf_anchor or d.fcf_ttm
        n  = n              or a.forecast_years

        if not f0:
            return {"error": "FCF anchor unavailable — cannot run DCF."}
        if r <= tg:
            return {"error": f"WACC ({_fmt_pct(r)}) must exceed "
                              f"terminal growth ({_fmt_pct(tg)})."}

        s1_yrs = n // 2          # e.g. 5
        s2_yrs = n - s1_yrs      # e.g. 5

        # Build growth schedule
        growth_schedule = []
        for _ in range(s1_yrs):
            growth_schedule.append(g1)                        # Stage 1: flat
        for k in range(1, s2_yrs + 1):
            g_t = g1 - k * (g1 - tg) / s2_yrs                # Stage 2: linear decay
            growth_schedule.append(max(g_t, tg))

        # Project FCFs and discount
        fcf_t         = f0
        fcf_projected = []
        pv_fcf        = []
        growth_display = []

        for t, g_t in enumerate(growth_schedule, 1):
            fcf_t  = fcf_t * (1 + g_t)
            pv_t   = fcf_t / (1 + r) ** t
            stage  = "Stage 1" if t <= s1_yrs else "Stage 2"
            fcf_projected.append(fcf_t)
            pv_fcf.append(pv_t)
            growth_display.append((t, stage, g_t, fcf_t, pv_t))

        # Terminal Value
        tv      = fcf_projected[-1] * (1 + tg) / (r - tg)
        pv_tv   = tv / (1 + r) ** n

        # EV bridge
        sum_pv  = sum(pv_fcf)
        ev      = sum_pv + pv_tv
        equity  = ev - (d.total_debt or 0) + (d.cash or 0)
        shares  = d.shares_outstanding
        fv      = equity / shares if shares else None

        return {
            "stage1_growth":        g1,
            "discount_rate":        r,
            "terminal_growth":      tg,
            "fcf_0":                f0,
            "n":                    n,
            "stage1_years":         s1_yrs,
            "stage2_years":         s2_yrs,
            "growth_display":       growth_display,
            "fcf_projected":        fcf_projected,
            "pv_fcf":               pv_fcf,
            "sum_pv_fcf":           sum_pv,
            "terminal_value":       tv,
            "pv_terminal_value":    pv_tv,
            "pv_terminal_pct":      pv_tv / ev if ev else None,
            "enterprise_value":     ev,
            "equity_value":         equity,
            "shares_outstanding":   shares,
            "fair_value_per_share": fv,
        }

    def dcf(self, **kwargs):
        """Backward-compatible alias for two_stage_dcf()."""
        return self.two_stage_dcf(**kwargs)

    # ── Relative Valuation ───────────────────────────────────────

    def relative_valuation(self):
        d, a = self.data, self.asmp

        pe_fv = None
        if d.eps_ttm and a.target_pe_ratio and d.eps_ttm > 0:
            pe_fv = d.eps_ttm * a.target_pe_ratio

        eveb_fv = None
        if d.ebitda_ttm and a.target_ev_ebitda and d.ebitda_ttm > 0:
            ev_impl = d.ebitda_ttm * a.target_ev_ebitda
            eq_impl = ev_impl - (d.total_debt or 0) + (d.cash or 0)
            if d.shares_outstanding:
                eveb_fv = eq_impl / d.shares_outstanding

        return {
            "pe_fair_value":        pe_fv,
            "target_pe":            a.target_pe_ratio,
            "sector_source":        a.sector_source,
            "eps_ttm":              d.eps_ttm,
            "ev_ebitda_fair_value": eveb_fv,
            "target_ev_ebitda":     a.target_ev_ebitda,
            "ebitda_ttm":           d.ebitda_ttm,
        }

    # ── Scenario Analysis ────────────────────────────────────────

    def scenario_analysis(self):
        results = {}
        for name, params in [("bear", self.asmp.bear),
                               ("base", self.asmp.base),
                               ("bull", self.asmp.bull)]:
            res = self.two_stage_dcf(
                stage1_growth   = params["stage1_growth"],
                discount_rate   = params["discount_rate"],
                terminal_growth = params["terminal_growth"],
            )
            res["probability"]    = params["probability"]
            res["growth_rate"]    = params["stage1_growth"]   # alias for printer
            res["discount_rate"]  = params["discount_rate"]
            res["terminal_growth"]= params["terminal_growth"]
            results[name] = res

        total_prob  = sum(results[n].get("probability", 0) for n in ("bear", "base", "bull"))
        weighted_fv = sum(
            (results[n].get("fair_value_per_share") or 0) * results[n].get("probability", 0)
            for n in ("bear", "base", "bull")
        ) / total_prob if total_prob > 0 else None

        results["weighted_fair_value"] = weighted_fv
        return results

    # ── ROIC vs WACC ─────────────────────────────────────────────

    def roic_check(self):
        d    = self.data
        wacc = d.wacc_computed or self.asmp.base_discount_rate

        if not d.nopat or not d.invested_capital or d.invested_capital <= 0:
            return {"available": False,
                    "reason": "NOPAT or Invested Capital data unavailable."}

        roic   = d.nopat / d.invested_capital
        spread = roic - wacc

        if   roic >= wacc:              signal = "VALUE CREATING"
        elif roic >= wacc * 0.80:       signal = "BORDERLINE"
        else:                           signal = "VALUE DESTROYING"

        return {
            "available":        True,
            "nopat":            d.nopat,
            "invested_capital": d.invested_capital,
            "roic":             roic,
            "wacc":             wacc,
            "spread":           spread,
            "signal":           signal,
        }

    # ── Sensitivity Table ─────────────────────────────────────────

    def sensitivity_table(self):
        rows = []
        for g in self.asmp.sensitivity_growth_rates:
            row = {}
            for r in self.asmp.sensitivity_discount_rates:
                res  = self.two_stage_dcf(stage1_growth=g, discount_rate=r)
                row[r] = res.get("fair_value_per_share")
            rows.append(row)
        df = pd.DataFrame(rows, index=self.asmp.sensitivity_growth_rates)
        df.index.name   = "Stage1 Growth"
        df.columns.name = "Discount Rate"
        return df

    # ── Sensitivity Driver ───────────────────────────────────────

    def sensitivity_impact(self, base_fv):
        """
        Shock each input by +10% and measure the resulting change in
        fair value.  Returns a dict sorted highest impact first.
        """
        if not base_fv or base_fv == 0:
            return {}

        a  = self.asmp
        g1 = a.stage1_growth
        r  = a.base_discount_rate
        tg = a.base_terminal_growth
        f0 = a.fcf_anchor or self.data.fcf_ttm
        delta = 0.10

        def _fv(**kw):
            params = dict(stage1_growth=g1, discount_rate=r,
                          terminal_growth=tg, fcf_0=f0)
            params.update(kw)
            # Guard terminal growth vs discount rate
            if params["terminal_growth"] >= params["discount_rate"]:
                params["terminal_growth"] = params["discount_rate"] - 0.001
            res = self.two_stage_dcf(**params)
            return res.get("fair_value_per_share") or base_fv

        impacts = {
            "Stage-1 Growth Rate":   abs(_fv(stage1_growth=g1*(1+delta)) - base_fv) / base_fv,
            "WACC / Discount Rate":  abs(_fv(discount_rate=r*(1+delta))  - base_fv) / base_fv,
            "Terminal Growth Rate":  abs(_fv(terminal_growth=min(tg*(1+delta), r-0.001)) - base_fv) / base_fv,
            "FCF Base Estimate":     abs(_fv(fcf_0=f0*(1+delta))         - base_fv) / base_fv,
        }
        return dict(sorted(impacts.items(), key=lambda x: x[1], reverse=True))

    # ── Margin of Safety ─────────────────────────────────────────

    def margin_of_safety(self, fair_value):
        price = self.data.current_price
        if not fair_value or not price:
            return {"mos": None, "signal": "N/A"}
        mos = (fair_value - price) / fair_value
        if   mos >= 0.30:   signal = "STRONG BUY"
        elif mos >= 0.10:   signal = "BUY"
        elif mos >= -0.10:  signal = "FAIR VALUE"
        elif mos >= -0.25:  signal = "SLIGHTLY OVERVALUED"
        else:               signal = "OVERVALUED"
        return {"mos": mos, "signal": signal, "price": price, "fair_value": fair_value}


# ──────────────────────────────────────────────────────────────────
#  4.  OUTPUT / REPORT LAYER  (v2.0)
# ──────────────────────────────────────────────────────────────────

class ReportPrinter:

    def __init__(self, data, engine, quality, wacc_calc):
        self.data    = data
        self.engine  = engine
        self.quality = quality
        self.wacc_c  = wacc_calc

    def print_quality(self):
        _print_subheader("QUALITY CHECKS & DIAGNOSTICS")
        self.quality.print_report()

    def print_wacc(self):
        self.wacc_c.print_breakdown()

    def print_growth_curve(self, res):
        _print_subheader("GROWTH SCHEDULE  (Two-Stage DCF)")
        if "error" in res:
            return
        rows = []
        for (t, stage, g_t, fcf_t, pv_t) in res.get("growth_display", []):
            col = "cyan" if stage == "Stage 1" else "white"
            rows.append([_c(str(t), col), _c(stage, col), _c(_fmt_pct(g_t), col),
                          _fmt_currency(fcf_t), _fmt_currency(pv_t)])
        _print_table(rows, ["Year", "Stage", "Growth Rate", "Projected FCF", "PV of FCF"])

        # ASCII bar chart of growth decay
        print()
        all_g = [row[2] for row in res.get("growth_display", [])]
        all_g_f = [float(x.replace("%",""))/100 for x in all_g]
        max_g = max(all_g_f) if all_g_f else 0.10
        print(_c("  Growth Rate Decay:", "white"))
        for t, stage, g_t, _, _ in res.get("growth_display", []):
            bar_w = int(g_t / max(max_g, 0.001) * 35)
            col   = "cyan" if stage == "Stage 1" else "white"
            print(f"    Yr {t:>2}  {_c(('|' * bar_w).ljust(36), col)}  {_fmt_pct(g_t)}")

    def print_dcf(self, res):
        _print_subheader("TWO-STAGE DCF VALUATION -- BASE CASE")
        if "error" in res:
            _error(res["error"])
            return

        d = self.data
        print(f"\n  FCF Anchor : {_c(_fmt_currency(res['fcf_0']), 'cyan')}  "
              f"({d.fcf_norm_method or 'TTM'})")
        print(f"  Stage 1    : years 1-{res['stage1_years']}  "
              f"@ {_c(_fmt_pct(res['stage1_growth']), 'cyan')} constant growth")
        print(f"  Stage 2    : years {res['stage1_years']+1}-{res['n']}  "
              f"linear decay to {_fmt_pct(res['terminal_growth'])}")
        print(f"  WACC       : {_c(_fmt_pct(res['discount_rate']), 'cyan')}")
        print()

        bridge = [
            ["Sum PV of FCFs (yr 1-10)",      _fmt_currency(res["sum_pv_fcf"])],
            ["Terminal Value (Gordon Growth)", _fmt_currency(res["terminal_value"])],
            ["PV of Terminal Value",           _fmt_currency(res["pv_terminal_value"])],
            ["  Terminal Value % of EV",       _fmt_pct(res.get("pv_terminal_pct"))],
            ["--",                             "--"],
            ["Enterprise Value (EV)",          _fmt_currency(res["enterprise_value"])],
            ["  - Total Debt",                 _fmt_currency(-(d.total_debt or 0))],
            ["  + Cash & Equivalents",         _fmt_currency(d.cash or 0)],
            ["Equity Value",                   _fmt_currency(res["equity_value"])],
            ["Shares Outstanding",             _fmt_currency(res["shares_outstanding"],
                                                               currency="")],
            ["--",                             "--"],
            ["DCF Fair Value / Share",         _c(_fmt_currency(res["fair_value_per_share"]),
                                                   "green")],
        ]
        _print_table(bridge, ["Metric", "Value"])

    def print_relative(self, res):
        _print_subheader("RELATIVE VALUATION")
        print(f"  Multiples reference sector: "
              f"{_c(res.get('sector_source', 'default'), 'cyan')}")
        rows = []
        if res["pe_fair_value"] is not None:
            rows.append(["P/E",       _fmt_x(res["target_pe"]),
                          _fmt_currency(res["eps_ttm"]),
                          _fmt_currency(res["pe_fair_value"])])
        if res["ev_ebitda_fair_value"] is not None:
            rows.append(["EV/EBITDA", _fmt_x(res["target_ev_ebitda"]),
                          _fmt_currency(res["ebitda_ttm"]),
                          _fmt_currency(res["ev_ebitda_fair_value"])])
        if rows:
            _print_table(rows, ["Model", "Target Multiple",
                                  "Base Metric", "Fair Value / Share"])
        else:
            print("  [INFO] No EPS or EBITDA available for relative valuation.")

    def print_scenarios(self, res):
        _print_subheader("SCENARIO ANALYSIS  (Two-Stage DCF)")
        rows = []
        for name in ("bear", "base", "bull"):
            s  = res[name]
            fv = s.get("fair_value_per_share")
            rows.append([name.upper(),
                          _fmt_pct(s.get("growth_rate")),
                          _fmt_pct(s.get("discount_rate")),
                          _fmt_pct(s.get("terminal_growth")),
                          _fmt_pct(s.get("probability")),
                          _fmt_currency(fv) if fv else "N/A"])
        _print_table(rows, ["Scenario", "Stage-1 g", "WACC", "Terminal g",
                              "Probability", "Fair Value"])
        wfv = res.get("weighted_fair_value")
        if wfv:
            m   = self.engine.margin_of_safety(wfv)
            col = ("green"  if "BUY"  in m["signal"] else
                   "yellow" if "FAIR" in m["signal"] else "red")
            print(f"\n  Probability-Weighted FV: {_c(_fmt_currency(wfv), 'cyan')}  "
                  f"-->  {_c(m['signal'], col)}")

    def print_roic(self, r):
        _print_subheader("ROIC vs WACC  (Value-Creation Check)")
        if not r.get("available"):
            print(f"  [INFO] {r.get('reason', 'Data unavailable.')}")
            return
        col = ("green"  if r["signal"] == "VALUE CREATING"
               else "yellow" if r["signal"] == "BORDERLINE"
               else "red")
        rows = [
            ["NOPAT  (EBIT x (1-t))",   _fmt_currency(r["nopat"])],
            ["Invested Capital",         _fmt_currency(r["invested_capital"])],
            ["ROIC",                     _c(_fmt_pct(r["roic"]), col)],
            ["WACC",                     _fmt_pct(r["wacc"])],
            ["Spread  (ROIC - WACC)",    _c(_fmt_pct(r["spread"]), col)],
            ["Signal",                   _c(r["signal"], col)],
        ]
        _print_table(rows, ["Metric", "Value"])
        if r["signal"] == "VALUE CREATING":
            print(_c(f"\n  Company earns above its cost of capital — "
                      f"ROIC spread of {_fmt_pct(r['spread'])} supports premium valuation.", col))
        elif r["signal"] == "BORDERLINE":
            print(_c("\n  ROIC barely covers WACC — "
                      "competitive advantages may be weakening.", col))
        else:
            print(_c("\n  ROIC < WACC — the company may be destroying economic value.  "
                      "Discount the DCF result.", col))

    def print_mos(self, results):
        _print_subheader("MARGIN OF SAFETY SUMMARY")
        dcf_fv   = results["dcf"].get("fair_value_per_share")
        pe_fv    = results["relative"].get("pe_fair_value")
        eveb_fv  = results["relative"].get("ev_ebitda_fair_value")
        scen_wfv = results["scenarios"].get("weighted_fair_value")
        bear_fv  = results["scenarios"]["bear"].get("fair_value_per_share")
        bull_fv  = results["scenarios"]["bull"].get("fair_value_per_share")
        price    = self.data.current_price

        rows = []
        for label, fv in [("DCF Two-Stage (Base)", dcf_fv),
                            ("Scenario Weighted",   scen_wfv),
                            ("Relative -- P/E",     pe_fv),
                            ("Relative -- EV/EBITDA",eveb_fv)]:
            if fv and fv > 0:
                m   = self.engine.margin_of_safety(fv)
                col = ("green"  if "BUY"  in m["signal"] else
                        "yellow" if "FAIR" in m["signal"] else "red")
                rows.append([label, _fmt_currency(price), _fmt_currency(fv),
                              _c(_fmt_pct(m["mos"]), col), _c(m["signal"], col)])
        if rows:
            _print_table(rows, ["Model", "Current Price", "Fair Value",
                                  "Margin of Safety", "Signal"])

        if bear_fv and bull_fv:
            print(f"\n  Fair Value Range (Bear to Bull): "
                  f"{_c(_fmt_currency(bear_fv), 'red')}  -->  "
                  f"{_c(_fmt_currency(bull_fv), 'green')}")

    def print_sensitivity(self, df):
        _print_subheader("SENSITIVITY TABLE  (Stage-1 Growth x WACC --> Fair Value)")
        print("  Rows = Stage-1 Growth Rate  |  Columns = WACC\n")
        price = self.data.current_price
        formatted_rows = []
        for g_rate, row in df.iterrows():
            formatted = [_fmt_pct(g_rate)]
            for r_rate, fv in row.items():
                cell = _fmt_currency(fv) if fv else "N/A"
                if fv and price:
                    mos = (fv - price) / fv
                    if mos >= 0.10:
                        cell = _c(cell, "green")
                    elif mos <= -0.10:
                        cell = _c(cell, "red")
                formatted.append(cell)
            formatted_rows.append(formatted)
        headers = ["Growth \\ WACC"] + [_fmt_pct(c) for c in df.columns]
        _print_table(formatted_rows, headers)
        print(_c("  Green = >10% upside  |  Red = >10% downside", "white"))

    def print_driver(self, impacts):
        _print_subheader("SENSITIVITY DRIVER ANALYSIS  (+10% input shock)")
        if not impacts:
            print("  [INFO] Cannot compute — fair value unavailable.")
            return
        rows = []
        for rank, (var, impact) in enumerate(impacts.items(), 1):
            bar = "|" * int(impact * 200)
            tag = _c("  <-- most sensitive", "cyan") if rank == 1 else ""
            rows.append([str(rank), var, _fmt_pct(impact), bar + tag])
        _print_table(rows, ["Rank", "Variable", "FV Impact (+10% shock)", "Relative Bar"])
        top = list(impacts.keys())[0]
        print(f"\n  {_c('Most sensitive: ' + top, 'bold')}  "
              "-- focus scenario testing here.")

    def print_full_report(self, results):
        _print_header(f"FAIR VALUE CALCULATOR v2.0  --  "
                      f"{self.data.company_name} ({self.data.ticker})")

        self.data.print_summary()
        self.print_quality()
        self.print_wacc()
        self.engine.asmp.print_summary()
        self.print_growth_curve(results["dcf"])
        self.print_dcf(results["dcf"])
        self.print_relative(results["relative"])
        self.print_scenarios(results["scenarios"])
        self.print_roic(results["roic"])
        self.print_mos(results)
        self.print_sensitivity(results["sensitivity"])
        self.print_driver(results.get("sensitivity_impact", {}))

        print("\n" + _c("=" * 72, "cyan"))
        print(_c("  Research purposes only -- not financial advice.", "yellow"))
        print(_c("  Verify all figures against official filings before investing.", "yellow"))
        print(_c("=" * 72 + "\n", "cyan"))


# ──────────────────────────────────────────────────────────────────
#  5.  ORCHESTRATOR  (v2.0 -- backward-compatible public API)
# ──────────────────────────────────────────────────────────────────

def run_valuation(ticker=None, interactive_mode=True, customize_assumptions=True):
    """
    Run a complete v2.0 valuation.

    Parameters
    ----------
    ticker                : str  -- e.g. "AAPL".  Prompts if None.
    interactive_mode      : bool -- show CLI prompts and colour output
    customize_assumptions : bool -- offer assumption override before run

    Returns
    -------
    dict: dcf, relative, scenarios, sensitivity, sensitivity_impact,
          roic, quality, data
    """
    _print_header("FAIR VALUE CALCULATOR  v2.0", width=72)
    print("  Two-Stage DCF  *  Dynamic WACC  *  FCF Normalisation  *  "
          "ROIC Check\n")

    # Ticker
    if ticker is None:
        ticker = input("  Enter stock ticker (e.g. AAPL, MSFT, TSLA): ").strip().upper()

    # Data
    print()
    data = StockData(ticker).fetch()

    # FCF Normalisation
    fcf_norm, fcf_method   = data.compute_normalized_fcf()
    data.fcf_normalized    = fcf_norm
    data.fcf_norm_method   = fcf_method
    if interactive_mode:
        print(f"  FCF normalisation: {_c(fcf_method, 'cyan')}")
        print(f"  Normalised FCF:    {_c(_fmt_currency(fcf_norm), 'cyan')}")

    # WACC
    wacc_calc  = WACCCalculator(data)
    wacc       = wacc_calc.compute()
    if interactive_mode:
        print(f"  Computed WACC:     {_c(_fmt_pct(wacc), 'cyan')}")

    # Assumptions
    assumptions = Assumptions(data)

    if customize_assumptions and interactive_mode:
        print(_c("\n  Seeded assumptions (based on data):", "cyan"))
        assumptions.print_summary()
        ans = input("\n  Customise assumptions? [y/N]: ").strip().lower()
        if ans == "y":
            assumptions.interactive_override()

    # Quality checks (may block)
    quality = QualityChecker(data, assumptions)
    safe    = quality.run_all()
    if not safe:
        _print_subheader("BLOCKING ERRORS DETECTED")
        quality.print_report()
        print(_c("\n  Cannot proceed. Resolve errors above and re-run.", "red"))
        return {"error": "quality_check_failed", "data": data, "quality": quality}

    # Run models
    engine  = ValuationEngine(data, assumptions)
    printer = ReportPrinter(data, engine, quality, wacc_calc)

    print(_c("\n  Running valuation models ...", "cyan"))

    dcf_res = engine.two_stage_dcf()
    base_fv = dcf_res.get("fair_value_per_share")

    results = {
        "dcf":               dcf_res,
        "relative":          engine.relative_valuation(),
        "scenarios":         engine.scenario_analysis(),
        "sensitivity":       engine.sensitivity_table(),
        "sensitivity_impact":engine.sensitivity_impact(base_fv) if base_fv else {},
        "roic":              engine.roic_check(),
        "quality":           quality,
        "data":              data,
    }

    printer.print_full_report(results)
    return results


# ──────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fair Value Calculator v2.0")
    parser.add_argument("ticker",  nargs="?", default=None,
                        help="Ticker symbol (e.g. AAPL)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip all interactive prompts")
    parser.add_argument("--batch", nargs="+", metavar="TICKER",
                        help="Value multiple tickers without prompts")
    args = parser.parse_args()

    if args.batch:
        summary_rows = []
        for t in args.batch:
            try:
                res = run_valuation(t, interactive_mode=False,
                                     customize_assumptions=False)
                if "error" in res and res["error"] == "quality_check_failed":
                    summary_rows.append([t.upper(), "--", "--", "--", "ERRORS", "--"])
                    continue
                dcf_fv = res["dcf"].get("fair_value_per_share")
                wfv    = res["scenarios"].get("weighted_fair_value")
                price  = res["data"].current_price
                mos    = ((dcf_fv - price) / dcf_fv) if dcf_fv and price else None
                roic_r = res["roic"]
                roic_s = roic_r.get("signal", "N/A") if roic_r.get("available") else "N/A"
                summary_rows.append([t.upper(), _fmt_currency(price),
                                      _fmt_currency(dcf_fv), _fmt_currency(wfv),
                                      _fmt_pct(mos), roic_s])
            except Exception as exc:
                summary_rows.append([t.upper(), "ERROR", str(exc)[:40], "--", "--", "--"])

        _print_header("BATCH VALUATION SUMMARY  v2.0")
        _print_table(summary_rows,
                      ["Ticker", "Price", "DCF Fair Value",
                        "Weighted FV", "Margin of Safety", "ROIC Signal"])

    else:
        run_valuation(
            ticker=args.ticker,
            interactive_mode=not args.no_interactive,
            customize_assumptions=not args.no_interactive,
        )
