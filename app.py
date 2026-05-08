
# app.py  —  NSE Portfolio Optimizer
# Built on the notebook: Portfolio_Optimization_Using_DOCplex_on_NSE_equities 



import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date
from docplex.mp.model import Model


st.set_page_config(
    page_title="NSE Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


NIFTY50_TICKERS = [
    "RELIANCE.NS",  "TCS.NS",        "HDFCBANK.NS",   "ICICIBANK.NS",
    "INFY.NS",      "HINDUNILVR.NS", "ITC.NS",         "SBIN.NS",
    "BHARTIARTL.NS","KOTAKBANK.NS",  "LT.NS",          "AXISBANK.NS",
    "ASIANPAINT.NS","MARUTI.NS",     "SUNPHARMA.NS",   "TITAN.NS",
    "BAJFINANCE.NS","WIPRO.NS",      "ONGC.NS",        "NTPC.NS",
    "POWERGRID.NS", "NESTLEIND.NS",  "TECHM.NS",       "BAJAJ-AUTO.NS",
    "HCLTECH.NS",   "COALINDIA.NS",  "DRREDDY.NS",     "DIVISLAB.NS",
    "CIPLA.NS",     "EICHERMOT.NS",  "GRASIM.NS",      "JSWSTEEL.NS",
    "TATASTEEL.NS", "TATAMOTORS.NS", "INDUSINDBK.NS",  "APOLLOHOSP.NS",
    "BPCL.NS",      "BRITANNIA.NS",  "HEROMOTOCO.NS",  "HINDALCO.NS",
    "M&M.NS",       "TATACONSUM.NS",
]

DEFAULT_TICKERS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "TCS.NS",      "INFY.NS",    "HINDUNILVR.NS",
    "RELIANCE.NS", "SBIN.NS",      "SUNPHARMA.NS","DRREDDY.NS", "WIPRO.NS",
]


def optimize_portfolio(
        mu,
        sigma,
        target_ret,
        max_weight=1.0,
        allow_short=True,
        short_lower_bound=-0.30,
        Total_marketexposure=1.5,
        prev_weights=None,
        lambda_tc=0.001,
        max_turnover=0.20,
        risk_free_rate=0.07,
):
    mu    = np.array(mu)
    sigma = np.array(sigma)
    n     = len(mu)
    tickers = list(range(n))

    mdl = Model(name="Portfolio Optimizer")
    mdl.parameters.optimalitytarget = 3

    lb = short_lower_bound if allow_short else 0.0
    ub = max_weight
    w  = mdl.continuous_var_list(n, lb=lb, ub=ub, name="w")

    if allow_short:
        g = mdl.continuous_var_list(n, lb=0, name="g")
        for i in tickers:
            mdl.add_constraint(g[i] >= w[i])
            mdl.add_constraint(g[i] >= -w[i])
        mdl.add_constraint(mdl.sum(g[i] for i in tickers) <= Total_marketexposure)

    portfolio_variance = mdl.sum(
        sigma[i][j] * w[i] * w[j]
        for i in tickers for j in tickers
    )

    if prev_weights is not None:
        prev_weights = np.array(prev_weights)
        t = mdl.continuous_var_list(n, lb=0, name="t")
        for i in tickers:
            mdl.add_constraint(t[i] >= (w[i] - prev_weights[i]))
            mdl.add_constraint(t[i] >= -(w[i] - prev_weights[i]))
        mdl.add_constraint(mdl.sum(t) <= max_turnover)
        tc_penalty = lambda_tc * mdl.sum(t)
    else:
        tc_penalty = 0

    mdl.minimize(portfolio_variance + tc_penalty)
    mdl.add_constraint(mdl.sum(w[i] for i in tickers) == 1)
    mdl.add_constraint(mdl.sum(mu[i] * w[i] for i in tickers) >= target_ret)

    solution = mdl.solve(log_output=False)
    if solution is None:
        return None

    weights            = np.array([w[i].solution_value for i in tickers])
    portfolio_return   = float(np.dot(mu, weights))
    portfolio_variance = float(weights @ sigma @ weights)
    portfolio_risk     = float(np.sqrt(portfolio_variance))
    sharpe_ratio       = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0

    return {
        "weights"           : weights,
        "portfolio_return"  : portfolio_return,
        "portfolio_risk"    : portfolio_risk,
        "portfolio_variance": portfolio_variance,
        "sharpe_ratio"      : sharpe_ratio,
    }


def run_robust_delta(
        mu,
        sigma,
        target_ret,
        max_weight=1.0,
        allow_short=True,
        short_lower_bound=-0.30,
        Total_marketexposure=1.5,
        prev_weights=None,
        lambda_tc=0.001,
        max_turnover=0.20,
        risk_free_rate=0.07,
        robust_delta=0.15,          # notebook default = 0.15; slider can override
):
    mu    = np.array(mu)
    sigma = np.array(sigma)
    n     = len(mu)
    tickers = list(range(n))

    mdl = Model(name="Portfolio Optimizer")
    mdl.parameters.optimalitytarget = 3

    lb = short_lower_bound if allow_short else 0.0
    ub = max_weight
    w  = mdl.continuous_var_list(n, lb=lb, ub=ub, name="w")

    if allow_short:
        g = mdl.continuous_var_list(n, lb=0, name="g")
        for i in tickers:
            mdl.add_constraint(g[i] >= w[i])
            mdl.add_constraint(g[i] >= -w[i])
        mdl.add_constraint(mdl.sum(g[i] for i in tickers) <= Total_marketexposure)

    portfolio_variance = mdl.sum(
        sigma[i][j] * w[i] * w[j]
        for i in tickers for j in tickers
    )

    if prev_weights is not None:
        prev_weights = np.array(prev_weights)
        t = mdl.continuous_var_list(n, lb=0, name="t")
        for i in tickers:
            mdl.add_constraint(t[i] >= (w[i] - prev_weights[i]))
            mdl.add_constraint(t[i] >= -(w[i] - prev_weights[i]))
        mdl.add_constraint(mdl.sum(t) <= max_turnover)
        tc_penalty = lambda_tc * mdl.sum(t)
    else:
        tc_penalty = 0

    mdl.minimize(portfolio_variance + tc_penalty)
    mdl.add_constraint(mdl.sum(w[i] for i in tickers) == 1)

    std_i = np.sqrt(np.diag(sigma))
    # Robust return constraint: penalise uncertain returns by subtracting delta * volatility
    # This makes the optimizer more conservative — same as notebook
    if robust_delta > 0:
        effective_mu = mu - robust_delta * std_i   # pessimistic expected return
    else:
        effective_mu = mu                           # same as basic Markowitz

    mdl.add_constraint(mdl.sum(effective_mu[i] * w[i] for i in tickers) >= target_ret)

    solution = mdl.solve(log_output=False)
    if solution is None:
        return None

    weights            = np.array([w[i].solution_value for i in tickers])
    portfolio_return   = float(np.dot(mu, weights))
    portfolio_variance = float(weights @ sigma @ weights)
    portfolio_risk     = float(np.sqrt(portfolio_variance))
    sharpe_ratio       = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0

    return {
        "weights"           : weights,
        "portfolio_return"  : portfolio_return,
        "portfolio_risk"    : portfolio_risk,
        "portfolio_variance": portfolio_variance,
        "sharpe_ratio"      : sharpe_ratio,
    }


def solve_cvar(
        mu, sigma, target_ret, scenarios, cvar_alpha,
        allow_short, short_lower_bound, Total_marketexposure,
        max_weight, risk_free_rate,
):
    mu    = np.array(mu)
    sigma = np.array(sigma)
    n     = len(mu)
    tickers = list(range(n))
    S = scenarios.shape[0]
    R = scenarios.values

    mdl = Model(name="CVaR_Optimizer")

    lb = short_lower_bound if allow_short else 0.0
    w  = mdl.continuous_var_list(n, lb=lb, ub=max_weight, name="w")

    # defining var - loss that we can suffer
    var_threshold = mdl.continuous_var(lb=-1.0, ub=1.0, name="VaR")
    # defining z - excess loss beyond VaR
    z = mdl.continuous_var_list(S, lb=0, name="z")

    # gross exposure
    if allow_short:
        g = mdl.continuous_var_list(n, lb=0, name="g")
        for i in tickers:
            mdl.add_constraint(g[i] >= w[i])
            mdl.add_constraint(g[i] >= -w[i])
        mdl.add_constraint(mdl.sum(g[i] for i in tickers) <= Total_marketexposure)

    # cvar objective function
    # cvar_alpha = 0.05 = % of worst-case scenarios we're focusing on
    scale = 1.0 / (cvar_alpha * S)
    mdl.minimize(var_threshold + scale * mdl.sum(z))

    # scenario-wise constraints — each scenario = one trading day
    for s in range(S):
        portfolio_return_s = mdl.sum(w[i] * float(R[s, i]) for i in tickers)
        mdl.add_constraint(z[s] >= -portfolio_return_s - var_threshold)

    mdl.add_constraint(mdl.sum(w) == 1)
    mdl.add_constraint(mdl.sum(mu[i] * w[i] for i in tickers) >= target_ret)

    solution = mdl.solve(log_output=False)
    if solution is None:
        return None

    weights    = np.array([w[i].solution_value for i in tickers])
    cvar_value = solution.get_objective_value()
    port_return = float(np.dot(mu, weights))
    port_risk   = float(np.sqrt(weights @ sigma @ weights))
    sharpe      = (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0

    return {
        "weights"           : weights,
        "portfolio_return"  : port_return,
        "portfolio_risk"    : port_risk,
        "sharpe_ratio"      : sharpe,
        "portfolio_variance": port_risk ** 2,
        "cvar"              : cvar_value,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Download adjusted closing prices. Cached 1 hour to avoid repeated downloads."""
    raw    = yf.download(list(tickers), start=start, end=end,
                         auto_adjust=True, progress=False)
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    min_rows = int(0.95 * len(prices))
    prices   = prices.dropna(axis=1, thresh=min_rows).ffill().dropna()
    return prices


def compute_mu_sigma(train_prices: pd.DataFrame, alpha: float = 0.6):
    """
    Hybrid expected return and covariance — exactly as in notebook:
        mu_stock       = daily_returns.mean() * 252
        momentum_60d   = train_prices.pct_change(125).iloc[-1]
        mu_momentum    = momentum_60d * (252/125)
        mu             = 0.6 * mu_stock + 0.4 * mu_momentum
    alpha parameter lets the slider override the 0.6 blend.
    """
    daily_ret    = train_prices.pct_change().dropna()
    mu_hist      = daily_ret.mean() * 252                      # annualised historical mean
    momentum_raw = train_prices.pct_change(125).iloc[-1]       # 6-month momentum
    mu_momentum  = momentum_raw * (252 / 125)                  # annualised
    mu           = alpha * mu_hist + (1 - alpha) * mu_momentum # hybrid blend
    sigma        = (daily_ret.cov() * 252).values              # annualised cov matrix
    return mu, sigma, daily_ret


def compute_test_metrics(weights, test_returns, rf):
    """
    Out-of-sample test metrics: annualised return, vol, Sharpe,
    max drawdown (same formula as notebook's maxpeak_toughloss),
    and the cumulative return series for plotting.
    """
    daily  = test_returns.values @ weights
    ann_r  = daily.mean() * 252
    ann_v  = daily.std()  * np.sqrt(252)
    sharpe = (ann_r - rf) / ann_v if ann_v > 0 else 0.0
    cum    = pd.Series((1 + daily).cumprod(), index=test_returns.index)
    peak   = cum.cummax()
    mdd    = ((cum - peak) / peak).min()
    return {
        "ann_return" : ann_r,
        "ann_vol"    : ann_v,
        "sharpe"     : sharpe,
        "max_dd"     : mdd,
        "cum_series" : cum,
    }

with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("📈 Select Stocks")
    selected_tickers = st.multiselect(
        label="Pick up to 15 Nifty 50 stocks",
        options=NIFTY50_TICKERS,
        default=DEFAULT_TICKERS,
        max_selections=15,
        format_func=lambda x: x.replace(".NS", ""),
        help="Max 15 to stay within DOCplex free-tier (1000 variables)",
    )

    st.divider()

    st.subheader("📅 Date Range")
    col_l, col_r = st.columns(2)
    train_start = col_l.date_input("Train start", value=date(2020, 1, 1))
    train_end   = col_r.date_input("Train end",   value=date(2024, 12, 31))
    test_end    = st.date_input(
        "Test end (out-of-sample)",
        value=date(2025, 12, 31),
        help="Model is trained on Train start→end, evaluated on Train end→Test end",
    )

    st.divider()

    st.subheader("🎯 Parameters")

    target_pct = st.slider(
        "Target Annual Return (%)",
        min_value=5, max_value=50, value=20, step=1,
        help="Optimizer meets this return target while minimising risk",
    )
    target_ret = target_pct / 100.0

    robust_d = st.slider(
        "Robust Delta (δ)",
        min_value=0.0, max_value=1.5, value=0.85, step=0.05,
        help=(
            "δ=0 → pure Markowitz. Higher δ = more conservative. "
            "Notebook tests δ=0.05 to 1.10; δ=0.85 gave the best Sharpe."
        ),
    )

    rf_pct = st.slider(
        "Risk-Free Rate (%)",
        min_value=4, max_value=12, value=7, step=1,
        help="Used in Sharpe ratio. ~7% = Indian 10Y G-Sec yield",
    )
    rf = rf_pct / 100.0

    alpha_hybrid = st.slider(
        "Hybrid μ weight on History (α)",
        min_value=0.0, max_value=1.0, value=0.6, step=0.1,
        help="μ = α × historical_mean + (1−α) × momentum. Notebook uses α=0.6.",
    )

    st.divider()

    run_clicked = st.button(
        "🚀  Run Optimization",
        type="primary",
        use_container_width=True,
    )


st.title("📊 NSE Portfolio Optimizer")
st.caption(
    "Markowitz  ·  Robust  ·  CVaR  —  IBM DOCplex QP solver  |  "
    "Live NSE data via Yahoo Finance"
)
st.divider()

if not run_clicked:
    a, b, c = st.columns(3)
    a.info("**Step 1**\nSelect stocks in the sidebar (10 pre-selected by default)")
    b.info("**Step 2**\nSet training dates and target return")
    c.info("**Step 3**\nClick **Run Optimization** ← big blue button")
    st.stop()

if len(selected_tickers) < 3:
    st.error("Please select at least 3 stocks.")
    st.stop()

if train_end >= test_end:
    st.error("Test end date must be after train end date.")
    st.stop()


with st.spinner("📥  Downloading NSE price data from Yahoo Finance..."):
    prices = download_prices(
        tickers=tuple(sorted(selected_tickers)),
        start=str(train_start),
        end=str(test_end),
    )

if prices.empty or prices.shape[1] < 2:
    st.error("Could not download enough data. Try different tickers or dates.")
    st.stop()

train_prices = prices[str(train_start) : str(train_end)]
test_prices  = prices[str(train_end)   : str(test_end)].iloc[1:]

tickers_clean = [t.replace(".NS", "") for t in prices.columns]
N = len(tickers_clean)

st.success(
    f"✓ {N} stocks downloaded  |  "
    f"Train: {train_prices.index[0].date()} → {train_prices.index[-1].date()} "
    f"({len(train_prices)} days)  |  "
    f"Test: {test_prices.index[0].date()} → {test_prices.index[-1].date()} "
    f"({len(test_prices)} days)"
)


# ── 7b — COMPUTE mu AND sigma ────────────────────────────────
with st.spinner("🔢  Computing expected returns and covariance matrix..."):
    mu, sigma, daily_ret = compute_mu_sigma(train_prices, alpha=alpha_hybrid)

test_returns = test_prices.pct_change().dropna()

# CVaR scenario count — notebook comment explains the limit:
# Free CPLEX: 1000 variables max.
# CVaR variables = N(w) + N(g) + S(z) + 1(VaR)
# For N=10 stocks that's 21 + S  →  S ≤ 979 technically, but
# the notebook CAPS at 800 for safety. We do the same.
# We go lower than 800 only when daily_ret has fewer rows.
MAX_CVaR_SCENARIOS = 800   # same as notebook
n_scenarios = min(MAX_CVaR_SCENARIOS, len(daily_ret))


# ── 7c — RUN ALL THREE MODELS ────────────────────────────────
all_results = {}
all_metrics = {}

with st.spinner("⚙️  Running Basic Markowitz..."):
    r = optimize_portfolio(
        mu=mu.values, sigma=sigma,
        target_ret=target_ret,
        risk_free_rate=rf,
    )
    if r:
        all_results["Basic Markowitz"] = r
        all_metrics["Basic Markowitz"] = compute_test_metrics(
            r["weights"], test_returns, rf
        )
    else:
        st.warning("⚠️ Basic Markowitz: no feasible solution at this target return.")

with st.spinner(f"⚙️  Running Robust Optimisation (δ={robust_d})..."):
    r = run_robust_delta(
        mu=mu.values, sigma=sigma,
        target_ret=target_ret,
        robust_delta=robust_d,
        risk_free_rate=rf,
    )
    label_r = f"Robust δ={robust_d}"
    if r:
        all_results[label_r] = r
        all_metrics[label_r] = compute_test_metrics(
            r["weights"], test_returns, rf
        )
    else:
        st.warning(f"⚠️ Robust (δ={robust_d}): no feasible solution. Try a lower target return.")

with st.spinner(f"⚙️  Running CVaR 5% ({n_scenarios} scenarios)..."):
    cvar_scenarios = daily_ret.sample(n=n_scenarios, random_state=42)
    r = solve_cvar(
        mu=mu.values, sigma=sigma,
        target_ret=target_ret,
        scenarios=cvar_scenarios,
        cvar_alpha=0.05,
        allow_short=True,
        short_lower_bound=-0.30,
        Total_marketexposure=1.5,
        max_weight=1.0,
        risk_free_rate=rf,
    )
    if r:
        all_results["CVaR 5%"] = r
        all_metrics["CVaR 5%"] = compute_test_metrics(
            r["weights"], test_returns, rf
        )
    else:
        st.warning("⚠️ CVaR 5%: no feasible solution. Try a lower target return.")

if not all_results:
    st.error("All three models were infeasible. Lower the target return and try again.")
    st.stop()

st.success(f"✅  {len(all_results)}/3 models solved successfully.")
st.divider()


# ── 7d — SHOW RESULTS IN 4 TABS ──────────────────────────────
tab_table, tab_cum, tab_frontier, tab_weights = st.tabs([
    "📋  Performance Table",
    "📈  Cumulative Returns",
    "🔵  Efficient Frontier",
    "🥧  Portfolio Weights",
])


# ════════════════════════════════════════════════════════════
# TAB 1 — PERFORMANCE TABLE
# ════════════════════════════════════════════════════════════
with tab_table:
    st.subheader("Model Comparison — Out-of-Sample Test Period")
    st.caption(
        f"Trained on {train_start} → {train_end}  |  "
        f"Tested on {train_end} → {test_end}  |  "
        f"Risk-free rate = {rf_pct}%"
    )

    rows = []
    for name, res in all_results.items():
        m = all_metrics[name]
        rows.append({
            "Model"           : name,
            "Train Return"    : f"{res['portfolio_return']:.1%}",
            "Train Risk"      : f"{res['portfolio_risk']:.1%}",
            "Train Sharpe"    : f"{res['sharpe_ratio']:.2f}",
            "Test Return"     : f"{m['ann_return']:.1%}",
            "Test Volatility" : f"{m['ann_vol']:.1%}",
            "Test Sharpe"     : f"{m['sharpe']:.2f}",
            "Max Drawdown"    : f"{m['max_dd']:.1%}",
        })

    df_compare = pd.DataFrame(rows).set_index("Model")
    st.dataframe(df_compare, use_container_width=True)

    best_name = max(all_metrics, key=lambda k: all_metrics[k]["sharpe"])
    best_m    = all_metrics[best_name]
    best_r    = all_results[best_name]

    st.markdown(f"### 🏆 Best model: **{best_name}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Return",     f"{best_m['ann_return']:.1%}")
    c2.metric("Test Volatility", f"{best_m['ann_vol']:.1%}")
    c3.metric("Test Sharpe",     f"{best_m['sharpe']:.2f}")
    c4.metric("Max Drawdown",    f"{best_m['max_dd']:.1%}")

    with st.expander("📌 Show parameters used in this run"):
        st.json({
            "stocks"          : tickers_clean,
            "train_period"    : f"{train_start} → {train_end}",
            "test_period"     : f"{train_end} → {test_end}",
            "target_return"   : f"{target_pct}%",
            "robust_delta"    : robust_d,
            "risk_free_rate"  : f"{rf_pct}%",
            "alpha_hybrid_mu" : alpha_hybrid,
            "cvar_scenarios"  : n_scenarios,
        })


# ════════════════════════════════════════════════════════════
# TAB 2 — CUMULATIVE RETURNS
# ════════════════════════════════════════════════════════════
with tab_cum:
    st.subheader("Growth of ₹1 Invested — Out-of-Sample Test Period")

    COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    fig    = go.Figure()

    for (name, res), color in zip(all_results.items(), COLORS):
        cum = all_metrics[name]["cum_series"]
        fig.add_trace(go.Scatter(
            x    = cum.index,
            y    = cum.values,
            name = name,
            mode = "lines",
            line = dict(color=color, width=2.5),
        ))

    # Equal-weight benchmark (1/N) — same idea as notebook's comparison
    ew_weights = np.ones(N) / N
    ew_daily   = test_returns.values @ ew_weights
    ew_cum     = pd.Series((1 + ew_daily).cumprod(), index=test_returns.index)
    fig.add_trace(go.Scatter(
        x    = ew_cum.index,
        y    = ew_cum.values,
        name = "Equal-Weight (1/N)",
        mode = "lines",
        line = dict(color="gray", width=1.5, dash="dash"),
    ))

    fig.add_hline(y=1.0, line_dash="dot", line_color="white",
                  line_width=1, annotation_text="Starting ₹1")

    fig.update_layout(
        xaxis_title = "Date",
        yaxis_title = "Portfolio Value (₹1 = start)",
        hovermode   = "x unified",
        height      = 460,
        legend      = dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin      = dict(l=50, r=20, t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — EFFICIENT FRONTIER
# ════════════════════════════════════════════════════════════
with tab_frontier:
    st.subheader("Mean-Variance Efficient Frontier")
    st.caption(
        "Each dot = one DOCplex Markowitz portfolio at a different target return. "
        "Colour = Sharpe ratio. Your three model portfolios are plotted as large symbols."
    )

    min_target = float(mu.min()) + 0.02
    max_target = float(mu.max()) - 0.02
    ef_targets = np.linspace(min_target, max_target, 25)

    ef_risks, ef_rets, ef_sharpes = [], [], []

    progress_bar = st.progress(0, text="Computing frontier points...")
    for k, tr in enumerate(ef_targets):
        r = optimize_portfolio(mu=mu.values, sigma=sigma,
                               target_ret=tr, risk_free_rate=rf)
        if r:
            ef_risks.append(r["portfolio_risk"])
            ef_rets.append(r["portfolio_return"])
            ef_sharpes.append(r["sharpe_ratio"])
        progress_bar.progress(
            int((k + 1) / len(ef_targets) * 100),
            text=f"Computing frontier... {k+1}/{len(ef_targets)}"
        )
    progress_bar.empty()

    fig_ef = go.Figure()

    if ef_risks:
        fig_ef.add_trace(go.Scatter(
            x    = [r * 100 for r in ef_risks],
            y    = [r * 100 for r in ef_rets],
            mode = "markers+lines",
            name = "Efficient Frontier",
            marker = dict(
                color      = ef_sharpes,
                colorscale = "Viridis",
                size       = 9,
                showscale  = True,
                colorbar   = dict(title="Sharpe", len=0.6),
            ),
            line = dict(color="rgba(180,180,180,0.4)", width=1.5),
            hovertemplate = "Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        ))

    stock_risks   = np.sqrt(np.diag(sigma))
    stock_returns = mu.values
    fig_ef.add_trace(go.Scatter(
        x    = stock_risks * 100,
        y    = stock_returns * 100,
        mode = "markers+text",
        name = "Individual Stocks",
        text = tickers_clean,
        textposition = "top center",
        textfont     = dict(size=9),
        marker = dict(symbol="x", size=9, color="salmon"),
    ))

    marker_symbols = ["circle", "square", "triangle-up"]
    marker_colors  = ["dodgerblue", "tomato", "limegreen"]
    for (name, res), sym, col in zip(all_results.items(),
                                      marker_symbols, marker_colors):
        fig_ef.add_trace(go.Scatter(
            x    = [res["portfolio_risk"]   * 100],
            y    = [res["portfolio_return"] * 100],
            mode = "markers+text",
            name = name,
            text = [name], textposition = "bottom right",
            textfont = dict(size=10),
            marker   = dict(symbol=sym, size=14, color=col,
                            line=dict(color="white", width=2)),
        ))

    fig_ef.add_hline(
        y=rf * 100, line_dash="dot", line_color="gold", line_width=1.5,
        annotation_text=f"Risk-free rate ({rf_pct}%)",
    )

    fig_ef.update_layout(
        xaxis_title = "Annual Volatility (Risk) %",
        yaxis_title = "Expected Annual Return %",
        hovermode   = "closest",
        height      = 570,
        margin      = dict(l=50, r=20, t=30, b=50),
    )
    st.plotly_chart(fig_ef, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 4 — PORTFOLIO WEIGHTS
# ════════════════════════════════════════════════════════════
with tab_weights:
    st.subheader("Optimal Portfolio Weight Allocations")

    chosen_model = st.selectbox(
        "Select model to inspect:",
        options=list(all_results.keys()),
    )
    w = all_results[chosen_model]["weights"]

    idx_sorted = np.argsort(np.abs(w))[::-1]
    w_sorted   = w[idx_sorted]
    t_sorted   = [tickers_clean[i] for i in idx_sorted]

    bar_colors = ["#2196F3" if wi >= 0 else "#F44336" for wi in w_sorted]

    fig_w = go.Figure(go.Bar(
        x            = w_sorted * 100,
        y            = t_sorted,
        orientation  = "h",
        marker_color = bar_colors,
        text         = [f"{wi*100:.1f}%" for wi in w_sorted],
        textposition = "outside",
    ))
    fig_w.add_vline(x=0, line_color="white", line_width=1)
    fig_w.update_layout(
        title       = f"Portfolio Weights — {chosen_model}",
        xaxis_title = "Weight (%)    Blue = Long   |   Red = Short",
        height      = max(400, N * 26),
        bargap      = 0.3,
        margin      = dict(l=80, r=80, t=50, b=30),
    )
    st.plotly_chart(fig_w, use_container_width=True)

    long_exp  = float(w[w > 0].sum()) * 100
    short_exp = float(w[w < 0].sum()) * 100
    hhi       = float(np.sum(w ** 2))
    eff_n     = round(1 / hhi, 1) if hhi > 0 else N

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Long Exposure",       f"{long_exp:.1f}%")
    c2.metric("Short Exposure",      f"{short_exp:.1f}%")
    c3.metric("HHI (concentration)", f"{hhi:.4f}")
    c4.metric("Effective N (1/HHI)", f"{eff_n} stocks")


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Streamlit · IBM DOCplex · yfinance · Plotly  |  "
    "Data: Yahoo Finance  |  DOCplex Community Edition: 1000 var limit"
)
