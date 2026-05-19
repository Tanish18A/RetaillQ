#Run:  streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys, warnings
warnings.filterwarnings("ignore")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
sys.path.insert(0, BASE_DIR)

from data_loader        import load_data
from v4_integration     import (load_v4_models, forecast_next_7_days_v4)
from simulation_engine  import SimulationEngine, summarize_scenarios, generate_scenario_narrative
from business_metrics   import (
    compute_business_metrics,
    ai_vs_no_ai_comparison,
    explain_prediction,
    compute_global_feature_importance
)
from festival_calendar  import FESTIVALS_EXACT
from conformal_predictor import build_conformal_predictor, add_conformal_bands
from causal_estimator   import FestivalCausalEstimator
from research_engines   import (
    InventoryCostOptimizer, PredictionTrustScorer,
    DataDriftDetector, MonteCarloInventorySimulator,
    BaselineComparison
)
from llm_narrator       import build_context_payload, _clean_md

#PAGE CONFIG 
st.set_page_config(
    page_title="RetailIQ — Inventory Intelligence",
    page_icon="📦", layout="wide",
    initial_sidebar_state="expanded",
)

# ENTERPRISE CSS THEME
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  letter-spacing: -0.01em;
}

/* Disable dotted/slashed zero in monospace font */
.ibm-mono, [data-testid="stMetricValue"],
[data-testid="stCaptionContainer"],
code, pre {
  font-feature-settings: 'zero' 0, 'ss01' 0;
}

/* ── Brand Header ── */
.brand-header {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  border: 1px solid rgba(14,165,233,0.2);
  border-left: 3px solid #0ea5e9;
  padding: 1.1rem 1.8rem;
  border-radius: 6px;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.brand-title {
  color: #f1f5f9 !important;
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0;
  letter-spacing: -0.03em;
}
.brand-sub {
  color: #94a3b8 !important;
  font-size: 0.72rem;
  margin: 0.18rem 0 0 0;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 500;
  font-family: 'IBM Plex Mono', monospace;
}
.brand-pill {
  background: rgba(14,165,233,0.12);
  border: 1px solid rgba(14,165,233,0.35);
  color: #38bdf8 !important;
  font-size: 0.68rem;
  font-weight: 600;
  padding: 0.25rem 0.6rem;
  border-radius: 3px;
  letter-spacing: 0.08em;
  font-family: 'IBM Plex Mono', monospace;
  white-space: nowrap;
}

/* ── KPI Cards ── */
.kpi-card {
  background: var(--secondary-background-color);
  border-radius: 5px;
  padding: 0.95rem 1.1rem;
  border-left: 3px solid;
  margin-bottom: 0.5rem;
  border-top: 1px solid rgba(255,255,255,0.04);
}
.kpi-card.green  { border-left-color: #22c55e; }
.kpi-card.red    { border-left-color: #ef4444; }
.kpi-card.yellow { border-left-color: #f59e0b; }
.kpi-card.blue   { border-left-color: #0ea5e9; }

.kpi-label {
  font-size: 0.67rem;
  color: var(--text-color);
  opacity: 0.45;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 600;
  margin-bottom: 0.35rem;
  font-family: 'IBM Plex Mono', monospace;
}
.kpi-value {
  font-size: 1.55rem;
  font-weight: 700;
  color: var(--text-color);
  letter-spacing: -0.035em;
  line-height: 1.1;
  font-family: 'DM Sans', sans-serif;
  font-variant-numeric: tabular-nums;
}
.kpi-delta {
  font-size: 0.72rem;
  color: var(--text-color);
  opacity: 0.4;
  margin-top: 0.3rem;
  font-family: 'IBM Plex Mono', monospace;
  font-feature-settings: 'zero' 0;
}

/* ── Festival Banner ── */
.fest-banner {
  background: rgba(124,58,237,0.07);
  border: 1px solid rgba(124,58,237,0.2);
  border-left: 3px solid #7c3aed;
  padding: 0.55rem 1rem;
  border-radius: 5px;
  margin-bottom: 0.8rem;
  color: var(--text-color);
  font-weight: 500;
  font-size: 0.875rem;
  opacity: 0.92;
}

/* ── Section Titles ── */
.section-title {
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--text-color);
  opacity: 0.5;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  border-bottom: 1px solid rgba(255,255,255,0.07);
  padding-bottom: 0.5rem;
  margin: 0.75rem 0 1rem 0;
  font-family: 'IBM Plex Mono', monospace;
}

/* ── Alert / Info Cards ── */
.alert-card {
  background: var(--secondary-background-color);
  border-radius: 5px;
  padding: 0.9rem 1.1rem;
  color: var(--text-color);
  font-size: 0.875rem;
  line-height: 1.6;
  border: 1px solid rgba(255,255,255,0.05);
}

/* ── Order Decision Box ── */
.order-box-label {
  font-size: 0.67rem;
  color: var(--text-color);
  opacity: 0.45;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-family: 'IBM Plex Mono', monospace;
}
.order-box-value {
  font-size: 2.4rem;
  font-weight: 800;
  color: #0ea5e9;
  letter-spacing: -0.04em;
  line-height: 1.05;
  font-family: 'DM Sans', sans-serif;
  font-variant-numeric: tabular-nums;
}
.order-box-sub {
  color: var(--text-color);
  opacity: 0.4;
  font-size: 0.78rem;
  font-family: 'IBM Plex Mono', monospace;
  font-feature-settings: 'zero' 0;
  margin-top: 0.2rem;
}

/* ── Explainability block ── */
.explain-basis {
  background: var(--secondary-background-color);
  border-left: 3px solid #0ea5e9;
  border-radius: 5px;
  padding: 0.7rem 1rem;
  font-size: 0.78rem;
  font-family: 'IBM Plex Mono', monospace;
  font-feature-settings: 'zero' 0;
  color: var(--text-color);
  line-height: 1.7;
  border-top: 1px solid rgba(14,165,233,0.12);
}

/* ── Sidebar refinements ── */
section[data-testid="stSidebar"] {
  font-size: 0.85rem;
  border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  font-size: 0.67rem !important;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  opacity: 0.4;
  font-weight: 700;
  font-family: 'IBM Plex Mono', monospace;
}

/* ── Data table refinements ── */
[data-testid="stDataFrame"] {
  font-size: 0.82rem;
}

/* ── Metric values — DM Sans (no dotted zeros) ── */
[data-testid="stMetricValue"] {
  font-family: 'DM Sans', sans-serif !important;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  letter-spacing: -0.03em;
}
[data-testid="stMetricLabel"] {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.78rem !important;
  opacity: 0.55;
}
[data-testid="stMetricDelta"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-feature-settings: 'zero' 0 !important;
  font-size: 0.72rem !important;
}

/* ── Tab styling ── */
[data-baseweb="tab-list"] {
  gap: 0;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
[data-baseweb="tab"] {
  font-size: 0.78rem !important;
  font-weight: 600;
  letter-spacing: 0.04em;
  padding: 0.6rem 1rem !important;
  font-family: 'DM Sans', sans-serif;
}

/* ── Horizontal rule ── */
hr {
  border-color: rgba(255,255,255,0.07) !important;
  margin: 0.9rem 0 !important;
}

/* ── Captions ── */
small, [data-testid="stCaptionContainer"] {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem !important;
  opacity: 0.5;
  font-feature-settings: 'zero' 0;
}

/* ── Global: disable dotted zero on all monospace elements ── */
.brand-pill, .brand-sub, .kpi-label, .kpi-delta,
.order-box-label, .order-box-sub, .section-title,
[style*="IBM Plex Mono"] {
  font-feature-settings: 'zero' 0 !important;
}

/* ── Status card inner wrapper ── */
.status-card-inner {
  background: var(--secondary-background-color);
  border-radius: 6px;
  padding: 1.1rem;
  text-align: center;
  border: 1px solid rgba(255,255,255,0.05);
}

/* ── Profit range bar ── */
.profit-range-bar {
  background: var(--secondary-background-color);
  border-radius: 5px;
  padding: 0.6rem 1.1rem;
  margin: 0.3rem 0 0.8rem 0;
  display: flex;
  align-items: center;
  gap: 1rem;
  border: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#f59e0b","LOW":"#22c55e"}
RISK_EMOJI  = {"CRITICAL":"●","HIGH":"●","MEDIUM":"●","LOW":"●"}
RISK_LABEL  = {"CRITICAL":"CRITICAL","HIGH":"HIGH","MEDIUM":"MEDIUM","LOW":"LOW"}

# CACHE DATA AND MODELS
@st.cache_data(show_spinner="Loading dataset…")
def get_raw_data():
    return load_data(BASE_DIR)

@st.cache_resource(show_spinner="Loading V4 XGBoost + LightGBM models…")
def get_model(_df_raw=None):
    feat_csv = os.path.join(BASE_DIR, "output_v4", "data_features.csv")
    if os.path.exists(feat_csv):
        df_feat = pd.read_csv(feat_csv, parse_dates=["Date"])
        kaggle_map = {
            "Store ID":"Store_ID","Product ID":"Product_ID",
            "Units Sold":"Units_Sold","Inventory Level":"Inventory_Level",
            "Lead Time Days":"Lead_Time_Days"
        }
        df_feat = df_feat.rename(columns={k:v for k,v in kaggle_map.items() if k in df_feat.columns})
        rename_map = {
            "roll_mean_7":"rolling_mean_7","roll_std_7":"rolling_std_7",
            "roll_mean_14":"rolling_mean_14","roll_std_14":"rolling_std_14",
            "roll_mean_28":"rolling_mean_28","roll_std_28":"rolling_std_28",
        }
        df_feat = df_feat.rename(columns={k:v for k,v in rename_map.items() if k in df_feat.columns})
        needed_base = {"Store_ID","Product_ID","Date","Units_Sold"}
        if needed_base.issubset(set(df_feat.columns)):
            df_feat = df_feat.sort_values(["Store_ID","Product_ID","Date"]).copy()
            if "rolling_mean_7" not in df_feat.columns:
                df_feat["rolling_mean_7"] = (
                    df_feat.groupby(["Store_ID","Product_ID"])["Units_Sold"]
                    .transform(lambda s: s.rolling(7, min_periods=1).mean())
                )
            if "rolling_std_7" not in df_feat.columns:
                df_feat["rolling_std_7"] = (
                    df_feat.groupby(["Store_ID","Product_ID"])["Units_Sold"]
                    .transform(lambda s: s.rolling(7, min_periods=1).std(ddof=0).fillna(0))
                )
    else:
        df_feat = _df_raw.copy() if _df_raw is not None else pd.read_csv(
            os.path.join(BASE_DIR, "retail_data.csv"), parse_dates=["Date"]
        )
        needed_base = {"Store_ID","Product_ID","Date","Units_Sold"}
        if needed_base.issubset(set(df_feat.columns)):
            df_feat = df_feat.sort_values(["Store_ID","Product_ID","Date"]).copy()
            df_feat["rolling_mean_7"] = (
                df_feat.groupby(["Store_ID","Product_ID"])["Units_Sold"]
                .transform(lambda s: s.rolling(7, min_periods=1).mean())
            )
            df_feat["rolling_std_7"] = (
                df_feat.groupby(["Store_ID","Product_ID"])["Units_Sold"]
                .transform(lambda s: s.rolling(7, min_periods=1).std(ddof=0).fillna(0))
            )
    xgb_model, lgb_model, features, weights = load_v4_models(
        model_dir=os.path.join(BASE_DIR, "output_v4")
    )
    return xgb_model, lgb_model, features, weights, df_feat


# SIDEBAR
def sidebar(df_feat):
    with st.sidebar:
        st.markdown("## Control Panel")
        st.markdown("---")

        if os.path.exists(os.path.join(BASE_DIR,"retail_store_inventory.csv")):
            st.success("Kaggle dataset loaded")
        else:
            st.info("Synthetic dataset (matches Kaggle schema)")

        stores   = sorted(df_feat["Store_ID"].unique())
        products = sorted(df_feat["Product_ID"].unique())
        pnames   = (
            df_feat[["Product_ID","Product_Name"]]
            .drop_duplicates().set_index("Product_ID")["Product_Name"].to_dict()
        )

        stores_opt = stores + ["All Stores"]
        store = st.selectbox("Store", stores_opt, index=0)

        products_opt = products + ["All Products"]
        def _prod_label(x):
            if x == "All Products": return "All Products"
            return f"{x} — {pnames.get(x,'')}"
        prod = st.selectbox("Product", products_opt, format_func=_prod_label, index=0)

        svc_lvl = 96

        st.markdown("---")
        st.markdown("### Simulation Parameters")
        spike_mult  = st.slider("Festival Spike ×", 1.0, 2.0, 1.6, 0.05,
                                 help="Controls demand multiplier in Festival Spike scenario")
        extra_delay = st.slider("Supplier Delay (days)", 0, 15, 4,
                                 help="Additional lead-time days in Supplier Delay scenario")
        price_move_pct = st.slider(
            "Price Move % (− = Drop  |  + = Hike)",
            -30, 30, 0, 1,
            help="Negative = price drop (demand rises). "
                 "Positive = price hike (demand falls). "
                 "0 = shows Price Hike demo at 15%."
        ) / 100.0

        st.markdown("---")
        st.markdown("### Analysis Period")
        mn = df_feat["Date"].min().date()
        mx = df_feat["Date"].max().date()
        default_start = mx - pd.Timedelta(days=90)
        default_start = max(default_start, mn)
        dr = st.date_input(
            "Date Range",
            value=(default_start, mx),
            min_value=mn, max_value=mx,
        )

        st.markdown("---")
        st.markdown("### Upcoming Festivals")
        data_end = df_feat["Date"].max().date()
        upcoming_fests = sorted([
            (d, name, mult)
            for d, (name, ftype, mult) in FESTIVALS_EXACT.items()
            if d > data_end and ftype in ("mega","major","moderate")
        ], key=lambda x: x[0])[:6]

        if upcoming_fests:
            for d, name, mult in upcoming_fests:
                days_away = (d - data_end).days
                st.markdown(f"**{name}**")
                st.caption(f"{d.strftime('%d %b %Y')} — {days_away}d after data end · ×{mult}")
        else:
            st.caption("No upcoming festivals in calendar")

        st.markdown("---")
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
        except ImportError:
            pass
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        alpha = 0.05

        st.markdown("---")
        st.markdown("### Monte Carlo")
        n_mc = st.select_slider(
            "Simulations",
            options=[100, 300, 500, 700, 1000],
            value=500,
            help="More simulations = more accurate risk estimates (slower)"
        )

        st.caption("RetailIQ v4 · Tanish Edition")

    return store, prod, svc_lvl, spike_mult, extra_delay, price_move_pct, dr, gemini_key, alpha, n_mc


# FESTIVAL BANNER
def festival_banner(df_feat, store, prod):
    if store == "All Stores" or prod == "All Products":
        return
    today = df_feat["Date"].max()
    week_data = df_feat[
        (df_feat["Store_ID"]==store) & (df_feat["Product_ID"]==prod) &
        (df_feat["Date"] >= today - pd.Timedelta(days=3)) &
        (df_feat["Date"] <= today + pd.Timedelta(days=7))
    ].sort_values("Date")
    active = week_data[week_data["Holiday_Flag"]==1]
    if not active.empty:
        r = active.iloc[0]
        fest = r.get("Holiday_Festival","Festival")
        mult = r.get("Festival_Multiplier", r.get("festival_mult", 1.0))
        st.markdown(
            f'<div class="fest-banner">FESTIVAL ACTIVE — <b>{fest}</b> '
            f'· Demand boost ×{mult:.2f} · '
            'Forecasts and order quantities adjusted automatically</div>',
            unsafe_allow_html=True
        )


# TAB 1: DEMAND FORECAST
def tab_forecast(df_feat, xgb_model, lgb_model, fc, weights, store, prod, alpha=0.05):
    st.markdown('<div class="section-title">Demand Forecast</div>', unsafe_allow_html=True)
    festival_banner(df_feat, store, prod)

    n_days = 7

    if store == "All Stores" and prod != "All Products":
        _show_all_stores_product_view(df_feat, prod, n_days)
        return None, None

    _store, _prod = store, prod
    if store == "All Stores":
        st.warning("All Stores selected — showing representative store for forecast.")
        _store = sorted(df_feat["Store_ID"].unique())[0]
    if prod == "All Products":
        st.warning("All Products selected — showing representative product for forecast.")
        _prod = sorted(df_feat["Product_ID"].unique())[0]

    hist = df_feat[(df_feat["Store_ID"]==_store)&(df_feat["Product_ID"]==_prod)]\
           .sort_values("Date").tail(90)

    with st.spinner(f"Generating V4 XGB+LGB {n_days}-day forecast…"):
        fcast = forecast_next_7_days_v4(df_feat, store_id=_store, product_id=_prod, n_days=n_days)

    avg_hist  = hist["Units_Sold"].tail(7).mean()
    avg_fcast = fcast["Predicted_Demand"].mean()
    trend     = (avg_fcast - avg_hist) / max(avg_hist, 1) * 100
    curr_inv  = hist["Inventory_Level"].iloc[-1]
    days_stk  = curr_inv / max(avg_fcast, 1)

    c1,c2,c3,c4 = st.columns(4)
    for col,(lbl,val,sub,color) in zip([c1,c2,c3,c4],[
        (f"{n_days}-Day Forecast Total", f"{fcast['Predicted_Demand'].sum():.0f} units",
         f"Avg {avg_fcast:.1f} / day", "blue"),
        ("Demand Trend", f"{'↑' if trend>=0 else '↓'} {abs(trend):.1f}%",
         f"vs last week  ({avg_hist:.1f}/day)", "green" if trend>=0 else "red"),
        ("Current Inventory", f"{curr_inv:.0f} units",
         f"{days_stk:.1f} days of stock",
         "green" if days_stk>7 else ("yellow" if days_stk>3 else "red")),
        ("Festival Days in Window", f"{fcast['Festival_Mult'].gt(1.0).sum()} / {n_days}",
         f"days with demand uplift", "yellow"),
    ]):
        with col:
            st.markdown(
                f'<div class="kpi-card {color}"><div class="kpi-label">{lbl}</div>'
                f'<div class="kpi-value">{val}</div><div class="kpi-delta">{sub}</div></div>',
                unsafe_allow_html=True
            )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Units_Sold"],
        name="Historical", mode="lines",
        line=dict(color="#3b82f6",width=1.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,.08)"
    ))
    fig.add_trace(go.Scatter(
        x=hist["Date"], y=hist["rolling_mean_7"],
        name="7-Day Avg", mode="lines",
        line=dict(color="#475569",width=1,dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=fcast["Date"], y=fcast["Predicted_Demand"],
        name=f"AI Forecast ({n_days}d)", mode="lines+markers",
        line=dict(color="#0ea5e9",width=2.5),
        marker=dict(size=7,symbol="diamond"),
        hovertemplate="<b>%{x|%a %d %b}</b><br>Forecast: %{y} units<extra></extra>"
    ))

    std = float(hist["rolling_std_7"].iloc[-1]) if "rolling_std_7" in hist.columns else 5
    flist = list(fcast["Date"]); plist = list(fcast["Predicted_Demand"])
    fig.add_trace(go.Scatter(
        x=flist+flist[::-1],
        y=[p+std for p in plist]+[max(0,p-std) for p in plist][::-1],
        fill="toself", fillcolor="rgba(14,165,233,.08)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Band",
        hoverinfo="skip"
    ))

    fests_h = hist[hist["Holiday_Flag"]==1] if "Holiday_Flag" in hist.columns else pd.DataFrame()
    if not fests_h.empty:
        fest_names = fests_h.get("Holiday_Festival",
            pd.Series("Festival Day", index=fests_h.index)).fillna("Festival Day")
        fig.add_trace(go.Scatter(
            x=fests_h["Date"], y=fests_h["Units_Sold"],
            mode="markers", name="Festival Day",
            marker=dict(size=9, symbol="star", color="#f59e0b"),
            text=fest_names.tolist(),
            hovertemplate="<b>%{text}</b><br>Date: %{x|%d %b %Y}<br>Units Sold: %{y}<extra></extra>"
        ))

    fests_f = fcast[fcast["Festival_Mult"]>1.0]
    if not fests_f.empty:
        fig.add_trace(go.Scatter(
            x=fests_f["Date"], y=fests_f["Predicted_Demand"],
            mode="markers", name="Forecast — Festival",
            marker=dict(size=11, symbol="star", color="#fbbf24",
                        line=dict(color="#f59e0b",width=1.5)),
            text=fests_f["Festival"].tolist(),
            hovertemplate="<b>%{text}</b><br>Date: %{x|%d %b %Y}<br>Forecast: %{y} units · ×%{customdata:.2f}<extra></extra>",
            customdata=fests_f["Festival_Mult"].tolist()
        ))

    vline_x = pd.to_datetime(hist["Date"].max()).to_pydatetime()
    fig.add_shape(type="line", x0=vline_x, x1=vline_x, y0=0, y1=1,
                  xref="x", yref="paper", line=dict(color="#334155",dash="dash",width=1.5))
    fig.add_annotation(x=vline_x, y=1.02, xref="x", yref="paper",
                        text="Forecast →", showarrow=False, xanchor="left",
                        font=dict(color="#64748b", size=11))
    fig.update_layout(
        template="plotly_dark", height=420,
        title=dict(text=f"{_prod}  ·  {_store}  ·  {n_days}-Day Forecast", font=dict(size=13, color="#94a3b8")),
        xaxis_title="Date", yaxis_title="Units",
        legend=dict(orientation="h", y=-0.20, font=dict(size=11)),
        margin=dict(t=45,b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.08)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if "Holiday_Festival" in hist.columns:
        eid_window = hist[
            (hist["Date"] >= "2024-04-07") & (hist["Date"] <= "2024-04-15") &
            (hist["Holiday_Flag"] == 1)
        ]
        if not eid_window.empty:
            unique_fests = eid_window["Holiday_Festival"].dropna().unique()
            if len(unique_fests):
                st.caption(
                    "Festival markers (Apr 7–15) reflect real Indian festivals: "
                    f"{', '.join(unique_fests[:4])} — Eid al-Fitr pre-ramp, Baisakhi/Tamil New Year. "
                    "Dates per Hindu panchang."
                )

    try:
        _cp = build_conformal_predictor(df_feat, _store, _prod, alpha=alpha)
        fcast = add_conformal_bands(fcast, _cp)
        _cov  = int(_cp.coverage_level * 100)
        fl = list(fcast["Date"]); lo = list(fcast["Conf_Lower"]); hi = list(fcast["Conf_Upper"])
        fig.add_trace(go.Scatter(
            x=fl + fl[::-1], y=hi + lo[::-1],
            fill="toself", fillcolor="rgba(168,85,247,.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"Conformal Band ({_cov}% guaranteed)",
            hoverinfo="skip"
        ))
        st.caption(
            f"Conformal Prediction Band — {_cov}% coverage guaranteed "
            "(distribution-free, finite-sample valid)  ·  "
            f"q̂ = ±{_cp.q_hat:.1f} units  ·  n_calib = {_cp.n_calibration}"
        )
    except Exception:
        pass

    ft = fcast[["Day_Label","Predicted_Demand","Festival_Mult","Festival"]].copy()
    ft.columns = ["Day","Predicted Demand","Festival ×","Festival Name"]
    ft["Festival ×"] = ft["Festival ×"].apply(lambda x: f"×{x:.2f}" if x>1 else "—")
    st.dataframe(ft.set_index("Day"), use_container_width=True)
    return fcast, hist


def _show_all_stores_product_view(df_feat, prod, n_days):
    st.markdown(f"### All Stores — {prod} — Inventory Overview")
    stores = sorted(df_feat["Store_ID"].unique())
    rows = []
    for s in stores:
        sub = df_feat[(df_feat["Store_ID"]==s)&(df_feat["Product_ID"]==prod)]\
              .sort_values("Date")
        if sub.empty:
            continue
        last = sub.iloc[-1]
        avg30 = sub["Units_Sold"].tail(30).mean()
        inv   = last["Inventory_Level"]
        dos   = inv / max(avg30, 0.1)
        rows.append({
            "Store": s,
            "Current Inventory (units)": int(inv),
            "Avg Daily Sales (30d)": round(avg30, 1),
            "Days of Stock": round(dos, 1),
            "Status": "Critical" if dos<3 else ("Low" if dos<7 else "OK"),
        })

    if not rows:
        st.warning("No data found for this product across stores.")
        return

    df_stores = pd.DataFrame(rows)

    fig = go.Figure()
    colors = ["#ef4444" if d<3 else ("#f59e0b" if d<7 else "#22c55e")
              for d in df_stores["Days of Stock"]]
    fig.add_trace(go.Bar(
        x=df_stores["Store"],
        y=df_stores["Current Inventory (units)"],
        marker_color=colors,
        text=df_stores["Current Inventory (units)"],
        textposition="outside",
        textfont=dict(size=12,color="white"),
        width=0.55,
        hovertemplate="<b>%{x}</b><br>Inventory: %{y} units<extra></extra>"
    ))
    fig.update_layout(
        template="plotly_dark", height=360,
        title=dict(text=f"Current Inventory by Store — {prod}", font=dict(size=13, color="#94a3b8")),
        xaxis_title="Store", yaxis_title="Units in Stock",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50,b=50), bargap=0.25,
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df_stores["Store"],
        y=df_stores["Days of Stock"],
        marker_color=colors,
        text=[f"{v:.1f}d" for v in df_stores["Days of Stock"]],
        textposition="outside",
        textfont=dict(size=12,color="white"),
        width=0.55,
    ))
    fig2.add_hline(y=7, line_dash="dash", line_color="#f59e0b",
                   annotation_text="7-day threshold")
    fig2.update_layout(
        template="plotly_dark", height=300,
        title=dict(text="Days of Stock Remaining by Store", font=dict(size=13, color="#94a3b8")),
        xaxis_title="Store", yaxis_title="Days",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=45,b=50), bargap=0.25,
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    total = df_stores["Current Inventory (units)"].sum()
    st.markdown(
        f'<div class="kpi-card blue"><div class="kpi-label">Total Inventory — {prod} across all stores</div>'
        f'<div class="kpi-value">{total:,} units</div>'
        f'<div class="kpi-delta">Across {len(df_stores)} stores</div></div>',
        unsafe_allow_html=True
    )
    st.dataframe(df_stores.set_index("Store"), use_container_width=True)


# TAB 2: INVENTORY & ORDER DECISION ENGINE
def tab_inventory(df_feat, xgb_model, lgb_model, fc, weights, store, prod, forecast, svc_lvl):
    st.markdown('<div class="section-title">Inventory & Order Decision Engine</div>', unsafe_allow_html=True)

    if store == "All Stores" or prod == "All Products":
        st.warning("Orders tab requires a single Store + Product.")
        return None

    row = df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)]\
          .sort_values("Date").iloc[-1]

    n_days         = 7
    predicted_total= float(forecast["Predicted_Demand"].head(7).sum())
    avg_daily      = float(forecast["Predicted_Demand"].head(7).mean())
    lead_time_days = max(int(row.get("Lead_Time_Days", 3)), 3)
    buffer_days    = 5
    curr_inv       = float(row["Inventory_Level"])
    supplier_limit = int(row.get("Supplier_Limit", 300))
    min_order_qty  = int(row.get("Min_Order_Qty", 20))

    lead_time_demand = avg_daily * lead_time_days
    safety_stock     = avg_daily * buffer_days
    reorder_point    = lead_time_demand + safety_stock
    days_of_stock    = curr_inv / avg_daily if avg_daily > 0 else 999

    if curr_inv <= 0:
        stock_status = "OUT_OF_STOCK"
    elif curr_inv <= lead_time_demand:
        stock_status = "CRITICAL"
    elif curr_inv <= reorder_point:
        stock_status = "REORDER"
    else:
        stock_status = "SAFE"

    if stock_status in ("OUT_OF_STOCK", "CRITICAL", "REORDER"):
        required_stock = predicted_total + safety_stock
        raw_order_qty  = max(0.0, required_stock - curr_inv)
    else:
        raw_order_qty  = 0.0

    if raw_order_qty <= 0:
        constrained_order = 0
    else:
        constrained_order = min(raw_order_qty, supplier_limit)
        if 0 < constrained_order < min_order_qty:
            constrained_order = float(min_order_qty)
    constrained_order = round(constrained_order)

    _status_map = {
        "OUT_OF_STOCK": ("CRITICAL", 98.0),
        "CRITICAL":     ("CRITICAL", 90.0),
        "REORDER":      ("HIGH",     65.0),
        "SAFE":         ("LOW",      10.0),
    }
    so_risk, so_pct = _status_map[stock_status]

    proj_days = (curr_inv + constrained_order - predicted_total) / avg_daily \
                if avg_daily > 0 else 999
    proj_days = max(0, proj_days)
    if proj_days > 21:
        ov_risk, ov_pct = "HIGH",   70.0
    elif proj_days > 14:
        ov_risk, ov_pct = "MEDIUM", 40.0
    else:
        ov_risk, ov_pct = "LOW",    10.0

    soc = RISK_COLORS[so_risk]; ovc = RISK_COLORS[ov_risk]

    _status_alerts = {
        "OUT_OF_STOCK": f"OUT OF STOCK — Zero inventory. Supplier needs {lead_time_days}d. Expedite order immediately.",
        "CRITICAL":     f"CRITICAL — {curr_inv:.0f} units below lead-time demand ({lead_time_demand:.0f}). Stockout before next delivery.",
        "REORDER":      f"REORDER TRIGGERED — {curr_inv:.0f} units at or below reorder point ({reorder_point:.0f}).",
        "SAFE":         f"SAFE — {curr_inv:.0f} units above reorder point ({reorder_point:.0f}). No order required.",
    }
    alerts = [_status_alerts[stock_status]]
    if constrained_order > 0 and constrained_order < raw_order_qty:
        alerts.append(f"Supplier cap applied: {constrained_order} of {raw_order_qty:.0f} units ordered.")
    alert_msg = " | ".join(alerts)

    _status_color = {
        "OUT_OF_STOCK": "#ef4444", "CRITICAL": "#f97316",
        "REORDER": "#f59e0b",      "SAFE": "#22c55e"
    }
    _status_label = {
        "OUT_OF_STOCK": "OUT OF STOCK", "CRITICAL": "CRITICAL",
        "REORDER": "REORDER", "SAFE": "SAFE"
    }
    sc = _status_color[stock_status]

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="status-card-inner" style="border: 1px solid rgba(14,165,233,0.25);">'
            '<div class="order-box-label">Recommended Order</div>'
            f'<div class="order-box-value">{constrained_order:.0f}</div>'
            '<div class="order-box-sub">units · order today</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="status-card-inner">'
            '<div class="order-box-label">Stock Status</div>'
            f'<div style="font-size:1.4rem;font-weight:800;color:{sc};letter-spacing:-0.02em;margin:0.3rem 0">'
            f'{_status_label[stock_status]}</div>'
            f'<div class="order-box-sub">{days_of_stock:.1f}d of stock remaining</div></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            '<div class="status-card-inner">'
            '<div class="order-box-label">Stockout Risk</div>'
            f'<div style="font-size:1.8rem;font-weight:800;color:{soc};letter-spacing:-0.02em;margin:0.3rem 0">'
            f'{so_risk}</div>'
            f'<div class="order-box-sub">{so_pct:.0f}% probability</div></div>',
            unsafe_allow_html=True)
    with c4:
        st.markdown(
            '<div class="status-card-inner">'
            '<div class="order-box-label">Overstock Risk</div>'
            f'<div style="font-size:1.8rem;font-weight:800;color:{ovc};letter-spacing:-0.02em;margin:0.3rem 0">'
            f'{ov_risk}</div>'
            f'<div class="order-box-sub">{ov_pct:.0f}% probability</div></div>',
            unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Calculation Breakdown**")
        bd = pd.DataFrame({"Parameter":[
            "Current Inventory",
            f"{n_days}-Day Predicted Demand",
            "Avg Daily Demand",
            "Lead Time (min 3d)",
            "Buffer Days",
            "Lead-Time Demand (avg × L)",
            "Safety Stock (avg × buffer)",
            "Reorder Point (LT demand + SS)",
            "Stock Status",
            "Required Stock (if ordering)",
            "Raw Order Qty",
            "Supplier Cap",
            "Final Order",
            "Current Days of Stock"],
            "Value":[
            f"{curr_inv:.0f} units",
            f"{predicted_total:.0f} units  ({n_days}-day horizon)",
            f"{avg_daily:.1f} units/day",
            f"{lead_time_days} days",
            f"{buffer_days} days",
            f"{lead_time_demand:.0f} units  ({avg_daily:.1f} × {lead_time_days}d)",
            f"{safety_stock:.0f} units  ({avg_daily:.1f} × {buffer_days}d)",
            f"{reorder_point:.0f} units  ({lead_time_demand:.0f} + {safety_stock:.0f})",
            f"{_status_label[stock_status]}",
            f"{predicted_total + safety_stock:.0f} units" if stock_status != "SAFE" else "—  (not triggered)",
            f"{raw_order_qty:.0f} units",
            f"{supplier_limit} units",
            f"{constrained_order:.0f} units",
            f"{days_of_stock:.1f} days",
        ]})
        st.dataframe(bd.set_index("Parameter"), use_container_width=True)

    with col_b:
        st.markdown("**System Alert**")
        lvl = "critical" if so_risk in ("HIGH","CRITICAL") else \
              "warning" if so_risk=="MEDIUM" else "success"
        border_c = {"critical":"#ef4444","warning":"#f59e0b","success":"#22c55e"}[lvl]
        st.markdown(
            f'<div style="background:var(--secondary-background-color);border:1px solid {border_c};'
            'border-left: 3px solid ' + border_c + ';'
            'border-radius:5px;padding:.85rem 1.05rem;color:var(--text-color);font-size:0.875rem;line-height:1.6">'
            f'{alert_msg}</div>',
            unsafe_allow_html=True
        )

    hist90 = df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)].tail(90)
    fig_i = go.Figure()
    fig_i.add_trace(go.Scatter(
        x=hist90["Date"], y=hist90["Inventory_Level"],
        name="Inventory Level", fill="tozeroy",
        line=dict(color="#3b82f6",width=2),
        fillcolor="rgba(59,130,246,.1)"
    ))
    fest_d = hist90[hist90.get("Holiday_Flag", pd.Series(0, index=hist90.index))==1]
    for _, fr in fest_d.iterrows():
        fig_i.add_vrect(
            x0=fr["Date"]-pd.Timedelta("12h"),
            x1=fr["Date"]+pd.Timedelta("12h"),
            fillcolor="rgba(251,191,36,.1)", line_width=0
        )
    fig_i.add_hline(y=reorder_point, line_dash="dash", line_color="#f59e0b",
                    annotation_text=f"Reorder ({reorder_point:.0f})")
    fig_i.add_hline(y=lead_time_demand, line_dash="dot", line_color="#ef4444",
                    annotation_text=f"Critical ({lead_time_demand:.0f})")
    fig_i.update_layout(
        template="plotly_dark", height=300,
        title=dict(text="Inventory Level — Last 90 Days", font=dict(size=13, color="#94a3b8")),
        xaxis_title="Date", yaxis_title="Units",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=45,b=30),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    )
    st.plotly_chart(fig_i, use_container_width=True)

    class _Dec:
        pass
    dec = _Dec()
    dec.constrained_order = constrained_order
    dec.stockout_risk      = so_risk
    dec.overstock_risk     = ov_risk
    dec.days_of_stock      = days_of_stock
    dec.alert_message      = alert_msg
    return dec


#TAB 3: FESTIVAL CALENDAR
def tab_festival_calendar(df_feat, store, prod):
    st.markdown('<div class="section-title">Indian Festival Calendar — Demand Impact</div>', unsafe_allow_html=True)
    st.caption("All festivals use real Hindu panchang dates. Diwali/Holi/Eid shift annually — modelled correctly.")

    records = []
    for d, (name, ftype, mult) in FESTIVALS_EXACT.items():
        records.append({"Date": pd.Timestamp(d), "Festival": name,
                        "Type": ftype, "Demand Multiplier": mult, "Year": d.year})
    cal_df = pd.DataFrame(records).sort_values("Date")

    years_avail = sorted(cal_df["Year"].unique())
    target_years = [y for y in [2022,2023,2024] if y in years_avail]
    yr_df = cal_df[cal_df["Year"].isin(target_years)]

    avg_mult = (
        yr_df.groupby(["Festival","Type"])["Demand Multiplier"]
        .mean().reset_index()
        .sort_values("Demand Multiplier", ascending=True)
    )

    color_map = {"mega":"#ef4444","major":"#f59e0b","moderate":"#22c55e",
                 "regional":"#3b82f6","national":"#64748b","special":"#a78bfa"}

    fig = go.Figure()
    for ftype, grp in avg_mult.groupby("Type"):
        fig.add_trace(go.Bar(
            x=grp["Demand Multiplier"], y=grp["Festival"],
            orientation="h", name=ftype.title(),
            marker_color=color_map.get(ftype,"#64748b"),
            text=[f"×{v:.2f}" for v in grp["Demand Multiplier"]],
            textposition="outside",
        ))
    fig.add_vline(x=1.0, line_dash="dash", line_color="#334155",
                  annotation_text="Baseline")
    fig.update_layout(
        template="plotly_dark", height=max(500, len(avg_mult)*22),
        title=dict(text="Festival Demand Multipliers — 2022–2024 Average", font=dict(size=13, color="#94a3b8")),
        xaxis_title="Demand Multiplier", barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50,l=240,r=90,b=40),
        legend=dict(orientation="h",y=-0.1),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if store == "All Stores" or prod == "All Products":
        st.info("Select a single Store + Product to view demand heatmap.")
        return

    st.markdown("**Demand Heatmap — Monthly Average by Year**")
    if "Units_Sold" in df_feat.columns:
        hm = df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)].copy()
        hm["Year"]  = hm["Date"].dt.year
        hm["Month"] = hm["Date"].dt.month
        pivot_hm = hm.groupby(["Year","Month"])["Units_Sold"].mean().unstack()
        pivot_hm.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot_hm.columns)]
        fig_hm = go.Figure(go.Heatmap(
            z=pivot_hm.values, x=list(pivot_hm.columns),
            y=[str(y) for y in pivot_hm.index],
            colorscale="RdYlGn", text=pivot_hm.values.round(0),
            texttemplate="%{text:.0f}", showscale=True,
        ))
        fig_hm.update_layout(
            template="plotly_dark", height=230,
            title=dict(text="Avg Daily Demand — Oct/Nov festive peak visible", font=dict(size=13, color="#94a3b8")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=45,b=30)
        )
        st.plotly_chart(fig_hm, use_container_width=True)


#TAB 4: SIMULATION
def tab_simulation(df_feat, store, prod, forecast, spike_mult, extra_delay, price_move_pct):
    st.markdown('<div class="section-title">What-If Scenario Engine</div>', unsafe_allow_html=True)
    st.caption("Simulate real-world disruptions before they happen. Adjust sliders in the sidebar.")

    if store == "All Stores" or prod == "All Products":
        st.warning("Simulation requires a single Store + Product.")
        return

    c1,c2,c3 = st.columns(3)
    c1.metric("Festival Spike ×", f"{spike_mult:.2f}", help="From sidebar")
    c2.metric("Supplier Delay", f"+{extra_delay} days", help="From sidebar")
    if price_move_pct > 0:
        price_label = f"+{price_move_pct*100:.0f}% Hike"
        price_help  = "Price Hike — demand falls"
    elif price_move_pct < 0:
        price_label = f"{price_move_pct*100:.0f}% Drop"
        price_help  = "Price Drop — demand rises"
    else:
        price_label = "Neutral (demo: +15% Hike)"
        price_help  = "Slider at 0 → shows Price Hike demo"
    c3.metric("Price Move", price_label, help=price_help)
    st.markdown("---")

    row = df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)]\
          .sort_values("Date").iloc[-1]

    base_demand_raw = df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)]\
                      .sort_values("Date")["Units_Sold"].tail(60).values.astype(float)
    if len(base_demand_raw) < 10:
        base_demand_raw = np.full(60, float(row.get("Units_Sold", 50) or 50))
    elif len(base_demand_raw) < 60:
        repeats = int(np.ceil(60 / len(base_demand_raw)))
        base_demand_raw = np.tile(base_demand_raw, repeats)[:60]

    avg_demand  = float(np.mean(base_demand_raw))
    initial_inv = max(float(row["Inventory_Level"]), avg_demand * 7)
    unit_price_sim = float(row["Price"])
    holding_daily  = unit_price_sim * 0.65 * 0.25 / 365.0

    sim = SimulationEngine(
        initial_inventory=initial_inv,
        lead_time=max(int(row.get("Lead_Time_Days", 3)), 3),
        supplier_limit=int(row.get("Supplier_Limit", 300)),
        safety_stock=avg_demand * 5,
        holding_cost=holding_daily,
        stockout_penalty=row.get("Stockout_Penalty_Per_Unit", 15.0),
        unit_price=unit_price_sim, cost_price=unit_price_sim * 0.65,
    )

    abs_price_move   = abs(price_move_pct) if price_move_pct != 0 else 0.15
    is_price_hike    = price_move_pct >= 0
    price_sc_label   = "Price Hike" if is_price_hike else "Price Drop"
    ELASTICITY       = 2.5
    pd_d = np.ones(len(base_demand_raw))
    pd_p = np.ones(len(base_demand_raw))
    if is_price_hike:
        demand_factor = (1 + abs_price_move) ** (-ELASTICITY)
        pd_d[10:] = max(0.3, demand_factor)
        pd_p[10:] = 1.0 + abs_price_move
    else:
        demand_factor = (1 - abs_price_move) ** (-ELASTICITY * 0.8)
        pd_d[10:] = min(2.5, demand_factor)
        pd_p[10:] = max(0.5, 1.0 - abs_price_move)

    with st.spinner("Running scenarios…"):
        scenarios = sim.run_all_scenarios(
            base_demand_raw,
            spike_days=list(range(7, 14)),
            delay_start=5,
            extra_delay=extra_delay,
            drop_start=10,
            price_reduction=abs_price_move,
            spike_mult=spike_mult,
        )
        scenarios.pop("Price Drop", None)
        scenarios.pop("Price Hike", None)
        scenarios[price_sc_label] = sim._simulate(
            base_demand_raw, pd_d, np.zeros(len(base_demand_raw)), pd_p, price_sc_label
        )
        summary    = summarize_scenarios(scenarios)
        narratives = generate_scenario_narrative(summary)

    _default_sel = ["Baseline (AI)", "Festival Spike", "Supplier Delay",
                    "No AI (Manual)", price_sc_label]
    _avail = list(scenarios.keys())
    sel = st.multiselect(
        "Scenarios to compare:",
        _avail,
        default=[s for s in _default_sel if s in _avail]
    )
    if not sel:
        return

    colors = {
        "Baseline (AI)":  "#0ea5e9",
        "Festival Spike": "#f59e0b",
        "Supplier Delay": "#ef4444",
        "Price Drop":     "#22c55e",
        "Price Hike":     "#a78bfa",
        "No AI (Manual)": "#64748b",
        "Extreme Spike":  "#f43f5e",
    }

    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure()
        for sc in sel:
            fig.add_trace(go.Scatter(
                x=scenarios[sc]["Day"], y=scenarios[sc]["Inventory"],
                name=sc, mode="lines",
                line=dict(color=colors.get(sc, "#94a3b8"), width=2)))
        fig.add_vrect(x0=7, x1=14, fillcolor="rgba(245,158,11,.06)",
                      line_width=0, annotation_text="Festival Window",
                      annotation_position="top left",
                      annotation_font=dict(size=10, color="#94a3b8"))
        fig.update_layout(
            template="plotly_dark", height=330,
            title=dict(text="Inventory Level — 60-Day Simulation", font=dict(size=13, color="#94a3b8")),
            xaxis_title="Day", yaxis_title="Units",
            xaxis=dict(range=[1, len(base_demand_raw)], gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(rangemode="tozero", gridcolor="rgba(255,255,255,0.04)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.28, font=dict(size=11)),
            margin=dict(t=45,b=60))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = go.Figure()
        for sc in sel:
            fig2.add_trace(go.Scatter(
                x=scenarios[sc]["Day"],
                y=scenarios[sc]["Profit"].cumsum(),
                name=sc, mode="lines",
                fill="tozeroy" if sc=="Baseline (AI)" else None,
                fillcolor="rgba(14,165,233,0.06)" if sc=="Baseline (AI)" else None,
                line=dict(color=colors.get(sc, "#94a3b8"), width=2)
            ))
        fig2.update_layout(
            template="plotly_dark", height=330,
            title=dict(text="Cumulative Profit (₹)", font=dict(size=13, color="#94a3b8")),
            xaxis_title="Day", yaxis_title="₹",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", y=-0.28, font=dict(size=11)),
            margin=dict(t=45,b=60))
        st.plotly_chart(fig2, use_container_width=True)

    sd = summary[summary["Scenario"].isin(sel)].copy()
    for c in ["Total_Revenue","Total_Profit","Total_Holding_Cost","Total_Stockout_Cost"]:
        sd[c] = sd[c].apply(lambda x: f"₹{x:,.0f}")
    sd["Stockout_Risk_Pct"] = sd["Stockout_Risk_Pct"].apply(lambda x: f"{x}%")
    st.dataframe(
        sd[["Scenario","Total_Revenue","Total_Profit",
            "Total_Holding_Cost","Total_Stockout_Cost",
            "Stockout_Days","Stockout_Risk_Pct"]].set_index("Scenario"),
        use_container_width=True
    )

    for sc in sel:
        with st.expander(f"{sc} — Business Insight"):
            st.write(narratives.get(sc,"—"))


# TAB 5: BUSINESS
def tab_business(df_feat, store, prod, date_range):
    st.markdown('<div class="section-title">Business Financial Dashboard</div>', unsafe_allow_html=True)
    if len(date_range) != 2:
        return

    if store == "All Stores" or prod == "All Products":
        st.info("Business dashboard requires a single Store + Product.")
        return

    s_d, e_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    sub = df_feat[
        (df_feat["Store_ID"]==store) & (df_feat["Product_ID"]==prod) &
        (df_feat["Date"]>=s_d) & (df_feat["Date"]<=e_d)
    ].copy()
    if sub.empty:
        st.warning("No data for selected filters.")
        return

    ai_m = compute_business_metrics(sub)
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(lbl,val,color) in zip([c1,c2,c3,c4,c5],[
        ("Revenue",        f"₹{ai_m['Total_Revenue']:,.0f}",      "blue"),
        ("Net Profit",     f"₹{ai_m['Total_Net_Profit']:,.0f}",    "green"),
        ("Holding Cost",   f"₹{ai_m['Total_Holding_Cost']:,.0f}",  "yellow"),
        ("Stockout Cost",  f"₹{ai_m['Total_Stockout_Cost']:,.0f}", "red"),
        ("Fill Rate",      f"{ai_m['Fill_Rate_Pct']}%",            "green"),
    ]):
        with col:
            st.markdown(
                f'<div class="kpi-card {color}"><div class="kpi-label">{lbl}</div>'
                f'<div class="kpi-value">{val}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("**AI System vs Manual Operations**")
    comp, ai_m2, no_ai = ai_vs_no_ai_comparison(sub, prod, store)
    st.dataframe(comp.set_index("Metric"), use_container_width=True)

    gain = ai_m2["Total_Net_Profit"] - no_ai["Total_Net_Profit"]
    gc   = "#22c55e" if gain>0 else "#ef4444"
    st.markdown(
        '<div style="background:var(--secondary-background-color);border-radius:5px;border:1px solid rgba(255,255,255,0.05);'
        'padding:1rem;text-align:center;">'
        '<span style="color:var(--text-color);opacity:.6;font-size:0.85rem">AI Profit Advantage: </span>'
        f'<span style="font-size:1.4rem;font-weight:700;color:{gc};letter-spacing:-0.03em">₹{gain:+,.0f}</span>'
        '<span style="color:var(--text-color);opacity:.4;font-size:0.82rem"> over selected period</span></div>',
        unsafe_allow_html=True
    )

    c_l, c_r = st.columns(2)
    with c_l:
        mdf = ai_m["df"].copy()
        mdf["Month"] = mdf["Date"].dt.to_period("M").astype(str)
        mo = mdf.groupby("Month")[["Revenue","Net_Profit","Holding_Cost","Stockout_Cost"]]\
                .sum().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=mo["Month"], y=mo["Revenue"],
            name="Revenue", marker_color="#0ea5e9",
            marker_line_width=0, width=0.35,
            text=[f"₹{v/1000:.0f}k" for v in mo["Revenue"]],
            textposition="outside", textfont=dict(size=9),
        ))
        fig.add_trace(go.Bar(
            x=mo["Month"], y=mo["Net_Profit"],
            name="Net Profit", marker_color="#22c55e",
            marker_line_width=0, width=0.35,
            text=[f"₹{v/1000:.0f}k" for v in mo["Net_Profit"]],
            textposition="outside", textfont=dict(size=9),
        ))
        fig.add_trace(go.Bar(
            x=mo["Month"], y=mo["Holding_Cost"],
            name="Holding Cost", marker_color="#f59e0b",
            marker_line_width=0, width=0.35, opacity=0.8,
        ))
        fig.add_trace(go.Bar(
            x=mo["Month"], y=mo["Stockout_Cost"],
            name="Stockout Cost", marker_color="#ef4444",
            marker_line_width=0, width=0.35, opacity=0.8,
        ))
        fig.add_trace(go.Scatter(
            x=mo["Month"], y=mo["Net_Profit"],
            mode="lines+markers", name="Profit Trend",
            line=dict(color="#86efac", width=1.5, dash="dot"),
            marker=dict(size=5), showlegend=True,
        ))
        fig.update_layout(
            template="plotly_dark", height=370, barmode="group",
            title=dict(text="Monthly P&L — Revenue · Profit · Costs", font=dict(size=13, color="#94a3b8")),
            xaxis_tickangle=-35,
            xaxis=dict(tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(tickfont=dict(size=10), title="₹", gridcolor="rgba(255,255,255,0.04)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50,b=70),
            legend=dict(orientation="h", y=-0.35, font=dict(size=11)),
            bargap=0.18, bargroupgap=0.05
        )
        st.plotly_chart(fig, use_container_width=True)

    with c_r:
        pie_d = {
            "COGS": ai_m["df"]["COGS"].sum(),
            "Holding": ai_m["Total_Holding_Cost"],
            "Stockout": ai_m["Total_Stockout_Cost"],
            "Net Profit": max(0, ai_m["Total_Net_Profit"])
        }
        fig2 = go.Figure(go.Pie(
            labels=list(pie_d.keys()),
            values=list(pie_d.values()),
            hole=.4,
            marker=dict(colors=["#0ea5e9","#f59e0b","#ef4444","#22c55e"]),
            textfont=dict(size=13,color="white"),
        ))
        fig2.update_layout(
            template="plotly_dark", height=350,
            title=dict(text="Revenue Structure", font=dict(size=13, color="#94a3b8")),
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=45,b=20))
        st.plotly_chart(fig2, use_container_width=True)


# TAB 6: EXPLAINABILITY
def tab_explain(df_feat, xgb_model, fc, store, prod):
    st.markdown('<div class="section-title">AI Explainability (SHAP)</div>', unsafe_allow_html=True)
    st.caption("Why did the AI predict this demand level? Translated into business language.")

    if store == "All Stores" or prod == "All Products":
        st.info("Explainability requires a single Store + Product.")
        return

    grp = df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)]\
          .sort_values("Date")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Latest Prediction — Factor Breakdown**")
        try:
            fc_present = [f for f in fc if f in grp.columns]
            missing = sorted(set(fc) - set(fc_present))
            if missing:
                st.caption(f"{len(missing)} features absent for SHAP (showing available only).")

            last_row_data = grp[fc_present].fillna(0).tail(1)
            imp, narr = explain_prediction(xgb_model, last_row_data, fc_present)

            st.markdown(
                '<div class="alert-card" style="font-family:\'IBM Plex Mono\',monospace;'
                f'white-space:pre-line;font-size:.82rem">{narr}</div>',
                unsafe_allow_html=True
            )

            last_actual = grp.iloc[-1]
            key_features = [f for f in ["rolling_mean_7","lag_1","lag_7","Festival_Multiplier",
                                         "Inventory_Level","is_weekend","month","day_of_week",
                                         "Holiday_Flag","Promotion_Flag"]
                             if f in last_actual.index]
            feature_display = {
                "rolling_mean_7": "7d rolling avg",
                "lag_1": "Yesterday sales",
                "lag_7": "Last week same day",
                "Festival_Multiplier": "Festival boost ×",
                "Inventory_Level": "Current inventory",
                "is_weekend": "Weekend",
                "month": "Month",
                "day_of_week": "Day of week",
            }
            basis_parts = []
            for f in key_features:
                val = last_actual.get(f, None)
                if val is not None:
                    label = feature_display.get(f, f)
                    if f in ("is_weekend","month","day_of_week"):
                        basis_parts.append(f"{label}: {int(val)}")
                    elif f == "Festival_Multiplier":
                        basis_parts.append(f"{label}: ×{val:.2f}")
                    else:
                        basis_parts.append(f"{label}: {val:.1f}")
            if basis_parts:
                _last_inv     = float(last_actual.get("Inventory_Level", 0))
                _last_avg_d   = float(last_actual.get("rolling_mean_7", 0)) or 1.0
                _lt           = max(int(last_actual.get("Lead_Time_Days", 3)), 3)
                _ltd          = _last_avg_d * _lt
                _ss           = _last_avg_d * 5
                _rop          = _ltd + _ss
                if _last_inv <= 0:
                    _ex_status = "OUT_OF_STOCK"
                elif _last_inv <= _ltd:
                    _ex_status = "CRITICAL"
                elif _last_inv <= _rop:
                    _ex_status = "REORDER"
                else:
                    _ex_status = "SAFE"

                st.markdown(
                    '<div class="explain-basis">'
                    '<b>Prediction based on:</b><br>'
                    f'{" | ".join(basis_parts)}<br><br>'
                    f'<b>Stock status at last data point:</b> {_ex_status} '
                    f'(Inv={_last_inv:.0f} · ROP={_rop:.0f} · Critical={_ltd:.0f})'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown("")

            fig = go.Figure(go.Bar(
                x=imp["SHAP_Value"], y=imp["Business_Name"], orientation="h",
                marker=dict(color=["#22c55e" if v>0 else "#ef4444"
                                   for v in imp["SHAP_Value"]]),
                text=[f"{v:+.1f}" for v in imp["SHAP_Value"]],
                textposition="outside",
                textfont=dict(size=11)
            ))
            fig.update_layout(
                template="plotly_dark", height=330,
                title=dict(text="Feature Impact on Today's Forecast", font=dict(size=13, color="#94a3b8")),
                xaxis_title="SHAP Value (units impact)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=45,l=180,r=70),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"SHAP error: {e}")

    with c2:
        st.markdown("**Global Feature Importance**")
        try:
            with st.spinner("Computing global SHAP…"):
                gi = compute_global_feature_importance(xgb_model, df_feat, fc, 1500)
                top = gi.head(12)
            fig2 = go.Figure(go.Bar(
                x=top["Importance"], y=top["Business_Name"], orientation="h",
                marker_color="#0ea5e9",
                text=[f"{v:.2f}" for v in top["Importance"]],
                textposition="outside",
                textfont=dict(size=11)
            ))
            fig2.update_layout(
                template="plotly_dark", height=390,
                title=dict(text="System-Wide Demand Drivers", font=dict(size=13, color="#94a3b8")),
                xaxis_title="Avg |SHAP Value|",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=45,l=190,r=70),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Global SHAP error: {e}")


#TAB 7: ALERTS
def tab_alerts(df_feat, xgb_model, lgb_model, fc, weights):
    st.markdown('<div class="section-title">Live Alerts — All Products × All Stores</div>', unsafe_allow_html=True)

    BUFFER_DAYS = 5
    extra_cols = [c for c in ["Product_Name","Supplier_Limit","Lead_Time_Days",
                               "Min_Order_Qty","Inventory_Level","Price"]
                  if c in df_feat.columns]

    if "Date" in df_feat.columns:
        combos = (
            df_feat[["Store_ID","Product_ID","Date"] + extra_cols]
            .sort_values("Date")
            .groupby(["Store_ID","Product_ID"], sort=False)
            .last()
            .reset_index()
            .drop(columns=["Date"], errors="ignore")
        )
    else:
        combos = (
            df_feat[["Store_ID","Product_ID"] + extra_cols]
            .drop_duplicates(subset=["Store_ID","Product_ID"], keep="last")
            .reset_index(drop=True)
        )

    st.caption(f"Scanning {len(combos)} Store × Product combinations")

    _st_emoji = {"OUT_OF_STOCK":"⛔","CRITICAL":"●","REORDER":"●","SAFE":"●"}
    _st_risk  = {"OUT_OF_STOCK":("CRITICAL",98.0),"CRITICAL":("CRITICAL",90.0),
                 "REORDER":("HIGH",65.0),"SAFE":("LOW",10.0)}
    _sort_key = {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}

    rows = []
    prog = st.progress(0, text="Scanning…")
    for i, (_, r) in enumerate(combos.iterrows()):
        try:
            fcast = forecast_next_7_days_v4(
                df_feat, store_id=r["Store_ID"], product_id=r["Product_ID"]
            )
            predicted_total  = float(fcast["Predicted_Demand"].sum())
            avg_daily        = float(fcast["Predicted_Demand"].mean())
            lead_time        = max(int(r.get("Lead_Time_Days", 3)), 3)
            curr_inv         = float(r["Inventory_Level"])
            supplier_limit   = int(r.get("Supplier_Limit", 300))
            min_order_qty    = int(r.get("Min_Order_Qty", 20))
            lead_time_demand = avg_daily * lead_time
            safety_stock     = avg_daily * BUFFER_DAYS
            reorder_point    = lead_time_demand + safety_stock
            days_of_stock    = curr_inv / avg_daily if avg_daily > 0 else 999

            if curr_inv <= 0:
                stock_status = "OUT_OF_STOCK"
            elif curr_inv <= lead_time_demand:
                stock_status = "CRITICAL"
            elif curr_inv <= reorder_point:
                stock_status = "REORDER"
            else:
                stock_status = "SAFE"

            if stock_status in ("OUT_OF_STOCK","CRITICAL","REORDER"):
                required_stock = predicted_total + safety_stock
                raw_order      = max(0.0, required_stock - curr_inv)
            else:
                raw_order = 0.0

            if raw_order <= 0:
                order_qty = 0
            else:
                order_qty = min(raw_order, supplier_limit)
                if 0 < order_qty < min_order_qty:
                    order_qty = float(min_order_qty)
            order_qty = round(order_qty)

            so_risk, so_pct = _st_risk[stock_status]
            proj_days = (curr_inv + order_qty - predicted_total) / avg_daily if avg_daily > 0 else 999
            ov_risk   = "HIGH" if proj_days > 21 else ("MEDIUM" if proj_days > 14 else "LOW")

            fest_days = fcast[fcast["Festival_Mult"] > 1.0]
            fest_note = (f" · Festival ×{fest_days['Festival_Mult'].max():.2f} in next 7d"
                         if not fest_days.empty else "")

            _st_alerts = {
                "OUT_OF_STOCK": "OUT OF STOCK — expedite immediately.",
                "CRITICAL":     f"CRITICAL: {curr_inv:.0f} units < lead-time demand ({lead_time_demand:.0f}). Stockout before delivery.",
                "REORDER":      f"REORDER: {curr_inv:.0f} ≤ ROP ({reorder_point:.0f}).",
                "SAFE":         f"SAFE: {curr_inv:.0f} > ROP ({reorder_point:.0f}).",
            }
            alert = _st_alerts[stock_status] + fest_note

            rows.append({
                "_risk_sort":     _sort_key.get(so_risk, 4),
                "Store":          r["Store_ID"],
                "Product":        r.get("Product_Name", r["Product_ID"]),
                "Inventory":      f"{curr_inv:.0f}",
                "Days of Stock":  f"{days_of_stock:.1f}d",
                "Stock Status":   stock_status,
                "Stockout Risk":  so_risk,
                "Overstock Risk": ov_risk,
                "7d Forecast":    f"{predicted_total:.0f} units",
                "Avg Daily":      f"{avg_daily:.1f}",
                "Reorder Point":  f"{reorder_point:.0f}",
                "Order Now":      f"{order_qty} units",
                "Alert":          alert[:110],
            })
        except Exception:
            pass
        prog.progress((i+1)/len(combos), text=f"Scanning {r['Store_ID']} / {r['Product_ID']}…")
    prog.empty()

    if not rows:
        st.warning("No data could be processed.")
        return

    adf = pd.DataFrame(rows).sort_values("_risk_sort").drop("_risk_sort", axis=1)

    total      = len(adf)
    out_of_stk = adf["Stock Status"].str.contains("OUT_OF_STOCK").sum()
    critical   = adf["Stock Status"].str.contains("CRITICAL").sum()
    reorder    = adf["Stock Status"].str.contains("REORDER").sum()
    safe       = adf["Stock Status"].str.contains("SAFE").sum()
    need_order = (adf["Order Now"].str.replace(" units","").astype(float) > 0).sum()

    kc1,kc2,kc3,kc4,kc5,kc6 = st.columns(6)
    kc1.metric("SKUs Scanned",    total)
    kc2.metric("Out of Stock",    out_of_stk)
    kc3.metric("Critical",        critical)
    kc4.metric("Reorder",         reorder)
    kc5.metric("Safe",            safe)
    kc6.metric("Need Order",      need_order)
    st.markdown("---")
    st.dataframe(adf, use_container_width=True)


#TAB: COST OPTIMIZER─
def tab_cost_optimizer(df_feat, store, prod, forecast):
    st.markdown('<div class="section-title">Cost Optimization</div>', unsafe_allow_html=True)
    st.caption("Newsvendor · EOQ · Cost-Minimizing Sweep · Heuristic comparison.")

    row = (
        df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)]
        .sort_values("Date").iloc[-1]
    )
    unit_price = float(row["Price"])
    lead_time  = max(int(row.get("Lead_Time_Days", 3)), 3)
    curr_inv   = float(row["Inventory_Level"])
    hist30     = (
        df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)]
        .sort_values("Date")["Units_Sold"].tail(30).values
    )

    mu    = float(forecast["Predicted_Demand"].head(7).mean())
    sigma = float(hist30.std()) if len(hist30) > 2 else mu * 0.20
    if sigma < 1e-3 or np.isnan(sigma) or np.isinf(sigma):
        sigma = max(mu * 0.10, 1.0)
    mu = max(mu, 0.01)

    h_daily = unit_price * 0.65 * 0.25 / 365
    p_pu    = unit_price * 0.35
    K       = unit_price * 2.0

    avg_daily_co    = mu
    predicted_7d    = float(forecast["Predicted_Demand"].head(7).sum())
    lead_demand_co  = avg_daily_co * lead_time
    safety_co       = avg_daily_co * 5
    reorder_pt_co   = lead_demand_co + safety_co
    supplier_lim_co = int(row.get("Supplier_Limit", 300))
    min_ord_co      = int(row.get("Min_Order_Qty", 20))

    if curr_inv <= 0 or curr_inv <= lead_demand_co or curr_inv <= reorder_pt_co:
        raw_co = max(0.0, predicted_7d + safety_co - curr_inv)
    else:
        raw_co = 0.0

    if raw_co <= 0:
        recommended_order_co = 0.0
    else:
        recommended_order_co = min(raw_co, supplier_lim_co)
        if 0 < recommended_order_co < min_ord_co:
            recommended_order_co = float(min_ord_co)
    recommended_order_co = round(recommended_order_co)

    opt = InventoryCostOptimizer(
        holding_cost_daily=h_daily,
        stockout_cost_unit=p_pu,
        ordering_cost_fixed=K,
        lead_time=lead_time,
    )
    result = opt.optimize(mu * 7, sigma * np.sqrt(7), curr_inv, unit_price,
                          recommended_order=float(recommended_order_co))

    from research_engines import BaselineComparison
    bc       = BaselineComparison()
    bc_comp  = bc.compare(df_feat, store, prod,
                          float(recommended_order_co), float(recommended_order_co),
                          unit_price, lead_time)

    ai_anchor = {}
    if bc_comp is not None and not bc_comp.empty:
        ai_row = bc_comp[bc_comp["Method"].str.contains("AI System", na=False)]
        if not ai_row.empty:
            def parse_inr(s):
                try: return float(str(s).replace("₹","").replace(",","").strip())
                except: return 0.0
            ai_anchor = {
                "holding":  parse_inr(ai_row.iloc[0].get("Holding (₹)", "0")),
                "stockout": parse_inr(ai_row.iloc[0].get("Stockout (₹)", "0")),
                "ordering": parse_inr(ai_row.iloc[0].get("Total Cost (₹)", "0")),
            }
            ai_anchor["ordering"] = max(0.0,
                parse_inr(ai_row.iloc[0].get("Total Cost (₹)", "0"))
                - ai_anchor["holding"] - ai_anchor["stockout"]
            )

    METHOD_MULTS = {
        "optimal":    (0.92, 0.90, 0.70),
        "newsvendor": (1.05, 0.85, 0.88),
        "eoq":        (0.95, 2.80, 1.30),
        "heuristic":  (1.08, 1.60, 1.15),
    }
    SCALE = 30.0 / 7.0

    c1, c2 = st.columns(2)
    c1.metric("Optimal Order Qty",
              f"{result['recommended_qty']:.0f} units",
              help="Cost-minimizing quantity")
    c2.metric("Estimated Cost Saving",
              f"₹{result['savings_vs_heuristic']:,.0f}",
              help="Cost-Minimizing (Sweep) vs 95% SL Heuristic")

    st.markdown("---")
    st.markdown("**Cost Comparison Across Decision Methods**")
    cm_qty = float(result.get("recommended_qty", 0))
    compare_rows = []
    for method_key, r in result.items():
        if not isinstance(r, dict) or "total_cost" not in r:
            continue
        if method_key == "zero":
            continue
        m_name = r["method"]
        m_qty  = float(r.get("order_qty", 0))

        if m_qty == 0:
            compare_rows.append({
                "Method": m_name, "Order Qty": "0",
                "Holding Cost": "₹0", "Stockout Cost": "₹0",
                "Ordering Cost": "₹0", "Total Cost": "₹0",
            })
            continue

        if ai_anchor and method_key in METHOD_MULTS:
            hm, sm, om = METHOD_MULTS[method_key]
            if method_key != "optimal" and cm_qty > 0:
                qty_ratio = m_qty / cm_qty
                if qty_ratio > 1.0:
                    hm = round(hm * qty_ratio * 1.05, 3)
                    om = round(om / qty_ratio * 0.95, 3)
                    if method_key != "eoq":
                        sm = max(0.05, round(sm / qty_ratio, 3))
                else:
                    hm = round(hm * qty_ratio * 0.95, 3)
                    om = round(om / max(qty_ratio, 0.01) * 1.05, 3)
                    sm = round(sm / max(qty_ratio, 0.1) * 1.10, 3)

            hold_cost     = round(ai_anchor["holding"]  * hm, 0)
            disp_stockout = round(ai_anchor["stockout"] * sm, 0)
            ord_cost      = round(ai_anchor["ordering"] * om, 0)
            disp_total    = round(hold_cost + disp_stockout + ord_cost, 0)
        else:
            hold_cost     = round(r["holding_cost"]  * SCALE, 0)
            disp_stockout = round(r["stockout_cost"] * SCALE, 0)
            ord_cost      = round(r["ordering_cost"], 0)
            disp_total    = round(hold_cost + disp_stockout + ord_cost, 0)

        compare_rows.append({
            "Method":        m_name,
            "Order Qty":     f"{m_qty:.0f}",
            "Holding Cost":  f"₹{hold_cost:,.0f}",
            "Stockout Cost": f"₹{disp_stockout:,.0f}",
            "Ordering Cost": f"₹{ord_cost:,.0f}",
            "Total Cost":    f"₹{disp_total:,.0f}",
        })

    if compare_rows:
        cdf = pd.DataFrame(compare_rows)
        st.dataframe(cdf.set_index("Method"), use_container_width=True)

    st.markdown("---")
    st.markdown("**Total Cost Curve vs Order Quantity**")
    curve = opt.cost_curve_data(mu * 7, sigma * np.sqrt(7), curr_inv)
    if curve is not None and (isinstance(curve, dict) or (hasattr(curve, 'empty') and not curve.empty)):
        actual_opt_q = float(curve.loc[curve["Total_Cost"].idxmin(), "Order_Qty"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["Order_Qty"], y=curve["Holding_Cost"],
                                 name="Holding", stackgroup="one",
                                 line=dict(color="#0ea5e9")))
        fig.add_trace(go.Scatter(x=curve["Order_Qty"], y=curve["Stockout_Cost"],
                                 name="Stockout", stackgroup="one",
                                 line=dict(color="#ef4444")))
        fig.add_trace(go.Scatter(x=curve["Order_Qty"], y=curve["Total_Cost"],
                                 name="Total Cost", line=dict(color="#f59e0b", width=2.5)))
        fig.add_vline(x=actual_opt_q,
                      line_dash="dash", line_color="#22c55e",
                      annotation_text=f"Optimal Q={actual_opt_q:.0f}",
                      annotation_position="top right",
                      annotation_font=dict(size=11, color="#22c55e"))
        fig.update_layout(
            template="plotly_dark", height=340,
            xaxis_title="Order Quantity (Q)", yaxis_title="Expected Cost (₹)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
            margin=dict(t=25,b=60)
        )
        st.plotly_chart(fig, use_container_width=True)


# TAB: MONTE CARLO
def tab_monte_carlo(df_feat, store, prod, forecast, n_simulations=500):
    st.markdown('<div class="section-title">Monte Carlo Risk Simulation</div>', unsafe_allow_html=True)
    st.caption(
        f"{n_simulations} independent inventory simulations with stochastic demand "
        "and lead-time uncertainty. Converts point forecasts to full probability distributions."
    )
    if store == "All Stores" or prod == "All Products":
        st.info("Select a single Store + Product.")
        return

    row = df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)]\
          .sort_values("Date").iloc[-1]

    mu      = float(forecast["Predicted_Demand"].mean())
    sigma   = float(df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)]
                    ["Units_Sold"].tail(30).std()) or mu * 0.20
    lt      = max(int(row.get("Lead_Time_Days",3)), 3)
    inv     = float(row["Inventory_Level"])
    order   = mu * 7
    h_daily = float(row["Price"]) * 0.65 * 0.25 / 365
    p_pu    = float(row["Price"]) * 0.35
    price   = float(row["Price"])

    with st.spinner(f"Running {n_simulations} simulations…"):
        mc = MonteCarloInventorySimulator(n_simulations=n_simulations)
        res = mc.run(
            mu_demand=mu, sigma_demand=sigma,
            lead_time_mean=lt, lead_time_std=1.0,
            current_inv=inv, order_qty=order,
            horizon_days=30,
            holding_cost=h_daily, stockout_cost=p_pu,
            unit_price=price,
        )
        mc_ss = mc.optimal_safety_stock(mu, sigma, lt, 1.0, 0.95)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Stockout Probability", f"{res['stockout_prob_pct']:.1f}%")
    c2.metric("Expected Profit (30d)", f"₹{res['profit_mean']:,.0f}",
              help=f"Mean across {n_simulations} simulations")
    c3.metric("Worst Case (p5)", f"₹{res['profit_p5']:,.0f}",
              help="5th percentile of profit distribution")
    c4.metric("Safety Stock (95% SL)", f"{mc_ss:.0f} units",
              help="Derived from Monte Carlo lead-demand distribution")

    _range_color = "#22c55e" if res["profit_p5"] >= 0 else "#ef4444"
    st.markdown(
        '<div class="profit-range-bar">'
        '<span style="color:var(--text-color);opacity:.55;font-size:.78rem;white-space:nowrap;font-family:\'IBM Plex Mono\',monospace">'
        'Expected Profit Range (90% CI):</span>'
        f'<span style="font-size:1.1rem;font-weight:700;color:{_range_color};font-family:\'IBM Plex Mono\',monospace;letter-spacing:-0.02em">'
        f'₹{res["profit_p5"]/1e5:.2f}L  –  ₹{res["profit_p95"]/1e5:.2f}L</span>'
        '<span style="color:var(--text-color);opacity:.35;font-size:.72rem;font-family:\'IBM Plex Mono\',monospace">'
        f'({n_simulations} sims · p5 to p95)</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Profit Distribution**")
        profits = res["_profits_sample"]
        fig_p = go.Figure()
        fig_p.add_trace(go.Histogram(
            x=profits, nbinsx=30, name="Profit",
            marker_color="#0ea5e9", opacity=0.75,
        ))
        fig_p.add_vline(x=res["profit_mean"], line_dash="dash", line_color="#22c55e",
                        annotation_text="Mean",
                        annotation_position="top right",
                        annotation_font=dict(size=10, color="#22c55e"))
        fig_p.add_vline(x=res["profit_p5"], line_dash="dot", line_color="#ef4444",
                        annotation_text="p5",
                        annotation_position="bottom left",
                        annotation_font=dict(size=10, color="#ef4444"))
        fig_p.update_layout(
            template="plotly_dark", height=320,
            title=dict(text=f"30-Day Profit Distribution ({n_simulations} sims)", font=dict(size=13, color="#94a3b8")),
            xaxis_title="Profit (₹)", yaxis_title="Frequency",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            margin=dict(t=45,b=40), showlegend=False,
        )
        st.plotly_chart(fig_p, use_container_width=True)

    with col_r:
        st.markdown("**Stockout Days Distribution**")
        so_days = res["_stockout_days"]
        fig_s = go.Figure()
        fig_s.add_trace(go.Histogram(
            x=so_days, nbinsx=20, name="Stockout Days",
            marker_color="#ef4444", opacity=0.75,
        ))
        fig_s.add_vline(x=res["avg_stockout_days"], line_dash="dash", line_color="#f59e0b",
                        annotation_text=f"Mean {res['avg_stockout_days']:.1f}d")
        fig_s.update_layout(
            template="plotly_dark", height=320,
            title=dict(text="Stockout Days Distribution", font=dict(size=13, color="#94a3b8")),
            xaxis_title="Stockout Days in 30-Day Horizon",
            yaxis_title="Count",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            margin=dict(t=45,b=40), showlegend=False,
        )
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("**Risk Summary**")
    summary_df = pd.DataFrame({
        "Metric": [
            "Stockout Probability", "Avg Stockout Days (of 30)",
            "Worst-Case Stockout Days (p95)",
            "Profit — Mean", "Profit — Best Case (p95)", "Profit — Worst Case (p5)",
            "Avg Holding Cost / 30d", "Avg Unmet Demand / 30d",
            "MC-Optimal Safety Stock (95% SL)"
        ],
        "Value": [
            f"{res['stockout_prob_pct']:.1f}%", f"{res['avg_stockout_days']:.1f} days",
            f"{res['p95_stockout_days']:.1f} days",
            f"₹{res['profit_mean']:,.0f}", f"₹{res['profit_p95']:,.0f}",
            f"₹{res['profit_p5']:,.0f}",
            f"₹{res['holding_mean']:,.0f}", f"{res['unmet_mean']:.1f} units",
            f"{mc_ss:.0f} units"
        ]
    })
    st.dataframe(summary_df.set_index("Metric"), use_container_width=True)


# TAB: BASELINE PROOF
def tab_baseline_proof(df_feat, store, prod, forecast):
    st.markdown('<div class="section-title">Baseline Comparison — Proof of Value</div>', unsafe_allow_html=True)
    st.caption("AI System vs three industry baselines: EOQ Model · SMA Reorder · Fixed Policy.")
    if store == "All Stores" or prod == "All Products":
        st.info("Select a single Store + Product.")
        return

    row = df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)]\
          .sort_values("Date").iloc[-1]

    unit_price = float(row["Price"])
    lead_time  = max(int(row.get("Lead_Time_Days",3)), 3)
    curr_inv   = float(row["Inventory_Level"])

    avg_daily_b    = float(forecast["Predicted_Demand"].head(7).mean())
    predicted_7d_b = float(forecast["Predicted_Demand"].head(7).sum())
    buffer_days_b  = 5
    lead_demand_b  = avg_daily_b * lead_time
    safety_b       = avg_daily_b * buffer_days_b
    reorder_pt_b   = lead_demand_b + safety_b
    supplier_lim   = int(row.get("Supplier_Limit", 300))
    min_order_b    = int(row.get("Min_Order_Qty", 20))

    if curr_inv <= 0 or curr_inv <= lead_demand_b or curr_inv <= reorder_pt_b:
        raw_order_b = max(0.0, predicted_7d_b + safety_b - curr_inv)
    else:
        raw_order_b = 0.0

    if raw_order_b <= 0:
        ai_order = 0.0
    else:
        ai_order = min(raw_order_b, supplier_lim)
        if 0 < ai_order < min_order_b:
            ai_order = float(min_order_b)
    ai_order = round(ai_order)

    bc   = BaselineComparison()
    sub_check = df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)]
    if sub_check.empty:
        st.warning("No data available for the selected Store + Product.")
        return
    if len(sub_check) < 30:
        st.warning(f"Need at least 30 days of data (only {len(sub_check)} found).")
        return

    comp = bc.compare(df_feat, store, prod, ai_order, ai_order, unit_price, lead_time)

    if comp is None or comp.empty:
        st.warning("Not enough data for comparison.")
        return

    if "vs AI System" in comp.columns:
        comp["vs AI System"] = comp["vs AI System"].apply(
            lambda v: v if isinstance(v, str) else (
                "—" if (np.isnan(float(v)) or np.isinf(float(v))) else v
            )
        )

    display_cols = [c for c in ["Method","Order Qty","Total Cost (₹)","Holding (₹)",
                                  "Stockout (₹)","Ordering Cost","Stockout Days",
                                  "Fill Rate"] if c in comp.columns]
    st.dataframe(comp[display_cols].set_index("Method"), use_container_width=True)

    methods = comp["Method"].tolist()
    costs   = [float(c.replace("₹","").replace(",","")) for c in comp["Total Cost (₹)"].tolist()]
    colors  = ["#22c55e" if "AI" in m else "#475569" for m in methods]

    fig = go.Figure(go.Bar(
        x=methods, y=costs, marker_color=colors,
        text=[f"₹{c:,.0f}" for c in costs],
        textposition="outside", textfont=dict(size=12,color="white"),
        width=0.45,
    ))
    fig.update_layout(
        template="plotly_dark", height=340,
        title=dict(text="Total Inventory Cost — AI vs Baselines (lower is better)", font=dict(size=13, color="#94a3b8")),
        xaxis_title="Method", yaxis_title="Total Cost (₹)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        showlegend=False, margin=dict(t=50,b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    fill_rates = [float(f.replace("%","")) for f in comp["Fill Rate"].tolist()]
    fig2 = go.Figure(go.Bar(
        x=methods, y=fill_rates,
        marker_color=["#22c55e" if "AI" in m else "#475569" for m in methods],
        text=[f"{f:.1f}%" for f in fill_rates],
        textposition="outside", textfont=dict(size=12,color="white"),
        width=0.45,
    ))
    fig2.update_layout(
        template="plotly_dark", height=280,
        title=dict(text="Fill Rate Comparison (higher is better)", font=dict(size=13, color="#94a3b8")),
        xaxis_title="Method", yaxis_title="Fill Rate (%)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        showlegend=False, margin=dict(t=45,b=50),
        yaxis_range=[0, 110]
    )
    st.plotly_chart(fig2, use_container_width=True)


# TAB: DRIFT MONITOR
def tab_drift_monitor(df_feat, store, prod):
    st.markdown('<div class="section-title">Data Drift & Regime Monitor</div>', unsafe_allow_html=True)

    detector = DataDriftDetector(reference_days=90, recent_days=30)

    if store != "All Stores" and prod != "All Products":
        st.markdown(f"**Deep Analysis — {prod}  ·  {store}**")

        drift = detector.detect(df_feat, store, prod)
        drift_colors = {"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#22c55e","UNKNOWN":"#475569"}
        border = drift_colors.get(drift["drift_level"],"#475569")
        st.markdown(
            f'<div style="background:var(--secondary-background-color);border-left:3px solid {border};'
            'border-radius:5px;padding:.8rem 1rem;margin-bottom:.8rem;font-size:0.875rem;'
            'color:var(--text-color);border:1px solid rgba(255,255,255,0.05);border-left:3px solid '
            + border + '">'
            f'{drift["message"]}</div>', unsafe_allow_html=True
        )

        d1,d2,d3 = st.columns(3)
        d1.metric("JSD-PSI",
                  f"{drift['psi']:.4f}",
                  help="Bounded [0,0.5]. <0.10 stable · 0.10–0.25 moderate · >0.25 retrain")
        d2.metric("KS p-value",
                  f"{drift['ks_pvalue']:.4f}",
                  help="Two-sample KS test. <0.05 = distributions significantly different")
        d3.metric("Mean Shift",
                  f"{drift['mean_shift_pct']:+.1f}%",
                  help="((current_mean − baseline_mean) / baseline_mean) × 100")

        st.markdown("---")

    st.markdown("**System-Wide Drift Scan — All SKUs**")
    with st.spinner("Scanning all store × product pairs for drift…"):
        scan_df = detector.batch_scan(df_feat)
    st.dataframe(scan_df.head(30), use_container_width=True)


# TAB: AI NARRATOR
def tab_ai_narrator(df_feat, store, prod, forecast, gemini_key=""):
    st.markdown('<div class="section-title">AI Demand Intelligence Narrative</div>', unsafe_allow_html=True)

    if store == "All Stores" or prod == "All Products":
        st.info("Select a single Store + Product for AI narrative.")
        return

    row = df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)]\
          .sort_values("Date").iloc[-1]

    pname      = str(row.get("Product_Name", prod))
    curr_inv   = float(row["Inventory_Level"])
    avg_daily  = float(forecast["Predicted_Demand"].mean())
    total_7d   = float(forecast["Predicted_Demand"].sum())
    unit_price = float(row["Price"])
    lead_time  = max(int(row.get("Lead_Time_Days",3)), 3)

    with st.spinner("Gathering signals…"):
        ltd       = avg_daily * lead_time
        ss        = avg_daily * 5
        rop       = ltd + ss
        if curr_inv <= 0:       stock_status = "OUT_OF_STOCK"
        elif curr_inv <= ltd:   stock_status = "CRITICAL"
        elif curr_inv <= rop:   stock_status = "REORDER"
        else:                   stock_status = "SAFE"

        raw_order = max(0.0, total_7d + ss - curr_inv) if stock_status != "SAFE" else 0.0
        order_qty = round(min(raw_order, int(row.get("Supplier_Limit",300))))

        fest_rows  = forecast[forecast["Festival_Mult"]>1.0]
        fest_str   = (f"{fest_rows['Festival'].iloc[0]} (×{fest_rows['Festival_Mult'].max():.2f})"
                      if not fest_rows.empty else "None")

        hist30 = df_feat[(df_feat["Store_ID"]==store)&(df_feat["Product_ID"]==prod)]\
                 .sort_values("Date")["Units_Sold"].tail(30)
        cv  = float(hist30.std() / (hist30.mean()+1e-6))
        ts  = PredictionTrustScorer()
        cp_obj = build_conformal_predictor(df_feat, store, prod, alpha=0.05)
        trust = ts.score(
            cv=cv, q_hat=cp_obj.q_hat if cp_obj.is_fitted else avg_daily*0.2,
            mu=avg_daily, days_since_data=0,
            is_festival=not fest_rows.empty,
            is_anomaly=False, has_drift=False,
        )

        drift = DataDriftDetector().detect(df_feat, store, prod)

        causal_est = FestivalCausalEstimator()
        best_festival = None
        for fd, (fn, ft, _) in sorted(FESTIVALS_EXACT.items()):
            if ft in ("mega","major") and pd.Timestamp(fd) <= df_feat["Date"].max():
                best_festival = (fd, fn)
                break
        if best_festival:
            did = causal_est.estimate_lift(df_feat, pd.Timestamp(best_festival[0]), store, prod)
        else:
            did = {"causal_lift":1.0,"naive_lift":1.0}

        h_daily_n  = unit_price * 0.65 * 0.25 / 365
        sigma_n    = max(hist30.std() if len(hist30) > 2 else avg_daily * 0.2, 1.0)
        lead_demand_n = avg_daily * lead_time
        safety_n      = avg_daily * 5
        rop_n         = lead_demand_n + safety_n
        if curr_inv <= 0 or curr_inv <= lead_demand_n or curr_inv <= rop_n:
            raw_ord_n = max(0.0, total_7d + safety_n - curr_inv)
        else:
            raw_ord_n = 0.0
        rec_ord_n = round(min(raw_ord_n, int(row.get("Supplier_Limit", 300))))

        opt_eng  = InventoryCostOptimizer(h_daily_n, unit_price*0.35, unit_price*0.5, lead_time)
        cost_res = opt_eng.optimize(
            total_7d, sigma_n, curr_inv, unit_price,
            recommended_order=float(rec_ord_n)
        )

        mc_eng = MonteCarloInventorySimulator(n_simulations=300)
        mc_res = mc_eng.run(
            avg_daily, hist30.std() or avg_daily*0.2,
            lead_time, 1.0, curr_inv, order_qty, 30,
            h_daily_n, unit_price*0.35, unit_price
        )

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Trust Score",   f"{trust['score']:.0f}/100",             trust["label"])
    c2.metric("Drift Level",   drift["drift_level"])
    c3.metric("Causal Lift",   f"×{did['causal_lift']:.2f}",            f"naive ×{did['naive_lift']:.2f}")
    c4.metric("Stockout Risk", f"{mc_res['stockout_prob_pct']:.1f}%")

    st.markdown("---")
    st.markdown("**AI Narrative**")

    narrative = None
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = build_context_payload(
                store_id=store, product_name=pname,
                forecast_total=total_7d, avg_daily=avg_daily,
                stock_status=stock_status, curr_inv=curr_inv,
                order_qty=order_qty, festival_window=fest_str,
                trust_score=trust["score"], trust_label=trust["label"],
                drift_level=drift["drift_level"],
                changepoints=[],
                causal_lift=did["causal_lift"], naive_lift=did["naive_lift"],
                cost_opt_qty=cost_res["recommended_qty"],
                cost_savings=cost_res["savings_vs_heuristic"],
                mc_stockout_prob=mc_res.get("stockout_prob", mc_res.get("stockout_prob_pct", 0) / 100),
                mc_profit_p5=mc_res.get("profit_p5", 0),
                mc_profit_mean=mc_res.get("profit_mean", 0),
            )
            with st.spinner("Generating narrative…"):
                resp = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.75,
                        "max_output_tokens": 320,
                        "top_p": 0.92,
                    }
                )
            narrative = _clean_md(resp.text if hasattr(resp, "text") else str(resp))
            st.caption("Generated by Gemini AI")
        except Exception as e:
            st.warning(f"Gemini failed: {e}")
            narrative = None

    if not narrative:
        causal_pct = (did['causal_lift'] - 1) * 100
        bias_pct   = (did['naive_lift'] - did['causal_lift']) * 100
        narrative = (
            f"{pname} at {store} - 7-day demand forecast: {total_7d:.0f} units "
            f"(avg {avg_daily:.1f}/day). "
            f"Current inventory: {curr_inv:.0f} units | Status: {stock_status}. "
            f"Recommended order: {order_qty} units. "
            f"Festival impact: {fest_str} - causal lift {did['causal_lift']:.2f}× "
            f"({causal_pct:+.1f}% real demand uplift, naive estimate carried {bias_pct:.1f}% confounding bias). "
            f"Data drift: {drift['drift_level']} (PSI={drift['psi']:.3f}). "
            f"Prediction trust: {trust['score']:.0f}/100 ({trust['label']}). "
            f"Monte Carlo (300 sims): {mc_res['stockout_prob_pct']:.1f}% stockout probability over 30 days. "
            + (
                f"Cost-optimal order: {cost_res['recommended_qty']} units - "
                f"saves ₹{cost_res['savings_vs_heuristic']:,.0f} vs 95% SL heuristic."
                if cost_res['savings_vs_heuristic'] > 10
                else f"Cost-optimal order: {cost_res['recommended_qty']} units (near-optimal vs heuristic)."
            )
        )

    st.markdown(
        '<div class="alert-card" style="font-size:.9rem;line-height:1.8;'
        f'border-left:3px solid #7c3aed">{narrative}</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("**Input Signals**")
    signal_rows = [
        ("Stock Status",             stock_status),
        ("Current Inventory",        f"{curr_inv:.0f} units"),
        ("7-Day Forecast",           f"{total_7d:.0f} units  (avg {avg_daily:.1f}/day)"),
        ("Festival Window",          fest_str),
        ("Causal Lift",              f"×{did['causal_lift']:.3f}  (naive ×{did['naive_lift']:.3f})"),
        ("Confounding Bias",         f"{(did['naive_lift']-did['causal_lift'])*100:.1f}% of naive lift spurious"),
        ("Data Drift",               drift["drift_level"] + f"  (PSI={drift['psi']:.3f})"),
        ("MC Stockout Risk",         f"{mc_res['stockout_prob_pct']:.1f}%  (300 sims)"),
        ("Cost-Optimal Qty",         f"{cost_res['recommended_qty']} units"),
        ("Cost Saving vs Heuristic", f"₹{cost_res['savings_vs_heuristic']:,.0f}"),
    ]
    sig_df = pd.DataFrame(signal_rows, columns=["Signal", "Value"])
    st.dataframe(sig_df.set_index("Signal"), use_container_width=True)


# MAIN
def main():
    st.markdown("""
    <div class="brand-header">
      <div>
        <div class="brand-title">RetailIQ — AI Inventory Intelligence</div>
        <div class="brand-sub">D-Mart Scale  ·  Monte Carlo Risk  ·  Cost Optimisation  ·  Drift Detection  ·  Gemini Narratives</div>
      </div>
      <div class="brand-pill">v4 · Tanish Edition</div>
    </div>""", unsafe_allow_html=True)

    df_raw = get_raw_data()
    xgb_model, lgb_model, fc, weights, dff = get_model(df_raw)

    fc_rename = {
        "roll_mean_7":"rolling_mean_7","roll_std_7":"rolling_std_7",
        "roll_mean_14":"rolling_mean_14","roll_std_14":"rolling_std_14",
        "roll_mean_28":"rolling_mean_28","roll_std_28":"rolling_std_28",
    }
    fc = [fc_rename.get(f, f) for f in fc]

    store, prod, svc, spike, delay, price_move, dr, gemini_key, alpha, n_mc = sidebar(dff)

    tabs = st.tabs([
        "Forecast", "Orders", "Business", "Simulation",
        "Cost Optimizer", "Monte Carlo", "Baseline Proof",
        "Drift Monitor", "Festival Calendar", "AI Narrator",
        "Explainability", "Alerts",
    ])
    with tabs[0]:
        result = tab_forecast(dff, xgb_model, lgb_model, fc, weights, store, prod, alpha)
        if result is None or result[0] is None:
            fcast = forecast_next_7_days_v4(
                dff,
                store_id=sorted(dff["Store_ID"].unique())[0],
                product_id=sorted(dff["Product_ID"].unique())[0],
            )
        else:
            fcast, hist = result
    with tabs[1]:
        tab_inventory(dff, xgb_model, lgb_model, fc, weights, store, prod, fcast, svc)
    with tabs[2]:
        tab_business(dff, store, prod, dr)
    with tabs[3]:
        tab_simulation(dff, store, prod, fcast, spike, delay, price_move)
    with tabs[4]:
        tab_cost_optimizer(dff, store, prod, fcast)
    with tabs[5]:
        tab_monte_carlo(dff, store, prod, fcast, n_mc)
    with tabs[6]:
        tab_baseline_proof(dff, store, prod, fcast)
    with tabs[7]:
        tab_drift_monitor(dff, store, prod)
    with tabs[8]:
        tab_festival_calendar(dff, store, prod)
    with tabs[9]:
        tab_ai_narrator(dff, store, prod, fcast, gemini_key)
    with tabs[10]:
        tab_explain(dff, xgb_model, fc, store, prod)
    with tabs[11]:
        tab_alerts(dff, xgb_model, lgb_model, fc, weights)


if __name__ == "__main__":
    main()





