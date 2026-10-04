"""
Page 3 — Production Forecast: trend, forecast, shortfall analysis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_loader import load_operational_data, load_weather_data, MOIL_MINES
from src.production_model import (
    train_production_model, predict_production_bulk,
    forecast_mine, predict_production, get_production_shap,
    REG_FEATURES, RISK_LABELS,
)
from src.visualization import (
    production_trend_chart, shortfall_chart, forecast_chart,
    feature_importance_chart, shap_bar_chart, rainfall_shortfall_scatter,
    PALETTE, RISK_COLOURS,
)

st.set_page_config(page_title="Production Forecast — MOIL AI", page_icon="📉", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0a0e1a;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a 0%,#1a2744 100%);border-right:1px solid #1e3a5f;}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
.kpi-card{background:linear-gradient(135deg,#0d1b2a,#1a2744);border:1px solid #1e3a5f;border-radius:12px;padding:1.2rem;text-align:center;}
.kpi-value{font-size:1.9rem;font-weight:700;line-height:1.2;}
.kpi-label{color:#94a3b8;font-size:.75rem;font-weight:500;margin-top:.25rem;text-transform:uppercase;letter-spacing:.05em;}
.risk-critical,.risk-high,.risk-medium,.risk-low{padding:.2rem .6rem;border-radius:6px;font-size:.75rem;font-weight:700;text-transform:uppercase;}
.risk-critical{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.4);color:#f87171;}
.risk-high{background:rgba(251,146,60,.15);border:1px solid rgba(251,146,60,.4);color:#fb923c;}
.risk-medium{background:rgba(250,204,21,.15);border:1px solid rgba(250,204,21,.4);color:#facc15;}
.risk-low{background:rgba(74,222,128,.15);border:1px solid rgba(74,222,128,.4);color:#4ade80;}
.info-box{background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.25);border-left:3px solid #38bdf8;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#94a3b8;margin:.75rem 0;}
.warning-box{background:rgba(251,146,60,.08);border:1px solid rgba(251,146,60,.25);border-left:3px solid #fb923c;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#94a3b8;margin:.75rem 0;}
.section-header{font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:.75rem;}
.pred-panel{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:12px;padding:1.25rem;}
.pred-row{display:flex;justify-content:space-between;align-items:center;padding:.4rem 0;border-bottom:1px solid #1a2744;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}[data-testid="stToolbar"]{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent !important;visibility:visible !important;display:block !important;z-index:99999 !important;}
[data-testid="collapsedControl"],[data-testid="collapsedControl"] button,[data-testid="stSidebarCollapseButton"],header button{visibility:visible !important;display:inline-flex !important;opacity:1 !important;color:#38bdf8 !important;}
[data-testid="collapsedControl"]{background:rgba(13,27,42,.95) !important;border:1px solid #1e3a5f !important;border-radius:8px !important;box-shadow:0 4px 12px rgba(0,0,0,.5) !important;}
[data-testid="collapsedControl"]:hover{border-color:#38bdf8 !important;box-shadow:0 0 10px rgba(56,189,248,.5) !important;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("---")
    selected_mine = st.selectbox("Mine", list(MOIL_MINES.keys()), index=0)
    forecast_days = st.radio("Forecast Horizon", [7, 14, 30], index=1, horizontal=True)
    trend_days    = st.slider("Historical Trend (days)", 30, 180, 90)

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1rem;">
    <h1 style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:700;
    background:linear-gradient(135deg,#fb923c,#f87171);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;margin:0;">
    📉 Production Forecast & Shortfall Intelligence
    </h1>
    <p style="color:#94a3b8;margin:.2rem 0 0;">
    XGBoost-powered production prediction and risk quantification
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
⚠️ <strong>Prototype operational dataset — synthetic due to non-public operational records.</strong><br>
The dataset is generated with realistic mining variables. Design allows substitution with actual MOIL data.
AI predicts production risk and supports corrective planning — not a production guarantee.
</div>
""", unsafe_allow_html=True)

# ── Load & train ───────────────────────────────────────────────────────────────
with st.spinner("Loading data and training production models…"):
    ops_df = load_operational_data()
    wx_df  = load_weather_data()
    reg, clf, metrics, fi = train_production_model(ops_df)
    ops_pred = predict_production_bulk(reg, clf, ops_df)

# ── Single-day prediction panel ────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎛️ Interactive Production Predictor")
st.markdown("Adjust operational parameters to see the AI prediction in real time:")

col_inputs, col_output = st.columns([2, 1], gap="medium")

mine_info = MOIL_MINES[selected_mine]
with col_inputs:
    c1, c2, c3 = st.columns(3)
    with c1:
        planned_tpd   = st.number_input("Planned (t/day)", 100, 5000,
                                         int(mine_info["capacity_tpd"] * 0.9), step=50)
        equip_avail   = st.slider("Equipment Availability (%)", 40, 100, 82)
    with c2:
        downtime_hrs  = st.slider("Equipment Downtime (hrs)", 0.0, 12.0, 2.5, 0.5)
        blasting_delay= st.slider("Blasting Delay (hrs)", 0.0, 8.0, 1.5, 0.5)
    with c3:
        rainfall_mm   = st.slider("Rainfall (mm)", 0, 100, 8)
        ore_grade     = st.slider("Ore Grade (%Mn)", 25, 55, 38)

    c4, c5 = st.columns(2)
    with c4:
        maintenance   = st.selectbox("Maintenance Event", [0, 1], format_func=lambda x: "Yes" if x else "No")
        working_hrs   = st.slider("Working Hours", 8, 24, 18)
    with c5:
        prev_actual   = st.number_input("Previous Day Actual (t)", 0, 5000,
                                         int(mine_info["capacity_tpd"] * 0.85), step=50)

    predict_btn = st.button("🔮 Predict Production", use_container_width=True)

with col_output:
    if predict_btn or True:  # Always show prediction
        input_dict = {
            "planned_tpd":           planned_tpd,
            "equipment_avail_pct":   equip_avail,
            "equipment_downtime_hrs":downtime_hrs,
            "blasting_delay_hrs":    blasting_delay,
            "working_hrs":           working_hrs,
            "rainfall_mm":           rainfall_mm,
            "ore_grade_pct":         ore_grade,
            "maintenance_event":     maintenance,
            "prev_actual_tpd":       prev_actual,
        }
        result = predict_production(reg, clf, input_dict)
        risk   = result["risk_label"]
        badge_class = f"risk-{risk.lower()}"

        shortfall_col = (
            "#f87171" if result["shortfall_pct"] > 25 else
            "#fb923c" if result["shortfall_pct"] > 15 else
            "#facc15" if result["shortfall_pct"] > 5 else "#4ade80"
        )

        st.markdown(f"""
        <div class="pred-panel">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                <span style="color:#e2e8f0;font-weight:700;font-size:1rem;">⛏️ {selected_mine}</span>
                <span class="{badge_class}">{risk} RISK</span>
            </div>
            <div class="pred-row">
                <span style="color:#94a3b8;font-size:.85rem;">📋 Planned</span>
                <span style="color:#e2e8f0;font-weight:600;">{result['planned_tpd']:,.0f} t/day</span>
            </div>
            <div class="pred-row">
                <span style="color:#94a3b8;font-size:.85rem;">🤖 AI Predicted</span>
                <span style="color:#38bdf8;font-weight:700;font-size:1.05rem;">{result['predicted_tpd']:,.0f} t/day</span>
            </div>
            <div class="pred-row">
                <span style="color:#94a3b8;font-size:.85rem;">⚠️ Expected Shortfall</span>
                <span style="color:{shortfall_col};font-weight:600;">{result['shortfall_tpd']:,.0f} t/day</span>
            </div>
            <div class="pred-row" style="border:none;">
                <span style="color:#94a3b8;font-size:.85rem;">📊 Shortfall %</span>
                <span style="color:{shortfall_col};font-weight:700;font-size:1.05rem;">{result['shortfall_pct']:.1f}%</span>
            </div>
            <div style="margin-top:1rem;padding-top:.75rem;border-top:1px solid #1e3a5f;">
                <div style="color:#94a3b8;font-size:.75rem;margin-bottom:.5rem;font-weight:600;">RISK PROBABILITIES</div>
        """, unsafe_allow_html=True)

        risk_keys = ["LOW","MEDIUM","HIGH","CRITICAL"]
        for i, (rk, prob) in enumerate(zip(risk_keys, result["risk_proba"])):
            bar_col = RISK_COLOURS.get(rk, "#fff")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.5rem;margin:.25rem 0;font-size:.78rem;">
                <div style="width:60px;color:#94a3b8;">{rk}</div>
                <div style="flex:1;background:#1e3a5f;border-radius:4px;height:8px;overflow:hidden;">
                    <div style="width:{prob*100:.0f}%;background:{bar_col};height:100%;border-radius:4px;"></div>
                </div>
                <div style="width:35px;color:{bar_col};text-align:right;">{prob*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

# ── Historical trend + Forecast ────────────────────────────────────────────────
st.markdown("---")
col_trend, col_stats = st.columns([3, 1], gap="medium")

with col_trend:
    st.markdown(f'<div class="section-header">📈 Historical Trend — {selected_mine}</div>',
                unsafe_allow_html=True)
    mine_hist = ops_pred[ops_pred["mine"] == selected_mine]
    st.plotly_chart(
        production_trend_chart(mine_hist, n_days=trend_days),
        use_container_width=True, key="prod_trend_page",
    )

with col_stats:
    st.markdown('<div class="section-header">📊 Period Stats</div>', unsafe_allow_html=True)
    recent_mine = mine_hist.tail(trend_days)
    stats = [
        ("Avg Actual", f"{recent_mine['actual_tpd'].mean():,.0f} t/day", "#38bdf8"),
        ("Avg Planned", f"{recent_mine['planned_tpd'].mean():,.0f} t/day", "#94a3b8"),
        ("Avg Shortfall", f"{recent_mine['shortfall_pct'].mean():.1f}%", "#fb923c"),
        ("Max Downtime", f"{recent_mine['equipment_downtime_hrs'].max():.1f} h", "#f87171"),
        ("Avg Ore Grade", f"{recent_mine['ore_grade_pct'].mean():.1f}% Mn", "#818cf8"),
        ("Rain Days", f"{(recent_mine['rainfall_mm'] > 10).sum()}", "#38bdf8"),
    ]
    for label, val, col_hex in stats:
        st.markdown(f"""
        <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;
        padding:.6rem .9rem;margin:.3rem 0;display:flex;justify-content:space-between;align-items:center;">
            <span style="color:#94a3b8;font-size:.8rem;">{label}</span>
            <span style="color:{col_hex};font-weight:600;font-size:.9rem;">{val}</span>
        </div>""", unsafe_allow_html=True)

# ── Forecast chart ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f'<div class="section-header">🔮 {forecast_days}-Day Production Forecast — {selected_mine}</div>',
            unsafe_allow_html=True)

with st.spinner(f"Generating {forecast_days}-day forecast…"):
    forecast_df = forecast_mine(reg, clf, ops_df, selected_mine, n_days=forecast_days)

if not forecast_df.empty:
    hist_30 = mine_hist.tail(30)
    st.plotly_chart(
        forecast_chart(hist_30, forecast_df),
        use_container_width=True, key="forecast_chart",
    )

    # Forecast table
    st.markdown('<div class="section-header">📋 Forecast Details</div>', unsafe_allow_html=True)
    show_cols = ["date","mine","planned_tpd","predicted_tpd","shortfall_tpd","shortfall_pct","risk_label"]
    disp_df = forecast_df[show_cols].copy()
    disp_df["date"] = disp_df["date"].dt.strftime("%Y-%m-%d")
    disp_df.columns = ["Date","Mine","Planned (t)","Predicted (t)","Shortfall (t)","Shortfall %","Risk"]
    st.dataframe(
        disp_df.style.applymap(
            lambda v: f"color: {RISK_COLOURS.get(v,'#fff')}" if v in RISK_COLOURS else "",
            subset=["Risk"],
        ),
        use_container_width=True,
        height=300,
    )

# ── Shortfall analysis ─────────────────────────────────────────────────────────
st.markdown("---")
col_sf, col_rain = st.columns(2, gap="medium")

with col_sf:
    st.markdown(f'<div class="section-header">📉 Shortfall History — {selected_mine}</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        shortfall_chart(mine_hist, n_days=trend_days),
        use_container_width=True, key="shortfall_prod",
    )

with col_rain:
    st.markdown('<div class="section-header">🌧️ Rainfall vs Shortfall</div>', unsafe_allow_html=True)
    st.plotly_chart(
        rainfall_shortfall_scatter(mine_hist),
        use_container_width=True, key="rain_scatter",
    )

# ── Feature importance ─────────────────────────────────────────────────────────
st.markdown("---")
col_fi, col_shap = st.columns(2, gap="medium")

with col_fi:
    st.markdown('<div class="section-header">📈 Feature Importance — Production Model</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        feature_importance_chart(fi, "XGBoost Feature Importance — Production Regressor"),
        use_container_width=True, key="fi_prod",
    )

with col_shap:
    st.markdown('<div class="section-header">🔬 SHAP Analysis</div>', unsafe_allow_html=True)
    with st.spinner("Computing SHAP…"):
        sv, ss, fn = get_production_shap(reg, ops_df)
    if sv is not None:
        st.plotly_chart(
            shap_bar_chart(sv, ss, fn, "SHAP Impact — Production Regressor"),
            use_container_width=True, key="shap_prod",
        )
    else:
        st.info("Install `shap` for interpretability charts.")

st.markdown("---")
st.caption(
    "⚠️ Prototype operational dataset — synthetic (not official MOIL records). "
    "Production forecast is AI-assisted, not a production guarantee. "
    "Prototype metric on synthetic/limited data — not field-validated."
)
