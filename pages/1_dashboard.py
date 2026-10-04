"""
Page 1 — Dashboard: KPIs, prospectivity map, production trend, risk summary.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

from src.data_loader import (
    load_satellite_data, load_operational_data, MOIL_MINES,
)
from src.prospectivity_model import train_prospectivity_model, predict_prospectivity
from src.production_model import train_production_model, predict_production_bulk
from src.risk_engine import generate_mine_summary
from src.visualization import (
    production_trend_chart, shortfall_chart, risk_donut,
    mine_production_bar, PALETTE, RISK_COLOURS, PROSP_COLOURS,
    gmp_map_html
)
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Dashboard — MOIL AI Mining",
    page_icon="📊", layout="wide",
)

# ── Load CSS from app.py via shared helper ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0a0e1a;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a 0%,#1a2744 100%);border-right:1px solid #1e3a5f;}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
.kpi-card{background:linear-gradient(135deg,#0d1b2a,#1a2744);border:1px solid #1e3a5f;border-radius:12px;padding:1.2rem;text-align:center;}
.kpi-value{font-size:1.9rem;font-weight:700;color:#38bdf8;line-height:1.2;}
.kpi-label{color:#94a3b8;font-size:0.75rem;font-weight:500;margin-top:0.25rem;text-transform:uppercase;letter-spacing:.05em;}
.risk-critical,.risk-high,.risk-medium,.risk-low{padding:.2rem .6rem;border-radius:6px;font-size:.75rem;font-weight:700;text-transform:uppercase;}
.risk-critical{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.4);color:#f87171;}
.risk-high{background:rgba(251,146,60,.15);border:1px solid rgba(251,146,60,.4);color:#fb923c;}
.risk-medium{background:rgba(250,204,21,.15);border:1px solid rgba(250,204,21,.4);color:#facc15;}
.risk-low{background:rgba(74,222,128,.15);border:1px solid rgba(74,222,128,.4);color:#4ade80;}
.info-box{background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.25);border-left:3px solid #38bdf8;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#94a3b8;margin:.75rem 0;}
.section-header{font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:.75rem;}
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
    st.markdown("**Filters**")
    selected_mine = st.selectbox("Mine", ["All"] + list(MOIL_MINES.keys()))
    date_range = st.date_input(
        "Date Range",
        value=(pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31")),
    )
    st.markdown("---")
    st.caption("MOIL AI Mining Intelligence\nSIH 2024 — Problem #26009")

# ── Page title ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem;">
    <h1 style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:700;
    background:linear-gradient(135deg,#38bdf8,#818cf8);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;margin:0;">
    📊 Operations Dashboard
    </h1>
    <p style="color:#94a3b8;margin:0.2rem 0 0;">
    Real-time mining intelligence — prospectivity, production, and risk at a glance
    </p>
</div>
""", unsafe_allow_html=True)

if st.session_state.get('demo_mode', True):
    st.markdown("""
    <div style="background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.3);
    border-radius:8px;padding:.6rem 1rem;font-size:.8rem;color:#38bdf8;margin-bottom:1rem;">
    🚀 <strong>Demo Mode Active</strong> — Prototype operational dataset (synthetic). 
    Satellite indicators and geological data are prototype proxies.
    </div>
    """, unsafe_allow_html=True)

# ── Load & train ───────────────────────────────────────────────────────────────
with st.spinner("Loading data and training models…"):
    geo_df  = load_satellite_data()
    ops_df  = load_operational_data()
    p_model, p_metrics, p_fi, rf_model = train_prospectivity_model(geo_df)
    r_model, c_model, r_metrics, r_fi = train_production_model(ops_df)

    geo_pred  = predict_prospectivity(p_model, geo_df)
    ops_pred  = predict_production_bulk(r_model, c_model, ops_df)

    st.session_state.models_trained = True

# Filter by date
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt = pd.Timestamp(date_range[0])
    end_dt   = pd.Timestamp(date_range[1])
    ops_filt = ops_pred[
        (ops_pred["date"] >= start_dt) & (ops_pred["date"] <= end_dt)
    ]
else:
    ops_filt = ops_pred

if selected_mine != "All":
    ops_filt = ops_filt[ops_filt["mine"] == selected_mine]

# ── KPIs ───────────────────────────────────────────────────────────────────────
high_zones     = int((geo_pred["prospectivity_class"] == "HIGH").sum())
total_area_km2 = 500 * 0.01 * 1.0   # proxy: 500 pts × 1 km² each
avg_pred_tpd   = ops_filt["predicted_tpd"].mean() if not ops_filt.empty else 0
avg_shortfall  = ops_filt["shortfall_pct"].mean() if not ops_filt.empty else 0
critical_alerts = int((ops_filt["risk_label"] == "CRITICAL").sum()) if not ops_filt.empty else 0
high_alerts     = int((ops_filt["risk_label"] == "HIGH").sum()) if not ops_filt.empty else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#38bdf8;">{high_zones}</div>
        <div class="kpi-label">🗺️ High Prospectivity Zones</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#818cf8;">{len(geo_pred)}</div>
        <div class="kpi-label">📍 Analysed Sample Points</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#4ade80;">{avg_pred_tpd:,.0f} t</div>
        <div class="kpi-label">📦 Avg Predicted Production/Day</div>
    </div>""", unsafe_allow_html=True)
with k4:
    risk_col = "#f87171" if avg_shortfall > 20 else "#fb923c" if avg_shortfall > 10 else "#facc15"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{risk_col};">{avg_shortfall:.1f}%</div>
        <div class="kpi-label">⚠️ Avg Shortfall Risk</div>
    </div>""", unsafe_allow_html=True)
with k5:
    alert_col = "#f87171" if critical_alerts > 0 else "#fb923c"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{alert_col};">{critical_alerts + high_alerts}</div>
        <div class="kpi-label">🚨 High/Critical Days (Period)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Layout: Map (L) + Risk Summary (R) ────────────────────────────────────
col_map, col_risk = st.columns([3, 2], gap="medium")

with col_map:
    st.markdown('<div class="section-header">🗺️ Manganese Prospectivity Map</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="font-size:0.75rem;">
    Satellite + geological indicators assess surface-level manganese prospectivity.
    This is <strong>NOT</strong> a confirmed reserve estimate — field investigation required.
    </div>
    """, unsafe_allow_html=True)

    # Build GMP HTML map
    api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
    map_html = gmp_map_html(geo_pred, api_key, height=480)
    components.html(map_html, height=480)

    # Legend
    st.markdown("""
    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:.5rem;font-size:0.8rem;color:#94a3b8;">
        <span>🔵 HIGH Prospectivity</span>
        <span>🟣 MEDIUM Prospectivity</span>
        <span>⚫ LOW Prospectivity</span>
        <span>🏭 Mine (colour = risk)</span>
    </div>
    """, unsafe_allow_html=True)

with col_risk:
    st.markdown('<div class="section-header">⚠️ Risk Summary</div>', unsafe_allow_html=True)

    # Risk donut
    st.plotly_chart(risk_donut(ops_filt), use_container_width=True, key="donut_dash")

    # Mine summary table
    st.markdown('<div class="section-header">Mine-Level Risk Overview</div>', unsafe_allow_html=True)
    mine_summary = generate_mine_summary(ops_filt, ops_filt)
    if not mine_summary.empty:
        for _, row in mine_summary.iterrows():
            risk = row.get("Risk Level", "LOW")
            badge_class = f"risk-{risk.lower()}"
            st.markdown(f"""
            <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;padding:.6rem .9rem;margin:.35rem 0;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="color:#e2e8f0;font-weight:600;font-size:.85rem;">⛏️ {row['Mine']}</span>
                    <span class="{badge_class}">{risk}</span>
                </div>
                <div style="color:#94a3b8;font-size:.75rem;margin-top:.3rem;">
                    Shortfall: {row['Expected Shortfall (t)']:.0f} t ({row['Shortfall %']:.1f}%) · {row['Main Cause']}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Production trend ───────────────────────────────────────────────────────────
st.markdown("---")
col_t1, col_t2 = st.columns([3, 2], gap="medium")

with col_t1:
    st.markdown('<div class="section-header">📈 Production Trend</div>', unsafe_allow_html=True)
    plot_mine = None if selected_mine == "All" else selected_mine
    st.plotly_chart(
        production_trend_chart(ops_filt, mine=plot_mine, n_days=90),
        use_container_width=True, key="prod_trend_dash",
    )

with col_t2:
    st.markdown('<div class="section-header">🏭 Mine Production Comparison</div>', unsafe_allow_html=True)
    st.plotly_chart(mine_production_bar(ops_filt), use_container_width=True, key="mine_bar_dash")

# ── Shortfall + Recommendations preview ───────────────────────────────────────
col_sf, col_rec = st.columns([3, 2], gap="medium")

with col_sf:
    st.markdown('<div class="section-header">📉 Daily Shortfall (Last 90 Days)</div>', unsafe_allow_html=True)
    st.plotly_chart(
        shortfall_chart(ops_filt, mine=plot_mine),
        use_container_width=True, key="shortfall_dash",
    )

with col_rec:
    st.markdown('<div class="section-header">🔔 Top Alerts</div>', unsafe_allow_html=True)
    top_alerts = ops_filt[ops_filt["risk_label"].isin(["CRITICAL","HIGH"])].sort_values("date", ascending=False).head(6)
    if not top_alerts.empty:
        for _, row in top_alerts.iterrows():
            risk = row["risk_label"]
            badge_class = f"risk-{risk.lower()}"
            st.markdown(f"""
            <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;padding:.6rem .9rem;margin:.35rem 0;">
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:#e2e8f0;font-size:.8rem;">{row['mine']} — {row['date'].strftime('%d %b %Y')}</span>
                    <span class="{badge_class}">{risk}</span>
                </div>
                <div style="color:#94a3b8;font-size:.75rem;">
                    Shortfall: {row['shortfall_tpd']:.0f} t · DT: {row['equipment_downtime_hrs']:.1f} h · Rain: {row['rainfall_mm']:.0f} mm
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:#4ade80;padding:1rem;text-align:center;font-size:.85rem;">
        ✅ No critical/high-risk days in selected period
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "📌 Prototype metric on synthetic/limited data — not field-validated. "
    "Satellite indicators assess surface prospectivity only; confirmed reserves require drilling. "
    "Production data is synthetic (not official MOIL records)."
)
