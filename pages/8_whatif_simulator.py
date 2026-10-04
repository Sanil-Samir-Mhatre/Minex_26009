"""
Page 8 -- What-if Production Simulator (USP-7)
Sliders -> instant production / shortfall / risk update with SHAP explanation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_loader import load_operational_data, MOIL_MINES
from src.production_model import (
    train_production_model, predict_whatif, WHATIF_DEFAULTS,
)

st.set_page_config(
    page_title="What-if Simulator -- MOIL AI",
    page_icon="⚠️", layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0a0e1a;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a 0%,#1a2744 100%);border-right:1px solid #1e3a5f;}
.kpi-card{background:linear-gradient(135deg,#0d1b2a,#1a2744);border:1px solid #1e3a5f;border-radius:12px;padding:1.2rem;text-align:center;}
.kpi-value{font-size:2rem;font-weight:700;line-height:1.2;}
.kpi-label{color:#94a3b8;font-size:0.75rem;font-weight:500;margin-top:0.25rem;text-transform:uppercase;letter-spacing:.05em;}
.risk-critical{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.4);color:#f87171;padding:.3rem .8rem;border-radius:8px;font-weight:700;text-align:center;}
.risk-high{background:rgba(251,146,60,.15);border:1px solid rgba(251,146,60,.4);color:#fb923c;padding:.3rem .8rem;border-radius:8px;font-weight:700;text-align:center;}
.risk-medium{background:rgba(250,204,21,.15);border:1px solid rgba(250,204,21,.4);color:#facc15;padding:.3rem .8rem;border-radius:8px;font-weight:700;text-align:center;}
.risk-low{background:rgba(74,222,128,.15);border:1px solid rgba(74,222,128,.4);color:#4ade80;padding:.3rem .8rem;border-radius:8px;font-weight:700;text-align:center;}
.info-box{background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.25);border-left:3px solid #38bdf8;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#94a3b8;margin:.75rem 0;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}[data-testid="stToolbar"]{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent !important;visibility:visible !important;display:block !important;z-index:99999 !important;}
[data-testid="collapsedControl"],[data-testid="collapsedControl"] button,[data-testid="stSidebarCollapseButton"],header button{visibility:visible !important;display:inline-flex !important;opacity:1 !important;color:#38bdf8 !important;}
[data-testid="collapsedControl"]{background:rgba(13,27,42,.95) !important;border:1px solid #1e3a5f !important;border-radius:8px !important;box-shadow:0 4px 12px rgba(0,0,0,.5) !important;}
[data-testid="collapsedControl"]:hover{border-color:#38bdf8 !important;box-shadow:0 0 10px rgba(56,189,248,.5) !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:1.5rem;">
    <h1 style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:700;
    background:linear-gradient(135deg,#fb923c,#f87171);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;margin:0;">
    ⚠️ What-if Production Simulator
    </h1>
    <p style="color:#94a3b8;margin:0.2rem 0 0;">
    Adjust operational parameters and instantly see how production, shortfall and risk respond.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    ⚗️ <strong>Prototype Scenario Model:</strong> This simulator uses the trained XGBoost production model
    to evaluate hypothetical operational scenarios. Change any slider to see immediate impact.
    The <strong>SHAP waterfall</strong> below explains exactly which factors are driving the result.
</div>
""", unsafe_allow_html=True)

# Load & train
with st.spinner("Loading models..."):
    ops_df = load_operational_data()
    r_model, c_model, r_metrics, r_fi = train_production_model(ops_df)

# ── Sidebar: mine selector
with st.sidebar:
    st.markdown("### Mine Context")
    sel_mine = st.selectbox("Base Mine", list(MOIL_MINES.keys()))
    planned_tpd = float(MOIL_MINES[sel_mine]["capacity_tpd"])
    st.markdown(f"**Capacity:** {planned_tpd:.0f} t/day")
    st.markdown("---")
    st.markdown("### Adjust Parameters")

# ── Two column layout
col_sliders, col_results = st.columns([1, 1], gap="large")

with col_sliders:
    st.markdown("### 🎛️ Operational Parameters")
    st.caption("Move sliders to simulate different conditions")

    downtime = st.slider("Equipment Downtime (hrs)", 0.0, 12.0, 3.0, 0.5,
                          help="Total hours of equipment unavailability per day")
    avail    = st.slider("Equipment Availability (%)", 50.0, 100.0, 88.0, 1.0,
                          help="Percentage of equipment fleet operational")
    blast_delay = st.slider("Blasting Delay (hrs)", 0.0, 8.0, 2.0, 0.5,
                             help="Delay in scheduled blasting operations")
    rainfall = st.slider("Rainfall (mm)", 0.0, 120.0, 10.0, 1.0,
                          help="Daily rainfall at the mine site")
    ore_grade = st.slider("Ore Grade (%Mn)", 30.0, 48.0, 38.0, 0.5,
                           help="Average manganese grade of ore extracted")
    maintenance = st.checkbox("Maintenance Event Today", value=False,
                               help="Scheduled or unscheduled maintenance shutdown")
    working_hrs = max(8.0, 22.0 - downtime - blast_delay)

    st.markdown(f"**Effective Working Hours:** {working_hrs:.1f} hrs")

# Run prediction
overrides = {
    "planned_tpd":            planned_tpd,
    "equipment_avail_pct":    avail,
    "equipment_downtime_hrs": downtime,
    "blasting_delay_hrs":     blast_delay,
    "working_hrs":            working_hrs,
    "rainfall_mm":            rainfall,
    "ore_grade_pct":          ore_grade,
    "maintenance_event":      int(maintenance),
    "prev_actual_tpd":        planned_tpd * 0.95,
}
result = predict_whatif(r_model, c_model, overrides)

with col_results:
    st.markdown("### 📊 Predicted Outcome")

    risk = result["risk_label"]
    risk_colours = {"LOW": "#4ade80", "MEDIUM": "#facc15", "HIGH": "#fb923c", "CRITICAL": "#f87171"}
    risk_col = risk_colours.get(risk, "#fff")

    k1, k2 = st.columns(2)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:#38bdf8;">{result['predicted_tpd']:,.0f}</div>
            <div class="kpi-label">Predicted Production (t/day)</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:{risk_col};">{result['shortfall_tpd']:,.0f}</div>
            <div class="kpi-label">Expected Shortfall (t/day)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="risk-{risk.lower()}" style="font-size:1.2rem;padding:.6rem 1rem;margin:.5rem 0;">
        Risk Level: {risk} &nbsp; ({result['shortfall_pct']:.1f}% shortfall)
    </div>""", unsafe_allow_html=True)

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result["predicted_tpd"],
        delta={"reference": planned_tpd, "valueformat": ".0f",
               "decreasing": {"color": "#f87171"}, "increasing": {"color": "#4ade80"}},
        title={"text": "Predicted vs Planned (t/day)", "font": {"color": "#94a3b8", "size": 13}},
        gauge={
            "axis": {"range": [0, planned_tpd * 1.1], "tickcolor": "#94a3b8"},
            "bar": {"color": risk_col},
            "bgcolor": "#0d1b2a",
            "steps": [
                {"range": [0, planned_tpd * 0.75], "color": "rgba(239,68,68,0.15)"},
                {"range": [planned_tpd * 0.75, planned_tpd * 0.9], "color": "rgba(251,146,60,0.15)"},
                {"range": [planned_tpd * 0.9, planned_tpd * 1.1], "color": "rgba(74,222,128,0.1)"},
            ],
            "threshold": {"line": {"color": "#38bdf8", "width": 2}, "value": planned_tpd},
        },
        number={"font": {"color": risk_col, "size": 36}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
        height=260, margin=dict(l=20, r=20, t=20, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True, key="whatif_gauge")

st.markdown("---")

# ── SHAP Waterfall (USP-6)
st.markdown("### 🔍 SHAP Explanation — Why This Prediction?")
st.caption("Positive values (red) push production DOWN from baseline. Negative values (blue) push it UP.")

if result.get("shap_values") is not None:
    sv   = result["shap_values"]
    feats = result["shap_features"]
    base = result["shap_base"]

    feat_labels = {
        "planned_tpd":            "Planned Production",
        "equipment_avail_pct":    "Equipment Availability",
        "equipment_downtime_hrs": "Equipment Downtime",
        "blasting_delay_hrs":     "Blasting Delay",
        "working_hrs":            "Working Hours",
        "rainfall_mm":            "Rainfall",
        "ore_grade_pct":          "Ore Grade",
        "maintenance_event":      "Maintenance Event",
        "prev_actual_tpd":        "Previous Day Production",
    }

    shap_df = pd.DataFrame({"feature": feats, "shap_value": sv})
    shap_df["label"] = shap_df["feature"].map(feat_labels).fillna(shap_df["feature"])
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=True)

    colours = ["#f87171" if v < 0 else "#38bdf8" for v in shap_df["shap_value"]]

    fig_shap = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["label"],
        orientation="h",
        marker_color=colours,
        text=[f"{v:+.1f} t/day" for v in shap_df["shap_value"]],
        textposition="outside",
        hovertemplate="%{y}: %{x:+.1f} t/day<extra></extra>",
    ))
    fig_shap.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e1a",
        height=340, margin=dict(l=10, r=80, t=30, b=10),
        xaxis_title="SHAP Contribution to Production (t/day)",
        font=dict(color="#e2e8f0", size=12),
        shapes=[{"type": "line", "x0": 0, "x1": 0, "y0": -0.5,
                 "y1": len(shap_df) - 0.5, "line": {"color": "#475569", "width": 1}}],
    )
    st.plotly_chart(fig_shap, use_container_width=True, key="whatif_shap")

    # Annotated summary
    top_neg = shap_df[shap_df["shap_value"] < 0].tail(2)
    top_pos = shap_df[shap_df["shap_value"] > 0].tail(2)
    factors_hurting = ", ".join(top_neg["label"].tolist()) if not top_neg.empty else "none"
    factors_helping = ", ".join(top_pos["label"].tolist()) if not top_pos.empty else "none"
    st.markdown(f"""
    <div class="info-box">
        <strong>Model says:</strong> Production of <strong>{result['predicted_tpd']:,.0f} t/day</strong>
        is driven down primarily by <strong style="color:#f87171">{factors_hurting}</strong>,
        partially offset by <strong style="color:#38bdf8">{factors_helping}</strong>.
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Install `shap` package to see SHAP explanations: `pip install shap`")

st.markdown("---")
st.caption("MOIL AI Mining Intelligence Platform | USP-7: What-if Production Simulator | SIH 2026 #26009")
