"""
Page 9 -- Exploration to Operations Link (USP-8)
Connects high-prospectivity zones with nearest mine operational risk.
The central architectural differentiator of this platform.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

from src.data_loader import load_satellite_data, load_operational_data, MOIL_MINES
from src.prospectivity_model import (
    train_prospectivity_model, predict_prospectivity, get_top_k_targets, bootstrap_uncertainty
)
from src.production_model import train_production_model, predict_production_bulk
from src.visualization import gmp_map_html

st.set_page_config(
    page_title="Exploration-Ops Link -- MOIL AI",
    page_icon="🔗", layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0a0e1a;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a 0%,#1a2744 100%);border-right:1px solid #1e3a5f;}
.link-card{background:linear-gradient(135deg,#0d1b2a,#1a2744);border:1px solid #1e3a5f;
    border-radius:12px;padding:1rem 1.2rem;margin:.4rem 0;}
.link-card-high{border-color:#38bdf8;}
.link-card-medium{border-color:#818cf8;}
.link-card-low{border-color:#475569;}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:999px;font-size:.72rem;font-weight:600;margin:.1rem;}
.badge-blue{background:rgba(56,189,248,.15);color:#38bdf8;border:1px solid rgba(56,189,248,.3);}
.badge-red{background:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.3);}
.badge-yellow{background:rgba(250,204,21,.15);color:#facc15;border:1px solid rgba(250,204,21,.3);}
.badge-green{background:rgba(74,222,128,.15);color:#4ade80;border:1px solid rgba(74,222,128,.3);}
.info-box{background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.25);border-left:3px solid #38bdf8;
    border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#94a3b8;margin:.75rem 0;}
.section-header{font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:.75rem;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:1.5rem;">
    <h1 style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:700;
    background:linear-gradient(135deg,#38bdf8,#818cf8,#4ade80);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;margin:0;">
    🔗 Exploration → Operations Intelligence Link
    </h1>
    <p style="color:#94a3b8;margin:0.2rem 0 0;">
    From discovering where manganese may exist to predicting where production may fall short.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    🌟 <strong>Platform Differentiator:</strong> Most AI systems treat exploration and operations as separate domains.
    This page connects them — showing how each high-prospectivity zone relates to its nearest active mine's
    current operational risk, enabling <em>joint spatial-operational</em> decision making.
</div>
""", unsafe_allow_html=True)

# Load everything
with st.spinner("Loading and linking datasets..."):
    geo_df   = load_satellite_data()
    ops_df   = load_operational_data()
    p_model, p_metrics, p_fi, rf_model = train_prospectivity_model(geo_df)
    r_model, c_model, r_metrics, r_fi  = train_production_model(ops_df)
    geo_pred  = predict_prospectivity(p_model, geo_df)
    ops_pred  = predict_production_bulk(r_model, c_model, ops_df)

# Compute mine risk summary
mine_risk_map = {}
for mine in MOIL_MINES:
    mdf = ops_pred[ops_pred["mine"] == mine]
    if not mdf.empty:
        mine_risk_map[mine] = {
            "risk_label":   mdf["risk_label"].mode()[0],
            "avg_shortfall": mdf["shortfall_pct"].mean(),
            "avg_pred_tpd": mdf["predicted_tpd"].mean(),
            "capacity_tpd": MOIL_MINES[mine]["capacity_tpd"],
            "lat": MOIL_MINES[mine]["lat"],
            "lon": MOIL_MINES[mine]["lon"],
        }

# Get top-K targets with uncertainty
geo_unc = bootstrap_uncertainty(rf_model, geo_pred)
top_k   = get_top_k_targets(geo_unc, k=15)

# Link each target to nearest mine
def nearest_mine(lat, lon):
    best, best_d = None, 1e9
    for m, info in MOIL_MINES.items():
        d = np.sqrt((info["lat"] - lat)**2 + (info["lon"] - lon)**2) * 111
        if d < best_d:
            best_d = d
            best = m
    return best, round(best_d, 1)

top_k["nearest_mine"]    = [nearest_mine(r.latitude, r.longitude)[0] for _, r in top_k.iterrows()]
top_k["dist_to_mine_km2"] = [nearest_mine(r.latitude, r.longitude)[1] for _, r in top_k.iterrows()]
top_k["mine_risk"]       = top_k["nearest_mine"].map(lambda m: mine_risk_map.get(m, {}).get("risk_label", "N/A"))
top_k["mine_shortfall"]  = top_k["nearest_mine"].map(lambda m: mine_risk_map.get(m, {}).get("avg_shortfall", 0))

# ── Architecture flow diagram
st.markdown("---")
st.markdown("### 🗺️ System Architecture: The End-to-End Intelligence Chain")

st.markdown("""
<div style="display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:.5rem;padding:1.5rem;
    background:linear-gradient(135deg,rgba(13,27,42,0.8),rgba(26,39,68,0.8));
    border:1px solid #1e3a5f;border-radius:16px;text-align:center;font-size:.85rem;color:#94a3b8;">
    <div style="background:rgba(56,189,248,.1);border:1px solid #38bdf8;border-radius:10px;padding:.7rem 1.2rem;color:#38bdf8;font-weight:600;">
        🛰️ Satellite Data<br><small style="color:#94a3b8;">Sentinel-2 Indices</small>
    </div>
    <div style="color:#475569;font-size:1.2rem;">+</div>
    <div style="background:rgba(129,140,248,.1);border:1px solid #818cf8;border-radius:10px;padding:.7rem 1.2rem;color:#818cf8;font-weight:600;">
        🗿 Geology<br><small style="color:#94a3b8;">GSI Lithology + Faults</small>
    </div>
    <div style="color:#475569;font-size:1.2rem;">→</div>
    <div style="background:rgba(74,222,128,.1);border:1px solid #4ade80;border-radius:10px;padding:.7rem 1.2rem;color:#4ade80;font-weight:600;">
        🤝 Ensemble AI<br><small style="color:#94a3b8;">RF + XGB + NB</small>
    </div>
    <div style="color:#475569;font-size:1.2rem;">→</div>
    <div style="background:rgba(251,146,60,.1);border:1px solid #fb923c;border-radius:10px;padding:.7rem 1.2rem;color:#fb923c;font-weight:600;">
        🎯 Top-K Targets<br><small style="color:#94a3b8;">Ranked by Score</small>
    </div>
    <div style="color:#475569;font-size:1.2rem;">→</div>
    <div style="background:rgba(239,68,68,.1);border:1px solid #f87171;border-radius:10px;padding:.7rem 1.2rem;color:#f87171;font-weight:600;">
        ⚠️ Mine Risk<br><small style="color:#94a3b8;">Operational AI</small>
    </div>
    <div style="color:#475569;font-size:1.2rem;">→</div>
    <div style="background:rgba(250,204,21,.1);border:1px solid #facc15;border-radius:10px;padding:.7rem 1.2rem;color:#facc15;font-weight:600;">
        🧠 Action<br><small style="color:#94a3b8;">Evidence-based Rec.</small>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main split: map + linked table
col_map, col_link = st.columns([3, 2], gap="medium")

with col_map:
    st.markdown('<div class="section-header">🗺️ Prospectivity Zones Linked to Mine Risk</div>', unsafe_allow_html=True)
    api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
    map_html = gmp_map_html(geo_pred, api_key, height=460)
    components.html(map_html, height=460)

with col_link:
    st.markdown('<div class="section-header">🎯 Top-15 Targets → Nearest Mine Risk</div>', unsafe_allow_html=True)

    risk_colours = {"LOW": "#4ade80", "MEDIUM": "#facc15", "HIGH": "#fb923c", "CRITICAL": "#f87171", "N/A": "#475569"}

    for rank, row in top_k.iterrows():
        prosp_col  = "#38bdf8" if row["prospectivity_class"] == "HIGH" else \
                     "#818cf8" if row["prospectivity_class"] == "MEDIUM" else "#475569"
        mine_risk  = row["mine_risk"]
        mine_col   = risk_colours.get(mine_risk, "#475569")
        unc        = f"±{row.get('prosp_std', 0)*100:.1f}%" if "prosp_std" in row else ""

        st.markdown(f"""
        <div class="link-card link-card-{row['prospectivity_class'].lower()}">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:#e2e8f0;font-weight:600;font-size:.85rem;">
                    #{rank} &nbsp;
                    <span style="color:{prosp_col};">{row['prospectivity_class']}</span>
                    &nbsp;Score: <strong>{row['prospectivity_score']:.3f}</strong>
                    <span style="color:#475569;font-size:.75rem;"> {unc}</span>
                </span>
                <span class="badge badge-{'blue' if mine_risk == 'LOW' else 'yellow' if mine_risk == 'MEDIUM' else 'red'}">
                    {row['nearest_mine']}: {mine_risk}
                </span>
            </div>
            <div style="color:#94a3b8;font-size:.75rem;margin-top:.3rem;">
                📍 {row['latitude']:.3f}°N, {row['longitude']:.3f}°E &nbsp;|&nbsp;
                {row.get('lithology','N/A')} &nbsp;|&nbsp;
                {row['dist_to_mine_km2']:.1f} km to mine &nbsp;|&nbsp;
                Mine shortfall: <span style="color:{mine_col};">{row['mine_shortfall']:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Mine-level summary
st.markdown("---")
st.markdown("### 🏭 Mine-Level: Prospectivity Potential vs Operational Risk")

mine_rows = []
for mine, info in mine_risk_map.items():
    near_zones = top_k[top_k["nearest_mine"] == mine]
    high_zones = (near_zones["prospectivity_class"] == "HIGH").sum() if not near_zones.empty else 0
    mine_rows.append({
        "Mine":         mine,
        "State":        MOIL_MINES[mine]["state"],
        "High Prosp. Zones Nearby": int(high_zones),
        "Avg Shortfall %": round(info["avg_shortfall"], 1),
        "Risk Level":   info["risk_label"],
        "Capacity (t/d)": info["capacity_tpd"],
    })

mine_df = pd.DataFrame(mine_rows).sort_values("High Prosp. Zones Nearby", ascending=False)

# Bubble chart: prospectivity potential vs shortfall risk
fig_bubble = go.Figure()
for _, row in mine_df.iterrows():
    r_col = risk_colours.get(row["Risk Level"], "#fff")
    fig_bubble.add_trace(go.Scatter(
        x=[row["Avg Shortfall %"]],
        y=[row["High Prosp. Zones Nearby"]],
        mode="markers+text",
        marker=dict(
            size=max(row["Capacity (t/d)"] / 100, 20),
            color=r_col, opacity=0.8,
            line=dict(color="white", width=1),
        ),
        text=[row["Mine"]],
        textposition="top center",
        textfont=dict(color="#e2e8f0", size=11),
        name=row["Mine"],
        hovertemplate=(
            f"<b>{row['Mine']}</b><br>"
            f"Risk: {row['Risk Level']}<br>"
            f"Shortfall: {row['Avg Shortfall %']:.1f}%<br>"
            f"High-Prosp Zones: {row['High Prosp. Zones Nearby']}<br>"
            f"Capacity: {row['Capacity (t/d)']} t/day<extra></extra>"
        ),
        showlegend=False,
    ))

fig_bubble.add_vline(x=15, line_dash="dash", line_color="#fb923c",
                     annotation_text="High shortfall threshold", annotation_font_color="#fb923c")
fig_bubble.add_hline(y=2, line_dash="dash", line_color="#38bdf8",
                     annotation_text="High exploration potential", annotation_font_color="#38bdf8")

fig_bubble.update_layout(
    template="plotly_dark", paper_bgcolor="#0a0e1a",
    height=380, margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Average Production Shortfall %  (higher = worse ops risk)",
    yaxis_title="High-Prospectivity Zones Nearby  (higher = better exploration)",
    font=dict(color="#e2e8f0"),
    title=dict(text="Mine Positioning: Exploration Upside vs Operational Risk", font=dict(color="#94a3b8", size=13)),
)
st.plotly_chart(fig_bubble, use_container_width=True, key="exp_ops_bubble")

st.markdown("""
<div class="info-box">
    <strong>How to read this chart:</strong>
    <strong style="color:#38bdf8;">Top-left</strong> = High exploration upside, low operational risk → <em>Priority expansion zones</em>.
    <strong style="color:#f87171;">Bottom-right</strong> = Low exploration upside, high operational risk → <em>Operational intervention needed</em>.
    Bubble size represents mine capacity.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("MOIL AI Mining Intelligence Platform | USP-8: Exploration-Operations Link | SIH 2024 #26009")
