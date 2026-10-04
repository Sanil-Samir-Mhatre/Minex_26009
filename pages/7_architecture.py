"""
Page 7 — Architecture: visual system diagram and methodology explanation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Architecture — MOIL AI", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0a0e1a;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a 0%,#1a2744 100%);border-right:1px solid #1e3a5f;}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
.info-box{background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.25);border-left:3px solid #38bdf8;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#94a3b8;margin:.75rem 0;}
.arch-node{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;padding:.8rem 1.2rem;text-align:center;font-size:.82rem;color:#e2e8f0;}
.arch-node-blue{border-color:#38bdf8;box-shadow:0 0 12px rgba(56,189,248,0.2);}
.arch-node-purple{border-color:#818cf8;box-shadow:0 0 12px rgba(129,140,248,0.2);}
.arch-node-orange{border-color:#fb923c;box-shadow:0 0 12px rgba(251,146,60,0.2);}
.arch-node-green{border-color:#4ade80;box-shadow:0 0 12px rgba(74,222,128,0.2);}
.section-header{font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:.75rem;}
.tech-pill{background:rgba(56,189,248,.1);border:1px solid rgba(56,189,248,.25);color:#38bdf8;padding:.2rem .6rem;border-radius:999px;font-size:.75rem;margin:.15rem;display:inline-block;}
.tech-pill-purple{background:rgba(129,140,248,.1);border:1px solid rgba(129,140,248,.25);color:#818cf8;}
.tech-pill-orange{background:rgba(251,146,60,.1);border:1px solid rgba(251,146,60,.25);color:#fb923c;}
.tech-pill-green{background:rgba(74,222,128,.1);border:1px solid rgba(74,222,128,.25);color:#4ade80;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1rem;">
    <h1 style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:700;
    background:linear-gradient(135deg,#4ade80,#38bdf8,#818cf8);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;margin:0;">
    🏗️ System Architecture & Methodology
    </h1>
    <p style="color:#94a3b8;margin:.2rem 0 0;">
    How satellite data, AI/ML, and operational intelligence connect into actionable mining insight
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
💡 <strong>Core Philosophy:</strong>
"Space data tells us <em>WHERE</em> potential exists.
Operational data tells us <em>WHERE</em> production may fall short.
AI connects both into actionable mining intelligence."
</div>
""", unsafe_allow_html=True)

# ── Architecture Diagram ───────────────────────────────────────────────────────
st.markdown("### 🔗 End-to-End Data Flow Architecture")

# Plotly-based architecture diagram using shapes + annotations
fig = go.Figure()

fig.update_layout(
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0a0e1a",
    width=1100,
    height=720,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(visible=False, range=[0, 10]),
    yaxis=dict(visible=False, range=[0, 14]),
    font=dict(family="Inter", color="#e2e8f0"),
)

# Helper for adding boxes
def add_box(fig, x0, y0, x1, y1, fill, line_col, text, font_size=11, font_col="#e2e8f0"):
    fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                  fillcolor=fill, line=dict(color=line_col, width=1.5),
                  layer="below")
    fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=text,
                       showarrow=False, font=dict(size=font_size, color=font_col),
                       align="center")

def add_arrow(fig, x0, y0, x1, y1, col="#1e3a5f"):
    fig.add_annotation(
        x=x1, y=y1, ax=x0, ay=y0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.2,
        arrowwidth=1.5, arrowcolor=col,
    )

# ── Layer 0: Data Sources ──────────────────────────────────────────────────────
add_box(fig, 0.1, 12.8, 2.4, 13.8, "rgba(56,189,248,0.12)", "#38bdf8",
        "🛰️ Sentinel-2<br>Satellite Imagery<br><span style='font-size:9px;color:#94a3b8'>Copernicus Data Space</span>", 10)
add_box(fig, 2.6, 12.8, 5.0, 13.8, "rgba(129,140,248,0.12)", "#818cf8",
        "🗺️ GSI Geological<br>Maps & Occurrence<br><span style='font-size:9px;color:#94a3b8'>gsi.gov.in</span>", 10)
add_box(fig, 5.2, 12.8, 7.6, 13.8, "rgba(56,189,248,0.12)", "#38bdf8",
        "🌦️ ERA5 Weather<br>Reanalysis<br><span style='font-size:9px;color:#94a3b8'>Copernicus CDS</span>", 10)
add_box(fig, 7.8, 12.8, 9.9, 13.8, "rgba(251,146,60,0.12)", "#fb923c",
        "⚙️ Operational Data<br>(MOIL / Synthetic)<br><span style='font-size:9px;color:#94a3b8'>Mine Records</span>", 10)

# ── Layer 1: Data Ingestion ────────────────────────────────────────────────────
add_box(fig, 0.8, 11.4, 9.2, 12.3, "rgba(30,58,95,0.5)", "#1e3a5f",
        "📥  DATA INGESTION  —  src/data_loader.py  |  Caching  |  Fallback to prototype data", 10)

# arrows from sources to ingestion
for cx in [1.25, 3.8, 6.4, 8.85]:
    add_arrow(fig, cx, 12.8, cx, 12.3, "#475569")

# ── Layer 2: Pre-processing ────────────────────────────────────────────────────
add_box(fig, 0.8, 10.0, 9.2, 10.9, "rgba(30,58,95,0.5)", "#1e3a5f",
        "⚙️  DATA PREPROCESSING  —  Band normalization  |  Index computation  |  Missing-value handling", 10)
add_arrow(fig, 5.0, 11.4, 5.0, 10.9, "#475569")

# ── Layer 3: Feature Engineering split ────────────────────────────────────────
add_box(fig, 0.5, 8.5, 4.8, 9.5, "rgba(56,189,248,0.1)", "#38bdf8",
        "🔬 FEATURE ENGINEERING — PROSPECTIVITY<br>NDVI · NDWI · BSI · Clay/Iron Ratio · SWIR<br>Lithology · Formation · Fault Proximity · Elevation", 9)
add_box(fig, 5.2, 8.5, 9.5, 9.5, "rgba(251,146,60,0.1)", "#fb923c",
        "🔬 FEATURE ENGINEERING — PRODUCTION<br>Planned · Downtime · Blasting Delay<br>Rainfall · Grade · Availability · Prev Actual", 9)

add_arrow(fig, 2.65, 10.0, 2.65, 9.5, "#38bdf8")
add_arrow(fig, 7.35, 10.0, 7.35, 9.5, "#fb923c")

# ── Layer 4: ML Models ────────────────────────────────────────────────────────
add_box(fig, 0.5, 6.8, 4.8, 7.9, "rgba(56,189,248,0.15)", "#38bdf8",
        "🤖  XGBoost Classifier<br>src/prospectivity_model.py<br>Binary: Mn Occurrence vs Background", 10)
add_box(fig, 5.2, 6.8, 9.5, 7.9, "rgba(251,146,60,0.15)", "#fb923c",
        "🤖  XGBoost Regressor + Classifier<br>src/production_model.py<br>Predict tonnes/day + Risk Class", 10)

add_arrow(fig, 2.65, 8.5, 2.65, 7.9, "#38bdf8")
add_arrow(fig, 7.35, 8.5, 7.35, 7.9, "#fb923c")

# ── Layer 5: Model Outputs ────────────────────────────────────────────────────
add_box(fig, 0.5, 5.3, 4.8, 6.3, "rgba(56,189,248,0.2)", "#38bdf8",
        "📍  MANGANESE PROSPECTIVITY SCORE<br>HIGH  ·  MEDIUM  ·  LOW<br>⚠️ Surface indicator — not a reserve estimate", 10)
add_box(fig, 5.2, 5.3, 9.5, 6.3, "rgba(251,146,60,0.2)", "#fb923c",
        "📉  PRODUCTION SHORTFALL FORECAST<br>Predicted t/day  ·  Shortfall %  ·  Risk Level<br>LOW  ·  MEDIUM  ·  HIGH  ·  CRITICAL", 10)

add_arrow(fig, 2.65, 6.8, 2.65, 6.3, "#38bdf8")
add_arrow(fig, 7.35, 6.8, 7.35, 6.3, "#fb923c")

# ── Layer 6: GIS + Risk Engine ────────────────────────────────────────────────
add_box(fig, 0.5, 3.8, 4.8, 4.8, "rgba(56,189,248,0.1)", "#38bdf8",
        "🗺️  GIS PROSPECTIVITY MAP<br>Folium / PyDeck  |  Click to inspect zone<br>Known occurrences  ·  Mine markers", 10)
add_box(fig, 5.2, 3.8, 9.5, 4.8, "rgba(74,222,128,0.1)", "#4ade80",
        "⚠️  RISK & DECISION ENGINE<br>src/risk_engine.py  |  Rule-based (no LLM)<br>Thresholds → Evidence → Recommended Action", 10)

add_arrow(fig, 2.65, 5.3, 2.65, 4.8, "#38bdf8")
add_arrow(fig, 7.35, 5.3, 7.35, 4.8, "#4ade80")

# Horizontal merge arrows → Recommendation layer
add_arrow(fig, 4.8, 4.3, 5.2, 4.3, "#94a3b8")

# ── Layer 7: Recommendations ──────────────────────────────────────────────────
add_box(fig, 2.0, 2.4, 8.0, 3.3, "rgba(74,222,128,0.15)", "#4ade80",
        "💡  RECOMMENDATIONS ENGINE  —  src/recommendations.py<br>IF downtime high → Reallocate equipment  |  IF rainfall high → Drainage protocol<br>IF shortfall critical → Trigger Recovery Plan  |  Compound risk detection", 10)

add_arrow(fig, 2.65, 3.8, 4.0, 3.3, "#4ade80")
add_arrow(fig, 7.35, 3.8, 6.0, 3.3, "#4ade80")

# ── Layer 8: Streamlit UI ─────────────────────────────────────────────────────
add_box(fig, 0.5, 0.8, 9.5, 1.9, "rgba(129,140,248,0.15)", "#818cf8",
        "🖥️  STREAMLIT DASHBOARD  —  app.py  +  7 pages<br>Dashboard  ·  Prospectivity  ·  Production  ·  Risk & Rec  ·  Data Explorer  ·  Model Perf  ·  Architecture", 10)

add_arrow(fig, 5.0, 2.4, 5.0, 1.9, "#818cf8")

st.plotly_chart(fig, use_container_width=True, key="arch_diagram")

# ── Methodology Explanation ────────────────────────────────────────────────────
st.markdown("---")
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("### 🛰️ Path A: Prospectivity Pipeline")
    st.markdown("""
    <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:12px;padding:1.25rem;">

    **1. Satellite Data Acquisition**
    - Sentinel-2 Level-2A surface reflectance
    - Bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11/B12 (SWIR)
    - Source: Copernicus Data Space (`dataspace.copernicus.eu`)

    **2. Spectral Index Computation**
    - **NDVI** = (NIR − Red) / (NIR + Red) → vegetation cover proxy
    - **NDWI** = (Green − NIR) / (Green + NIR) → water/moisture
    - **BSI** (Bare Soil Index) = [(SWIR1+Red) − (NIR+Blue)] / [...] → exposed rock/soil
    - **Iron oxide ratio** = Red / Blue → Mn-associated iron oxides
    - **Clay mineral ratio** = SWIR1 / SWIR2 → clay-rich alteration

    **3. Geological Integration**
    - GSI lithology classification (Gondite/Kodurite = primary Mn host rock)
    - Geological formation (Sausar Group = primary Mn-bearing terrane)
    - Fault/lineament proximity
    - Distance to known Mn occurrences

    **4. XGBoost Classifier**
    - Label: known_occurrence = 1 | background = 0
    - Output: Prospectivity Score ∈ [0, 1]
    - Classes: HIGH (≥0.65) | MEDIUM (0.40–0.65) | LOW (<0.40)
    - Evaluation: ROC-AUC, Precision, Recall, F1 + 5-fold CV

    **⚠️ Limitation:** This is a surface indicator model.
    Underground reserve estimation requires drilling + assay data.
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("### ⚙️ Path B: Production Intelligence Pipeline")
    st.markdown("""
    <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:12px;padding:1.25rem;">

    **1. Operational Data Collection**
    - Daily production records per mine
    - Equipment availability & downtime (hours)
    - Blasting schedule adherence (delay hours)
    - Rainfall, working hours, ore grade, maintenance events
    - Source: MOIL operational records (synthetic for prototype)

    **2. Weather Integration**
    - ERA5 daily rainfall for mine region
    - Monsoon season adjustment
    - Source: Copernicus Climate Data Store

    **3. XGBoost Regressor**
    - Predicts: actual production (tonnes/day)
    - Input features: 9 operational variables
    - Evaluation: MAE, RMSE, R²

    **4. XGBoost Classifier (Risk)**
    - LOW (<5% shortfall)
    - MEDIUM (5–15%)
    - HIGH (15–25%)
    - CRITICAL (>25%)
    - Evaluation: Weighted Accuracy, Precision, Recall, F1

    **5. Rule-Based Recommendation Engine**
    - Transparent IF-THEN logic — no LLM
    - Configurable thresholds
    - Maps: Factor → Evidence → Action
    - Compound risk detection (≥3 simultaneous high factors)

    **⚠️ Limitation:** Forecasts are AI estimates.
    Actual production depends on ground conditions and human decisions.
    </div>
    """, unsafe_allow_html=True)

# ── Technology Stack ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🛠️ Technology Stack")

col_t1, col_t2, col_t3, col_t4 = st.columns(4)

with col_t1:
    st.markdown("**🤖 AI / ML**")
    for tech in ["XGBoost", "scikit-learn", "SHAP", "NumPy", "Pandas"]:
        st.markdown(f'<span class="tech-pill">{tech}</span>', unsafe_allow_html=True)

with col_t2:
    st.markdown("**🗺️ GIS / Visualization**")
    for tech in ["Folium", "streamlit-folium", "Plotly", "GeoPandas"]:
        st.markdown(f'<span class="tech-pill tech-pill-purple">{tech}</span>', unsafe_allow_html=True)

with col_t3:
    st.markdown("**🛰️ Remote Sensing**")
    for tech in ["Sentinel-2 (Copernicus)", "ERA5 (Climate CDS)", "GSI Geological Maps", "Rasterio"]:
        st.markdown(f'<span class="tech-pill tech-pill-orange">{tech}</span>', unsafe_allow_html=True)

with col_t4:
    st.markdown("**🖥️ Frontend / Deployment**")
    for tech in ["Streamlit", "Joblib (model caching)", "Python 3.10+", "Plotly Dark Theme"]:
        st.markdown(f'<span class="tech-pill tech-pill-green">{tech}</span>', unsafe_allow_html=True)

# ── Future Deployment ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🚀 Future Production Deployment Architecture")

st.markdown("""
```
┌──────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                              │
├──────────────┬────────────────────────┬─────────────────────────────┤
│  DATA LAYER  │    COMPUTE LAYER       │   PRESENTATION LAYER        │
│              │                        │                             │
│  Sentinel-2  │  Apache Airflow DAGs   │  Streamlit Cloud / AWS EC2  │
│  ERA5 API    │  Model Retraining      │  Role-based access          │
│  MOIL SCADA  │  MLflow Experiment     │  Mobile-responsive UI       │
│  GSI WMS     │  Tracking              │  SMS/Email Alerts           │
│  PostgreSQL  │  Docker Containers     │  PDF Report Generation      │
│  PostGIS     │  GPU Inference (opt.)  │  API Integration w/ MOIL   │
└──────────────┴────────────────────────┴─────────────────────────────┘
```
""")

st.markdown("""
<div class="info-box">
🔒 <strong>Data Security:</strong> All MOIL operational data would be handled within MOIL's secure network perimeter.
No sensitive production or reserve data leaves the organisation's infrastructure.
The Streamlit frontend would be deployed on MOIL's internal server or a government-approved cloud.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("MOIL AI Mining Intelligence Platform | SIH 2024 Problem #26009 | Ministry of Steel — MOIL Ltd.")
