"""
MOIL AI Mining Intelligence Platform
Main entry point for the Streamlit application.
"""
import streamlit as st

st.set_page_config(
    page_title="MOIL AI Mining Intelligence Platform",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Orbitron:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background: #0a0e1a; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1a2744 100%);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        border: 1px solid #1e3a5f;
        margin-bottom: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.05) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8, #818cf8, #fb923c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .hero-badges {
        display: flex;
        gap: 0.75rem;
        justify-content: center;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-orange {
        background: rgba(251, 146, 60, 0.1);
        border: 1px solid rgba(251, 146, 60, 0.3);
        color: #fb923c;
    }
    .badge-purple {
        background: rgba(129, 140, 248, 0.1);
        border: 1px solid rgba(129, 140, 248, 0.3);
        color: #818cf8;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #0d1b2a, #1a2744);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1.2;
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-delta {
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }
    .kpi-up { color: #4ade80; }
    .kpi-down { color: #f87171; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Alert/Risk badges */
    .risk-critical {
        background: rgba(239,68,68,0.15);
        border: 1px solid rgba(239,68,68,0.4);
        color: #f87171;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .risk-high {
        background: rgba(251,146,60,0.15);
        border: 1px solid rgba(251,146,60,0.4);
        color: #fb923c;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .risk-medium {
        background: rgba(250,204,21,0.15);
        border: 1px solid rgba(250,204,21,0.4);
        color: #facc15;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .risk-low {
        background: rgba(74,222,128,0.15);
        border: 1px solid rgba(74,222,128,0.4);
        color: #4ade80;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    /* Prospectivity badges */
    .prosp-high {
        background: rgba(56,189,248,0.15);
        border: 1px solid rgba(56,189,248,0.4);
        color: #38bdf8;
        padding: 0.2rem 0.7rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .prosp-medium {
        background: rgba(129,140,248,0.15);
        border: 1px solid rgba(129,140,248,0.4);
        color: #818cf8;
        padding: 0.2rem 0.7rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .prosp-low {
        background: rgba(148,163,184,0.15);
        border: 1px solid rgba(148,163,184,0.4);
        color: #94a3b8;
        padding: 0.2rem 0.7rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
    }

    /* Info box */
    .info-box {
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 3px solid #38bdf8;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0.75rem 0;
    }
    .warning-box {
        background: rgba(251,146,60,0.08);
        border: 1px solid rgba(251,146,60,0.25);
        border-left: 3px solid #fb923c;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0.75rem 0;
    }
    .caution-box {
        background: rgba(250,204,21,0.08);
        border: 1px solid rgba(250,204,21,0.25);
        border-left: 3px solid #facc15;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0.75rem 0;
    }

    /* Metric boxes */
    .metric-box {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Stmetric override */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d1b2a, #1a2744);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1rem;
    }

    /* Plotly chart container */
    .plot-container { border-radius: 12px; overflow: hidden; }

    /* Demo mode banner */
    .demo-banner {
        background: linear-gradient(90deg, rgba(56,189,248,0.15), rgba(129,140,248,0.15));
        border: 1px solid rgba(56,189,248,0.3);
        border-radius: 10px;
        padding: 0.75rem 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }

    /* Sidebar nav */
    .nav-item {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: 0.2rem 0;
        cursor: pointer;
        transition: background 0.2s;
        color: #94a3b8;
        font-size: 0.9rem;
    }
    .nav-item:hover { background: rgba(56,189,248,0.1); color: #38bdf8; }
    .nav-item.active { background: rgba(56,189,248,0.15); color: #38bdf8; border-left: 2px solid #38bdf8; }

    /* Stbutton override */
    .stButton button {
        background: linear-gradient(135deg, #1e3a5f, #0f3460);
        color: #e2e8f0;
        border: 1px solid #1e5f8f;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #0f3460, #1e3a5f);
        border-color: #38bdf8;
        color: #38bdf8;
    }

    /* Table styles */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = True
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False

# Hero header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">⛏️ MOIL AI Mining Intelligence Platform</div>
    <div class="hero-subtitle">Space-enabled Manganese Prospectivity &amp; Production Shortfall Intelligence</div>
    <div class="hero-badges">
        <span class="badge">🛰️ Sentinel-2 Satellite</span>
        <span class="badge badge-orange">🤖 XGBoost AI/ML</span>
        <span class="badge badge-purple">🗺️ GIS Mapping</span>
        <span class="badge">📊 Smart India Hackathon 2024</span>
        <span class="badge badge-orange">Problem #26009</span>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("""
    ### Welcome to MOIL AI Mining Intelligence
    This platform combines **satellite remote sensing**, **geological data**, and **operational intelligence** 
    to deliver two core capabilities:
    
    - 🗺️ **Manganese Prospectivity Mapping** — Identify areas with higher mineral potential using AI
    - 📉 **Production Shortfall Prediction** — Forecast and prevent production gaps proactively
    
    Navigate using the **sidebar** to explore all modules.
    """)

with col2:
    st.markdown("""
    #### 🏛️ Organization
    **Ministry of Steel**  
    MOIL Ltd.  
    Category: Software  
    Theme: Space Technology  
    SIH Problem #26009
    """)

with col3:
    demo_toggle = st.toggle("🚀 Demo Mode", value=st.session_state.demo_mode)
    st.session_state.demo_mode = demo_toggle
    if st.session_state.demo_mode:
        if "API_KEY" in st.secrets:
            api_status = "<br><small style='color:#4ade80'>🔑 API Key Detected in secrets.toml</small>"
        else:
            api_status = "<br><small style='color:#94a3b8'>Using sample data — no API credentials needed</small>"
            
        st.markdown(f"""
        <div style="background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.3);
        border-radius:8px;padding:0.6rem;font-size:0.8rem;color:#38bdf8;text-align:center;">
        ✅ Demo Mode Active{api_status}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
        ⚠️ Live Mode: Configure data sources in sidebar
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="info-box">
    📌 <strong>Scientific Note:</strong> Satellite and geological indicators are used to estimate <em>manganese prospectivity</em> 
    — not confirmed underground reserves. Prospectivity scores prioritize areas for field investigation and drilling confirmation. 
    Production forecasts are AI-assisted predictions, not guarantees.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🗺️ Module 1: Prospectivity Analysis")
    st.markdown("""
    Uses Sentinel-2 derived indices (NDVI, NDWI, Bare Soil Index, spectral ratios) 
    combined with GSI geological data to score manganese mineralization potential.
    """)
    if st.button("→ Open Prospectivity Analysis", key="goto_prosp"):
        st.switch_page("pages/2_prospectivity.py")

with col2:
    st.markdown("### 📉 Module 2: Production Intelligence")
    st.markdown("""
    Uses operational data (equipment uptime, blasting schedules, weather) 
    to predict production shortfalls and recommend corrective actions.
    """)
    if st.button("→ Open Production Forecast", key="goto_prod"):
        st.switch_page("pages/3_production.py")

st.markdown("---")
st.caption("MOIL AI Mining Intelligence Platform | Smart India Hackathon 2024 | Problem #26009 | Ministry of Steel — MOIL Ltd.")
