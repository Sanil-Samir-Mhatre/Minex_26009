"""
Page 5 — Data Explorer: inspect all datasets with distributions, stats, CSV download.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_satellite_data, load_weather_data, load_operational_data, MOIL_MINES

st.set_page_config(page_title="Data Explorer — MOIL AI", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0a0e1a;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a 0%,#1a2744 100%);border-right:1px solid #1e3a5f;}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
.info-box{background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.25);border-left:3px solid #38bdf8;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#94a3b8;margin:.75rem 0;}
.warning-box{background:rgba(251,146,60,.08);border:1px solid rgba(251,146,60,.25);border-left:3px solid #fb923c;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#94a3b8;margin:.75rem 0;}
.section-header{font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:.75rem;}
.stat-card{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;padding:.6rem .9rem;margin:.25rem 0;display:flex;justify-content:space-between;align-items:center;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}[data-testid="stToolbar"]{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent !important;visibility:visible !important;display:block !important;z-index:99999 !important;}
[data-testid="collapsedControl"],[data-testid="collapsedControl"] button,[data-testid="stSidebarCollapseButton"],header button{visibility:visible !important;display:inline-flex !important;opacity:1 !important;color:#38bdf8 !important;}
[data-testid="collapsedControl"]{background:rgba(13,27,42,.95) !important;border:1px solid #1e3a5f !important;border-radius:8px !important;box-shadow:0 4px 12px rgba(0,0,0,.5) !important;}
[data-testid="collapsedControl"]:hover{border-color:#38bdf8 !important;box-shadow:0 0 10px rgba(56,189,248,.5) !important;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1rem;">
    <h1 style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:700;
    background:linear-gradient(135deg,#818cf8,#38bdf8);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;margin:0;">
    🔍 Data Explorer
    </h1>
    <p style="color:#94a3b8;margin:.2rem 0 0;">
    Inspect all datasets used in the platform — satellite, geological, weather, operational
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load ───────────────────────────────────────────────────────────────────────
with st.spinner("Loading all datasets…"):
    sat_df = load_satellite_data()
    wx_df  = load_weather_data()
    ops_df = load_operational_data()

DARK_CLR = "#0d1b2a"
PLOT_BG  = "#1a2744"
TEXT_CLR = "#e2e8f0"
GRID_CLR = "#1e3a5f"


def dark_hist(df, col, title=""):
    fig = px.histogram(df, x=col, nbins=40, title=title)
    fig.update_traces(marker_color="#38bdf8", opacity=0.8)
    fig.update_layout(
        paper_bgcolor=DARK_CLR, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_CLR, family="Inter"),
        xaxis=dict(gridcolor=GRID_CLR), yaxis=dict(gridcolor=GRID_CLR),
        margin=dict(l=15, r=15, t=40, b=15), height=220,
        title=dict(font=dict(size=12)),
    )
    return fig


def missing_table(df):
    total   = len(df)
    missing = df.isnull().sum()
    pct     = (missing / total * 100).round(2)
    return pd.DataFrame({"Column": missing.index, "Missing": missing.values, "% Missing": pct.values})


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_sat, tab_geo, tab_wx, tab_ops = st.tabs([
    "🛰️ Satellite Features", "🗺️ Geological Data", "🌦️ Weather / ERA5", "⛏️ Operational Production"
])

# ────────────────────────────────────────────────────────────────────
# TAB 1 — Satellite
# ────────────────────────────────────────────────────────────────────
with tab_sat:
    st.markdown("""
    <div class="info-box">
    📡 <strong>Source:</strong> Sentinel-2 multispectral imagery (Copernicus Data Space — dataspace.copernicus.eu)<br>
    <strong>Derived indices:</strong> NDVI, NDWI, Bare Soil Index, Clay Mineral Ratio, Iron Oxide Ratio, SWIR Ratio<br>
    <strong>Status:</strong> Prototype — synthetic proxy data generated from realistic spectral distributions.
    In production, replace with actual Sentinel-2 Level-2A reflectance.
    </div>
    """, unsafe_allow_html=True)

    col_stat, col_tbl = st.columns([1, 2])
    with col_stat:
        st.markdown('<div class="section-header">📊 Dataset Statistics</div>', unsafe_allow_html=True)
        for lab, val in [
            ("Rows", len(sat_df)),
            ("Columns", len(sat_df.columns)),
            ("Numeric Columns", sat_df.select_dtypes(include=np.number).shape[1]),
            ("Lat Range", f"{sat_df['latitude'].min():.2f} – {sat_df['latitude'].max():.2f}"),
            ("Lon Range", f"{sat_df['longitude'].min():.2f} – {sat_df['longitude'].max():.2f}"),
            ("Known Occurrences", int(sat_df["known_occurrence"].sum())),
        ]:
            st.markdown(f"""
            <div class="stat-card">
                <span style="color:#94a3b8;font-size:.82rem;">{lab}</span>
                <span style="color:#38bdf8;font-weight:600;font-size:.88rem;">{val}</span>
            </div>""", unsafe_allow_html=True)

    with col_tbl:
        st.dataframe(sat_df.head(50), use_container_width=True, height=280)

    # Missing values
    st.markdown('<div class="section-header">🔎 Missing Values</div>', unsafe_allow_html=True)
    mv = missing_table(sat_df)
    st.dataframe(mv[mv["Missing"] > 0] if (mv["Missing"] > 0).any() else mv.head(5),
                 use_container_width=True, height=150)

    # Distributions
    st.markdown('<div class="section-header">📈 Index Distributions</div>', unsafe_allow_html=True)
    cols_to_plot = ["ndvi","ndwi","bsi","iron_ratio","clay_ratio","swir_ratio"]
    chart_cols   = st.columns(3)
    for i, col_name in enumerate(cols_to_plot):
        with chart_cols[i % 3]:
            st.plotly_chart(dark_hist(sat_df, col_name, col_name.upper()),
                            use_container_width=True, key=f"sat_hist_{col_name}")

    st.download_button(
        "⬇️ Download Satellite Dataset (CSV)", sat_df.to_csv(index=False), "satellite_features.csv", "text/csv"
    )

# ────────────────────────────────────────────────────────────────────
# TAB 2 — Geological
# ────────────────────────────────────────────────────────────────────
with tab_geo:
    st.markdown("""
    <div class="info-box">
    🗺️ <strong>Source:</strong> Geological Survey of India (gsi.gov.in) — public geological maps<br>
    <strong>Features:</strong> Lithology, formation, fault proximity, distance to known occurrence points<br>
    <strong>Status:</strong> Prototype — synthetic proxy data based on realistic geological distributions.
    In production, replace with GSI digital geological map data.
    </div>
    """, unsafe_allow_html=True)

    geo_cols = ["latitude","longitude","lithology","formation","dist_to_mine_km",
                "fault_proximity","known_occurrence","elevation_m","slope_deg"]
    geo_view = sat_df[geo_cols]

    col_stat2, col_tbl2 = st.columns([1, 2])
    with col_stat2:
        st.markdown('<div class="section-header">📊 Lithology Distribution</div>', unsafe_allow_html=True)
        lith_counts = sat_df["lithology"].value_counts()
        for lith, cnt in lith_counts.items():
            pct = cnt / len(sat_df) * 100
            st.markdown(f"""
            <div style="margin:.25rem 0;">
                <div style="display:flex;justify-content:space-between;font-size:.78rem;color:#94a3b8;margin-bottom:.1rem;">
                    <span>{lith}</span><span style="color:#e2e8f0;">{cnt} ({pct:.0f}%)</span>
                </div>
                <div style="background:#1e3a5f;border-radius:3px;height:6px;">
                    <div style="width:{pct}%;background:#38bdf8;height:100%;border-radius:3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_tbl2:
        st.dataframe(geo_view.head(50), use_container_width=True, height=280)

    st.markdown('<div class="section-header">📈 Geological Distributions</div>', unsafe_allow_html=True)
    geo_chart_cols = st.columns(3)
    for i, col_name in enumerate(["dist_to_mine_km","fault_proximity","elevation_m"]):
        with geo_chart_cols[i]:
            st.plotly_chart(dark_hist(sat_df, col_name, col_name),
                            use_container_width=True, key=f"geo_hist_{col_name}")

    st.download_button(
        "⬇️ Download Geological Dataset (CSV)", geo_view.to_csv(index=False), "geological_data.csv", "text/csv"
    )

# ────────────────────────────────────────────────────────────────────
# TAB 3 — Weather
# ────────────────────────────────────────────────────────────────────
with tab_wx:
    st.markdown("""
    <div class="info-box">
    🌦️ <strong>Source:</strong> ERA5 Reanalysis (Copernicus Climate Data Store — cds.climate.copernicus.eu)<br>
    <strong>Variables:</strong> Daily rainfall, max/min temperature, humidity, soil moisture<br>
    <strong>Status:</strong> Prototype — synthetic proxy with realistic monsoon seasonality.
    In production, replace with ERA5 API data for the mine region.
    </div>
    """, unsafe_allow_html=True)

    col_wstat, col_wtbl = st.columns([1, 2])
    with col_wstat:
        st.markdown('<div class="section-header">📊 Weather Statistics</div>', unsafe_allow_html=True)
        for lab, val in [
            ("Date Range", f"{wx_df['date'].min().date()} – {wx_df['date'].max().date()}"),
            ("Total Records", len(wx_df)),
            ("Avg Rainfall (mm)", f"{wx_df['rainfall_mm'].mean():.1f}"),
            ("Max Rainfall (mm)", f"{wx_df['rainfall_mm'].max():.1f}"),
            ("Avg Max Temp (°C)", f"{wx_df['temp_max_c'].mean():.1f}"),
            ("Avg Humidity (%)", f"{wx_df['humidity_pct'].mean():.1f}"),
            ("Avg Soil Moisture", f"{wx_df['soil_moisture'].mean():.3f}"),
        ]:
            st.markdown(f"""
            <div class="stat-card">
                <span style="color:#94a3b8;font-size:.82rem;">{lab}</span>
                <span style="color:#38bdf8;font-weight:600;font-size:.88rem;">{val}</span>
            </div>""", unsafe_allow_html=True)

    with col_wtbl:
        st.dataframe(wx_df.head(60), use_container_width=True, height=280)

    st.markdown('<div class="section-header">📈 Weather Distributions</div>', unsafe_allow_html=True)
    wx_chart_cols = st.columns(3)
    for i, col_name in enumerate(["rainfall_mm","temp_max_c","humidity_pct"]):
        with wx_chart_cols[i]:
            st.plotly_chart(dark_hist(wx_df, col_name, col_name),
                            use_container_width=True, key=f"wx_hist_{col_name}")

    # Rainfall time series
    st.markdown('<div class="section-header">📈 Monthly Rainfall Time Series</div>', unsafe_allow_html=True)
    wx_monthly = wx_df.copy()
    wx_monthly["month"] = wx_monthly["date"].dt.to_period("M").astype(str)
    monthly_rain = wx_monthly.groupby("month")["rainfall_mm"].sum().reset_index()

    fig_rain = go.Figure()
    fig_rain.add_trace(go.Bar(
        x=monthly_rain["month"], y=monthly_rain["rainfall_mm"],
        marker_color="#38bdf8", opacity=0.85, name="Monthly Rainfall",
    ))
    fig_rain.update_layout(
        paper_bgcolor="#0d1b2a", plot_bgcolor="#1a2744",
        font=dict(color="#e2e8f0", family="Inter"),
        xaxis=dict(gridcolor="#1e3a5f", tickangle=-45),
        yaxis=dict(gridcolor="#1e3a5f", title="Rainfall (mm)"),
        margin=dict(l=15, r=15, t=30, b=80), height=300,
    )
    st.plotly_chart(fig_rain, use_container_width=True, key="wx_monthly_rain")

    st.download_button(
        "⬇️ Download Weather Dataset (CSV)", wx_df.to_csv(index=False), "weather_era5.csv", "text/csv"
    )

# ────────────────────────────────────────────────────────────────────
# TAB 4 — Operational
# ────────────────────────────────────────────────────────────────────
with tab_ops:
    st.markdown("""
    <div class="warning-box">
    ⚠️ <strong>Prototype operational dataset — synthetic due to non-public operational records.</strong><br>
    This dataset is entirely synthetic and does NOT represent actual MOIL Ltd. operational records.
    Variables are based on realistic open-pit/underground manganese mining operations.
    This interface is designed so that actual MOIL data can be substituted by replacing the
    <code>generate_operational_data()</code> function in <code>src/data_loader.py</code>.
    </div>
    """, unsafe_allow_html=True)

    col_ostat, col_otbl = st.columns([1, 2])
    with col_ostat:
        st.markdown('<div class="section-header">📊 Operational Statistics</div>', unsafe_allow_html=True)
        for lab, val in [
            ("Date Range", f"{ops_df['date'].min().date()} – {ops_df['date'].max().date()}"),
            ("Total Records", f"{len(ops_df):,}"),
            ("Mines", ops_df["mine"].nunique()),
            ("Avg Planned (t/day)", f"{ops_df['planned_tpd'].mean():.0f}"),
            ("Avg Actual (t/day)", f"{ops_df['actual_tpd'].mean():.0f}"),
            ("Avg Shortfall %", f"{ops_df['shortfall_pct'].mean():.1f}%"),
            ("Avg Downtime (hrs)", f"{ops_df['equipment_downtime_hrs'].mean():.1f}"),
            ("Avg Ore Grade (%Mn)", f"{ops_df['ore_grade_pct'].mean():.1f}"),
        ]:
            st.markdown(f"""
            <div class="stat-card">
                <span style="color:#94a3b8;font-size:.82rem;">{lab}</span>
                <span style="color:#38bdf8;font-weight:600;font-size:.88rem;">{val}</span>
            </div>""", unsafe_allow_html=True)

    with col_otbl:
        # Filter controls
        sel_mine_exp = st.selectbox("Filter by Mine", ["All"] + list(MOIL_MINES.keys()), key="exp_mine")
        view_ops = ops_df if sel_mine_exp == "All" else ops_df[ops_df["mine"] == sel_mine_exp]
        st.dataframe(view_ops.head(80), use_container_width=True, height=260)

    st.markdown('<div class="section-header">📈 Operational Distributions</div>', unsafe_allow_html=True)
    ops_chart_cols = st.columns(4)
    for i, col_name in enumerate(["actual_tpd","equipment_downtime_hrs","blasting_delay_hrs","rainfall_mm"]):
        with ops_chart_cols[i]:
            st.plotly_chart(dark_hist(view_ops, col_name, col_name),
                            use_container_width=True, key=f"ops_hist_{col_name}")

    # Per-mine production summary
    st.markdown('<div class="section-header">📊 Per-Mine Production Summary</div>', unsafe_allow_html=True)
    mine_agg = ops_df.groupby("mine").agg(
        planned_avg=("planned_tpd","mean"),
        actual_avg=("actual_tpd","mean"),
        shortfall_avg=("shortfall_pct","mean"),
        downtime_avg=("equipment_downtime_hrs","mean"),
        rain_avg=("rainfall_mm","mean"),
    ).round(2).reset_index()
    mine_agg.columns = ["Mine","Avg Planned (t)","Avg Actual (t)","Avg Shortfall %","Avg Downtime (h)","Avg Rainfall (mm)"]
    st.dataframe(mine_agg, use_container_width=True)

    st.download_button(
        "⬇️ Download Operational Dataset (CSV)",
        view_ops.to_csv(index=False), "operational_prototype.csv", "text/csv",
    )

st.markdown("---")
st.caption(
    "All datasets are either publicly derived prototype data or clearly labelled synthetic prototypes. "
    "No confidential or official MOIL data is included."
)
