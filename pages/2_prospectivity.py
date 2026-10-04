"""
Page 2 -- Prospectivity Analysis (upgraded with all USPs):
  USP-2: Ensemble model badges (RF + XGB + NB AUCs)
  USP-3: Uncertainty-aware prospectivity (score +/- confidence)
  USP-4: Top-K exploration targets table
  USP-5: Temporal change monitoring tab
  USP-6: SHAP waterfall for individual zones
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

from src.data_loader import (
    load_satellite_data, load_temporal_satellite_data,
    MOIL_MINES, LITHOLOGY_CLASSES,
)
from src.prospectivity_model import (
    train_prospectivity_model, predict_prospectivity,
    bootstrap_uncertainty, get_top_k_targets,
    get_shap_values, get_shap_single, FEATURE_LABELS,
)
from src.visualization import (
    prospectivity_histogram, feature_importance_chart,
    shap_bar_chart, prospectivity_pie, PALETTE, PROSP_COLOURS,
    gmp_map_html,
)

st.set_page_config(page_title="Prospectivity -- MOIL AI", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0a0e1a;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a 0%,#1a2744 100%);border-right:1px solid #1e3a5f;}
.prosp-high{background:rgba(56,189,248,.15);border:1px solid rgba(56,189,248,.4);color:#38bdf8;
    padding:.25rem .7rem;border-radius:6px;font-size:.8rem;font-weight:700;}
.prosp-medium{background:rgba(129,140,248,.15);border:1px solid rgba(129,140,248,.4);color:#818cf8;
    padding:.25rem .7rem;border-radius:6px;font-size:.8rem;font-weight:700;}
.prosp-low{background:rgba(148,163,184,.15);border:1px solid rgba(148,163,184,.4);color:#94a3b8;
    padding:.25rem .7rem;border-radius:6px;font-size:.8rem;font-weight:700;}
.info-box{background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.25);border-left:3px solid #38bdf8;
    border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#94a3b8;margin:.75rem 0;}
.caution-box{background:rgba(250,204,21,.08);border:1px solid rgba(250,204,21,.25);border-left:3px solid #facc15;
    border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#94a3b8;margin:.75rem 0;}
.section-header{font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:.75rem;}
.metric-card{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;padding:1rem;text-align:center;}
.metric-val{font-size:1.5rem;font-weight:700;color:#38bdf8;}
.metric-lab{color:#94a3b8;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;}
.model-badge{display:inline-block;padding:.2rem .6rem;border-radius:6px;font-size:.72rem;
    font-weight:600;margin:.1rem;background:rgba(74,222,128,.15);color:#4ade80;border:1px solid rgba(74,222,128,.3);}
.unc-bar{background:rgba(56,189,248,.15);border:1px solid rgba(56,189,248,.3);border-radius:8px;
    padding:.5rem .8rem;font-size:.82rem;color:#94a3b8;margin:.2rem 0;}
.target-card{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;
    padding:.7rem 1rem;margin:.3rem 0;transition:border-color .2s;}
.target-card:hover{border-color:#38bdf8;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}[data-testid="stToolbar"]{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent !important;visibility:visible !important;display:block !important;z-index:99999 !important;}
[data-testid="collapsedControl"],[data-testid="collapsedControl"] button,[data-testid="stSidebarCollapseButton"],header button{visibility:visible !important;display:inline-flex !important;opacity:1 !important;color:#38bdf8 !important;}
[data-testid="collapsedControl"]{background:rgba(13,27,42,.95) !important;border:1px solid #1e3a5f !important;border-radius:8px !important;box-shadow:0 4px 12px rgba(0,0,0,.5) !important;}
[data-testid="collapsedControl"]:hover{border-color:#38bdf8 !important;box-shadow:0 0 10px rgba(56,189,248,.5) !important;}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("**Prospectivity Filters**")
    show_classes = st.multiselect("Show classes", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
    min_score    = st.slider("Min Prospectivity Score", 0.0, 1.0, 0.0, 0.05)
    lith_filter  = st.multiselect("Lithology filter", LITHOLOGY_CLASSES, default=[])
    top_k_n      = st.slider("Top-K targets", 5, 20, 10)
    st.markdown("---")
    n_bootstrap  = st.slider("Bootstrap samples (uncertainty)", 20, 60, 40, 10)

# Page header
st.markdown("""
<div style="margin-bottom:1rem;">
    <h1 style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:700;
    background:linear-gradient(135deg,#38bdf8,#818cf8);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;margin:0;">
    🗺️ Manganese Prospectivity Analysis
    </h1>
    <p style="color:#94a3b8;margin:.2rem 0 0;">
    Ensemble AI (RF + XGBoost + Naïve Bayes) with uncertainty quantification
    </p>
</div>
""", unsafe_allow_html=True)

# Caution notice
st.markdown("""
<div class="caution-box">
⚠️ <strong>Scientific Caution:</strong> Prospectivity does not confirm economically recoverable reserves.
Satellite imagery identifies surface indicators associated with manganese-bearing geology.
Field investigation, geochemical sampling, and drilling confirmation are required.
</div>
""", unsafe_allow_html=True)

# Load & train
with st.spinner("Training ensemble prospectivity model..."):
    geo_df  = load_satellite_data()
    model, metrics, feat_imp, rf_model = train_prospectivity_model(geo_df)
    geo_pred = predict_prospectivity(model, geo_df)

# Uncertainty quantification
with st.spinner("Computing bootstrap uncertainty..."):
    geo_unc = bootstrap_uncertainty(rf_model, geo_pred, n_bootstrap=n_bootstrap)

# Filters
filt_df = geo_unc[
    (geo_unc["prospectivity_class"].isin(show_classes)) &
    (geo_unc["prospectivity_score"] >= min_score)
]
if lith_filter:
    filt_df = filt_df[filt_df["lithology"].isin(lith_filter)]

# USP-2: Ensemble model badges
models_used = metrics.get("models_used", ["Random Forest"])
badges_html = " ".join([f'<span class="model-badge">✓ {m}</span>' for m in models_used])
smote_badge = '<span class="model-badge">✓ SMOTE Balanced</span>' if metrics.get("smote_used") else ""
st.markdown(f"""
<div class="info-box">
    🤝 <strong>Ensemble:</strong> {badges_html} {smote_badge} &nbsp;&nbsp;
    <strong>Ensemble AUC:</strong> <span style="color:#4ade80">{metrics['roc_auc']:.4f}</span> &nbsp;|&nbsp;
    <strong>RF:</strong> {metrics.get('rf_auc', 'N/A')} &nbsp;|&nbsp;
    <strong>XGB:</strong> {metrics.get('xgb_auc', 'N/A')} &nbsp;|&nbsp;
    <strong>NB:</strong> {metrics.get('nb_auc', 'N/A')} &nbsp;|&nbsp;
    <strong>CV AUC:</strong> {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}
</div>
""", unsafe_allow_html=True)

# Tabs
tab_map, tab_topk, tab_temporal, tab_shap, tab_stats = st.tabs([
    "🗺️ Prospectivity Map",
    "🎯 Top-K Targets",
    "🕒 Temporal Monitoring",
    "🔍 SHAP Explainability",
    "📊 Statistics",
])

# ─── TAB 1: MAP ───────────────────────────────────────────────────────────────
with tab_map:
    col_map, col_zone = st.columns([3, 2], gap="medium")
    with col_map:
        st.markdown('<div class="section-header">🗺️ Prospectivity Map (Google Maps)</div>', unsafe_allow_html=True)
        api_key  = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
        map_html = gmp_map_html(filt_df, api_key, height=500)
        components.html(map_html, height=500)

        # USP-3: Uncertainty legend
        st.markdown("""
        <div class="unc-bar">
            📊 <strong>Uncertainty (Bootstrap 95% CI):</strong>
            Each point's score is the <em>ensemble soft-vote mean</em>.
            Confidence interval computed from 40 bootstrap sub-forest runs.
            Wider CI = higher prediction uncertainty for that location.
        </div>
        """, unsafe_allow_html=True)

    with col_zone:
        st.markdown('<div class="section-header">📊 Zone Summary</div>', unsafe_allow_html=True)
        fig_pie = prospectivity_pie(filt_df)
        st.plotly_chart(fig_pie, use_container_width=True, key="prosp_pie")

        # Zone stats with uncertainty
        for cls, col in [("HIGH", "#38bdf8"), ("MEDIUM", "#818cf8"), ("LOW", "#94a3b8")]:
            sub = filt_df[filt_df["prospectivity_class"] == cls]
            if not sub.empty:
                mean_unc = sub["uncertainty_pct"].mean()
                st.markdown(f"""
                <div class="metric-card" style="border-color:{col};margin:.4rem 0;">
                    <div style="color:{col};font-weight:600;font-size:.9rem;">{cls}</div>
                    <div style="color:#e2e8f0;font-size:1.2rem;font-weight:700;">{len(sub)} zones</div>
                    <div style="color:#94a3b8;font-size:.75rem;">
                        Avg score: {sub['prospectivity_score'].mean():.3f} &nbsp;|&nbsp;
                        Uncertainty: ±{mean_unc:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Uncertainty scatter
    st.markdown("### 📊 Score vs Uncertainty (USP-3)")
    fig_unc = go.Figure()
    for cls, col in [("HIGH", "#38bdf8"), ("MEDIUM", "#818cf8"), ("LOW", "#475569")]:
        sub = filt_df[filt_df["prospectivity_class"] == cls]
        if not sub.empty:
            fig_unc.add_trace(go.Scatter(
                x=sub["prospectivity_score"],
                y=sub["uncertainty_pct"],
                mode="markers",
                marker=dict(color=col, size=5, opacity=0.6),
                name=cls,
                hovertemplate="Score: %{x:.3f}<br>Uncertainty: ±%{y:.1f}%<extra>" + cls + "</extra>",
            ))
    fig_unc.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e1a",
        height=280, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Prospectivity Score",
        yaxis_title="Uncertainty (±%)",
        legend=dict(orientation="h", y=1.1),
        font=dict(color="#e2e8f0"),
        title=dict(text="Uncertainty-Aware Prospectivity — each point is Score ± Bootstrap 95% CI",
                   font=dict(color="#94a3b8", size=12)),
    )
    st.plotly_chart(fig_unc, use_container_width=True, key="prosp_unc")

# ─── TAB 2: TOP-K TARGETS ─────────────────────────────────────────────────────
with tab_topk:
    st.markdown(f"### 🎯 Top-{top_k_n} Exploration Targets (USP-4)")
    st.markdown("""
    <div class="info-box">
        Zones ranked by prospectivity score. Each target includes uncertainty band, lithology,
        distance to nearest mine, and key spectral indicators.
        <strong>These are the highest-priority zones for field follow-up.</strong>
    </div>
    """, unsafe_allow_html=True)

    top_k = get_top_k_targets(geo_unc, k=top_k_n)

    # Card-style display
    for rank, row in top_k.iterrows():
        cls_col = "#38bdf8" if row["prospectivity_class"] == "HIGH" else \
                  "#818cf8" if row["prospectivity_class"] == "MEDIUM" else "#94a3b8"
        unc_str = f"±{row.get('prosp_std', 0)*100:.1f}%" if "prosp_std" in row else ""
        ci_str  = f"[{row.get('prosp_lower', row['prospectivity_score']):.3f}–{row.get('prosp_upper', row['prospectivity_score']):.3f}]" \
                  if "prosp_lower" in row else ""

        st.markdown(f"""
        <div class="target-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:#e2e8f0;font-weight:700;">
                    #{rank} &nbsp;
                    <span style="color:{cls_col};">{row['prospectivity_class']}</span>
                    &nbsp;Score: <strong style="color:{cls_col};">{row['prospectivity_score']:.4f}</strong>
                    <span style="color:#475569;font-size:.78rem;"> {unc_str} CI{ci_str}</span>
                </span>
                <span style="color:#94a3b8;font-size:.8rem;">
                    📍 {row['latitude']:.3f}°N, {row['longitude']:.3f}°E
                </span>
            </div>
            <div style="color:#94a3b8;font-size:.78rem;margin-top:.3rem;display:flex;gap:1rem;flex-wrap:wrap;">
                <span>🪨 {row.get('lithology','N/A')}</span>
                <span>🔩 Structural Density: {row.get('structural_density', 0):.3f}</span>
                <span>⛏️ Dist to mine: {row.get('dist_to_mine_km', 0):.1f} km</span>
                <span>🔴 Iron Ratio: {row.get('iron_ratio', 0):.2f}</span>
                <span>🌿 BSI: {row.get('bsi', 0):.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Full table:**")
    display_cols = [c for c in ["prospectivity_score", "prospectivity_class", "prosp_std",
                                "latitude", "longitude", "lithology", "structural_density",
                                "dist_to_mine_km", "iron_ratio", "bsi"] if c in top_k.columns]
    st.dataframe(top_k[display_cols].style.format({
        "prospectivity_score": "{:.4f}", "prosp_std": "{:.4f}",
        "latitude": "{:.4f}", "longitude": "{:.4f}",
        "structural_density": "{:.3f}", "iron_ratio": "{:.3f}",
        "dist_to_mine_km": "{:.1f}", "bsi": "{:.3f}",
    }), use_container_width=True)

    csv = top_k.to_csv(index=True)
    st.download_button("📥 Download Top-K Targets (CSV)", data=csv,
                       file_name="top_k_exploration_targets.csv", mime="text/csv")

# ─── TAB 3: TEMPORAL MONITORING ───────────────────────────────────────────────
with tab_temporal:
    st.markdown("### 🕒 Temporal Change Monitoring (USP-5)")
    st.markdown("""
    <div class="info-box">
        Compares satellite spectral indices between two acquisition dates (Dry Season vs Post-Monsoon).
        Changes in NDVI, Bare Soil Index, and Iron Ratio can reveal vegetation/exposure dynamics
        linked to near-surface mineralisation signatures. <em>Prototype: uses synthetic seasonal shift.</em>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading temporal snapshots..."):
        geo_t1, geo_t2, delta = load_temporal_satellite_data()

    sel_idx = st.select_slider(
        "Select spectral change index",
        options=["delta_ndvi", "delta_bsi", "delta_ndwi", "delta_iron_ratio", "delta_clay_ratio"],
        format_func=lambda x: x.replace("delta_", "Δ ").upper(),
    )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**T1 — Dry Season (Jan 2024)**")
        base_col = sel_idx.replace("delta_", "")
        if base_col in geo_t1.columns:
            fig_t1 = px.scatter(
                geo_t1, x="longitude", y="latitude", color=base_col,
                color_continuous_scale="RdYlGn",
                title=f"T1: {base_col.upper()}", height=320,
                template="plotly_dark",
            )
            fig_t1.update_layout(paper_bgcolor="#0a0e1a", margin=dict(l=0, r=0, t=30, b=0),
                                  font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_t1, use_container_width=True, key="t1_map")

    with col_t2:
        st.markdown("**T2 — Post-Monsoon (Oct 2024)**")
        if base_col in geo_t2.columns:
            fig_t2 = px.scatter(
                geo_t2, x="longitude", y="latitude", color=base_col,
                color_continuous_scale="RdYlGn",
                title=f"T2: {base_col.upper()}", height=320,
                template="plotly_dark",
            )
            fig_t2.update_layout(paper_bgcolor="#0a0e1a", margin=dict(l=0, r=0, t=30, b=0),
                                  font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_t2, use_container_width=True, key="t2_map")

    st.markdown("**Δ Change Map (T2 - T1)**")
    if sel_idx in delta.columns:
        fig_delta = px.scatter(
            delta, x="longitude", y="latitude", color=sel_idx,
            color_continuous_scale="RdBu", color_continuous_midpoint=0,
            title=f"Change: {sel_idx.replace('delta_','Δ ').upper()}  (Red=increase, Blue=decrease)",
            height=340, template="plotly_dark",
        )
        fig_delta.update_layout(paper_bgcolor="#0a0e1a", margin=dict(l=0, r=0, t=40, b=0),
                                 font=dict(color="#e2e8f0"))
        st.plotly_chart(fig_delta, use_container_width=True, key="delta_map")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Mean Change", f"{delta[sel_idx].mean():.4f}")
        with col_b:
            st.metric("Max Increase", f"+{delta[sel_idx].max():.4f}")
        with col_c:
            st.metric("Max Decrease", f"{delta[sel_idx].min():.4f}")

# ─── TAB 4: SHAP EXPLAINABILITY ───────────────────────────────────────────────
with tab_shap:
    st.markdown("### 🔍 Explainable AI — SHAP Analysis (USP-6)")

    subtab_global, subtab_local = st.tabs(["🌐 Global Feature Importance", "📍 Single-Zone Explanation"])

    with subtab_global:
        st.markdown("**Global SHAP — Which features drive prospectivity most?**")
        shap_vals, shap_sample, feat_cols = get_shap_values(rf_model, geo_pred, n_samples=150)
        if shap_vals is not None:
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_df = pd.DataFrame({
                "feature": feat_cols,
                "label":   [FEATURE_LABELS.get(f, f) for f in feat_cols],
                "mean_abs_shap": mean_abs_shap,
            }).sort_values("mean_abs_shap", ascending=True)

            fig_shap_global = go.Figure(go.Bar(
                x=shap_df["mean_abs_shap"],
                y=shap_df["label"],
                orientation="h",
                marker=dict(
                    color=shap_df["mean_abs_shap"],
                    colorscale="Blues", showscale=False,
                ),
                text=[f"{v:.4f}" for v in shap_df["mean_abs_shap"]],
                textposition="outside",
            ))
            fig_shap_global.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0e1a",
                height=420, margin=dict(l=10, r=60, t=30, b=10),
                xaxis_title="Mean |SHAP| (impact on prospectivity score)",
                font=dict(color="#e2e8f0", size=12),
                title=dict(text="Feature Importance via SHAP (Random Forest component)",
                           font=dict(color="#94a3b8", size=12)),
            )
            st.plotly_chart(fig_shap_global, use_container_width=True, key="shap_global")

            top_feat = shap_df.tail(3)["label"].tolist()
            st.markdown(f"""
            <div class="info-box">
                🏆 <strong>Top predictors:</strong> {', '.join(reversed(top_feat))}<br>
                Consistent with Zhao et al. (2025) who found <em>structural density and lithology</em>
                as the most influential features for manganese prospectivity.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Install `shap` to see SHAP analysis: `pip install shap`")

    with subtab_local:
        st.markdown("**Single-Zone SHAP Waterfall — Why did the model score this location HIGH/MEDIUM/LOW?**")

        top10 = geo_unc.nlargest(10, "prospectivity_score").reset_index(drop=True)
        zone_options = {
            f"Zone #{i+1} — {row['prospectivity_class']} (Score {row['prospectivity_score']:.3f}) @ {row['latitude']:.3f},{row['longitude']:.3f}": i
            for i, row in top10.iterrows()
        }
        sel_zone_label = st.selectbox("Select a zone to explain:", list(zone_options.keys()))
        sel_idx_local  = zone_options[sel_zone_label]
        sel_row = top10.iloc[[sel_idx_local]]

        ev, sv, labels, fvals = get_shap_single(rf_model, sel_row)
        if sv is not None:
            shap_row = pd.DataFrame({
                "label":      labels,
                "shap_value": sv,
                "feat_value": fvals,
                "abs_shap":   np.abs(sv),
            }).sort_values("abs_shap", ascending=True)

            colours = ["#38bdf8" if v > 0 else "#f87171" for v in shap_row["shap_value"]]

            fig_waterfall = go.Figure(go.Bar(
                x=shap_row["shap_value"],
                y=shap_row["label"],
                orientation="h",
                marker_color=colours,
                text=[f"{v:+.4f}  (val={fv:.3f})" for v, fv in zip(shap_row["shap_value"], shap_row["feat_value"])],
                textposition="outside",
                hovertemplate="%{y}<br>SHAP: %{x:+.4f}<extra></extra>",
            ))
            score = float(sel_row["prospectivity_score"].iloc[0])
            cls   = sel_row["prospectivity_class"].iloc[0]
            cls_col = "#38bdf8" if cls == "HIGH" else "#818cf8" if cls == "MEDIUM" else "#94a3b8"

            fig_waterfall.update_layout(
                template="plotly_dark", paper_bgcolor="#0a0e1a",
                height=420, margin=dict(l=10, r=120, t=50, b=10),
                xaxis_title="SHAP contribution to prospectivity score",
                font=dict(color="#e2e8f0", size=12),
                title=dict(
                    text=f"SHAP Waterfall — Score: <b style='color:{cls_col}'>{score:.4f} ({cls})</b>  |  Baseline: {ev:.4f}",
                    font=dict(color="#94a3b8", size=12),
                ),
                shapes=[{"type": "line", "x0": 0, "x1": 0, "y0": -0.5,
                         "y1": len(shap_row)-0.5, "line": {"color": "#475569", "width": 1}}],
            )
            st.plotly_chart(fig_waterfall, use_container_width=True, key="shap_waterfall")

            pos_feats = shap_row[shap_row["shap_value"] > 0].tail(2)["label"].tolist()
            neg_feats = shap_row[shap_row["shap_value"] < 0].tail(2)["label"].tolist()
            st.markdown(f"""
            <div class="info-box">
                <strong>Interpretation:</strong> For this zone (score {score:.4f}),
                <strong style="color:#38bdf8">{', '.join(reversed(pos_feats)) if pos_feats else 'none'}</strong>
                push the score <em>higher</em> (blue bars),
                while <strong style="color:#f87171">{', '.join(reversed(neg_feats)) if neg_feats else 'none'}</strong>
                pull it <em>lower</em> (red bars).
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Install `shap` to see zone-level explanation: `pip install shap`")

# ─── TAB 5: STATISTICS ────────────────────────────────────────────────────────
with tab_stats:
    st.markdown("### 📊 Model Performance & Zone Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{metrics['roc_auc']:.4f}</div>
            <div class="metric-lab">Ensemble AUC-ROC</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{metrics['cv_auc_mean']:.4f}</div>
            <div class="metric-lab">CV AUC (5-fold)</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{metrics['f1']:.4f}</div>
            <div class="metric-lab">F1 Score</div></div>""", unsafe_allow_html=True)
    with col4:
        smote_txt = "Active" if metrics.get("smote_used") else "Unavailable"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val" style="font-size:1.2rem;">{smote_txt}</div>
            <div class="metric-lab">SMOTE Balancing</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig_hist = prospectivity_histogram(geo_pred)
    st.plotly_chart(fig_hist, use_container_width=True, key="prosp_hist")
    fig_fi = feature_importance_chart(feat_imp)
    st.plotly_chart(fig_fi, use_container_width=True, key="feat_imp_chart")

st.markdown("---")
st.caption("MOIL AI Mining Intelligence Platform | USP-2,3,4,5,6 Active | SIH 2026 #26009")
