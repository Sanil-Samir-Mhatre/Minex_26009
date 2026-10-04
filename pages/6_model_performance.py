"""
Page 6 — Model Performance: full metrics with honest prototype labelling.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.data_loader import load_satellite_data, load_operational_data
from src.prospectivity_model import (
    train_prospectivity_model, predict_prospectivity, FEATURE_COLS as P_FEATS,
)
from src.production_model import (
    train_production_model, predict_production_bulk, REG_FEATURES,
)
from src.visualization import feature_importance_chart, PALETTE

st.set_page_config(page_title="Model Performance — MOIL AI", page_icon="🏅", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0a0e1a;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a 0%,#1a2744 100%);border-right:1px solid #1e3a5f;}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
.metric-card{background:linear-gradient(135deg,#0d1b2a,#1a2744);border:1px solid #1e3a5f;border-radius:12px;padding:1.1rem;text-align:center;}
.metric-val{font-size:1.8rem;font-weight:700;line-height:1.2;}
.metric-lab{color:#94a3b8;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;margin-top:.25rem;}
.caution-box{background:rgba(250,204,21,.08);border:1px solid rgba(250,204,21,.25);border-left:3px solid #facc15;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#94a3b8;margin:.75rem 0;}
.info-box{background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.25);border-left:3px solid #38bdf8;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#94a3b8;margin:.75rem 0;}
.section-header{font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:.75rem;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}[data-testid="stToolbar"]{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent !important;visibility:visible !important;display:block !important;z-index:99999 !important;}
[data-testid="collapsedControl"],[data-testid="collapsedControl"] button,[data-testid="stSidebarCollapseButton"],header button{visibility:visible !important;display:inline-flex !important;opacity:1 !important;color:#38bdf8 !important;}
[data-testid="collapsedControl"]{background:rgba(13,27,42,.95) !important;border:1px solid #1e3a5f !important;border-radius:8px !important;box-shadow:0 4px 12px rgba(0,0,0,.5) !important;}
[data-testid="collapsedControl"]:hover{border-color:#38bdf8 !important;box-shadow:0 0 10px rgba(56,189,248,.5) !important;}
</style>
""", unsafe_allow_html=True)


# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1rem;">
    <h1 style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:700;
    background:linear-gradient(135deg,#4ade80,#38bdf8);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;margin:0;">
    🏅 Model Performance Metrics
    </h1>
    <p style="color:#94a3b8;margin:.2rem 0 0;">
    Evaluation metrics for both prospectivity and production ML models
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="caution-box">
⚠️ <strong>Prototype metric on synthetic/limited data — not a field-validated performance measure.</strong><br>
All metrics below are computed on prototype/synthetic datasets. They demonstrate the methodology's 
capability but cannot be taken as real-world accuracy until trained on actual GSI occurrence data 
and official MOIL operational records.
</div>
""", unsafe_allow_html=True)

# ── Load & train ───────────────────────────────────────────────────────────────
with st.spinner("Training models…"):
    geo_df = load_satellite_data()
    ops_df = load_operational_data()
    p_model, p_metrics, p_fi, rf_model = train_prospectivity_model(geo_df)
    r_reg, r_clf, r_metrics, r_fi = train_production_model(ops_df)
    geo_pred = predict_prospectivity(p_model, geo_df)

# ── Tabs: Prospectivity | Production ──────────────────────────────────────────
tab_prosp, tab_prod = st.tabs(["🗺️ Prospectivity Model", "📉 Production Model"])

# ─────────────────────────────────────────────────────────────────────────────
#  PROSPECTIVITY
# ─────────────────────────────────────────────────────────────────────────────
with tab_prosp:
    st.markdown("### XGBoost Classifier — Manganese Prospectivity")
    st.markdown("""
    <div class="info-box">
    <strong>Task:</strong> Binary classification — known Mn occurrence (1) vs background (0)<br>
    <strong>Features:</strong> Sentinel-2 indices, geological attributes, geospatial features (16 total)<br>
    <strong>Train/Test Split:</strong> 75% / 25% | <strong>CV:</strong> 5-fold stratified
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    metrics_disp = [
        (f"{p_metrics['accuracy']:.3f}", "Accuracy", "#38bdf8"),
        (f"{p_metrics['precision']:.3f}", "Precision", "#818cf8"),
        (f"{p_metrics['recall']:.3f}", "Recall", "#fb923c"),
        (f"{p_metrics['f1']:.3f}", "F1 Score", "#4ade80"),
        (f"{p_metrics['roc_auc']:.3f}", "ROC-AUC", "#facc15"),
    ]
    for col, (val, lab, col_hex) in zip([k1, k2, k3, k4, k5], metrics_disp):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:{col_hex};">{val}</div>
                <div class="metric-lab">{lab}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box" style="margin-top:1rem;">
    <strong>Cross-Validation (5-fold) ROC-AUC:</strong> 
    {p_metrics['cv_auc_mean']:.4f} ± {p_metrics['cv_auc_std']:.4f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_roc, col_cm, col_fi_p = st.columns(3, gap="medium")

    with col_roc:
        st.markdown('<div class="section-header">📈 ROC Curve</div>', unsafe_allow_html=True)
        # Compute ROC
        X = geo_df[P_FEATS]
        y = geo_df["known_occurrence"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        proba = p_model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        roc_auc_val = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line=dict(color="#38bdf8", width=2),
            name=f"AUC = {roc_auc_val:.3f}",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(color="#475569", width=1, dash="dash"),
            name="Random",
        ))
        fig_roc.update_layout(
            paper_bgcolor="#0d1b2a", plot_bgcolor="#1a2744",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(title="FPR", gridcolor="#1e3a5f"),
            yaxis=dict(title="TPR", gridcolor="#1e3a5f"),
            margin=dict(l=15, r=15, t=30, b=15), height=280,
        )
        st.plotly_chart(fig_roc, use_container_width=True, key="roc_prosp")

    with col_cm:
        st.markdown('<div class="section-header">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
        y_pred = p_model.predict(X_te)
        cm = confusion_matrix(y_te, y_pred)
        labels_cm = ["Non-Occurrence", "Occurrence"]

        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels_cm, y=labels_cm,
            colorscale=[[0,"#0d1b2a"],[0.5,"#1a3a6f"],[1,"#38bdf8"]],
            text=cm, texttemplate="%{text}", showscale=False,
        ))
        fig_cm.update_layout(
            paper_bgcolor="#0d1b2a", plot_bgcolor="#1a2744",
            font=dict(color="#e2e8f0"), height=280,
            xaxis=dict(title="Predicted"), yaxis=dict(title="Actual"),
            margin=dict(l=15, r=15, t=30, b=15),
        )
        st.plotly_chart(fig_cm, use_container_width=True, key="cm_prosp")

    with col_fi_p:
        st.markdown('<div class="section-header">📊 Feature Importance</div>', unsafe_allow_html=True)
        st.plotly_chart(
            feature_importance_chart(p_fi.head(10), "Top 10 Features"),
            use_container_width=True, key="fi_prosp_perf",
        )

# ─────────────────────────────────────────────────────────────────────────────
#  PRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
with tab_prod:
    st.markdown("### XGBoost Regressor + Classifier — Production Shortfall")

    sub_tab_reg, sub_tab_clf = st.tabs(["📉 Regressor (Production Prediction)", "🚦 Classifier (Risk Class)"])

    with sub_tab_reg:
        st.markdown("""
        <div class="info-box">
        <strong>Task:</strong> Predict actual production (t/day) from operational features<br>
        <strong>Features:</strong> Planned, equipment metrics, rainfall, grade, maintenance (9 total)<br>
        <strong>Train/Test:</strong> 80/20 split
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        for col, (val, lab, col_hex) in zip(
            [k1, k2, k3],
            [
                (f"{r_metrics['reg_mae']:.1f} t/day", "MAE", "#38bdf8"),
                (f"{r_metrics['reg_rmse']:.1f} t/day", "RMSE", "#818cf8"),
                (f"{r_metrics['reg_r2']:.4f}", "R² Score", "#4ade80"),
            ]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color:{col_hex};">{val}</div>
                    <div class="metric-lab">{lab}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_scatter, col_fi_r = st.columns(2, gap="medium")

        with col_scatter:
            st.markdown('<div class="section-header">📈 Predicted vs Actual</div>', unsafe_allow_html=True)
            ops_pred_df = predict_production_bulk(r_reg, r_clf, ops_df)
            sample_pred = ops_pred_df.sample(min(500, len(ops_pred_df)), random_state=42)

            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=sample_pred["actual_tpd"], y=sample_pred["predicted_tpd"],
                mode="markers", marker=dict(color="#38bdf8", size=4, opacity=0.5),
                name="Predictions",
            ))
            mn = min(sample_pred["actual_tpd"].min(), sample_pred["predicted_tpd"].min())
            mx = max(sample_pred["actual_tpd"].max(), sample_pred["predicted_tpd"].max())
            fig_scatter.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx], mode="lines",
                line=dict(color="#f87171", dash="dash"), name="Perfect Fit",
            ))
            fig_scatter.update_layout(
                paper_bgcolor="#0d1b2a", plot_bgcolor="#1a2744",
                font=dict(color="#e2e8f0"),
                xaxis=dict(title="Actual (t/day)", gridcolor="#1e3a5f"),
                yaxis=dict(title="Predicted (t/day)", gridcolor="#1e3a5f"),
                margin=dict(l=15, r=15, t=30, b=15), height=320,
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_reg")

        with col_fi_r:
            st.markdown('<div class="section-header">📊 Feature Importance</div>', unsafe_allow_html=True)
            st.plotly_chart(
                feature_importance_chart(r_fi.head(9), "Production Regressor Features"),
                use_container_width=True, key="fi_reg_perf",
            )

        # Residual plot
        st.markdown('<div class="section-header">📉 Residuals Distribution</div>', unsafe_allow_html=True)
        residuals = sample_pred["actual_tpd"] - sample_pred["predicted_tpd"]
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=residuals, nbinsx=50, marker_color="#818cf8", opacity=0.8, name="Residuals",
        ))
        fig_res.add_vline(x=0, line_dash="dash", line_color="#f87171")
        fig_res.update_layout(
            paper_bgcolor="#0d1b2a", plot_bgcolor="#1a2744",
            font=dict(color="#e2e8f0"),
            xaxis=dict(title="Residual (Actual - Predicted)", gridcolor="#1e3a5f"),
            yaxis=dict(gridcolor="#1e3a5f"),
            margin=dict(l=15, r=15, t=30, b=15), height=250,
        )
        st.plotly_chart(fig_res, use_container_width=True, key="residuals_plot")

    with sub_tab_clf:
        st.markdown("""
        <div class="info-box">
        <strong>Task:</strong> Classify shortfall risk → LOW / MEDIUM / HIGH / CRITICAL<br>
        <strong>Thresholds:</strong> &lt;5% = LOW, 5–15% = MEDIUM, 15–25% = HIGH, &gt;25% = CRITICAL<br>
        <strong>Metric:</strong> Weighted (due to class imbalance)
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        for col, (val, lab, col_hex) in zip(
            [k1, k2, k3, k4],
            [
                (f"{r_metrics['clf_acc']:.3f}",  "Accuracy", "#38bdf8"),
                (f"{r_metrics['clf_prec']:.3f}", "Precision (wtd)", "#818cf8"),
                (f"{r_metrics['clf_rec']:.3f}",  "Recall (wtd)", "#fb923c"),
                (f"{r_metrics['clf_f1']:.3f}",   "F1 (wtd)", "#4ade80"),
            ]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color:{col_hex};">{val}</div>
                    <div class="metric-lab">{lab}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Class distribution
        ops_pred_df = predict_production_bulk(r_reg, r_clf, ops_df)
        clf_dist = ops_pred_df["risk_label"].value_counts()
        st.markdown('<div class="section-header">📊 Predicted Risk Class Distribution</div>',
                    unsafe_allow_html=True)
        colours_order = {"LOW":"#4ade80","MEDIUM":"#facc15","HIGH":"#fb923c","CRITICAL":"#f87171"}
        fig_clf_bar = go.Figure(go.Bar(
            x=list(clf_dist.index),
            y=list(clf_dist.values),
            marker_color=[colours_order.get(k,"#fff") for k in clf_dist.index],
        ))
        fig_clf_bar.update_layout(
            paper_bgcolor="#0d1b2a", plot_bgcolor="#1a2744",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="#1e3a5f"),
            yaxis=dict(gridcolor="#1e3a5f", title="Count"),
            margin=dict(l=15, r=15, t=30, b=15), height=260,
        )
        st.plotly_chart(fig_clf_bar, use_container_width=True, key="clf_bar")

st.markdown("---")
st.caption(
    "⚠️ All metrics are computed on synthetic prototype data. "
    "They illustrate the methodology only — real-world accuracy requires actual GSI and MOIL data."
)
