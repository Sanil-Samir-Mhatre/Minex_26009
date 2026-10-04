"""
Shared visualization helpers — Plotly charts used across pages.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "primary":    "#38bdf8",
    "secondary":  "#818cf8",
    "accent":     "#fb923c",
    "success":    "#4ade80",
    "warning":    "#facc15",
    "danger":     "#f87171",
    "bg":         "#0d1b2a",
    "surface":    "#1a2744",
    "border":     "#1e3a5f",
    "text":       "#e2e8f0",
    "subtext":    "#94a3b8",
}

RISK_COLOURS = {
    "LOW":      "#4ade80",
    "MEDIUM":   "#facc15",
    "HIGH":     "#fb923c",
    "CRITICAL": "#f87171",
}

PROSP_COLOURS = {
    "HIGH":   "#38bdf8",
    "MEDIUM": "#818cf8",
    "LOW":    "#475569",
}

DARK_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], family="Inter"),
        xaxis=dict(
            gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"],
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"],
            showgrid=True,
        ),
        margin=dict(l=20, r=20, t=40, b=20),
    )
)


def _dark_fig(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], family="Inter"),
        title=dict(text=title, font=dict(size=14, color=PALETTE["text"])),
        xaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        yaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PALETTE["subtext"])),
    )
    return fig


# ── Production trend chart ────────────────────────────────────────────────────
def production_trend_chart(
    df: pd.DataFrame,
    mine: str = None,
    n_days: int = 90,
) -> go.Figure:
    if mine:
        df = df[df["mine"] == mine]
    df = df.sort_values("date").tail(n_days)

    fig = _dark_fig("Actual vs Planned Production (tonnes/day)")
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["planned_tpd"],
        name="Planned", line=dict(color=PALETTE["subtext"], width=1.5, dash="dot"),
        fill=None,
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["actual_tpd"],
        name="Actual", line=dict(color=PALETTE["primary"], width=2),
        fill="tonexty", fillcolor="rgba(56,189,248,0.08)",
    ))
    if "predicted_tpd" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["predicted_tpd"],
            name="AI Predicted", line=dict(color=PALETTE["accent"], width=2, dash="dash"),
        ))
    fig.update_layout(height=350, legend=dict(orientation="h", y=-0.2))
    return fig


# ── Shortfall chart ───────────────────────────────────────────────────────────
def shortfall_chart(df: pd.DataFrame, mine: str = None, n_days: int = 90) -> go.Figure:
    if mine:
        df = df[df["mine"] == mine]
    df = df.sort_values("date").tail(n_days)
    colours = [RISK_COLOURS.get(r, "#fff") for r in df.get("risk_label", ["LOW"] * len(df))]

    fig = _dark_fig("Daily Production Shortfall (tonnes/day)")
    fig.add_trace(go.Bar(
        x=df["date"], y=df["shortfall_tpd"],
        marker_color=colours, name="Shortfall",
        hovertemplate="<b>%{x}</b><br>Shortfall: %{y:.0f} t<extra></extra>",
    ))
    fig.update_layout(height=280)
    return fig


# ── Forecast chart ────────────────────────────────────────────────────────────
def forecast_chart(hist_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = _dark_fig("Production Forecast — Historical + AI Prediction")
    hist = hist_df.sort_values("date").tail(30)
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["actual_tpd"],
        name="Historical Actual", line=dict(color=PALETTE["primary"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["planned_tpd"],
        name="Planned", line=dict(color=PALETTE["subtext"], width=1, dash="dot"),
    ))
    # Forecast
    fdf = forecast_df.sort_values("date")
    # Confidence band
    upper = fdf["predicted_tpd"] * 1.08
    lower = fdf["predicted_tpd"] * 0.92

    fig.add_trace(go.Scatter(
        x=pd.concat([fdf["date"], fdf["date"][::-1]]),
        y=pd.concat([upper, lower[::-1]]),
        fill="toself", fillcolor="rgba(251,146,60,0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=True, name="Confidence Band",
    ))
    fig.add_trace(go.Scatter(
        x=fdf["date"], y=fdf["predicted_tpd"],
        name="AI Forecast", line=dict(color=PALETTE["accent"], width=2, dash="dash"),
        hovertemplate="<b>%{x}</b><br>Forecast: %{y:.0f} t<extra></extra>",
    ))
    # Vertical divider
    if not fdf.empty:
        fig.add_vline(
            x=fdf["date"].iloc[0].timestamp() * 1000,
            line_dash="dot", line_color=PALETTE["border"], annotation_text="Forecast →",
            annotation_font_color=PALETTE["subtext"],
        )
    fig.update_layout(height=380, legend=dict(orientation="h", y=-0.25))
    return fig


# ── Feature importance chart ──────────────────────────────────────────────────
def feature_importance_chart(feat_imp: pd.DataFrame, title: str = "Feature Importance") -> go.Figure:
    fi = feat_imp.head(12).sort_values("importance")
    fig = _dark_fig(title)
    fig.add_trace(go.Bar(
        x=fi["importance"], y=fi["feature"], orientation="h",
        marker=dict(
            color=fi["importance"],
            colorscale=[[0, PALETTE["secondary"]], [0.5, PALETTE["primary"]], [1, PALETTE["accent"]]],
        ),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(height=380, yaxis=dict(tickfont=dict(size=11)))
    return fig


# ── Prospectivity histogram ───────────────────────────────────────────────────
def prospectivity_histogram(df: pd.DataFrame) -> go.Figure:
    fig = _dark_fig("Prospectivity Score Distribution")
    for cls, col in PROSP_COLOURS.items():
        sub = df[df["prospectivity_class"] == cls]
        if not sub.empty:
            fig.add_trace(go.Histogram(
                x=sub["prospectivity_score"], name=cls,
                marker_color=col, opacity=0.75, nbinsx=25,
            ))
    fig.update_layout(barmode="overlay", height=300,
                      xaxis_title="Prospectivity Score", yaxis_title="Count")
    return fig


# ── Risk donut ────────────────────────────────────────────────────────────────
def risk_donut(pred_df: pd.DataFrame) -> go.Figure:
    if "risk_label" not in pred_df.columns:
        return go.Figure()
    counts = pred_df["risk_label"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colours = [RISK_COLOURS.get(l, "#fff") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.6, marker=dict(colors=colours),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} records<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        font=dict(color=PALETTE["text"]),
        annotations=[dict(text="Risk<br>Mix", x=0.5, y=0.5, showarrow=False,
                          font=dict(size=14, color=PALETTE["subtext"]))],
    )
    return fig


# ── SHAP bar chart ────────────────────────────────────────────────────────────
def shap_bar_chart(shap_vals, sample_df, feature_names: list, title: str = "SHAP Mean |Value|") -> go.Figure:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)
    top_n = min(12, len(order))
    idx = order[-top_n:]

    fig = _dark_fig(title)
    fig.add_trace(go.Bar(
        x=mean_abs[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        marker=dict(
            color=mean_abs[idx],
            colorscale=[[0, PALETTE["secondary"]], [1, PALETTE["accent"]]],
        ),
    ))
    fig.update_layout(height=350, xaxis_title="Mean |SHAP Value|")
    return fig


# ── Prospectivity class pie ───────────────────────────────────────────────────
def prospectivity_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["prospectivity_class"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colours = [PROSP_COLOURS.get(l, "#fff") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55, marker=dict(colors=colours),
        textinfo="label+percent",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        height=280, margin=dict(l=10, r=10, t=20, b=10),
        showlegend=True,
        font=dict(color=PALETTE["text"]),
        annotations=[dict(text="Class<br>Mix", x=0.5, y=0.5, showarrow=False,
                          font=dict(size=12, color=PALETTE["subtext"]))],
    )
    return fig


# ── Mine production bar ───────────────────────────────────────────────────────
def mine_production_bar(ops_df: pd.DataFrame) -> go.Figure:
    recent = ops_df[ops_df["date"] >= ops_df["date"].max() - pd.Timedelta(days=30)]
    agg = recent.groupby("mine")[["planned_tpd", "actual_tpd"]].mean().reset_index()

    fig = _dark_fig("Average Daily Production by Mine (Last 30 Days)")
    fig.add_trace(go.Bar(name="Planned", x=agg["mine"], y=agg["planned_tpd"],
                         marker_color=PALETTE["subtext"]))
    fig.add_trace(go.Bar(name="Actual",  x=agg["mine"], y=agg["actual_tpd"],
                         marker_color=PALETTE["primary"]))
    fig.update_layout(barmode="group", height=340,
                      xaxis_tickangle=-30, legend=dict(orientation="h", y=-0.35))
    return fig


# ── Rainfall vs Shortfall scatter ─────────────────────────────────────────────
def rainfall_shortfall_scatter(ops_df: pd.DataFrame) -> go.Figure:
    sample = ops_df.sample(min(300, len(ops_df)), random_state=42)
    fig = _dark_fig("Rainfall vs Production Shortfall")
    colours = [RISK_COLOURS.get(r, "#fff") for r in sample.get("risk_label", ["LOW"] * len(sample))]
    fig.add_trace(go.Scatter(
        x=sample["rainfall_mm"], y=sample["shortfall_pct"],
        mode="markers",
        marker=dict(color=colours, size=6, opacity=0.7),
        hovertemplate="Rain: %{x:.1f} mm<br>Shortfall: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(height=300,
                      xaxis_title="Rainfall (mm)", yaxis_title="Shortfall %")
    return fig

# ── Google Maps Platform (gmp-map) ────────────────────────────────────────────
def gmp_map_html(geo_df: pd.DataFrame, api_key: str, height: int = 500) -> str:
    """Generates HTML using the gmp-map Web Component for Google Maps."""
    markers_html = ""
    colour_map = {"HIGH": "#38bdf8", "MEDIUM": "#818cf8", "LOW": "#475569"}
    
    for _, row in geo_df.iterrows():
        col = colour_map.get(row["prospectivity_class"], "#fff")
        title = f"Score: {row['prospectivity_score']:.2f} | Lithology: {row['lithology']}"
        markers_html += f'''
        <gmp-advanced-marker position="{row['latitude']},{row['longitude']}" title="{title}">
          <div class="dot" style="background:{col};"></div>
        </gmp-advanced-marker>
        '''

    html = f'''
    <!DOCTYPE html>
    <html>
      <head>
        <style>
          body, html {{ margin: 0; padding: 0; height: 100%; overflow: hidden; }}
          .dot {{
            width: 10px; height: 10px; border-radius: 50%;
            border: 1px solid rgba(255,255,255,0.4);
            box-shadow: 0 0 5px rgba(0,0,0,0.5);
          }}
        </style>
        <script type="module">
          import {{ Map }} from "https://unpkg.com/@googlemaps/extended-component-library@0.6";
        </script>
      </head>
      <body>
        <gmp-map center="21.3,79.6" zoom="8" map-id="DEMO_MAP_ID" style="height: {height}px; width: 100%;">
          {markers_html}
        </gmp-map>
        <script async src="https://maps.googleapis.com/maps/api/js?key={api_key}&v=alpha&libraries=maps,marker"></script>
      </body>
    </html>
    '''
    return html
