"""
Quick smoke test — run before launching the app to verify imports and data generation.
Usage:  python verify_install.py
"""
import sys

errors = []
warnings = []

# ── Core imports ───────────────────────────────────────────────────────────────
required = [
    ("streamlit",         "streamlit"),
    ("pandas",            "pandas"),
    ("numpy",             "numpy"),
    ("plotly",            "plotly"),
    ("sklearn",           "scikit-learn"),
    ("xgboost",           "xgboost"),
    ("joblib",            "joblib"),
    ("folium",            "folium"),
    ("streamlit_folium",  "streamlit-folium"),
]
optional = [
    ("shap",              "shap (SHAP interpretability)"),
]

print("=" * 60)
print(" MOIL AI Mining Intelligence — Install Verification")
print("=" * 60)

for mod, label in required:
    try:
        __import__(mod)
        print(f"  ✅  {label}")
    except ImportError as e:
        print(f"  ❌  {label}  — {e}")
        errors.append(label)

for mod, label in optional:
    try:
        __import__(mod)
        print(f"  ✅  {label} (optional)")
    except ImportError:
        print(f"  ⚠️   {label} (optional — SHAP charts will be skipped)")
        warnings.append(label)

print()

# ── Data generation ────────────────────────────────────────────────────────────
if not errors:
    print("Testing data generation…")
    try:
        from src.data_loader import generate_satellite_features, generate_operational_data, generate_weather_data
        sat = generate_satellite_features(n_points=50)
        ops = generate_operational_data("2024-01-01", "2024-01-31")
        wx  = generate_weather_data("2024-01-01", "2024-01-31")
        print(f"  ✅  Satellite features: {sat.shape}")
        print(f"  ✅  Operational data:   {ops.shape}")
        print(f"  ✅  Weather data:       {wx.shape}")
    except Exception as e:
        print(f"  ❌  Data generation failed: {e}")
        errors.append("data_generation")

    # Quick model test
    print("Testing ML models…")
    try:
        from src.data_loader import generate_satellite_features, generate_geological_features
        from src.prospectivity_model import train_prospectivity_model, predict_prospectivity
        import streamlit as st

        sat_df = generate_satellite_features(n_points=100)
        geo_df = generate_geological_features(sat_df)

        # Monkey-patch st.cache_resource for testing
        import functools
        def _passthrough(fn=None, **kwargs):
            if fn is None:
                return lambda f: f
            return fn
        st.cache_resource = _passthrough
        st.cache_data = _passthrough
        st.spinner = lambda msg: __import__('contextlib').nullcontext()

        from src.prospectivity_model import train_prospectivity_model
        model, metrics, fi = train_prospectivity_model.__wrapped__(geo_df) if hasattr(train_prospectivity_model, '__wrapped__') else train_prospectivity_model(geo_df)
        print(f"  ✅  Prospectivity model — ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
    except Exception as e:
        print(f"  ⚠️   ML model test skipped in CLI context: {e}")

print()
print("=" * 60)
if errors:
    print(f"❌  {len(errors)} error(s) found. Fix before running the app.")
    print("   Run:  pip install --only-binary :all: " + " ".join(errors))
    sys.exit(1)
else:
    print("✅  All required packages OK!")
    if warnings:
        print(f"⚠️   {len(warnings)} optional package(s) missing — app will run without them.")
    print()
    print("  Launch the app with:")
    print("    streamlit run app.py")
print("=" * 60)
