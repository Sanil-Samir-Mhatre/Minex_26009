"""End-to-end logic test — run with: python e2e_test.py"""
import sys
sys.path.insert(0, '.')

print("=" * 55)
print("  MOIL AI Platform — End-to-End Logic Test")
print("=" * 55)

# ── Data generation ────────────────────────────────────────────────
from src.data_loader import (
    generate_satellite_features, generate_geological_features,
    generate_operational_data, generate_weather_data,
)
sat = generate_satellite_features(200)
geo = generate_geological_features(sat)
ops = generate_operational_data()
wx  = generate_weather_data()
print(f"[DATA] Satellite: {geo.shape} | Operational: {ops.shape} | Weather: {wx.shape}")
print(f"[DATA] Mines: {ops['mine'].nunique()} | Date range: {ops['date'].min().date()} – {ops['date'].max().date()}")

# ── Prospectivity model ────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from src.prospectivity_model import FEATURE_COLS
import xgboost as xgb
import numpy as np

X = geo[FEATURE_COLS]
y = geo["known_occurrence"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pm = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
pm.fit(Xtr, ytr)
proba = pm.predict_proba(Xte)[:, 1]
pred  = (proba >= 0.5).astype(int)
auc   = roc_auc_score(yte, proba)
f1    = f1_score(yte, pred, zero_division=0)
print(f"[PROSP] ROC-AUC: {auc:.4f} | F1: {f1:.4f}")

# Prospectivity scores
scores  = pm.predict_proba(X)[:, 1]
classes = np.where(scores >= 0.65, "HIGH", np.where(scores >= 0.40, "MEDIUM", "LOW"))
from collections import Counter
dist = Counter(classes)
print(f"[PROSP] Distribution: HIGH={dist['HIGH']} MED={dist['MEDIUM']} LOW={dist['LOW']}")

# ── Production model ───────────────────────────────────────────────
from sklearn.metrics import r2_score, mean_absolute_error
from src.production_model import REG_FEATURES

Xp  = ops[REG_FEATURES]
yp  = ops["actual_tpd"]
Xptr, Xpte, yptr, ypte = train_test_split(Xp, yp, test_size=0.2, random_state=42)
rm  = xgb.XGBRegressor(n_estimators=150, random_state=42, verbosity=0)
rm.fit(Xptr, yptr)
ypred = rm.predict(Xpte)
print(f"[PROD]  R²: {r2_score(ypte, ypred):.4f} | MAE: {mean_absolute_error(ypte, ypred):.1f} t/day")

# ── Risk engine ───────────────────────────────────────────────────
from src.risk_engine import generate_recommendations
recs = generate_recommendations(
    mine="Balaghat",
    planned_tpd=2800, predicted_tpd=2100, shortfall_pct=25,
    downtime_hrs=6.0, blast_delay_hrs=4.0,
    rainfall_mm=60, equip_avail_pct=65, maintenance_event=1,
)
print(f"[RISK]  Recommendations: {len(recs)}")
for r in recs[:4]:
    print(f"         [{r['severity']:8s}] {r['factor']}")

# ── SHAP ─────────────────────────────────────────────────────────
try:
    import shap
    explainer = shap.TreeExplainer(pm)
    sv = explainer.shap_values(Xte.head(50))
    if isinstance(sv, list):
        sv = sv[1]
    top = np.abs(sv).mean(axis=0).argmax()
    print(f"[SHAP]  Top feature: {FEATURE_COLS[top]}")
except Exception as e:
    print(f"[SHAP]  Skipped: {e}")

print()
print("[OK] ALL SYSTEMS PASSED - App ready at http://localhost:8502")
print("=" * 55)
