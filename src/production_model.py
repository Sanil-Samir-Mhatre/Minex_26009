"""
Production shortfall ML model — XGBoost regressor + classifier.
Predicts daily production (tonnes) and classifies shortfall risk.
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    XGB_AVAILABLE = False

REG_FEATURES = [
    "planned_tpd",
    "equipment_avail_pct",
    "equipment_downtime_hrs",
    "blasting_delay_hrs",
    "working_hrs",
    "rainfall_mm",
    "ore_grade_pct",
    "maintenance_event",
    "prev_actual_tpd",
]

RISK_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
RISK_THRESHOLDS = [5, 15, 25]   # shortfall % thresholds

REG_MODEL_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "production_reg_model.pkl")
CLF_MODEL_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "production_clf_model.pkl")


def _risk_class(shortfall_pct: float) -> int:
    if shortfall_pct <= RISK_THRESHOLDS[0]:
        return 0
    elif shortfall_pct <= RISK_THRESHOLDS[1]:
        return 1
    elif shortfall_pct <= RISK_THRESHOLDS[2]:
        return 2
    return 3


def _risk_label(shortfall_pct):
    cls = _risk_class(float(shortfall_pct))
    return RISK_LABELS[cls]


def _build_regressor():
    if XGB_AVAILABLE:
        return xgb.XGBRegressor(
            n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)


def _build_classifier():
    if XGB_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)


@st.cache_resource(show_spinner="Training production model…")
def train_production_model(df: pd.DataFrame):
    """
    Trains regressor (actual_tpd) + classifier (risk class).
    NOTE: Prototype metric on synthetic data — not a field-validated measure.
    """
    df = df.copy()
    df["risk_class"] = df["shortfall_pct"].apply(_risk_class)

    X = df[REG_FEATURES]
    y_reg = df["actual_tpd"]
    y_clf = df["risk_class"]

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    # Regressor
    reg = _build_regressor()
    reg.fit(X_tr, yr_tr)
    yr_pred = reg.predict(X_te)

    mae  = mean_absolute_error(yr_te, yr_pred)
    rmse = np.sqrt(mean_squared_error(yr_te, yr_pred))
    r2   = r2_score(yr_te, yr_pred)

    # Classifier
    clf = _build_classifier()
    clf.fit(X_tr, yc_tr)
    yc_pred = clf.predict(X_te)

    clf_acc  = accuracy_score(yc_te, yc_pred)
    clf_prec = precision_score(yc_te, yc_pred, average="weighted", zero_division=0)
    clf_rec  = recall_score(yc_te, yc_pred, average="weighted", zero_division=0)
    clf_f1   = f1_score(yc_te, yc_pred, average="weighted", zero_division=0)

    metrics = {
        "reg_mae":   round(mae, 2),
        "reg_rmse":  round(rmse, 2),
        "reg_r2":    round(r2, 4),
        "clf_acc":   round(clf_acc, 4),
        "clf_prec":  round(clf_prec, 4),
        "clf_rec":   round(clf_rec, 4),
        "clf_f1":    round(clf_f1, 4),
    }

    # Feature importance
    fi_reg = pd.DataFrame({
        "feature":    REG_FEATURES,
        "importance": reg.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Save
    os.makedirs(os.path.dirname(REG_MODEL_PATH), exist_ok=True)
    joblib.dump(reg, REG_MODEL_PATH)
    joblib.dump(clf, CLF_MODEL_PATH)

    return reg, clf, metrics, fi_reg


def predict_production(reg, clf, input_dict: dict) -> dict:
    """
    Predict for a single row (dict of feature values).
    Returns predicted_tpd, shortfall_tpd, shortfall_pct, risk_label.
    """
    row = pd.DataFrame([{k: input_dict[k] for k in REG_FEATURES}])
    pred_tpd     = float(reg.predict(row)[0])
    planned      = float(input_dict["planned_tpd"])
    shortfall    = max(planned - pred_tpd, 0)
    shortfall_pct = shortfall / planned * 100 if planned > 0 else 0

    risk_cls   = int(clf.predict(row)[0])
    risk_proba = clf.predict_proba(row)[0]

    return {
        "predicted_tpd":  round(pred_tpd, 1),
        "planned_tpd":    round(planned, 1),
        "shortfall_tpd":  round(shortfall, 1),
        "shortfall_pct":  round(shortfall_pct, 2),
        "risk_label":     RISK_LABELS[risk_cls],
        "risk_class":     risk_cls,
        "risk_proba":     risk_proba,
    }


def predict_production_bulk(reg, clf, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict for all rows in df. Returns df with prediction columns appended.
    """
    X = df[REG_FEATURES]
    pred_tpd     = reg.predict(X)
    shortfall    = np.maximum(df["planned_tpd"].values - pred_tpd, 0)
    shortfall_pct = shortfall / df["planned_tpd"].values * 100

    risk_cls   = clf.predict(X)
    risk_labels = [RISK_LABELS[r] for r in risk_cls]

    out = df.copy()
    out["predicted_tpd"]  = np.round(pred_tpd, 1)
    out["shortfall_tpd"]  = np.round(shortfall, 1)
    out["shortfall_pct"]  = np.round(shortfall_pct, 2)
    out["risk_class"]     = risk_cls
    out["risk_label"]     = risk_labels
    return out


def forecast_mine(reg, clf, ops_df: pd.DataFrame, mine: str, n_days: int = 30) -> pd.DataFrame:
    """
    Simple rolling forecast for a specific mine.
    """
    mine_df = ops_df[ops_df["mine"] == mine].sort_values("date").tail(30).copy()
    if mine_df.empty:
        return pd.DataFrame()

    last_row = mine_df.iloc[-1]
    last_date = last_row["date"]

    rows = []
    prev_actual = last_row["actual_tpd"]

    for i in range(1, n_days + 1):
        fut_date = last_date + pd.Timedelta(days=i)
        doy = fut_date.timetuple().tm_yday
        import numpy as np
        rng = np.random.default_rng(seed=i + 1000)
        monsoon_factor = np.exp(-((doy - 240)**2) / (2 * 50**2))

        row_dict = {
            "planned_tpd":           last_row["planned_tpd"] * (1 + rng.uniform(-0.02, 0.02)),
            "equipment_avail_pct":   float(np.clip(last_row["equipment_avail_pct"] + rng.normal(0, 3), 60, 100)),
            "equipment_downtime_hrs":float(np.clip(rng.exponential(2.5) * (1 + monsoon_factor), 0, 10)),
            "blasting_delay_hrs":    float(np.clip(rng.exponential(1.5) * (1 + monsoon_factor), 0, 8)),
            "working_hrs":           float(np.clip(22 - float(np.clip(rng.exponential(2.5) * (1 + monsoon_factor), 0, 10)), 8, 22)),
            "rainfall_mm":           float(np.clip(rng.exponential(4) * (1 + 12 * monsoon_factor), 0, 100)),
            "ore_grade_pct":         float(rng.normal(38, 3)),
            "maintenance_event":     int(rng.random() < 0.05),
            "prev_actual_tpd":       prev_actual,
        }
        result = predict_production(reg, clf, row_dict)
        result["date"] = fut_date
        result["mine"] = mine
        rows.append(result)
        prev_actual = result["predicted_tpd"]

    return pd.DataFrame(rows)


def get_production_shap(reg, df: pd.DataFrame, n_samples: int = 100):
    """Return SHAP values for production regressor."""
    try:
        import shap
        sample = df[REG_FEATURES].sample(min(n_samples, len(df)), random_state=42)
        explainer = shap.TreeExplainer(reg)
        shap_vals = explainer.shap_values(sample)
        return shap_vals, sample, REG_FEATURES
    except Exception:
        return None, None, REG_FEATURES


# ── USP-7: What-if Production Simulator ───────────────────────────────────────
WHATIF_DEFAULTS = {
    "planned_tpd":            1500.0,
    "equipment_avail_pct":    88.0,
    "equipment_downtime_hrs": 3.0,
    "blasting_delay_hrs":     2.0,
    "working_hrs":            18.0,
    "rainfall_mm":            10.0,
    "ore_grade_pct":          38.0,
    "maintenance_event":      0,
    "prev_actual_tpd":        1450.0,
}

def predict_whatif(reg, clf, overrides: dict) -> dict:
    """
    USP-7: What-if simulator.
    Takes a dict of user-specified values (can be partial), fills defaults,
    returns prediction with risk label, SHAP contribution per feature.
    """
    row = {**WHATIF_DEFAULTS, **overrides}
    result = predict_production(reg, clf, row)

    # Compute SHAP for this single row
    try:
        import shap
        X = pd.DataFrame([{k: row[k] for k in REG_FEATURES}])
        explainer = shap.TreeExplainer(reg)
        shap_vals = explainer.shap_values(X)
        if hasattr(shap_vals, "values"):
            shap_vals = shap_vals.values
        sv = shap_vals[0] if not isinstance(shap_vals, list) else shap_vals[0]
        sv = np.ravel(sv)
        result["shap_values"]   = sv.tolist()
        result["shap_features"] = REG_FEATURES
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = ev[0]
        result["shap_base"]     = float(ev)
    except Exception:
        result["shap_values"]   = None
        result["shap_features"] = REG_FEATURES
        result["shap_base"]     = None

    return result

