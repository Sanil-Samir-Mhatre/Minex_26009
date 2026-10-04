"""
Prospectivity ML model -- upgraded with 9 USPs:
  USP-1: structural_density feature (Zhao et al. 2025 -- most influential predictor for Mn)
  USP-2: Multi-model ensemble (RF + XGBoost + NaiveBayes)
  USP-3: Bootstrap uncertainty quantification
  USP-4: Top-K exploration targeting
  USP-6: SHAP waterfall + force plot support
  SMOTE class balancing for sparse Mn deposit labels
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
import joblib, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

FEATURE_COLS = [
    "ndvi", "ndwi", "bsi", "clay_ratio", "iron_ratio", "swir_ratio",
    "b4_red", "b8_nir", "b11_swir1", "b12_swir2",
    "elevation_m", "slope_deg",
    "lithology_code", "formation_code",
    "dist_to_mine_km", "fault_proximity",
    "structural_density",          # USP-1: structural density (Zhao 2025)
]

FEATURE_LABELS = {
    "ndvi":               "Vegetation Index (NDVI)",
    "ndwi":               "Water Index (NDWI)",
    "bsi":                "Bare Soil Index (BSI)",
    "clay_ratio":         "Clay Mineral Ratio",
    "iron_ratio":         "Iron Oxide Ratio",
    "swir_ratio":         "SWIR Ratio",
    "b4_red":             "Red Band (B4)",
    "b8_nir":             "NIR Band (B8)",
    "b11_swir1":          "SWIR-1 Band (B11)",
    "b12_swir2":          "SWIR-2 Band (B12)",
    "elevation_m":        "Elevation (m)",
    "slope_deg":          "Slope (deg)",
    "lithology_code":     "Lithology",
    "formation_code":     "Formation",
    "dist_to_mine_km":    "Distance to Mine (km)",
    "fault_proximity":    "Fault Proximity",
    "structural_density": "Structural Density",
}

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "prospectivity_model.pkl")


def _build_ensemble():
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42,
        n_jobs=-1, class_weight="balanced",
    )
    nb = GaussianNB()
    estimators = [("rf", rf), ("nb", nb)]
    if XGB_AVAILABLE:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        estimators.append(("xgb", xgb_clf))
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    return ensemble, rf


@st.cache_resource(show_spinner="Training ensemble prospectivity model...")
def train_prospectivity_model(df: pd.DataFrame):
    """Train RF+XGBoost+NB ensemble with SMOTE balancing."""
    X = df[FEATURE_COLS].copy()
    y = df["known_occurrence"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # SMOTE class balancing
    if SMOTE_AVAILABLE:
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_train.sum()) - 1))
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        except Exception:
            X_train_bal, y_train_bal = X_train, y_train
    else:
        X_train_bal, y_train_bal = X_train, y_train

    ensemble, rf_model = _build_ensemble()
    ensemble.fit(X_train_bal, y_train_bal)
    rf_model.fit(X_train_bal, y_train_bal)   # standalone RF for SHAP

    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "precision":   round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":      round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":          round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":     round(roc_auc_score(y_test, y_prob), 4),
        "smote_used":  SMOTE_AVAILABLE,
        "models_used": ["Random Forest", "XGBoost", "Naive Bayes"] if XGB_AVAILABLE
                       else ["Random Forest", "Naive Bayes"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(rf_model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    metrics["cv_auc_mean"] = round(cv_aucs.mean(), 4)
    metrics["cv_auc_std"]  = round(cv_aucs.std(), 4)

    # Individual model AUCs for ensemble transparency
    metrics["rf_auc"] = round(roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]), 4)
    nb_tmp = GaussianNB().fit(X_train_bal, y_train_bal)
    metrics["nb_auc"] = round(roc_auc_score(y_test, nb_tmp.predict_proba(X_test)[:, 1]), 4)
    if XGB_AVAILABLE:
        xgb_tmp = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
        xgb_tmp.fit(X_train_bal, y_train_bal)
        metrics["xgb_auc"] = round(roc_auc_score(y_test, xgb_tmp.predict_proba(X_test)[:, 1]), 4)

    feat_imp = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": rf_model.feature_importances_,
        "label":      [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
    }).sort_values("importance", ascending=False)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"ensemble": ensemble, "rf": rf_model}, MODEL_PATH)
    return ensemble, metrics, feat_imp, rf_model


def predict_prospectivity(model, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    scores  = model.predict_proba(X)[:, 1]
    classes = np.where(scores >= 0.65, "HIGH", np.where(scores >= 0.40, "MEDIUM", "LOW"))
    out = df.copy()
    out["prospectivity_score"] = np.round(scores, 4)
    out["prospectivity_class"] = classes
    return out


def bootstrap_uncertainty(rf_model, df: pd.DataFrame, n_bootstrap: int = 40) -> pd.DataFrame:
    """USP-3: Bootstrap uncertainty via individual tree subsampling."""
    X = df[FEATURE_COLS].values
    rng = np.random.default_rng(42)
    n_trees = len(rf_model.estimators_)
    preds = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_trees, n_trees // 2)
        boot = np.mean(
            [rf_model.estimators_[j].predict_proba(X)[:, 1] for j in idx], axis=0
        )
        preds.append(boot)
    preds = np.array(preds)
    mean_s = preds.mean(axis=0)
    std_s  = preds.std(axis=0)
    out = df.copy()
    out["prosp_mean"]      = np.round(mean_s, 4)
    out["prosp_std"]       = np.round(std_s, 4)
    out["prosp_lower"]     = np.round(np.clip(mean_s - 1.96 * std_s, 0, 1), 4)
    out["prosp_upper"]     = np.round(np.clip(mean_s + 1.96 * std_s, 0, 1), 4)
    out["uncertainty_pct"] = np.round(std_s * 100, 1)
    return out


def get_top_k_targets(geo_pred: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """USP-4: Top-K exploration targets ranked by prospectivity score."""
    want = ["latitude", "longitude", "prospectivity_score", "prospectivity_class",
            "lithology", "formation", "dist_to_mine_km", "fault_proximity",
            "structural_density", "iron_ratio", "clay_ratio", "bsi"]
    if "prosp_mean" in geo_pred.columns:
        want += ["prosp_mean", "prosp_std"]
    avail = [c for c in want if c in geo_pred.columns]
    top_k = geo_pred.nlargest(k, "prospectivity_score")[avail].reset_index(drop=True)
    top_k.index += 1
    return top_k


def get_shap_values(rf_model, df: pd.DataFrame, n_samples: int = 200):
    try:
        import shap
        sample = df[FEATURE_COLS].sample(min(n_samples, len(df)), random_state=42)
        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer.shap_values(sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        return shap_vals, sample, FEATURE_COLS
    except Exception:
        return None, None, FEATURE_COLS


def get_shap_single(rf_model, row_df: pd.DataFrame):
    """USP-6: SHAP for a single location -- waterfall-style data."""
    try:
        import shap
        X = row_df[FEATURE_COLS].copy()
        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer.shap_values(X)
        sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        ev = explainer.expected_value
        ev = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
        labels = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]
        return float(ev), sv, labels, X.iloc[0].values
    except Exception:
        return None, None, [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS], None
