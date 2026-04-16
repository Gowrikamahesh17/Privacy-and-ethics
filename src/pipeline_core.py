"""
pipeline_core.py — Core pipeline functions for pytest coverage measurement.
Extracted from the Responsible AI Healthcare Notebook (Week 3).
No LLM / ChromaDB / API dependencies — runs standalone.
"""
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

# ── Privacy ──────────────────────────────────────────────────────────────────
def generate_patient_id(row, key_columns):
    """SHA-256 pseudonymisation. Same input always gives same ID."""
    composite = "|".join(
        "" if pd.isna(row[col]) else str(row[col]).lower().strip()
        for col in key_columns if col in row.index
    )
    h = hashlib.sha256(composite.encode()).hexdigest()
    return f"PAT_{h[:12].upper()}"

def scrub_pii(df, pii_cols, key_cols):
    """Pseudonymise key_cols into patient_id column, then drop pii_cols."""
    df = df.copy()
    available = [c for c in key_cols if c in df.columns]
    if available:
        df["patient_id"] = df.apply(lambda r: generate_patient_id(r, available), axis=1)
    for col in pii_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df

# ── Feature engineering ──────────────────────────────────────────────────────
def cap_outliers_iqr(df, columns):
    """Clip values to [Q1-1.5*IQR, Q3+1.5*IQR] for each column."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
    return df

def add_composite_features(df):
    """Add domain-specific medical composite features."""
    df = df.copy()
    if "HAEMOGLOBINS" in df.columns and "HAEMATOCRIT" in df.columns:
        df["HGB_HCT_ratio"] = df["HAEMOGLOBINS"] / (df["HAEMATOCRIT"] + 1e-6)
    if "MCHC" in df.columns:
        df["MCHC_deviation"] = (df["MCHC"] - 34.0).abs()
    if "LEUCOCYTE" in df.columns:
        df["leukocytosis_flag"] = (df["LEUCOCYTE"] > 11.0).astype(int)
    if "THROMBOCYTE" in df.columns:
        df["thrombocytopenia_flag"] = (df["THROMBOCYTE"] < 150).astype(int)
    return df

def drop_low_variance(df, threshold=0.01, protect=None):
    """Drop columns with variance below threshold (except protected cols)."""
    protect = protect or []
    var = df.var(numeric_only=True)
    to_drop = [c for c in var[var < threshold].index if c not in protect]
    return df.drop(columns=to_drop, errors="ignore")

def prepare_features(df, target_col="readmitted", drop_cols=None):
    """Encode categoricals, fill NaN with median. Returns (X, y, feature_names)."""
    drop_cols = drop_cols or ["patient_id", "SOURCE", "source"]
    X = df.drop(columns=[target_col] + [c for c in drop_cols if c in df.columns], errors="ignore")
    y = df[target_col]
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(X.median(numeric_only=True))
    return X, y, X.columns.tolist()

# ── Imbalance ────────────────────────────────────────────────────────────────
def imbalance_ratio(y):
    """Return majority/minority class count ratio."""
    counts = pd.Series(y).value_counts()
    return counts.max() / counts.min()

def apply_smote(X_train, y_train, k_neighbors=5):
    """Apply SMOTE oversampling. Returns (X_balanced, y_balanced)."""
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    return smote.fit_resample(X_train, y_train)

# ── Model helpers ────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train, n_estimators=100):
    """Train a Random Forest with balanced class weights."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=10,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    """Return dict with f1_weighted and roc_auc."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    n_classes = len(np.unique(y_test))
    auc = (roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
           if n_classes > 2 else roc_auc_score(y_test, y_prob[:, 1]))
    return {
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "roc_auc":     round(float(auc), 4),
    }

# ── Robustness helpers ────────────────────────────────────────────────────────
def prediction_flip_rate(model, X, noise_std=0.01, seed=42):
    """Fraction of predictions that change under small Gaussian noise."""
    rng = np.random.default_rng(seed)
    X_noisy = X.copy()
    num_cols = X_noisy.select_dtypes(include=[np.number]).columns
    X_noisy[num_cols] = X_noisy[num_cols] + rng.normal(0, noise_std, (len(X_noisy), len(num_cols)))
    return float((model.predict(X) != model.predict(X_noisy)).mean())

