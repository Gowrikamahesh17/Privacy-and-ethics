"""
test_pipeline.py
================
pytest test suite for pipeline_core.py.
Run with:
    pytest test_pipeline.py --cov=pipeline_core --cov-report=term-missing -v
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pipeline_core as pc


@pytest.fixture(scope="module")
def sample_df():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "first_name":   [f"Name{i}" for i in range(n)],
        "last_name":    [f"Sur{i}"  for i in range(n)],
        "HAEMATOCRIT":  np.random.normal(40, 5, n),
        "HAEMOGLOBINS": np.random.normal(13, 2, n),
        "LEUCOCYTE":    np.clip(np.random.normal(7, 3, n), 1, 25),
        "THROMBOCYTE":  np.random.normal(250, 60, n),
        "MCHC":         np.random.normal(34, 2, n),
        "age":          np.random.randint(20, 90, n),
        "readmitted":   np.random.choice([0, 1, 2], n, p=[0.6, 0.2, 0.2]),
    })


@pytest.fixture(scope="module")
def ml_fixtures():
    X, y = make_classification(n_samples=400, n_features=10, n_classes=2,
                                n_informative=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    y_s  = pd.Series(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_s, test_size=0.2,
                                               random_state=42, stratify=y_s)
    X_bal, y_bal = pc.apply_smote(X_tr, y_tr)
    model = pc.train_random_forest(X_bal, y_bal)
    return model, X_tr, X_te, y_tr, y_te, X_bal, y_bal


# ── Privacy ──────────────────────────────────────────────────────────────────
class TestPrivacy:
    def test_patient_id_format(self, sample_df):
        pid = pc.generate_patient_id(sample_df.iloc[0], ["first_name","last_name"])
        assert pid.startswith("PAT_")
        assert len(pid) == 16

    def test_patient_id_deterministic(self, sample_df):
        p1 = pc.generate_patient_id(sample_df.iloc[0], ["first_name","last_name"])
        p2 = pc.generate_patient_id(sample_df.iloc[0], ["first_name","last_name"])
        assert p1 == p2

    def test_patient_id_unique_across_patients(self, sample_df):
        p1 = pc.generate_patient_id(sample_df.iloc[0], ["first_name","last_name"])
        p2 = pc.generate_patient_id(sample_df.iloc[1], ["first_name","last_name"])
        assert p1 != p2

    def test_scrub_pii_removes_columns(self, sample_df):
        result = pc.scrub_pii(sample_df, ["first_name","last_name"], ["first_name","last_name"])
        assert "first_name" not in result.columns
        assert "last_name"  not in result.columns

    def test_scrub_pii_adds_patient_id(self, sample_df):
        result = pc.scrub_pii(sample_df, ["first_name"], ["first_name","last_name"])
        assert "patient_id" in result.columns

    def test_scrub_pii_no_mutation(self, sample_df):
        cols_before = list(sample_df.columns)
        pc.scrub_pii(sample_df, ["first_name"], ["first_name"])
        assert list(sample_df.columns) == cols_before

    def test_patient_id_not_plain_integer(self, sample_df):
        result = pc.scrub_pii(sample_df, [], ["first_name","last_name"])
        for pid in result["patient_id"].head(10):
            assert not str(pid).isdigit()


# ── Feature engineering ──────────────────────────────────────────────────────
class TestFeatureEngineering:
    def test_cap_outliers_clips_extreme(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "LEUCOCYTE"] = 9999
        result = pc.cap_outliers_iqr(df, ["LEUCOCYTE"])
        assert result["LEUCOCYTE"].max() < 9999

    def test_cap_outliers_no_mutation(self, sample_df):
        original = sample_df["LEUCOCYTE"].max()
        df = sample_df.copy()
        df.loc[0, "LEUCOCYTE"] = 9999
        pc.cap_outliers_iqr(df, ["LEUCOCYTE"])
        assert sample_df["LEUCOCYTE"].max() == original

    def test_cap_outliers_skips_missing_col(self, sample_df):
        result = pc.cap_outliers_iqr(sample_df, ["NONEXISTENT_COL"])
        assert "NONEXISTENT_COL" not in result.columns

    def test_composite_hgb_hct_created(self, sample_df):
        result = pc.add_composite_features(sample_df)
        assert "HGB_HCT_ratio" in result.columns

    def test_composite_leukocytosis_binary(self, sample_df):
        result = pc.add_composite_features(sample_df)
        assert set(result["leukocytosis_flag"].unique()).issubset({0,1})

    def test_composite_thrombocytopenia_binary(self, sample_df):
        result = pc.add_composite_features(sample_df)
        assert set(result["thrombocytopenia_flag"].unique()).issubset({0,1})

    def test_composite_mchc_deviation_non_negative(self, sample_df):
        result = pc.add_composite_features(sample_df)
        assert (result["MCHC_deviation"] >= 0).all()

    def test_drop_low_variance_removes_constant(self, sample_df):
        df = sample_df.copy(); df["const"] = 1.0
        result = pc.drop_low_variance(df, threshold=0.01)
        assert "const" not in result.columns

    def test_drop_low_variance_protects_listed(self, sample_df):
        df = sample_df.copy(); df["const"] = 1.0
        result = pc.drop_low_variance(df, threshold=0.01, protect=["const"])
        assert "const" in result.columns

    def test_prepare_features_no_nan(self, sample_df):
        X, y, _ = pc.prepare_features(sample_df)
        assert X.isna().sum().sum() == 0

    def test_prepare_features_no_inf(self, sample_df):
        X, y, _ = pc.prepare_features(sample_df)
        assert not np.isinf(X.values).any()

    def test_prepare_features_target_not_in_X(self, sample_df):
        X, y, _ = pc.prepare_features(sample_df)
        assert "readmitted" not in X.columns

    def test_prepare_features_names_match_columns(self, sample_df):
        X, y, names = pc.prepare_features(sample_df)
        assert list(names) == list(X.columns)


# ── Imbalance ────────────────────────────────────────────────────────────────
class TestImbalance:
    def test_imbalance_ratio_balanced(self):
        y = pd.Series([0,0,1,1,0,1])
        assert pc.imbalance_ratio(y) == 1.0

    def test_imbalance_ratio_imbalanced(self):
        y = pd.Series([0]*80 + [1]*20)
        assert pc.imbalance_ratio(y) == 4.0

    def test_smote_reduces_imbalance(self):
        X = pd.DataFrame(np.random.randn(200,5))
        y = pd.Series([0]*160 + [1]*40)
        X_bal, y_bal = pc.apply_smote(X, y)
        assert pc.imbalance_ratio(pd.Series(y_bal)) < pc.imbalance_ratio(y)

    def test_smote_preserves_feature_count(self):
        X = pd.DataFrame(np.random.randn(200,7))
        y = pd.Series([0]*160 + [1]*40)
        X_bal, _ = pc.apply_smote(X, y)
        assert X_bal.shape[1] == 7


# ── Model ─────────────────────────────────────────────────────────────────────
class TestModels:
    def test_rf_has_predict(self, ml_fixtures):
        model, *_ = ml_fixtures
        assert hasattr(model, "predict")

    def test_evaluate_returns_required_keys(self, ml_fixtures):
        model, _, X_te, _, y_te, _, _ = ml_fixtures
        m = pc.evaluate_model(model, X_te, y_te)
        assert "f1_weighted" in m and "roc_auc" in m

    def test_f1_in_valid_range(self, ml_fixtures):
        model, _, X_te, _, y_te, _, _ = ml_fixtures
        m = pc.evaluate_model(model, X_te, y_te)
        assert 0.0 <= m["f1_weighted"] <= 1.0

    def test_auc_in_valid_range(self, ml_fixtures):
        model, _, X_te, _, y_te, _, _ = ml_fixtures
        m = pc.evaluate_model(model, X_te, y_te)
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_predict_determinism(self, ml_fixtures):
        model, _, X_te, _, _, _, _ = ml_fixtures
        assert (model.predict(X_te) == model.predict(X_te)).all()

    def test_proba_sums_to_one(self, ml_fixtures):
        model, _, X_te, _, _, _, _ = ml_fixtures
        probs = model.predict_proba(X_te)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_proba_non_negative(self, ml_fixtures):
        model, _, X_te, _, _, _, _ = ml_fixtures
        assert (model.predict_proba(X_te) >= 0).all()


# ── Robustness ────────────────────────────────────────────────────────────────
class TestRobustness:
    def test_flip_rate_low_under_noise(self, ml_fixtures):
        model, _, X_te, _, _, _, _ = ml_fixtures
        rate = pc.prediction_flip_rate(model, X_te, noise_std=0.01)
        assert rate < 0.10, f"Flip rate {rate:.2%} > 10%"

    def test_handles_nan_after_imputation(self, ml_fixtures):
        model, _, X_te, _, _, _, _ = ml_fixtures
        X_miss = X_te.copy(); X_miss.iloc[:, 0] = np.nan
        X_imp  = X_miss.fillna(X_miss.median(numeric_only=True))
        preds  = model.predict(X_imp)
        assert len(preds) == len(X_te)

