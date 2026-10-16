import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve,
)
 
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
 
 
def split_data(X, y, test_size=0.25):
    return train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=y)
 
 
def metrics_row(name, y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n=== {name} ===")
    print(f"Confusion matrix [Legit, Fraud] (rows=actual): {cm.tolist()}")
    return {
        "model": name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
 
 
def best_f1_threshold(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[int(np.argmax(f1))])
 
 
def make_random_forest():
    return RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=3,
                                  class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
 
 
def feature_importance(model, feature_names) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)
    return pd.Series(dtype=float)
 
 
def run_ml_pipeline(X: pd.DataFrame, y: pd.Series, feature_names, rule_score: pd.Series = None):
    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)
    results = []
 
    if rule_score is not None:
        rs_test = rule_score.loc[X_test.index]
        cut = rule_score.quantile(0.95)
        results.append(metrics_row("Rule engine (baseline)", y_test, (rs_test >= cut).astype(int), rs_test))
 
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED).fit(Xtr_s, y_train)
    lr_p = lr.predict_proba(Xte_s)[:, 1]
    lr_thr = best_f1_threshold(y_train, cross_val_predict(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        Xtr_s, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1])
    results.append(metrics_row("Logistic Regression", y_test, (lr_p >= lr_thr).astype(int), lr_p))
 
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(make_random_forest(), X_train, y_train, cv=skf, method="predict_proba")[:, 1]
    rf_thr = best_f1_threshold(y_train, oof)
    cv_f1 = f1_score(y_train, (oof >= rf_thr).astype(int))
    print(f"\nRandom Forest 5-fold CV (train set): F1 = {cv_f1:.4f}, tuned threshold = {rf_thr:.3f}")
    rf = make_random_forest().fit(X_train, y_train)
    rf_p = rf.predict_proba(X_test)[:, 1]
    results.append(metrics_row("Random Forest", y_test, (rf_p >= rf_thr).astype(int), rf_p))
 
    iso = IsolationForest(n_estimators=200, contamination=float(y_train.mean()), random_state=SEED, n_jobs=-1).fit(Xtr_s)
    iso_score = -iso.score_samples(Xte_s)
    results.append(metrics_row("Isolation Forest (unsupervised)", y_test, (iso.predict(Xte_s) == -1).astype(int), iso_score))
 
    print("\nTop 10 Random Forest feature importances (explainability, FR11):")
    print(feature_importance(rf, feature_names).head(10).round(4))
 
    joblib.dump({"model": rf, "threshold": rf_thr, "features": list(feature_names)}, MODEL_DIR / "random_forest_fraud_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "feature_scaler.joblib")
    res = pd.DataFrame(results)
    res.attrs["rf_threshold"] = rf_thr
    res.attrs["cv_f1"] = cv_f1
    return res, rf, scaler, rf_thr
 
 
def score_all_transactions(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, threshold: float) -> pd.DataFrame:
    proba = cross_val_predict(make_random_forest(), X, y,
                              cv=StratifiedKFold(3, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    df = df.copy()
    df["fraud_probability"] = proba
    df["fraud_risk_score"] = (proba * 100).round(2)
    df["risk_category"] = np.where(proba >= threshold, "High", np.where(proba >= threshold / 3, "Medium", "Low"))
    return df
 
 
if __name__ == "__main__":
    from data_preprocessing import get_clean_dataset
    from fraud_labeling import build_labelled_dataset
    from feature_engineering import build_feature_matrix
 
    data = build_labelled_dataset(get_clean_dataset())
    X, y, names, _ = build_feature_matrix(data)
    res, rf, scaler, thr = run_ml_pipeline(X, y, names, rule_score=data["rule_score"])
    print("\nModel comparison:")
    print(res.to_string(index=False))
