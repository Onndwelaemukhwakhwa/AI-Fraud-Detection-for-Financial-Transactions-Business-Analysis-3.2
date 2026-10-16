import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from src.ml_models import split_data, metrics_row, best_f1_threshold
except ImportError:
    from ml_models import split_data, metrics_row, best_f1_threshold

HAS_TF = False
if os.environ.get("FORCE_SKLEARN_DL") != "1":
    try:
        import tensorflow as tf
        try:
            import keras
        except ImportError:
            from tensorflow import keras  # type: ignore
        tf.random.set_seed(42)
        HAS_TF = True
    except Exception as exc:
        print(f"[deep_learning_model] TensorFlow unavailable ({type(exc).__name__}); "
              "using the scikit-learn neural-network fallback.")

np.random.seed(42)
MODEL_DIR = Path(__file__).resolve().parent.parent / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _tf_ann(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    Xfit, Xval, yfit, yval = train_test_split(Xtr, y_train, test_size=0.2, stratify=y_train, random_state=42)
    w = float((yfit == 0).sum() / max((yfit == 1).sum(), 1)) ** 0.5

    model = keras.Sequential([
        keras.layers.Input(shape=(Xfit.shape[1],)),
        keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"), keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="fraud_ann_classifier")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), keras.metrics.AUC(name="pr_auc", curve="PR")])
    model.fit(Xfit, yfit, validation_data=(Xval, yval), epochs=40, batch_size=256, verbose=0,
              class_weight={0: 1.0, 1: w},
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=6,
                                                       restore_best_weights=True)])
    thr = best_f1_threshold(yval, model.predict(Xval, verbose=0).ravel())
    proba = model.predict(Xte, verbose=0).ravel()
    m = metrics_row("ANN (TensorFlow/Keras)", y_test, (proba >= thr).astype(int), proba)
    model.save(MODEL_DIR / "fraud_ann_classifier.keras")
    return model, scaler, m


def _tf_autoencoder(X, y, q=0.99):
    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler = StandardScaler().fit(X_train[y_train == 0])
    Xn, Xte = scaler.transform(X_train[y_train == 0]), scaler.transform(X_test)
    Xfit, Xval = train_test_split(Xn, test_size=0.1, random_state=42)
    inp = keras.layers.Input(shape=(Xfit.shape[1],))
    h = keras.layers.Dense(16, activation="relu")(inp)
    h = keras.layers.Dense(6, activation="relu")(h)
    h = keras.layers.Dense(16, activation="relu")(h)
    ae = keras.Model(inp, keras.layers.Dense(Xfit.shape[1], activation="linear")(h), name="fraud_autoencoder")
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(Xfit, Xfit, validation_data=(Xval, Xval), epochs=30, batch_size=256, verbose=0,
           callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    err = lambda a: np.mean(np.square(a - ae.predict(a, verbose=0)), axis=1)
    thr = float(np.quantile(err(Xval), q))
    te = err(Xte)
    m = metrics_row("Autoencoder (unsupervised)", y_test, (te > thr).astype(int), te)
    ae.save(MODEL_DIR / "fraud_autoencoder.keras")
    return ae, scaler, m


def _sk_ann(X, y):
    from sklearn.neural_network import MLPClassifier
    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    Xfit, Xval, yfit, yval = train_test_split(Xtr, y_train, test_size=0.2, stratify=y_train, random_state=42)

    reps = max(int(np.sqrt((yfit == 0).sum() / max((yfit == 1).sum(), 1))), 1)
    pos = np.where(yfit.values == 1)[0]
    idx = np.concatenate([np.arange(len(yfit))] + [pos] * (reps - 1))
    model = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                          n_iter_no_change=6, max_iter=200, random_state=42)
    model.fit(Xfit[idx], yfit.values[idx])
    thr = best_f1_threshold(yval, model.predict_proba(Xval)[:, 1])
    proba = model.predict_proba(Xte)[:, 1]
    m = metrics_row("ANN (scikit-learn MLP fallback)", y_test, (proba >= thr).astype(int), proba)
    return model, scaler, m


def _sk_autoencoder(X, y, q=0.99):
    from sklearn.neural_network import MLPRegressor
    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler = StandardScaler().fit(X_train[y_train == 0])
    Xn, Xte = scaler.transform(X_train[y_train == 0]), scaler.transform(X_test)
    Xfit, Xval = train_test_split(Xn, test_size=0.1, random_state=42)
    ae = MLPRegressor(hidden_layer_sizes=(16, 6, 16), early_stopping=True, n_iter_no_change=4,
                      max_iter=100, random_state=42)
    ae.fit(Xfit, Xfit)
    err = lambda a: np.mean(np.square(a - ae.predict(a)), axis=1)
    thr = float(np.quantile(err(Xval), q))
    te = err(Xte)
    m = metrics_row("Autoencoder (scikit-learn fallback)", y_test, (te > thr).astype(int), te)
    return ae, scaler, m


def train_ann(X: pd.DataFrame, y: pd.Series):
    return _tf_ann(X, y) if HAS_TF else _sk_ann(X, y)


def train_autoencoder(X: pd.DataFrame, y: pd.Series, normal_quantile: float = 0.99):
    return _tf_autoencoder(X, y, normal_quantile) if HAS_TF else _sk_autoencoder(X, y, normal_quantile)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_preprocessing import get_clean_dataset
    from src.fraud_labeling import build_labelled_dataset
    from src.feature_engineering import build_feature_matrix

    data = build_labelled_dataset(get_clean_dataset())
    X, y, names, _ = build_feature_matrix(data)
    _, _, ann_m = train_ann(X, y)
    _, _, ae_m = train_autoencoder(X, y)
    print(pd.DataFrame([ann_m, ae_m]).to_string(index=False))
