"""Train and evaluate the crowd-demand regression model."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.preprocess import add_features, make_preprocessor, CATEGORICAL, NUMERIC, TARGET, save_preprocessor, ROOT, MODEL_DIR

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
from sklearn.neural_network import MLPRegressor


def train():
    data_path = ROOT / "dataset" / "transport_data.csv"
    if not data_path.exists():
        from src.data.generate_dataset import generate
        generate()
    df = add_features(pd.read_csv(data_path))
    X, y = df[CATEGORICAL + NUMERIC], df[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = make_preprocessor()
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    save_preprocessor(pre)
    if HAS_TF:
        X_train_t = X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t
        X_test_t = X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(X_train_t.shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1)])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
        history = model.fit(X_train_t, y_train, validation_split=0.15, epochs=60, batch_size=256, verbose=0, callbacks=callbacks)
        predictions = model.predict(X_test_t, verbose=0).ravel()
        model.save(MODEL_DIR / "crowd_dnn.keras")
        backend = "TensorFlow/Keras DNN"
        history_out = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    else:
        # Same dense neural-network family, available without the optional TensorFlow install.
        model = MLPRegressor(hidden_layer_sizes=(128, 64, 32, 16), activation="relu",
                             solver="adam", early_stopping=True, random_state=42,
                             max_iter=180, batch_size=256, validation_fraction=0.15)
        model.fit(X_train_t, y_train)
        predictions = model.predict(X_test_t)
        joblib.dump(model, MODEL_DIR / "crowd_dnn.pkl")
        backend = "scikit-learn MLPRegressor fallback (install tensorflow-cpu for Keras)"
        history_out = {"loss_curve": [float(v) for v in model.loss_curve_]}
    metrics = {"backend": backend, "mae": float(mean_absolute_error(y_test, predictions)),
               "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
               "r2": float(r2_score(y_test, predictions)), "test_rows": int(len(y_test))}
    MODEL_DIR.mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    (ROOT / "results" / "training_history.json").write_text(json.dumps(history_out, indent=2))
    pd.DataFrame({"actual": y_test, "predicted": predictions}).to_csv(ROOT / "results" / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
