"""Inference helpers used by the Streamlit app."""
from pathlib import Path
import joblib
import pandas as pd
from src.data.preprocess import ROOT, MODEL_DIR, add_features, load_preprocessor, crowd_level


def load_model():
    keras_path = MODEL_DIR / "crowd_dnn.keras"
    if keras_path.exists():
        import tensorflow as tf
        return tf.keras.models.load_model(keras_path), "tensorflow"
    return joblib.load(MODEL_DIR / "crowd_dnn.pkl"), "sklearn"


def predict_one(payload: dict) -> dict:
    pre = load_preprocessor()
    model, backend = load_model()
    row = add_features(pd.DataFrame([payload]))
    X = pre.transform(row[["route_id", "stop_id", "weather", "hour", "minute", "day_of_week", "weekend", "peak_hour", "holiday", "rainfall_mm", "temperature_c", "special_event", "vehicle_capacity", "previous_passengers"]])
    if backend == "tensorflow":
        X = X.toarray() if hasattr(X, "toarray") else X
    passengers = float(model.predict(X, verbose=0).ravel()[0] if backend == "tensorflow" else model.predict(X)[0])
    passengers = max(0, passengers)
    capacity = float(payload["vehicle_capacity"])
    return {"predicted_passengers": round(passengers, 1), "capacity": capacity,
            "occupancy_pct": round(100 * passengers / capacity, 1),
            "crowd_level": crowd_level(passengers, capacity)}


def recommend(payload: dict, routes: dict) -> list[dict]:
    results = []
    for route_id, info in routes.items():
        candidate = dict(payload, route_id=route_id, vehicle_capacity=info["capacity"])
        result = predict_one(candidate)
        results.append({"route_id": route_id, "capacity": info["capacity"], **result})
    return sorted(results, key=lambda x: (x["occupancy_pct"], x["predicted_passengers"]))
