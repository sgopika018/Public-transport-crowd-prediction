"""Feature engineering shared by training and inference."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

CATEGORICAL = ["route_id", "stop_id", "weather"]
NUMERIC = ["hour", "minute", "day_of_week", "weekend", "peak_hour", "holiday",
           "rainfall_mm", "temperature_c", "special_event", "vehicle_capacity",
           "previous_passengers"]
TARGET = "passenger_count"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["date"])
    t = pd.to_datetime(out["time"], format="%H:%M")
    out["hour"] = t.dt.hour
    out["minute"] = t.dt.minute
    out["day_of_week"] = dt.dt.dayofweek
    out["weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["peak_hour"] = (((out["hour"] >= 7) & (out["hour"] < 10)) |
                        ((out["hour"] >= 16) & (out["hour"] < 20))).astype(int)
    out["weather"] = out.apply(lambda r: "Rainy" if r["rainfall_mm"] >= 7 else "Cloudy" if r["rainfall_mm"] >= 3 else "Clear", axis=1)
    if "previous_passengers" not in out:
        out["previous_passengers"] = out.get(TARGET, 0).shift(1).fillna(out.get(TARGET, 0).median() if TARGET in out else 30)
    return out


def make_preprocessor():
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                              ("num", StandardScaler(), NUMERIC)])


def save_preprocessor(preprocessor):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")


def load_preprocessor():
    return joblib.load(MODEL_DIR / "preprocessor.pkl")


def crowd_level(passengers: float, capacity: float) -> str:
    occupancy = 100 * passengers / max(capacity, 1)
    if occupancy <= 40: return "Low"
    if occupancy <= 70: return "Medium"
    if occupancy <= 100: return "High"
    return "Overcrowded"


def crowd_color(level: str) -> str:
    return {"Low": "#2e9d63", "Medium": "#d59b19", "High": "#e87822", "Overcrowded": "#d64545"}[level]
