"""Generate a realistic, deterministic demo dataset for the transport prototype."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "transport_data.csv"


def generate(n_days: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    routes = {
        "21A": ("Central", "University", 50, 1.00),
        "21B": ("Central", "Tech Park", 60, 0.78),
        "21C": ("Market", "Airport", 70, 0.90),
        "14A": ("Railway Station", "Hospital", 45, 0.72),
        "8X": ("Old Town", "IT Corridor", 55, 0.84),
    }
    stops = ["Central", "Market", "University", "Tech Park", "Railway Station", "Hospital"]
    rows = []
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    for date in dates:
        dow = date.dayofweek
        weekend = int(dow >= 5)
        holiday = int((date.dayofyear % 23) == 0)
        for route_id, (_, _, capacity, route_factor) in routes.items():
            for stop in stops:
                for slot in range(0, 24 * 4, 2):  # every 30 minutes
                    hour, minute = divmod(slot * 15, 60)
                    peak = int((7 <= hour < 10) or (16 <= hour < 20))
                    rain = float(max(0, rng.normal(4 if date.dayofyear % 7 == 0 else 1.5, 3)))
                    temp = float(np.clip(rng.normal(28 - 0.06 * rain, 3), 18, 38))
                    event = int((date.dayofyear + hour + len(stop)) % 31 == 0)
                    time_wave = 14 * np.sin((hour + minute / 60) / 24 * 2 * np.pi - 1.2)
                    peak_demand = 32 * peak
                    stop_factor = 1.18 if stop in {"Central", "Railway Station"} else 0.88
                    weather_effect = 0.45 * rain
                    demand = (28 + time_wave + peak_demand + 12 * (1 - weekend) + 8 * event
                              + weather_effect) * route_factor * stop_factor
                    demand *= (1 - 0.18 * holiday)
                    passengers = int(np.clip(rng.normal(demand, 5), 3, capacity * 1.8))
                    rows.append({
                        "date": date.date().isoformat(), "time": f"{hour:02d}:{minute:02d}",
                        "route_id": route_id, "stop_id": stop, "vehicle_capacity": capacity,
                        "holiday": holiday, "weekend": weekend, "rainfall_mm": round(rain, 2),
                        "temperature_c": round(temp, 2), "special_event": event,
                        "passenger_count": passengers,
                    })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows to {OUT}")
    return df


if __name__ == "__main__":
    generate()

# This dataset is synthetic and intended for demonstration. Replace it with validated
# ticketing, APC, or GTFS-linked observations for a production deployment.

        
