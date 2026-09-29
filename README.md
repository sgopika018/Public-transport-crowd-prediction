# TransitPulse: AI-Based Public Transport Crowd Prediction

TransitPulse is a complete Python project that predicts public-transport passenger demand before a journey begins. It converts the predicted passenger count into four operational crowd levels and recommends an alternative route when another service is expected to be less crowded.

> **Project status:** runnable college-project prototype. The included dataset is synthetic and must be replaced with validated transport observations for production use.

## 1. Problem statement

Public transport crowding changes with time of day, route, stop, weekday, holidays, weather, and special events. Timetables describe when a vehicle should arrive, but they do not forecast how full it is likely to be. TransitPulse addresses this gap by forecasting passenger demand and occupancy for a selected route and stop.

## 2. Objectives

The system is designed to forecast passenger count, classify crowding, identify busy hours, compare route alternatives, and give operators a compact network-level view. The core model is a dense neural-network regressor. TensorFlow/Keras is used when installed; otherwise the project uses an equivalent scikit-learn multilayer perceptron so that the project remains runnable in a lightweight VS Code environment.

## 3. Architecture

```text
Transport observations
        ↓
Data generation / collection
        ↓
Feature engineering: time, calendar, weather, events, history
        ↓
One-hot encoding + standard scaling
        ↓
Dense neural network regression
        ↓
Passenger forecast
        ↓
Occupancy = forecast / vehicle capacity
        ↓
Low / Medium / High / Overcrowded
        ↓
Streamlit passenger and operator dashboards
```

## 4. Crowd definitions

| Occupancy | Level | Meaning |
|---:|---|---|
| 0–40% | Low | Ample capacity is expected. |
| 41–70% | Medium | Moderate demand is expected. |
| 71–100% | High | The service may be uncomfortable or near capacity. |
| >100% | Overcrowded | Demand is above nominal capacity. |

## 5. Folder structure

| Path | Purpose |
|---|---|
| `dataset/` | Input CSV and generated demonstration data. |
| `models/` | Saved preprocessor and trained model artifacts. |
| `src/data/` | Dataset generation and feature engineering. |
| `src/training/` | Model training and evaluation. |
| `src/prediction/` | Inference and route recommendation. |
| `app/` | Streamlit user interface. |
| `results/` | Metrics, predictions, and training outputs. |
| `notebooks/` | Optional experiments. |

## 6. Run in VS Code

Open the project folder in VS Code, create a virtual environment, and install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Generate the demo dataset and train the model:

```bash
python -m src.data.generate_dataset
python -m src.training.train_dnn
```

Launch the dashboard:

```bash
streamlit run app/app.py
```

If TensorFlow is installed, the trainer creates `models/crowd_dnn.keras`. Without TensorFlow, it creates `models/crowd_dnn.pkl` using the same dense-layer layout through `MLPRegressor`. The saved preprocessor is `models/preprocessor.pkl`.

## 7. Dataset schema

The generated CSV contains `date`, `time`, `route_id`, `stop_id`, `vehicle_capacity`, `holiday`, `weekend`, `rainfall_mm`, `temperature_c`, `special_event`, and `passenger_count`. During preprocessing, the system derives hour, minute, day of week, peak-hour flag, weather category, and previous passenger count.

For a real deployment, replace the generator with a data ingestion step using automatic passenger counters, validated ticketing data, or a carefully designed survey. Preserve a time-based validation split when possible to avoid leakage from future records.

## 8. Evaluation

This is a regression problem, so the project reports mean absolute error (MAE), root mean squared error (RMSE), and R². Accuracy alone is not an appropriate primary metric for passenger-count forecasting. The exact metrics are written to `results/model_metrics.json` after training.

## 9. Demonstration flow

Select a route, stop, date, time, weather condition, holiday/event flags, and recent passenger count. Click **Predict crowd**. The passenger view shows expected passengers, vehicle capacity, occupancy, crowd level, and alternative route estimates. The operator view shows average demand by hour and route and summarizes model performance.

## 10. Limitations and next improvements

The included data is synthetic, so the reported metrics demonstrate the software pipeline rather than real-world accuracy. A stronger study should add several months of timestamped observations, route-specific capacities, special-event calendars, missing-data handling, time-series cross-validation, uncertainty intervals, SHAP explanations, and a live API or database. Operational deployment should also include privacy review, monitoring for concept drift, and human oversight.

## 11. Suggested viva questions

**Why is this regression rather than classification?** The model predicts a numeric passenger count. Crowd labels are derived afterward from predicted occupancy.

**Why is capacity needed?** Passenger count alone is not enough to determine whether a vehicle is crowded. Occupancy relates demand to the capacity of the selected vehicle.

**Why use a DNN?** A multilayer perceptron can learn nonlinear interactions among route, time, calendar, weather, and event variables. It is appropriate for a structured-data prototype, although tree models should also be benchmarked in a rigorous study.

**What is the main limitation?** Synthetic training data cannot establish real-world performance. Real, representative observations are required before using the system for transport decisions.
