# AI-Powered Vehicle Health Monitoring and Predictive Maintenance

A Flask decision-support prototype for failure-risk prediction, anomaly detection, SHAP explanations, batch analysis, and saved prediction history. It uses the project's existing XGBoost and Isolation Forest artifacts; it does not retrain or replace them.

> **Scope:** Uses the synthetic AI4I 2020 Predictive Maintenance dataset. This is not a validated vehicle diagnostic, safety, or real-time monitoring system. Outputs require human review.

## Features

- XGBoost failure probability and Isolation Forest anomaly flag.
- Original engineered model inputs and saved feature order.
- Per-report TreeSHAP waterfall, additivity verification, and top three contributors.
- Global mean absolute SHAP chart from up to 500 dataset rows.
- SQLite history dashboard with interactive Plotly trends.
- CSV batch inference (up to 1,000 rows by default) and downloadable reports.
- Input validation and exception handling. No API key or paid AI service is required.

## Architecture

```text
Browser (Flask/Jinja + CSS + Plotly)
  ├── prediction and batch routes -> maintenance/routes.py
  │    ├── maintenance/ml.py -> existing models in models/
  │    └── maintenance/database.py -> SQLite
  └── dashboard and CSV downloads -> saved history
```

| Module | Responsibility |
| --- | --- |
| `maintenance/ml.py` | Validation, feature engineering, model inference, SHAP explanations |
| `maintenance/routes.py` | Flask pages, predictions, batch routes, downloads |
| `maintenance/database.py` | SQLite schema and prediction history |
| `maintenance/config.py` | Project paths and environment configuration |

## Setup

Python 3.11 is recommended. In PowerShell from this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The app initializes SQLite under `instance/` on first run. The original model files in `models/` must be present.

## Use

1. Open **Predict**, enter sensor values, and submit. The report saves model outputs and a local explanation.
2. Review the TreeSHAP waterfall. Positive raw-margin contributions move toward the failure class; negative ones move away. Their sum with the base value is checked against the XGBoost margin for the displayed probability.
3. Open **Dashboard** for the latest 200 saved reports. These are historical records, not streaming telemetry.
4. Upload a CSV from **Predict** for batch inference and download its report. Required columns: `product_type, air_temperature, process_temperature, rotational_speed, torque, tool_wear`. AI4I sensor labels are also accepted.

## Data and model notes

- Feature order from `models/features.pkl`: `Air_Temperature`, `Process_Temperature`, `Rotational_Speed`, `Torque`, `Tool_Wear`, `Temperature_Difference`, `Power`, `Temperature_Ratio`, `Type_L`, `Type_M`.
- Temperature difference, mechanical power, and temperature ratio are engineered; product type H is the reference category.
- “Health score” is `100 × (1 − failure probability)`, not an independently validated health measure.
- Global importance is mean absolute SHAP on a reproducible sample of at most 500 dataset rows, not a model-performance metric.
- Only load trusted pickle files; serialized models can depend on library versions.

## Tests

```powershell
python -m pytest -q
```

Tests cover inference and SHAP additivity, validation, routes, persistence, batch prediction, and CSV downloads.

## Screenshots

Capture the report, dashboard, and batch CSV section from the running app before publishing. Avoid personal data or credentials.
