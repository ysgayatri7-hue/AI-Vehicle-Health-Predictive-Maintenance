"""Feature engineering and inference for the original serialized models."""
from __future__ import annotations

from functools import cached_property
from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from .config import DATASET_PATH, MODEL_DIR

REQUIRED_INPUTS = ("product_type", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear")
MODEL_FEATURES = (
    "Air_Temperature", "Process_Temperature", "Rotational_Speed", "Torque", "Tool_Wear",
    "Temperature_Difference", "Power", "Temperature_Ratio", "Type_L", "Type_M",
)
INPUT_LIMITS = {
    "air_temperature": (250.0, 350.0), "process_temperature": (250.0, 400.0),
    "rotational_speed": (0.0, 10000.0), "torque": (0.0, 500.0), "tool_wear": (0.0, 10000.0),
}


class InputValidationError(ValueError):
    pass


def validate_input(values: dict) -> dict:
    if not isinstance(values, dict):
        raise InputValidationError("Provide one input object.")
    missing = [key for key in REQUIRED_INPUTS if key not in values or values[key] in (None, "")]
    if missing:
        raise InputValidationError("Missing required fields: " + ", ".join(missing))
    product_type = str(values["product_type"]).strip().upper()
    if product_type not in {"L", "M", "H"}:
        raise InputValidationError("product_type must be L, M, or H.")
    clean = {"product_type": product_type}
    for name, (lower, upper) in INPUT_LIMITS.items():
        try:
            number = float(values[name])
        except (TypeError, ValueError):
            raise InputValidationError(f"{name} must be a number.") from None
        if not math.isfinite(number) or not lower <= number <= upper:
            raise InputValidationError(f"{name} must be between {lower:g} and {upper:g}.")
        clean[name] = number
    if clean["air_temperature"] <= 0:
        raise InputValidationError("air_temperature must be above zero Kelvin.")
    if clean["process_temperature"] < clean["air_temperature"]:
        raise InputValidationError("process_temperature must be at least air_temperature.")
    return clean


def engineer_features(values: dict, feature_order: list[str] | tuple[str, ...] = MODEL_FEATURES) -> pd.DataFrame:
    values = validate_input(values)
    air = values["air_temperature"]
    process = values["process_temperature"]
    frame = pd.DataFrame({
        "Air_Temperature": [air], "Process_Temperature": [process],
        "Rotational_Speed": [values["rotational_speed"]], "Torque": [values["torque"]],
        "Tool_Wear": [values["tool_wear"]], "Temperature_Difference": [process - air],
        "Power": [values["torque"] * values["rotational_speed"] * 2 * np.pi / 60],
        "Temperature_Ratio": [process / air],
        "Type_L": [int(values["product_type"] == "L")],
        "Type_M": [int(values["product_type"] == "M")],
    })
    if set(feature_order) != set(frame.columns) or len(feature_order) != len(frame.columns):
        raise RuntimeError(f"Saved model feature order is incompatible: {feature_order}")
    return frame[list(feature_order)]


def dataset_features(dataset: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    """Apply exactly the same transforms to the original AI4I rows."""
    colmap = {c.lower(): c for c in dataset.columns}
    def col(name):
        found = colmap.get(name.lower())
        if not found:
            raise RuntimeError(f"AI4I dataset is missing expected column: {name}")
        return found
    air = dataset[col("Air temperature [K]")].astype(float)
    process = dataset[col("Process temperature [K]")].astype(float)
    torque = dataset[col("Torque [Nm]")].astype(float)
    rpm = dataset[col("Rotational speed [rpm]")].astype(float)
    product = dataset[col("Type")].astype(str)
    transformed = pd.DataFrame({
        "Air_Temperature": air, "Process_Temperature": process, "Rotational_Speed": rpm,
        "Torque": torque, "Tool_Wear": dataset[col("Tool wear [min]")].astype(float),
        "Temperature_Difference": process - air, "Power": torque * rpm * 2 * np.pi / 60,
        "Temperature_Ratio": process / air,
        "Type_L": (product == "L").astype(int), "Type_M": (product == "M").astype(int),
    })
    return transformed[list(feature_order)]


class PredictionService:
    def __init__(self, model_dir: Path = MODEL_DIR, dataset_path: Path = DATASET_PATH):
        self.model_dir = Path(model_dir)
        self.dataset_path = Path(dataset_path)
        # These are the original project artifacts. No model is retrained or replaced.
        self.xgb_model = joblib.load(self.model_dir / "xgboost_failure_model.pkl")
        self.isolation_model = joblib.load(self.model_dir / "isolation_forest.pkl")
        self.features = list(joblib.load(self.model_dir / "features.pkl"))
        if self.features != list(MODEL_FEATURES):
            raise RuntimeError(f"Saved feature order differs from the application contract: {self.features}")
        self.explainer = shap.TreeExplainer(self.xgb_model)

    def predict(self, values: dict) -> dict:
        clean = validate_input(values)
        row = engineer_features(clean, self.features)
        predicted_class = int(self.xgb_model.predict(row)[0])
        probability = float(self.xgb_model.predict_proba(row)[0][1])
        anomaly_class = int(self.isolation_model.predict(row)[0])
        explanation = self._explain(row, probability)
        recs = []
        if clean["tool_wear"] > 150:
            recs.append("Inspect component/tool wear.")
        if clean["torque"] > 60:
            recs.append("Check mechanical load and torque.")
        if clean["process_temperature"] - clean["air_temperature"] > 15:
            recs.append("Inspect thermal operating conditions.")
        if probability > 0.5:
            recs.append("Schedule maintenance inspection.")
        if not recs:
            recs.append("No immediate maintenance action indicated by the prototype rules.")
        return {
            "inputs": clean, "features": {name: float(row.iloc[0][name]) for name in self.features},
            "prediction": predicted_class, "failure_probability": probability,
            "anomaly_prediction": anomaly_class,
            "health_score": max(0.0, 100.0 * (1.0 - probability)),
            "maintenance_status": "Maintenance Recommended" if predicted_class == 1 else "Normal",
            "anomaly_status": "Anomaly Detected" if anomaly_class == -1 else "Normal",
            "recommendations": recs, "explanation": explanation,
        }

    def _explain(self, row: pd.DataFrame, probability: float) -> dict:
        raw_values = self.explainer.shap_values(row, check_additivity=False)
        if isinstance(raw_values, list):
            values = np.asarray(raw_values[-1], dtype=float).reshape(-1)
        else:
            values = np.asarray(raw_values, dtype=float)
            if values.ndim == 3:
                values = values[0, :, -1]
            else:
                values = values.reshape(-1)
        base = np.asarray(self.explainer.expected_value, dtype=float)
        base_value = float(base[-1] if base.ndim else base.item())
        contributions = [
            {"feature": feature, "value": float(row.iloc[0][feature]), "contribution": float(value)}
            for feature, value in zip(self.features, values)
        ]
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        margin = float(self.xgb_model.get_booster().predict(xgb.DMatrix(row), output_margin=True)[0])
        residual = float(base_value + np.sum(values) - margin)
        # SHAP uses raw log-odds for the binary tree model. Check additivity to ensure
        # the shown waterfall describes the same positive-class output as predict_proba.
        shap_probability = 1.0 / (1.0 + math.exp(-max(-709.0, min(709.0, base_value + float(np.sum(values))))))
        if abs(residual) > 1e-3 or abs(shap_probability - probability) > 1e-3:
            raise RuntimeError("SHAP contributions failed the model additivity check.")
        return {
            "base_value": base_value, "model_margin": margin, "probability": probability,
            "additivity_residual": residual, "verified": True,
            "contributions": contributions,
            "top_features": contributions[:3],
        }

    @cached_property
    def global_importance(self) -> list[dict]:
        if not self.dataset_path.exists():
            return []
        data = pd.read_csv(self.dataset_path)
        sample = data.sample(n=min(500, len(data)), random_state=42)
        frame = dataset_features(sample, self.features)
        values = self.explainer.shap_values(frame, check_additivity=False)
        if isinstance(values, list):
            matrix = np.asarray(values[-1], dtype=float)
        else:
            matrix = np.asarray(values, dtype=float)
            if matrix.ndim == 3:
                matrix = matrix[:, :, -1]
        means = np.mean(np.abs(matrix), axis=0)
        return sorted(
            [{"feature": name, "mean_abs_shap": float(value)} for name, value in zip(self.features, means)],
            key=lambda item: item["mean_abs_shap"], reverse=True,
        )
