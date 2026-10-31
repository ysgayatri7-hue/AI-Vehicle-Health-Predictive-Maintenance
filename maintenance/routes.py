"""Flask routes for predictions, dashboard, and reports."""
import csv
import io
from datetime import datetime

import pandas as pd
from flask import Blueprint, Response, abort, current_app, redirect, render_template, request, url_for

from . import database
from .ml import InputValidationError

bp = Blueprint("main", __name__)


def _db_path():
    return current_app.extensions.get("database_path") or database.DATABASE_PATH


@bp.get("/")
def home():
    return render_template("home.html")


@bp.get("/about")
def about():
    return render_template("about.html")


@bp.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")
    try:
        result = current_app.extensions["prediction_service"].predict(request.form.to_dict())
    except InputValidationError as exc:
        return render_template("predict.html", error=str(exc), submitted=request.form), 400
    except Exception:
        current_app.logger.exception("Prediction pipeline failed")
        return render_template("predict.html", error="The prediction could not be completed. Check server logs for details.", submitted=request.form), 500
    record_id = database.save_prediction(result, path=_db_path())
    return redirect(url_for("main.result", prediction_id=record_id))


@bp.get("/result/<int:prediction_id>")
def result(prediction_id):
    record = database.get_prediction(prediction_id, path=_db_path())
    if not record:
        abort(404)
    result = record["result"]
    chart = {"base": result["explanation"]["base_value"], "margin": result["explanation"]["model_margin"], "features": result["explanation"]["contributions"]}
    return render_template("result.html", result=result, prediction_id=record["id"], created_at=record["created_at"], shap_chart=chart)


@bp.get("/dashboard")
def dashboard():
    records = database.list_predictions(200, path=_db_path())
    chronological = list(reversed(records))
    service = current_app.extensions["prediction_service"]
    return render_template("dashboard.html", records=records, chronological=chronological,
                           global_importance=service.global_importance, total_count=len(records),
                           alert_count=sum(1 for x in records if x["prediction"] == 1 or x["anomaly_prediction"] == -1))


@bp.get("/predictions/<int:prediction_id>/download.csv")
def download_prediction(prediction_id):
    record = database.get_prediction(prediction_id, path=_db_path())
    if not record:
        abort(404)
    return _csv_response([record["result"]], f"prediction-{prediction_id}.csv")


@bp.post("/predict/batch")
def batch_predict():
    upload = request.files.get("file")
    if not upload or not upload.filename.lower().endswith(".csv"):
        return render_template("predict.html", error="Choose a CSV file to run batch predictions."), 400
    try:
        table = pd.read_csv(upload, nrows=current_app.config["MAX_BATCH_ROWS"] + 1)
        if table.empty or len(table) > current_app.config["MAX_BATCH_ROWS"]:
            raise InputValidationError(f"CSV must contain 1 to {current_app.config['MAX_BATCH_ROWS']} data rows.")
        aliases = {
            "product_type": "product_type", "type": "product_type", "air_temperature": "air_temperature", "air temperature [k]": "air_temperature",
            "process_temperature": "process_temperature", "process temperature [k]": "process_temperature", "rotational_speed": "rotational_speed",
            "rotational speed [rpm]": "rotational_speed", "torque": "torque", "torque [nm]": "torque", "tool_wear": "tool_wear", "tool wear [min]": "tool_wear",
        }
        table = table.rename(columns={c: aliases[str(c).strip().lower()] for c in table.columns if str(c).strip().lower() in aliases})
        required = ["product_type", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
        missing = set(required).difference(table.columns)
        if missing:
            raise InputValidationError("CSV is missing columns: " + ", ".join(sorted(missing)))
        service, results = current_app.extensions["prediction_service"], []
        for _, row in table.iterrows():
            result = service.predict(row[required].to_dict())
            database.save_prediction(result, source="batch", path=_db_path())
            results.append(result)
        filename = f"batch-predictions-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
        return _csv_response(results, filename)
    except (pd.errors.ParserError, UnicodeDecodeError, InputValidationError, ValueError) as exc:
        return render_template("predict.html", error=f"Batch prediction failed: {exc}"), 400
    except Exception:
        current_app.logger.exception("Batch prediction failed")
        return render_template("predict.html", error="Batch prediction failed. Check the CSV and server logs."), 500


@bp.get("/healthz")
def healthz():
    return jsonify(status="ok")


def _csv_response(results, filename):
    output = io.StringIO()
    fields = ["product_type", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear", "failure_probability", "prediction", "anomaly_prediction", "health_score", "top_shap_features"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for result in results:
        values = result["inputs"].copy()
        values.update({"failure_probability": result["failure_probability"], "prediction": result["prediction"],
                       "anomaly_prediction": result["anomaly_prediction"], "health_score": result["health_score"],
                       "top_shap_features": "; ".join(x["feature"] for x in result["explanation"]["top_features"])})
        writer.writerow(values)
    return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})
