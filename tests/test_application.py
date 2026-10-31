import io

import pytest

from app import create_app
from maintenance.ml import InputValidationError, PredictionService, engineer_features, validate_input


SAMPLE = {
    "product_type": "M", "air_temperature": 300.0, "process_temperature": 310.0,
    "rotational_speed": 1500, "torque": 40.0, "tool_wear": 100,
}


@pytest.fixture(scope="module")
def prediction_service():
    return PredictionService()




@pytest.fixture
def client(tmp_path, prediction_service):
    app = create_app(prediction_service, tmp_path / "test.db")
    app.config.update(TESTING=True)
    return app.test_client()


def test_feature_order_and_engineering():
    row = engineer_features(SAMPLE)
    assert list(row.columns) == [
        "Air_Temperature", "Process_Temperature", "Rotational_Speed", "Torque", "Tool_Wear",
        "Temperature_Difference", "Power", "Temperature_Ratio", "Type_L", "Type_M",
    ]
    assert row.loc[0, "Temperature_Difference"] == pytest.approx(10)
    assert row.loc[0, "Temperature_Ratio"] == pytest.approx(310 / 300)
    assert row.loc[0, "Type_M"] == 1


@pytest.mark.parametrize("change", [
    {"product_type": "Z"}, {"air_temperature": float("nan")},
    {"process_temperature": 299}, {"rotational_speed": -1},
])
def test_server_validates_sensor_inputs(change):
    with pytest.raises(InputValidationError):
        validate_input({**SAMPLE, **change})


def test_existing_models_predict_and_shap_is_additive(prediction_service):
    result = prediction_service.predict(SAMPLE)
    explanation = result["explanation"]
    assert result["prediction"] in (0, 1)
    assert 0 <= result["failure_probability"] <= 1
    assert result["anomaly_prediction"] in (-1, 1)
    assert explanation["verified"] is True
    assert len(explanation["top_features"]) == 3
    assert abs(explanation["additivity_residual"]) < 1e-3


def test_pages_and_saved_result(client):
    assert client.get("/").status_code == 200
    assert client.get("/about").status_code == 200
    assert client.get("/predict").status_code == 200
    response = client.post("/predict", data=SAMPLE)
    assert response.status_code == 302
    report = client.get(response.headers["Location"])
    assert report.status_code == 200
    assert b"Individual Prediction Explanation" in report.data
    assert b"ASK COPILOT" not in report.data
    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert b"Historical Failure Probability" in dashboard.data


def test_invalid_form_returns_explanation(client):
    response = client.post("/predict", data={**SAMPLE, "air_temperature": "nan"})
    assert response.status_code == 400
    assert b"must be between" in response.data


def test_batch_csv_returns_download_and_saves_history(client):
    payload = "product_type,air_temperature,process_temperature,rotational_speed,torque,tool_wear\nM,300,310,1500,40,100\n"
    response = client.post("/predict/batch", data={"file": (io.BytesIO(payload.encode()), "batch.csv")}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert b"failure_probability" in response.data
    assert b"top_shap_features" in response.data


def test_report_csv_download(client):
    response = client.post("/predict", data=SAMPLE)
    report_id = response.headers["Location"].rsplit("/", 1)[-1]
    download = client.get(f"/predictions/{report_id}/download.csv")
    assert download.status_code == 200
    assert download.mimetype == "text/csv"

