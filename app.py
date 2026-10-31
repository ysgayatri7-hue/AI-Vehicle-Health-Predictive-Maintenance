"""Flask entry point using the project's existing serialized ML artifacts."""
import os

from flask import Flask

from maintenance import database
from maintenance.config import MAX_BATCH_ROWS
from maintenance.ml import PredictionService
from maintenance.routes import bp


def create_app(prediction_service=None, database_path=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=12 * 1024 * 1024, MAX_BATCH_ROWS=MAX_BATCH_ROWS)
    resolved_db = database_path or database.DATABASE_PATH
    database.initialize(resolved_db)
    app.extensions["database_path"] = resolved_db
    app.extensions["prediction_service"] = prediction_service or PredictionService()
    app.register_blueprint(bp)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)
