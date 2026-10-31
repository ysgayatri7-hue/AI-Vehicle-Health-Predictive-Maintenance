from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "instance" / "maintenance.db"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
DATASET_PATH = Path(os.getenv("DATASET_PATH", BASE_DIR / "ai4i2020.csv"))
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", BASE_DIR / "knowledge"))
MAX_BATCH_ROWS = int(os.getenv("MAX_BATCH_ROWS", "1000"))
