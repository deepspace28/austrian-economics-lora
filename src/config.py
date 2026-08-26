from pathlib import Path
import os

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOOKS_DIR = PROJECT_ROOT / "books"
EXTRACTED_DIR = PROJECT_ROOT / "extracted"
CLEANED_DIR = PROJECT_ROOT / "cleaned"
DATASET_DIR = PROJECT_ROOT / "datasets"
CHUNKS_DIR = PROJECT_ROOT / "chunks"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
LOG_DIR = PROJECT_ROOT / "logs"

# ==========================================================
# MODEL
# ==========================================================

MODEL_NAME = os.environ.get("MODEL_NAME", "glm-5.2")

# Read from the environment; never commit a literal key here.
#   Windows:  set MODEL_ACCESS_KEY=...
#   bash:     export MODEL_ACCESS_KEY=...
MODEL_ACCESS_KEY = os.environ.get("MODEL_ACCESS_KEY", "")


# ==========================================================
# DATASET
# ==========================================================

MIN_QA = 3
MAX_QA = 8

CHUNK_SIZE = 1500

TEMPERATURE = 0.2

MAX_TOKENS = 2500