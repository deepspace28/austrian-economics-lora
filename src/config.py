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
# DATASET GENERATION
# ==========================================================
# These are read by src/dataset_generator.py. They used to be declared here and
# never read, while the generator hardcoded its own values, so the prompt asked
# for 3-8 pairs and the sampler used a different temperature than the file said.

# Upper bound on pairs kept per chunk. The system prompt asks for 3-8.
MAX_QA = 8

# Sampling. 0.2 keeps generation close to the source text; higher values drift
# into invented material, which the prompt forbids.
TEMPERATURE = 0.2

# Generation budget. 600 tokens truncated long chunks mid-JSON, and the parse
# then failed and the chunk was retried, which made failures cluster on the
# longest and richest chunks.
MAX_NEW_TOKENS = 1500

# Attempts per chunk before it is given up on.
MAX_ATTEMPTS = 3