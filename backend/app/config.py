"""
Backend Configuration for Personalized CVD Recoloring API.
"""

from pathlib import Path
import os


class Settings:
    PROJECT_NAME: str = "Personalized CVD Recoloring API"
    VERSION: str = "1.0.0"
    HOST: str = os.environ.get("CVD_HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("CVD_PORT", "8000"))

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    WEIGHTS_DIR: Path = BASE_DIR / "weights"
    DEFAULT_WEIGHTS_PATH: Path = Path(os.environ.get("CVD_WEIGHTS", str(WEIGHTS_DIR / "best_restormer.pt")))

    # Production Model Architecture
    PRODUCTION_MODEL_NAME: str = os.environ.get("CVD_MODEL_NAME", "Restormer_RDB")

    # Safety: Require trained checkpoint in strict production mode
    REQUIRE_TRAINED_WEIGHTS: bool = os.environ.get("REQUIRE_TRAINED_WEIGHTS", "0").lower() in ("1", "true", "yes")

    # Hardware & Performance
    DEVICE: str = os.environ.get("CVD_DEVICE", "cuda")  # Auto-falls back to cpu if cuda unavailable
    MAX_CACHE_SIZE: int = int(os.environ.get("CVD_CACHE_SIZE", "500"))


settings = Settings()
