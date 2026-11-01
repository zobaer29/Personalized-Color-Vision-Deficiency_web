"""
FastAPI Routes for Personalized CVD Recoloring Engine.
"""

import binascii
import logging
from PIL import UnidentifiedImageError
from fastapi import APIRouter, HTTPException, status
from .schemas import (
    RecolorRequest,
    RecolorResponse,
    HealthResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    FilterRequest,
)
from ..core.predictor import CVDRecolorPredictor
from ..core.page_filter import page_filter
from ..core.screening import public_catalog, score
from ..core.cache import recolor_cache
from ..config import settings

logger = logging.getLogger("cvd_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

router = APIRouter()

# Initialize predictor singleton on startup
predictor = CVDRecolorPredictor(
    weights_path=str(settings.DEFAULT_WEIGHTS_PATH) if settings.DEFAULT_WEIGHTS_PATH.exists() else None,
    model_name=settings.PRODUCTION_MODEL_NAME,
)

if settings.REQUIRE_TRAINED_WEIGHTS and not predictor.weights_loaded:
    logger.error("CRITICAL: REQUIRE_TRAINED_WEIGHTS is enabled, but no valid trained weights could be loaded.")


def ready():
    if not predictor.weights_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained Restormer_RDB checkpoint missing or incompatible. Set CVD_WEIGHTS and restart backend.",
        )


@router.post("/api/v1/recolor", response_model=RecolorResponse)
def recolor_image(payload: RecolorRequest):
    """
    Personalized neural image recoloring endpoint.
    Receives base64 image, condition ID (0, 1, 2), severity [0, 1], and return format.
    """
    ready()

    try:
        result = predictor.recolor(
            image_base64=payload.image_base64,
            condition_id_or_name=payload.condition_id,
            severity=payload.severity,
            return_format=payload.return_format,
        )

        logger.info(
            f"[Recolor] condition={payload.condition_id} severity={payload.severity:.2f} "
            f"fmt={payload.return_format} cached={result['cached']} "
            f"inf_ms={result['inference_time_ms']} total_ms={result['total_time_ms']}"
        )
        return RecolorResponse(**result)

    except ValueError as exc:
        msg = str(exc)
        if "payload too large" in msg.lower():
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=msg) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg) from exc
    except (binascii.Error, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid or corrupted image: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during recoloring inference")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Recoloring processing failed") from exc


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Returns real-time health, loaded architecture, weights status, and cache size."""
    health_status = "healthy" if predictor.weights_loaded else "checkpoint_required"
    return HealthResponse(
        status=health_status,
        model_loaded=True,
        trained_weights_loaded=predictor.weights_loaded,
        architecture=predictor.model_name,
        model_version=predictor.model_version,
        device=str(predictor.device),
        cache_enabled=True,
        cached_entries=recolor_cache.size(),
    )


@router.get("/api/v1/plates")
def get_screening_plates():
    """Returns non-clinical CVD estimation screening plates."""
    return public_catalog()


@router.post("/api/v1/diagnose", response_model=DiagnoseResponse)
def evaluate_screening_test(payload: DiagnoseRequest):
    """
    Evaluates responses to screening plates.
    Returns estimated condition profile and suggested personalization severity.
    """
    try:
        answers_dict = [a.model_dump() for a in payload.answers]
        result = score(answers_dict)
        return DiagnoseResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/api/v1/filter")
def browser_fallback_filter(payload: FilterRequest):
    """Fallback whole-page SVG color matrix for UI elements."""
    ready()
    return page_filter(predictor, payload.condition_id, payload.severity)
