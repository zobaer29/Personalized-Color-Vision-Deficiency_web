"""
Personalization and CVD Diagnostic Engine.
"""

from typing import Dict, Any, List, Optional
import numpy as np


CVD_TYPES = ("Protanopia", "Deuteranopia", "Tritanopia")

CONDITION_TO_ID = {
    "protanopia": 0,
    "protan": 0,
    "deuteranopia": 1,
    "deutan": 1,
    "tritanopia": 2,
    "tritan": 2,
}

ID_TO_CONDITION = {
    0: "Protanopia",
    1: "Deuteranopia",
    2: "Tritanopia",
}


def resolve_condition_id(condition_name_or_id: Any) -> int:
    """Resolves condition name or integer to validated ID [0, 1, 2]."""
    if isinstance(condition_name_or_id, int):
        if condition_name_or_id in ID_TO_CONDITION:
            return condition_name_or_id
        raise ValueError(f"Invalid condition_id: {condition_name_or_id}. Expected 0, 1, or 2.")

    if isinstance(condition_name_or_id, str):
        clean = condition_name_or_id.strip().lower()
        if clean in CONDITION_TO_ID:
            return CONDITION_TO_ID[clean]
        if clean.isdigit():
            val = int(clean)
            if val in ID_TO_CONDITION:
                return val

    raise ValueError(f"Unknown condition '{condition_name_or_id}'. Expected Protanopia, Deuteranopia, or Tritanopia.")


def apply_personalization_blend(
    original_np: np.ndarray,
    recolored_np: np.ndarray,
    severity: float = 1.0,
) -> np.ndarray:
    """
    Blends original image and recolored model output according to the user's severity/preference.
    severity = 1.0 -> 100% recolored
    severity = 0.0 -> 100% original
    """
    alpha = float(np.clip(severity, 0.0, 1.0))
    blended = alpha * recolored_np + (1.0 - alpha) * original_np
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)

