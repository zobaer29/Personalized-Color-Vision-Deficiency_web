"""
Screening and CVD Profile Estimation Engine.
Provides non-clinical profile estimation to help users initialize their extension settings.
Explicitly disclaims clinical diagnosis.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

CATALOG_PATH = Path(__file__).resolve().parents[2] / "screening" / "plates.json"


def catalog() -> List[Dict[str, Any]]:
    if not CATALOG_PATH.exists():
        return []
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        return data.get("plates", [])
    except Exception:
        return []


def public_catalog() -> Dict[str, Any]:
    plates = catalog()
    return {
        "available": bool(plates),
        "clinical_diagnosis": False,
        "message": (
            "CVD Profile Estimation Screening. This is a non-clinical calibration test "
            "to help personalize browser recoloring. It is not an official medical diagnosis."
            if plates
            else "Screening plate catalog is not installed. Please select your known profile manually."
        ),
        "plates": [
            {
                "id": p["id"],
                "type": p.get("type", "general"),
                "prompt": p.get("prompt", "What numeral do you perceive in the plate?"),
                "image": p["image"],
                "options": p["options"],
            }
            for p in plates
        ],
    }


def score(answers: List[Dict[str, str]]) -> Dict[str, Any]:
    plates = {p["id"]: p for p in catalog()}
    if not plates:
        raise RuntimeError("Screening plate catalog is not installed.")

    ids = [a["plate_id"] for a in answers]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate plate answers submitted.")

    if set(ids) != set(plates.keys()):
        raise ValueError(f"Answer count mismatch: received {len(ids)} answers, expected {len(plates)}.")

    protan_votes = 0
    deutan_votes = 0
    tritan_votes = 0
    errors = 0

    for a in answers:
        plate = plates[a["plate_id"]]
        user_ans = a["user_answer"]
        if user_ans not in plate["options"]:
            raise ValueError(f"Invalid answer '{user_ans}' for plate '{plate['id']}'.")

        normal_ans = plate.get("normal_answer")
        if user_ans != normal_ans:
            errors += 1
            if user_ans == plate.get("protan_answer"):
                protan_votes += 2
            if user_ans == plate.get("deutan_answer"):
                deutan_votes += 2
            if user_ans == plate.get("tritan_answer"):
                tritan_votes += 2
            if plate.get("type") == "protan":
                protan_votes += 3
            elif plate.get("type") == "deutan":
                deutan_votes += 3
            elif plate.get("type") == "blue-yellow":
                tritan_votes += 4

    total_plates = len(plates)

    if errors == 0:
        return {
            "diagnosed": False,
            "condition": "Normal Trichromacy (No errors detected)",
            "condition_id": None,
            "severity": 0.0,
            "errors": 0,
            "total": total_plates,
            "confidence": "High",
            "message": "No color confusion detected on screening plates. You can still select a profile manually if desired.",
        }

    # Determine most likely deficiency profile
    scores = [
        (protan_votes, 0, "Protanopia (Red-weak)"),
        (deutan_votes, 1, "Deuteranopia (Green-weak)"),
        (tritan_votes, 2, "Tritanopia (Blue-weak)"),
    ]
    scores.sort(key=lambda x: x[0], reverse=True)
    top_votes, est_id, est_name = scores[0]

    # Estimated personalization severity based on error proportion
    est_severity = round(min(1.0, max(0.5, errors / total_plates)), 2)

    return {
        "diagnosed": False,
        "condition": f"Estimated Profile: {est_name}",
        "condition_id": est_id,
        "severity": est_severity,
        "errors": errors,
        "total": total_plates,
        "confidence": "Moderate" if top_votes >= 3 else "Low",
        "message": (
            "Screening indicates a tendency toward this profile. "
            "Please confirm or adjust your profile in settings. "
            "This is an estimation for accessibility adjustments, not a clinical diagnosis."
        ),
    }
