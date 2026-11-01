"""
Pydantic API Request/Response Schemas.
Aligned with Task Specification 21 & 22.
"""

from typing import Literal, Optional, List, Any
from pydantic import BaseModel, Field, model_validator
from ..core.personalization import resolve_condition_id


class RecolorRequest(BaseModel):
    image_base64: str = Field(..., min_length=1, max_length=12_000_000, description="Base64 encoded image or data URL")
    condition: Optional[str] = Field(None, description="Optional condition name ('Protanopia', 'Deuteranopia', 'Tritanopia')")
    condition_id: Optional[int] = Field(None, ge=0, le=2, description="Canonical condition ID: 0 (Protan), 1 (Deutan), 2 (Tritan)")
    severity: float = Field(1.0, ge=0.0, le=1.0, allow_inf_nan=False, description="Personalization severity in range [0.0, 1.0]")
    return_format: Literal["webp", "jpeg", "png"] = Field("webp", description="Requested output image format")

    @model_validator(mode="after")
    def validate_and_resolve_condition(self):
        if self.condition_id is None:
            if self.condition is not None:
                self.condition_id = resolve_condition_id(self.condition)
            else:
                self.condition_id = 1  # Default to Deuteranopia
        elif self.condition is not None:
            resolved = resolve_condition_id(self.condition)
            if resolved != self.condition_id:
                raise ValueError(f"Conflicting inputs: condition '{self.condition}' resolves to ID {resolved}, but condition_id is {self.condition_id}")
        return self


class RecolorResponse(BaseModel):
    status: str = Field("success", description="Status string: 'success'")
    recolored_base64: str = Field(..., description="Base64 data URL of personalized recolored image")
    inference_time_ms: float = Field(..., description="Neural model forward pass execution time in milliseconds")
    total_time_ms: float = Field(..., description="Total request execution time in milliseconds")
    cached: bool = Field(..., description="Whether the result was retrieved from cache")
    condition_id: int = Field(..., ge=0, le=2, description="Condition ID applied")
    severity: float = Field(..., ge=0.0, le=1.0, description="Personalization severity applied")
    model_version: str = Field(..., description="Identifier and version of the active serving model")
    device: str = Field(..., description="Execution compute device ('cpu' or 'cuda')")


class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy' or 'degraded_untrained_fallback'")
    model_loaded: bool = Field(True, description="True if model architecture is initialized in memory")
    trained_weights_loaded: bool = Field(..., description="True if verified trained weights are loaded")
    architecture: str = Field(..., description="Active neural model architecture name")
    model_version: str = Field(..., description="Active model version tag")
    device: str = Field(..., description="Compute hardware ('cpu' or 'cuda')")
    cache_enabled: bool = Field(True, description="Whether inference cache is active")
    cached_entries: int = Field(0, description="Number of entries currently in cache")


class FilterRequest(BaseModel):
    condition_id: int = Field(..., ge=0, le=2)
    severity: float = Field(1.0, ge=0.0, le=1.0, allow_inf_nan=False)


class PlateAnswer(BaseModel):
    plate_id: str = Field(..., max_length=80)
    user_answer: str = Field(..., max_length=40)


class DiagnoseRequest(BaseModel):
    answers: List[PlateAnswer] = Field(..., min_length=1, max_length=50)


class DiagnoseResponse(BaseModel):
    diagnosed: bool = Field(False, description="Always False: screening is not a clinical medical diagnosis")
    condition: str = Field(..., description="Estimated condition profile name")
    condition_id: Optional[int] = Field(None, description="Estimated condition ID (0=Protan, 1=Deutan, 2=Tritan)")
    severity: Optional[float] = Field(None, description="Suggested personalization severity in range [0.0, 1.0]")
    errors: int = Field(..., description="Total screening errors")
    total: int = Field(..., description="Total plates evaluated")
    confidence: str = Field(..., description="Estimation confidence rating")
    message: str = Field(..., description="Disclaimer stating this is an estimation, not a clinical diagnosis")
