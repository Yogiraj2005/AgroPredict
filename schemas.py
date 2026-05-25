"""
Pydantic Data Schemas for AgroPredict API
==========================================
Defines the request/response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ==========================================
# Request Schemas
# ==========================================
class EnvironmentalData(BaseModel):
    """Soil and weather parameters required by Model C (Random Forest)."""
    n: float = Field(..., description="Nitrogen content in soil (kg/ha)")
    p: float = Field(..., description="Phosphorus content in soil (kg/ha)")
    k: float = Field(..., description="Potassium content in soil (kg/ha)")
    soil_ph: float = Field(..., description="Soil pH level")
    soil_moisture: float = Field(..., description="Soil moisture (%)")
    temperature: float = Field(..., description="Average temperature (°C)")
    humidity: float = Field(..., description="Humidity (%)")
    rainfall: float = Field(..., description="Rainfall (mm)")
    sunlight_hours: float = Field(..., description="Sunlight hours per day")


# ==========================================
# Response Schemas
# ==========================================
class DiseaseReport(BaseModel):
    """Output from Path A (Model A + Model B)."""
    disease_name: str
    confidence_score: float
    dsi: float
    affected_area_percent: float
    severity: str


class YieldReport(BaseModel):
    """Output from Path B (Model C + Model D)."""
    ideal_yield: float
    final_yield: float
    yield_loss: float
    yield_loss_percentage: float
    dsi_used: float
    crop_sensitivity_K: float


class LLMReport(BaseModel):
    """Output from the Ollama LLM Recommendations service."""
    status: str
    model_used: Optional[str] = None
    recommendation: str


class AnalysisResponse(BaseModel):
    """Full combined intelligence report returned by the /analyze endpoint."""
    disease_report: DiseaseReport
    yield_report: YieldReport
    ai_recommendation: LLMReport
