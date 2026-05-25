"""
AgroPredict - FastAPI Backend
==============================
Main application that orchestrates all AI models into a single REST API.

Endpoints:
    GET  /health   - Health check
    POST /analyze  - Full pipeline: Disease Detection → Yield Prediction → LLM Recommendations
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import uuid
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from schemas import EnvironmentalData, AnalysisResponse, DiseaseReport, YieldReport, LLMReport
from model_a import predict_image
from model_cRF import predict_yield, calculate_final_yield
from llm_service import get_recommendations

# ==========================================
# App Initialization
# ==========================================
app = FastAPI(
    title="AgroPredict API",
    description="AI-Driven Maize Disease Diagnostics & Yield Estimation",
    version="1.0.0",
)

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Dashboard UI (serves index.html at root)
# ==========================================
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the AgroPredict frontend dashboard."""
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)

@app.get("/data_flow", response_class=HTMLResponse)
def serve_data_flow():
    """Serves the visually animated Data Flow explanation page."""
    html_path = Path(__file__).parent / "data_flow.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)

# ==========================================
# Health Check
# ==========================================
@app.get("/health")
def health_check():
    """Returns a simple status to verify the API is running."""
    return {
        "status": "online",
        "models": {
            "model_a": os.path.exists("model_a_weights.pth"),
            "model_cRF": os.path.exists("model_cRF_weights.pkl"),
        }
    }

# ==========================================
# Full Analysis Pipeline
# ==========================================
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    image: UploadFile = File(..., description="Leaf image for disease detection (JPG/PNG)"),
    env_data: str = Form(..., description="JSON string of environmental data")
):
    """
    Runs the complete AgroPredict pipeline:
      1. Model A + B → Disease classification, DSI, severity
      2. Model C (RF) + D → Ideal yield → Health-adjusted final yield
      3. LLM (Ollama) → AI-generated agronomic recommendations
      
    The uploaded image is processed in-memory and instantly deleted.
    """
    import json

    # --------------------------------------------------
    # 1. Parse environmental data from the JSON string
    # --------------------------------------------------
    try:
        env_dict = json.loads(env_data)
        env = EnvironmentalData(**env_dict)
    except (json.JSONDecodeError, Exception) as e:
        raise HTTPException(status_code=422, detail=f"Invalid environmental data JSON: {str(e)}")

    # --------------------------------------------------
    # 2. Save uploaded image to a temp file, run Model A, then delete
    # --------------------------------------------------
    ext = os.path.splitext(image.filename)[1] or ".jpg"
    tmp_path = os.path.join(tempfile.gettempdir(), f"agro_{uuid.uuid4().hex}{ext}")

    try:
        # Write uploaded bytes to temp file
        contents = await image.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        # Run Path A (Model A + Model B)
        disease_result = predict_image(tmp_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model A error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease detection failed: {str(e)}")
    finally:
        # Instantly delete the temp image
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # --------------------------------------------------
    # 3. Run Path B (Model C RF → Model D)
    # --------------------------------------------------
    try:
        yield_result = predict_yield(
            n=env.n, p=env.p, k=env.k,
            soil_ph=env.soil_ph,
            soil_moisture=env.soil_moisture,
            temperature=env.temperature,
            humidity=env.humidity,
            rainfall=env.rainfall,
            sunlight_hours=env.sunlight_hours
        )
        ideal = yield_result["ideal_yield"]

        final_result = calculate_final_yield(
            ideal_yield=ideal,
            dsi=disease_result["dsi"]
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Yield model error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yield prediction failed: {str(e)}")

    # --------------------------------------------------
    # 4. Run LLM Recommendations (Ollama)
    # --------------------------------------------------
    soil_data = {
        "N": env.n, "P": env.p, "K": env.k,
        "Soil_pH": env.soil_ph, "Soil_Moisture": env.soil_moisture
    }
    weather_data = {
        "Rainfall": env.rainfall, "Temperature": env.temperature,
        "Humidity": env.humidity, "Sunlight_Hours": env.sunlight_hours
    }

    llm_result = get_recommendations(
        disease_name=disease_result["disease_name"],
        confidence=disease_result["confidence_score"],
        dsi=disease_result["dsi"],
        severity=disease_result["severity"],
        ideal_yield=final_result["ideal_yield"],
        final_yield=final_result["final_yield"],
        yield_loss_pct=final_result["yield_loss_percentage"],
        soil_data=soil_data,
        weather_data=weather_data
    )

    # --------------------------------------------------
    # 5. Build and return the unified response
    # --------------------------------------------------
    return AnalysisResponse(
        disease_report=DiseaseReport(
            disease_name=disease_result["disease_name"],
            confidence_score=disease_result["confidence_score"],
            dsi=disease_result["dsi"],
            affected_area_percent=disease_result["affected_area_percent"],
            severity=disease_result["severity"]
        ),
        yield_report=YieldReport(
            ideal_yield=final_result["ideal_yield"],
            final_yield=final_result["final_yield"],
            yield_loss=final_result["yield_loss"],
            yield_loss_percentage=final_result["yield_loss_percentage"],
            dsi_used=final_result["dsi_used"],
            crop_sensitivity_K=final_result["crop_sensitivity_K"]
        ),
        ai_recommendation=LLMReport(
            status=llm_result["status"],
            model_used=llm_result.get("model_used"),
            recommendation=llm_result["recommendation"]
        )
    )


# ==========================================
# Run with: python main.py
# ==========================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    is_production = "PORT" in os.environ
    uvicorn.run(
        "main:app",
        host="0.0.0.0" if is_production else "localhost",
        port=port,
        reload=not is_production
    )
