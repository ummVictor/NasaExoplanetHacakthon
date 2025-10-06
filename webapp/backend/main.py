from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os
from pathlib import Path

from backend.api.predictions import router as predictions_router, load_models as load_prediction_models
from backend.api.models import router as models_router

# Initialize FastAPI app
app = FastAPI(
    title="CodeLock API",
    description="AI-powered exoplanet classification using NASA datasets",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
load_prediction_models()

# Include routers
app.include_router(predictions_router, prefix="/api/predictions", tags=["predictions"])
app.include_router(models_router, prefix="/api/models", tags=["models"])

@app.get("/")
async def root():
    return {
        "message": "CodeLock API",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    from backend.api.predictions import models_loaded, models
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "available_models": list(models.keys()) if models_loaded else []
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
