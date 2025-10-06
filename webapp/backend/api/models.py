from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter()

@router.get("/info")
async def get_model_info():
    """Get information about loaded models"""
    from backend.api.predictions import models_loaded, models
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "models_loaded": models_loaded,
        "available_models": list(models.keys()),
        "primary_model": "stacking"
    }

@router.get("/available")
async def get_available_models():
    """Get list of available models"""
    from backend.api.predictions import models_loaded, models
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "models": list(models.keys()),
        "primary_model": "stacking",
        "recommended": "stacking"
    }

@router.get("/performance")
async def get_model_performance():
    """Get model performance metrics (placeholder for now)"""
    # In a real implementation, you'd load these from your training reports
    return {
        "stacking": {
            "accuracy": 0.89,
            "precision": 0.87,
            "recall": 0.85,
            "f1_score": 0.86,
            "description": "Ensemble model combining multiple algorithms"
        },
        "random_forest": {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.81,
            "f1_score": 0.82,
            "description": "Random Forest with 300 trees"
        },
        "lightgbm": {
            "accuracy": 0.87,
            "precision": 0.85,
            "recall": 0.83,
            "f1_score": 0.84,
            "description": "LightGBM gradient boosting"
        },
        "adaboost": {
            "accuracy": 0.82,
            "precision": 0.80,
            "recall": 0.78,
            "f1_score": 0.79,
            "description": "AdaBoost with decision trees"
        }
    }

@router.get("/features")
async def get_feature_info():
    """Get information about model features"""
    feature_descriptions = {
        "period_d": {
            "name": "Orbital Period",
            "unit": "days",
            "description": "Time for one complete orbit around the star",
            "typical_range": "0.5 - 1000 days"
        },
        "depth_ppm": {
            "name": "Transit Depth",
            "unit": "ppm",
            "description": "Fractional decrease in stellar brightness during transit",
            "typical_range": "10 - 10000 ppm"
        },
        "prad_re": {
            "name": "Planet Radius",
            "unit": "R⊕",
            "description": "Planet radius in Earth radii",
            "typical_range": "0.1 - 20 R⊕"
        },
        "teff_k": {
            "name": "Stellar Temperature",
            "unit": "K",
            "description": "Effective temperature of the host star",
            "typical_range": "3000 - 10000 K"
        },
        "radius_rsun": {
            "name": "Stellar Radius",
            "unit": "R☉",
            "description": "Radius of the host star in solar radii",
            "typical_range": "0.1 - 10 R☉"
        },
        "teq_k": {
            "name": "Equilibrium Temperature",
            "unit": "K",
            "description": "Planet's equilibrium temperature",
            "typical_range": "100 - 3000 K"
        },
        "insol_earth": {
            "name": "Insolation Flux",
            "unit": "S⊕",
            "description": "Stellar flux received by the planet",
            "typical_range": "0.1 - 10000 S⊕"
        },
        "snr": {
            "name": "Signal-to-Noise Ratio",
            "unit": "",
            "description": "Quality of the transit detection",
            "typical_range": "1 - 100"
        }
    }
    
    return {
        "features": feature_descriptions,
        "feature_count": len(feature_descriptions),
        "required_features": list(feature_descriptions.keys())
    }
