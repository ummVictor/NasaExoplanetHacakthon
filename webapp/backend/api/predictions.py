from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

router = APIRouter()

# Load models directly
models = {}
models_loaded = False

def load_models():
    global models, models_loaded
    try:
        import sys
        import os
        
        # Add backend directory to Python path so ml_utils can be found
        backend_dir = Path(__file__).parent.parent
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        
        model_path = Path(__file__).parent.parent.parent.parent / "Models"
        models['random_forest'] = joblib.load(model_path / "rf_pipeline.joblib")
        models['lightgbm'] = joblib.load(model_path / "lgb_pipeline.joblib")
        models['adaboost'] = joblib.load(model_path / "ada_pipeline.joblib")
        models['stacking'] = joblib.load(model_path / "stack_model.joblib")
        models_loaded = True
        print(f"✅ Loaded {len(models)} models successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e

class PredictionRequest(BaseModel):
    features: Dict[str, float]
    model_name: Optional[str] = "stacking"

class BatchPredictionRequest(BaseModel):
    data: List[Dict[str, float]]
    model_name: Optional[str] = "stacking"

class PredictionResponse(BaseModel):
    classification: str
    confidence_score: float
    probabilities: Dict[str, float]
    feature_importance: Dict[str, float]
    model_used: str
    prediction_reasoning: str

@router.post("/single", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Make a single prediction"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if request.model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model {request.model_name} not found")
    
    try:
        # Get the selected model
        model = models[request.model_name]
        
        # Convert features to DataFrame
        df = pd.DataFrame([request.features])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        # Handle binary classification (2 classes)
        num_classes = len(probabilities)
        if num_classes == 2:
            # Binary classification: 0 = False Positive, 1 = Confirmed
            class_mapping = {0: "False Positive", 1: "Confirmed"}
            classification = class_mapping.get(prediction, "Unknown")
            probabilities_dict = {
                "False Positive": float(probabilities[0]),
                "Confirmed": float(probabilities[1])
            }
        else:
            # Fallback for unexpected number of classes
            classification = f"Class_{prediction}"
            probabilities_dict = {f"Class_{i}": float(prob) for i, prob in enumerate(probabilities)}
        
        # Calculate confidence score
        confidence_score = float(np.max(probabilities) * 100)
        
        # Get feature importance from model only
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_names = df.columns.tolist()
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        else:
            # No fallback - return empty if model doesn't have feature importance
            feature_importance = {}
        
        # Generate reasoning
        reasoning = generate_prediction_reasoning({
            "classification": classification,
            "confidence_score": confidence_score
        }, request.features)
        
        return PredictionResponse(
            classification=classification,
            confidence_score=confidence_score,
            probabilities=probabilities_dict,
            feature_importance=feature_importance,
            model_used=request.model_name,
            prediction_reasoning=reasoning
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if request.model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model {request.model_name} not found")
    
    results = []
    for features in request.data:
        try:
            model = models[request.model_name]
            df = pd.DataFrame([features])
            prediction = model.predict(df)[0]
            probabilities = model.predict_proba(df)[0]

            num_classes = len(probabilities)
            if num_classes == 2:
                class_mapping = {0: "False Positive", 1: "Confirmed"}
                classification = class_mapping.get(prediction, "Unknown")
                probabilities_dict = {
                    "False Positive": float(probabilities[0]),
                    "Confirmed": float(probabilities[1])
                }
            else:
                classification = f"Class_{prediction}"
                probabilities_dict = {f"Class_{i}": float(prob) for i, prob in enumerate(probabilities)}

            confidence_score = float(np.max(probabilities) * 100)

            # Get feature importance from model only
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = df.columns.tolist()
                feature_importance = dict(zip(feature_names, model.feature_importances_))

            reasoning = generate_prediction_reasoning({
                "classification": classification,
                "confidence_score": confidence_score
            }, features)

            results.append(PredictionResponse(
                classification=classification,
                confidence_score=confidence_score,
                probabilities=probabilities_dict,
                feature_importance=feature_importance,
                model_used=request.model_name,
                prediction_reasoning=reasoning
            ))
        except Exception as e:
            results.append(PredictionResponse(
                classification="Error",
                confidence_score=0.0,
                probabilities={},
                feature_importance={},
                model_used=request.model_name,
                prediction_reasoning=f"Prediction failed for this entry: {str(e)}"
            ))
    return results

def generate_prediction_reasoning(result: Dict, features: Dict) -> str:
    """Generate human-readable reasoning for the prediction"""
    classification = result["classification"]
    confidence = result["confidence_score"]
    
    # Get top contributing features
    feature_importance = result.get("feature_importance", {})
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    
    reasoning_parts = []
    
    # Base reasoning based on classification
    if classification == "Confirmed":
        reasoning_parts.append("Strong evidence suggests this is a confirmed exoplanet.")
    elif classification == "Candidate":
        reasoning_parts.append("Promising candidate with some characteristics of an exoplanet.")
    else:
        reasoning_parts.append("Analysis suggests this is likely a false positive.")
    
    # Add feature-specific reasoning
    if top_features:
        top_feature_name = top_features[0][0].replace("_", " ").title()
        reasoning_parts.append(f"The {top_feature_name} parameter was most influential in this classification.")
    
    # Add confidence context
    if confidence > 80:
        reasoning_parts.append("High confidence in this classification.")
    elif confidence > 60:
        reasoning_parts.append("Moderate confidence in this classification.")
    else:
        reasoning_parts.append("Lower confidence - additional data may be helpful.")
    
    return " ".join(reasoning_parts)
