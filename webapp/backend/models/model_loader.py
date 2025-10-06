import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import os

class ModelLoader:
    def __init__(self):
        self.models = {}
        self.models_loaded = False
        self.model_path = Path(__file__).parent.parent.parent / "Models"
        self.feature_columns = [
            'period_d', 'dur_h', 'depth_ppm', 'impact', 'ror', 'prad_re', 
            'a_au', 'a_over_rstar', 'insol_earth', 'teq_k', 'teff_k', 
            'logg_cgs', 'radius_rsun', 'mass_msun', 'feh_dex', 'mes', 'snr',
            'fpflag_nt', 'fpflag_ss', 'fpflag_co', 'fpflag_ec'
        ]
    
    async def load_models(self):
        """Load all trained models"""
        try:
            # Load individual models
            self.models['random_forest'] = joblib.load(self.model_path / "rf_pipeline.joblib")
            self.models['lightgbm'] = joblib.load(self.model_path / "lgb_pipeline.joblib")
            self.models['adaboost'] = joblib.load(self.model_path / "ada_pipeline.joblib")
            self.models['stacking'] = joblib.load(self.model_path / "stack_model.joblib")
            
            # Set primary model (stacking ensemble for best performance)
            self.primary_model = self.models['stacking']
            
            self.models_loaded = True
            print(f"✅ Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise e
    
    def preprocess_features(self, features: Dict[str, float]) -> np.ndarray:
        """Preprocess features to match training format"""
        # Create DataFrame with expected column order
        df = pd.DataFrame([features])
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Reorder columns to match training
        df = df[self.feature_columns]
        
        return df.values
    
    def predict_single(self, features: Dict[str, float], model_name: str = "stacking") -> Dict[str, Any]:
        """Make prediction for single sample"""
        if not self.models_loaded:
            raise ValueError("Models not loaded")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        X = self.preprocess_features(features)
        
        # Get prediction and probabilities
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Map numeric labels to class names
        class_mapping = {0: "False Positive", 1: "Candidate", 2: "Confirmed"}
        classification = class_mapping.get(prediction, "Unknown")
        
        # Calculate confidence score (max probability * 100)
        confidence_score = float(np.max(probabilities) * 100)
        
        # Get feature importance if available
        feature_importance = self.get_feature_importance(model, X)
        
        return {
            "classification": classification,
            "confidence_score": confidence_score,
            "probabilities": {
                "False Positive": float(probabilities[0]),
                "Candidate": float(probabilities[1]),
                "Confirmed": float(probabilities[2])
            },
            "feature_importance": feature_importance,
            "model_used": model_name
        }
    
    def _rule_based_classification(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Fallback rule-based classification when ML models aren't available"""
        # Simple rule-based logic
        fpflags = [features.get('fpflag_nt', 0), features.get('fpflag_ss', 0), 
                  features.get('fpflag_co', 0), features.get('fpflag_ec', 0)]
        fpflags_sum = sum(fpflags)
        
        mes = features.get('mes', 0)
        snr = features.get('snr', 0)
        
        if fpflags_sum > 0:
            classification = "False Positive"
            confidence_score = 85.0
        elif mes > 7.1 and snr > 10:
            classification = "Confirmed"
            confidence_score = 80.0
        elif mes > 5.0:
            classification = "Candidate"
            confidence_score = 65.0
        else:
            classification = "False Positive"
            confidence_score = 70.0
        
        # Simple feature importance based on values
        feature_importance = {k: abs(v) / 100 for k, v in features.items() if isinstance(v, (int, float))}
        
        return {
            "classification": classification,
            "confidence_score": confidence_score,
            "probabilities": {
                "False Positive": 0.7 if classification == "False Positive" else 0.1,
                "Candidate": 0.7 if classification == "Candidate" else 0.1,
                "Confirmed": 0.7 if classification == "Confirmed" else 0.1
            },
            "feature_importance": feature_importance,
            "model_used": "rule-based"
        }
    
    def predict_batch(self, features_list: List[Dict[str, float]], model_name: str = "stacking") -> List[Dict[str, Any]]:
        """Make predictions for multiple samples"""
        results = []
        for features in features_list:
            try:
                result = self.predict_single(features, model_name)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "classification": "Error",
                    "confidence_score": 0.0
                })
        return results
    
    def get_feature_importance(self, model, X: np.ndarray) -> Dict[str, float]:
        """Extract feature importance from model"""
        try:
            # Try to get feature importance from the model
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'named_steps') and 'clf' in model.named_steps:
                # For pipeline models, get importance from the classifier
                clf = model.named_steps['clf']
                if hasattr(clf, 'feature_importances_'):
                    importances = clf.feature_importances_
                else:
                    # For stacking models, use equal weights
                    importances = np.ones(len(self.feature_columns)) / len(self.feature_columns)
            else:
                # Default to equal weights
                importances = np.ones(len(self.feature_columns)) / len(self.feature_columns)
            
            # Normalize to 0-1 range
            importances = importances / np.sum(importances)
            
            return dict(zip(self.feature_columns, importances.tolist()))
            
        except Exception:
            # Return equal weights if feature importance not available
            equal_weights = 1.0 / len(self.feature_columns)
            return {col: equal_weights for col in self.feature_columns}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        if not self.models_loaded:
            return {"error": "Models not loaded"}
        
        return {
            "models_loaded": self.models_loaded,
            "available_models": list(self.models.keys()),
            "primary_model": "stacking",
            "feature_columns": self.feature_columns,
            "model_count": len(self.models)
        }
