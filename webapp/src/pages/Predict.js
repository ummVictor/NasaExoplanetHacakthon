import React, { useState } from "react";
import { Loader2 } from "lucide-react";
import ParameterForm from "../components/predict/ParameterForm";
// Removed FileUploader import - CSV upload removed
import PredictionResult from "../components/predict/PredictionResult";
import ModelSelector from "../components/predict/ModelSelector";

export default function PredictPage() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  // Removed inputMethod state - only manual input now
  const [selectedModel, setSelectedModel] = useState("stacking");
  const [useBackend, setUseBackend] = useState(true);
  const [confidenceThreshold, setConfidenceThreshold] = useState(70);

  const adjustClassificationByConfidence = (classification, confidenceScore) => {
    if (confidenceScore < confidenceThreshold) {
      return "Uncertain";
    }
    return classification;
  };

  const runPrediction = async (params) => {
    setIsProcessing(true);
    setResult(null);

    try {
      let predictionData;

      if (useBackend) {
        // Try to use the ML backend
        try {
          const response = await fetch('http://localhost:8000/api/predictions/single', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              features: params,
              model_name: selectedModel
            })
          });

            if (response.ok) {
              const backendResult = await response.json();
              const adjustedClassification = adjustClassificationByConfidence(
                backendResult.classification, 
                backendResult.confidence_score
              );
              predictionData = {
                ...params,
                classification: adjustedClassification,
                confidence_score: backendResult.confidence_score,
                feature_importance: backendResult.feature_importance,
                prediction_reasoning: `ML Model (${selectedModel}): ${adjustedClassification} with ${backendResult.confidence_score.toFixed(1)}% confidence (threshold: ${confidenceThreshold}%)`,
                source: "manual",
                model_used: selectedModel
              };
          } else {
            throw new Error('Backend not available');
          }
        } catch (error) {
          console.error('Backend error:', error);
          throw new Error('Backend not available - ML models required');
        }
      } else {
        throw new Error('Backend required - no fallback available');
      }

      // No need to store predictions since history is removed

      setResult(predictionData);
    } catch (error) {
      console.error('Prediction error:', error);
      setResult({
        ...params,
        classification: "Error",
        confidence_score: 0,
        feature_importance: {},
        prediction_reasoning: "An error occurred during prediction",
        source: "manual",
        model_used: 'error'
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // Enhanced rule-based classification using all CSV features
  const classifyExoplanet = (params) => {
    const { period_d, dur_h, depth_ppm, impact, ror, prad_re, a_au, a_over_rstar, 
            insol_earth, teq_k, teff_k, logg_cgs, radius_rsun, mass_msun, 
            feh_dex, mes, snr, fpflag_nt, fpflag_ss, fpflag_co, fpflag_ec } = params;
    
    let score = 0;
    let falsePositiveFlags = 0;
    
    // Check false positive flags first (most important)
    if (fpflag_nt === 1) falsePositiveFlags += 1; // Not a transit
    if (fpflag_ss === 1) falsePositiveFlags += 1; // Stellar system issue
    if (fpflag_co === 1) falsePositiveFlags += 1; // Contamination
    if (fpflag_ec === 1) falsePositiveFlags += 1; // Eclipsing binary
    
    // If any false positive flags are set, likely false positive
    if (falsePositiveFlags > 0) {
      if (falsePositiveFlags >= 2) return "False Positive";
      score -= 2; // Heavy penalty for false positive indicators
    }
    
    // Core orbital and transit parameters
    if (period_d > 0.5 && period_d < 1000) score += 1;
    if (dur_h > 0.5 && dur_h < 24) score += 1;
    if (depth_ppm > 100 && depth_ppm < 50000) score += 1;
    if (impact >= 0 && impact <= 1) score += 1;
    if (ror > 0.001 && ror < 0.5) score += 1;
    
    // Planet characteristics
    if (prad_re > 0.1 && prad_re < 20) score += 1;
    if (a_au > 0.001 && a_au < 10) score += 1;
    if (a_over_rstar > 1 && a_over_rstar < 1000) score += 1;
    if (insol_earth > 0.1 && insol_earth < 10000) score += 1;
    if (teq_k > 100 && teq_k < 3000) score += 1;
    
    // Stellar characteristics
    if (teff_k > 2000 && teff_k < 10000) score += 1;
    if (logg_cgs > 3.0 && logg_cgs < 5.0) score += 1;
    if (radius_rsun > 0.1 && radius_rsun < 10) score += 1;
    if (mass_msun > 0.1 && mass_msun < 5) score += 1;
    if (feh_dex > -2.0 && feh_dex < 1.0) score += 1;
    
    // Detection quality
    if (mes > 7.1) score += 2; // MES > 7.1 is threshold for detection
    if (snr > 5) score += 1;
    
    // Classification based on score
    if (score >= 12) return "Confirmed";
    if (score >= 8) return "Candidate";
    return "False Positive";
  };

  const calculateConfidence = (params, classification) => {
    const { mes, snr, fpflag_nt, fpflag_ss, fpflag_co, fpflag_ec } = params;
    
    let confidence = 50; // Base confidence
    
    // Boost confidence for high-quality detections
    if (mes > 7.1) confidence += 15; // Above detection threshold
    if (mes > 10) confidence += 10; // Very high MES
    if (snr > 10) confidence += 10;
    if (snr > 20) confidence += 5;
    
    // Penalize for false positive flags
    if (fpflag_nt === 1) confidence -= 20;
    if (fpflag_ss === 1) confidence -= 15;
    if (fpflag_co === 1) confidence -= 15;
    if (fpflag_ec === 1) confidence -= 25; // Eclipsing binary is strong false positive indicator
    
    // Adjust based on classification
    if (classification === "Confirmed") confidence += 15;
    if (classification === "Candidate") confidence += 5;
    if (classification === "False Positive") confidence -= 10;
    
    return Math.min(Math.max(confidence, 15), 95);
  };

  // Feature importance now only comes from ML models - no fallback calculation

  const generateReasoning = (params, classification) => {
    const { period_d, dur_h, depth_ppm, mes, snr, fpflag_nt, fpflag_ss, fpflag_co, fpflag_ec } = params;
    
    let reasoning = `Based on comprehensive analysis of ${Object.keys(params).length} astronomical parameters: `;
    
    // Check false positive flags first
    const fpflags = [];
    if (fpflag_nt === 1) fpflags.push("not a transit");
    if (fpflag_ss === 1) fpflags.push("stellar system issue");
    if (fpflag_co === 1) fpflags.push("contamination");
    if (fpflag_ec === 1) fpflags.push("eclipsing binary");
    
    if (fpflags.length > 0) {
      reasoning += `False positive indicators detected: ${fpflags.join(", ")}. `;
    }
    
    if (classification === "Confirmed") {
      reasoning += `The orbital period of ${period_d} days, transit duration of ${dur_h} hours, and transit depth of ${depth_ppm} ppm are consistent with confirmed exoplanet characteristics. `;
    } else if (classification === "Candidate") {
      reasoning += `The parameters show promising characteristics but require additional verification. `;
    } else {
      reasoning += `The parameter values suggest this is likely a false positive detection. `;
    }
    
    // Detection quality assessment
    if (mes > 7.1) {
      reasoning += `The Multiple Event Statistic (${mes}) exceeds the detection threshold, indicating strong signal confidence. `;
    } else {
      reasoning += `The Multiple Event Statistic (${mes}) is below the detection threshold. `;
    }
    
    if (snr > 10) {
      reasoning += `The high signal-to-noise ratio (${snr}) indicates excellent detection quality.`;
    } else if (snr > 5) {
      reasoning += `The signal-to-noise ratio (${snr}) suggests good detection quality.`;
    } else {
      reasoning += `The signal-to-noise ratio (${snr}) indicates moderate detection quality.`;
    }
    
    return reasoning;
  };

  return (
    <div className="min-h-screen dark:bg-gradient-to-br dark:from-slate-950 dark:via-slate-900 dark:to-indigo-950 bg-gradient-to-br from-slate-50 to-indigo-50 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
              <h1 className="text-4xl font-bold dark:text-white text-slate-900 mb-3">
                CodeLock - Exoplanet Classification
              </h1>
          <p className="dark:text-slate-400 text-slate-600 text-lg">
            Predict whether a celestial body is a confirmed exoplanet, candidate, or false positive
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          <div className="space-y-6">
            <ModelSelector 
              selectedModel={selectedModel} 
              onModelChange={setSelectedModel} 
              isProcessing={isProcessing} 
            />
            
            <div className="dark:bg-slate-800/50 bg-white backdrop-blur-sm border dark:border-slate-700/50 border-slate-200 shadow-xl rounded-2xl">
              <div className="border-b dark:border-slate-700/50 border-slate-200 p-6">
                <h3 className="dark:text-white text-slate-900 text-lg font-semibold">
                  Confidence Threshold
                </h3>
              </div>
              <div className="p-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="dark:text-slate-300 text-slate-700 font-medium">
                      Threshold: {confidenceThreshold}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseInt(e.target.value))}
                    disabled={isProcessing}
                    className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
                    style={{
                      background: `linear-gradient(to right, #8b5cf6 0%, #8b5cf6 ${confidenceThreshold}%, #e2e8f0 ${confidenceThreshold}%, #e2e8f0 100%)`
                    }}
                  />
                  <div className="flex justify-between text-xs dark:text-slate-500 text-slate-500">
                    <span>0%</span>
                    <span>50%</span>
                    <span>100%</span>
                  </div>
                </div>
              </div>
            </div>
            
            <ParameterForm onSubmit={runPrediction} isProcessing={isProcessing} />
          </div>

          <div>
            {isProcessing && (
              <div className="dark:bg-slate-800/50 bg-white backdrop-blur-sm rounded-2xl p-8 border dark:border-slate-700/50 border-slate-200 shadow-xl">
                <div className="flex flex-col items-center justify-center h-64">
                  <Loader2 className="w-12 h-12 dark:text-violet-400 text-violet-600 animate-spin mb-4" />
                  <p className="dark:text-slate-300 text-slate-700 font-medium">Analyzing parameters...</p>
                  <p className="dark:text-slate-500 text-slate-500 text-sm mt-2">Running AI classification model</p>
                </div>
              </div>
            )}

            {!isProcessing && result && (
              <PredictionResult result={result} />
            )}

            {!isProcessing && !result && (
              <div className="dark:bg-slate-800/30 bg-white/50 backdrop-blur-sm rounded-2xl p-8 border dark:border-slate-700/30 border-slate-200">
                <div className="flex flex-col items-center justify-center h-64 text-center">
                  <div className="w-20 h-20 dark:bg-violet-500/10 bg-violet-100 rounded-full flex items-center justify-center mb-4">
                    <svg className="w-10 h-10 dark:text-violet-400 text-violet-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <p className="dark:text-slate-400 text-slate-600 mb-2">No prediction yet</p>
                  <p className="dark:text-slate-500 text-slate-500 text-sm">Enter parameters to start classification</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
