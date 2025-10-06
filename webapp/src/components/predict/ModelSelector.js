import React from "react";
import { Brain, ChevronDown } from "lucide-react";

const availableModels = [
  {
    id: "random_forest",
    name: "Random Forest"
  },
  {
    id: "lightgbm", 
    name: "LightGBM"
  },
  {
    id: "adaboost",
    name: "AdaBoost"
  },
  {
    id: "stacking",
    name: "Stacking Ensemble"
  }
];

export default function ModelSelector({ selectedModel, onModelChange, isProcessing }) {
  return (
    <div className="dark:bg-slate-800/50 bg-white backdrop-blur-sm border dark:border-slate-700/50 border-slate-200 shadow-xl rounded-2xl">
      <div className="border-b dark:border-slate-700/50 border-slate-200 p-6">
        <h3 className="flex items-center gap-2 dark:text-white text-slate-900 text-lg font-semibold">
          <Brain className="w-5 h-5 text-violet-500" />
          Select ML Model
        </h3>
        <p className="dark:text-slate-400 text-slate-600 text-sm mt-1">
          Choose which trained model to use for classification
        </p>
      </div>
      
      <div className="p-6">
        <div className="relative">
          <select
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
            disabled={isProcessing}
            className="w-full px-4 py-3 rounded-lg dark:bg-slate-900/50 bg-slate-50 dark:border-slate-600 border-slate-300 dark:text-white text-slate-900 focus:ring-2 focus:ring-violet-500 focus:border-transparent appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {availableModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 dark:text-slate-400 text-slate-500 pointer-events-none" />
        </div>
        
        {selectedModel && (
          <div className="mt-4 p-3 bg-violet-50 dark:bg-violet-900/20 rounded-lg border border-violet-200 dark:border-violet-800">
            <p className="text-sm text-violet-700 dark:text-violet-300">
              <strong>Selected:</strong> {availableModels.find(m => m.id === selectedModel)?.name}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
