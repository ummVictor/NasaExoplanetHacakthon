import React from "react";
import { CheckCircle, AlertCircle, XCircle, TrendingUp } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

export default function PredictionResult({ result }) {
  const getClassificationColor = (classification) => {
    switch (classification) {
      case "Confirmed":
        return { bg: "bg-green-500", text: "text-green-500", icon: CheckCircle };
      case "Candidate":
        return { bg: "bg-yellow-500", text: "text-yellow-500", icon: AlertCircle };
      case "False Positive":
        return { bg: "bg-red-500", text: "text-red-500", icon: XCircle };
      default:
        return { bg: "bg-gray-500", text: "text-gray-500", icon: AlertCircle };
    }
  };

  const classColor = getClassificationColor(result.classification);
  const ClassIcon = classColor.icon;

  const featureData = Object.entries(result.feature_importance || {}).map(([key, value]) => ({
    name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    importance: (value * 100).toFixed(1),
    rawValue: value
  })).sort((a, b) => b.rawValue - a.rawValue);

  const getBarColor = (index) => {
    const colors = ['#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe'];
    return colors[index % colors.length];
  };

  return (
    <div className="space-y-6">
      <div className="dark:bg-slate-800/50 bg-white backdrop-blur-sm border dark:border-slate-700/50 border-slate-200 shadow-xl overflow-hidden rounded-2xl">
        <div className={`h-2 ${classColor.bg}`} />
        <div className="border-b dark:border-slate-700/50 border-slate-200 p-6">
          <h3 className="flex items-center justify-between dark:text-white text-slate-900 text-lg font-semibold">
            <span>Classification Result</span>
            <ClassIcon className={`w-6 h-6 ${classColor.text}`} />
          </h3>
        </div>
        <div className="p-6">
          <div className="space-y-6">
            <div className="text-center py-6">
              <span className={`${classColor.bg} text-white text-lg px-6 py-2 rounded-lg mb-4 inline-block`}>
                {result.classification}
              </span>
              <div className="mt-4">
                <p className="dark:text-slate-400 text-slate-600 text-sm mb-2">Confidence Score</p>
                <div className="flex items-center justify-center gap-3">
                  <div className="text-4xl font-bold dark:text-white text-slate-900">
                    {result.confidence_score.toFixed(1)}%
                  </div>
                  <TrendingUp className={`w-6 h-6 ${classColor.text}`} />
                </div>
              </div>
            </div>

            <div className="dark:bg-slate-900/30 bg-slate-100 rounded-lg p-4">
              <p className="text-sm font-medium dark:text-slate-300 text-slate-700 mb-2">Analysis</p>
              <p className="text-sm dark:text-slate-400 text-slate-600 leading-relaxed">
                {result.prediction_reasoning}
              </p>
            </div>

            <div>
              <p className="text-sm font-medium dark:text-slate-300 text-slate-700 mb-1">Input Parameters</p>
              <div className="grid grid-cols-2 gap-3 mt-3">
                <div className="dark:bg-slate-900/30 bg-slate-100 rounded-lg p-3">
                  <p className="text-xs dark:text-slate-500 text-slate-500">Orbital Period</p>
                  <p className="font-semibold dark:text-white text-slate-900">{result.orbital_period} days</p>
                </div>
                <div className="dark:bg-slate-900/30 bg-slate-100 rounded-lg p-3">
                  <p className="text-xs dark:text-slate-500 text-slate-500">Planet Radius</p>
                  <p className="font-semibold dark:text-white text-slate-900">{result.planet_radius} R⊕</p>
                </div>
                <div className="dark:bg-slate-900/30 bg-slate-100 rounded-lg p-3">
                  <p className="text-xs dark:text-slate-500 text-slate-500">Transit Depth</p>
                  <p className="font-semibold dark:text-white text-slate-900">{result.transit_depth} ppm</p>
                </div>
                <div className="dark:bg-slate-900/30 bg-slate-100 rounded-lg p-3">
                  <p className="text-xs dark:text-slate-500 text-slate-500">S/N Ratio</p>
                  <p className="font-semibold dark:text-white text-slate-900">{result.signal_to_noise}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="dark:bg-slate-800/50 bg-white backdrop-blur-sm border dark:border-slate-700/50 border-slate-200 shadow-xl rounded-2xl">
        <div className="border-b dark:border-slate-700/50 border-slate-200 p-6">
          <h3 className="dark:text-white text-slate-900 text-lg font-semibold">Feature Importance</h3>
        </div>
        <div className="p-6">
          {featureData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={featureData} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
                <XAxis type="number" stroke="#94a3b8" />
                <YAxis dataKey="name" type="category" stroke="#94a3b8" width={110} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: '8px',
                    color: '#e2e8f0'
                  }}
                />
                <Bar dataKey="importance" radius={[0, 8, 8, 0]}>
                  {featureData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getBarColor(index)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <p className="dark:text-slate-400 text-slate-600 mb-2">Feature Importance Not Available</p>
                <p className="dark:text-slate-500 text-slate-500 text-sm">
                  This model doesn't provide feature importance data
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
