import React, { useState } from "react";
import { Search } from "lucide-react";

const defaultValues = {
  period_d: 3.52,
  dur_h: 2.5,
  depth_ppm: 1200,
  impact: 0.3,
  ror: 0.1,
  prad_re: 1.08,
  a_au: 0.05,
  a_over_rstar: 10.0,
  insol_earth: 388,
  teq_k: 1500,
  teff_k: 5777,
  logg_cgs: 4.4,
  radius_rsun: 1.0,
  mass_msun: 1.0,
  feh_dex: 0.0,
  mes: 15.0,        // High MES - above detection threshold
  snr: 15.3,        // High SNR - good signal quality
  fpflag_nt: 0,     // Not flagged as non-transit
  fpflag_ss: 0,     // No stellar system issues
  fpflag_co: 0,     // No contamination
  fpflag_ec: 0      // Not an eclipsing binary
};

export default function ParameterForm({ onSubmit, isProcessing }) {
  const [params, setParams] = useState(defaultValues);

  const handleChange = (field, value) => {
    // Allow any input - no restrictions
    setParams(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Convert to numbers for submission, defaulting to 0 for invalid inputs
    const submitParams = Object.fromEntries(
      Object.entries(params).map(([key, value]) => [key, parseFloat(value) || 0])
    );
    onSubmit(submitParams);
  };

  const parameters = [
    // Core orbital and transit parameters
    { key: "period_d", label: "Orbital Period", unit: "days", step: "0.01", min: "0.1", max: "1000" },
    { key: "dur_h", label: "Transit Duration", unit: "hours", step: "0.1", min: "0.5", max: "24" },
    { key: "depth_ppm", label: "Transit Depth", unit: "ppm", step: "1", min: "1", max: "100000" },
    { key: "impact", label: "Impact Parameter", unit: "", step: "0.01", min: "0", max: "1" },
    { key: "ror", label: "Planet-to-Star Radius Ratio", unit: "", step: "0.001", min: "0.001", max: "0.5" },
    
    // Planet characteristics
    { key: "prad_re", label: "Planet Radius", unit: "R⊕", step: "0.01", min: "0.1", max: "50" },
    { key: "a_au", label: "Semi-major Axis", unit: "AU", step: "0.001", min: "0.001", max: "10" },
    { key: "a_over_rstar", label: "Semi-major Axis / Stellar Radius", unit: "", step: "0.1", min: "1", max: "1000" },
    { key: "insol_earth", label: "Insolation Flux", unit: "S⊕", step: "0.1", min: "0.1", max: "10000" },
    { key: "teq_k", label: "Equilibrium Temperature", unit: "K", step: "1", min: "100", max: "3000" },
    
    // Stellar characteristics
    { key: "teff_k", label: "Stellar Temperature", unit: "K", step: "1", min: "2000", max: "10000" },
    { key: "logg_cgs", label: "Stellar Surface Gravity", unit: "log g", step: "0.1", min: "3.0", max: "5.0" },
    { key: "radius_rsun", label: "Stellar Radius", unit: "R☉", step: "0.01", min: "0.1", max: "10" },
    { key: "mass_msun", label: "Stellar Mass", unit: "M☉", step: "0.01", min: "0.1", max: "5" },
    { key: "feh_dex", label: "Stellar Metallicity", unit: "[Fe/H]", step: "0.1", min: "-2.0", max: "1.0" },
    
    // Detection quality
    { key: "mes", label: "Multiple Event Statistic", unit: "", step: "0.1", min: "1", max: "1000" },
    { key: "snr", label: "Signal-to-Noise Ratio", unit: "", step: "0.1", min: "1", max: "1000" },
    
    // False positive flags (binary)
    { key: "fpflag_nt", label: "Not Transit Flag", unit: "0/1", step: "1", min: "0", max: "1" },
    { key: "fpflag_ss", label: "Stellar System Flag", unit: "0/1", step: "1", min: "0", max: "1" },
    { key: "fpflag_co", label: "Contamination Flag", unit: "0/1", step: "1", min: "0", max: "1" },
    { key: "fpflag_ec", label: "Eclipsing Binary Flag", unit: "0/1", step: "1", min: "0", max: "1" }
  ];

  return (
    <div className="dark:bg-slate-800/50 bg-white backdrop-blur-sm border dark:border-slate-700/50 border-slate-200 shadow-xl rounded-2xl">
      <div className="border-b dark:border-slate-700/50 border-slate-200 p-6">
        <h3 className="flex items-center gap-2 dark:text-white text-slate-900 text-lg font-semibold">
          <Search className="w-5 h-5 text-violet-500" />
          Input Parameters
        </h3>
      </div>
      <div className="p-6">
        <form onSubmit={handleSubmit}>
          <div className="max-h-96 overflow-y-auto pr-2 space-y-4 mb-6">
            {parameters.map(param => (
              <div key={param.key} className="space-y-2">
                <label className="dark:text-slate-300 text-slate-700 font-medium block">
                  {param.label} {param.unit && <span className="dark:text-slate-500 text-slate-500">({param.unit})</span>}
                </label>
                <input
                  type="number"
                  value={params[param.key]}
                  onChange={(e) => handleChange(param.key, e.target.value)}
                  className="w-full px-3 py-2 rounded-lg dark:bg-slate-900/50 bg-slate-50 dark:border-slate-600 border-slate-300 dark:text-white text-slate-900 focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                />
              </div>
            ))}
          </div>

          <button
            type="submit"
            disabled={isProcessing}
            className="w-full bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white py-6 text-lg font-semibold shadow-lg shadow-violet-500/30 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isProcessing ? "Processing..." : "Classify Exoplanet"}
          </button>
        </form>
      </div>
    </div>
  );
}
