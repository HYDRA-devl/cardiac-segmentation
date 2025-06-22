'use client';

import React from 'react';
import { Zap, Eye, Sparkles } from 'lucide-react';
import { ProcessingParameters } from '@/types/pipeline';

interface ParametersPanelProps {
  parameters: ProcessingParameters;
  onParameterChange: (params: Partial<ProcessingParameters>) => void;
  disabled?: boolean;
}

interface SliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  icon: React.ComponentType<{ className?: string }>;
  disabled?: boolean;
  min?: number;
  max?: number;
}

function ParameterSlider({ 
  label, 
  value, 
  onChange, 
  icon: Icon, 
  disabled = false,
  min = 0,
  max = 100 
}: SliderProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Icon className={`w-4 h-4 ${disabled ? 'text-slate-300' : 'text-slate-400'}`} />
          <span className={`text-sm font-medium ${disabled ? 'text-slate-400' : 'text-slate-700'}`}>
            {label}
          </span>
        </div>
        <span className={`text-sm ${disabled ? 'text-slate-400' : 'text-slate-500'}`}>
          {value}%
        </span>
      </div>
      
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          value={value}
          onChange={(e) => onChange(parseInt(e.target.value))}
          disabled={disabled}
          className={`w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer slider transition-opacity ${
            disabled ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        />
        <div 
          className={`absolute top-0 left-0 h-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg pointer-events-none transition-all ${
            disabled ? 'opacity-50' : ''
          }`}
          style={{ width: `${(value - min) / (max - min) * 100}%` }}
        />
      </div>
    </div>
  );
}

export function ParametersPanel({ 
  parameters, 
  onParameterChange, 
  disabled = false 
}: ParametersPanelProps) {
  return (
    <div className="space-y-6">
      <ParameterSlider
        label="Contraste"
        value={parameters.contrast}
        onChange={(value) => onParameterChange({ contrast: value })}
        icon={Zap}
        disabled={disabled}
      />
      
      <ParameterSlider
        label="Luminosité"
        value={parameters.brightness}
        onChange={(value) => onParameterChange({ brightness: value })}
        icon={Eye}
        disabled={disabled}
      />
      
      <ParameterSlider
        label="Réduction du Bruit"
        value={parameters.noiseReduction}
        onChange={(value) => onParameterChange({ noiseReduction: value })}
        icon={Sparkles}
        disabled={disabled}
      />

      {/* Preset buttons */}
      <div className="pt-4 border-t border-slate-200">
        <p className="text-sm font-medium text-slate-700 mb-3">Préréglages</p>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => onParameterChange({ contrast: 30, brightness: 40, noiseReduction: 60 })}
            disabled={disabled}
            className={`px-3 py-2 text-xs rounded-lg border transition-colors ${
              disabled 
                ? 'border-slate-200 text-slate-400 cursor-not-allowed'
                : 'border-slate-300 text-slate-600 hover:border-blue-300 hover:bg-blue-50'
            }`}
          >
            Conservateur
          </button>
          <button
            onClick={() => onParameterChange({ contrast: 70, brightness: 60, noiseReduction: 80 })}
            disabled={disabled}
            className={`px-3 py-2 text-xs rounded-lg border transition-colors ${
              disabled 
                ? 'border-slate-200 text-slate-400 cursor-not-allowed'
                : 'border-slate-300 text-slate-600 hover:border-blue-300 hover:bg-blue-50'
            }`}
          >
            Agressif
          </button>
        </div>
      </div>
    </div>
  );
}
