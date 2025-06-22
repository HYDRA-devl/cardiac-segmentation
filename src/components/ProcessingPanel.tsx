'use client';

import React from 'react';
import { Sparkles, Zap, Eye, Target } from 'lucide-react';
import { ProcessingState } from '@/types/pipeline';
import { PIPELINE_CONFIG } from '@/lib/config';

interface ProcessingPanelProps {
  processingState: ProcessingState;
  hasResults: boolean;
}

const STEP_ICONS = {
  Sparkles,
  Zap,
  Eye,
  Target,
};

export function ProcessingPanel({ processingState, hasResults }: ProcessingPanelProps) {
  const { isProcessing, currentStep } = processingState;

  return (
    <div className="grid grid-cols-5 gap-2 md:gap-3">
      {PIPELINE_CONFIG.steps.map((step, index) => {
        const IconComponent = STEP_ICONS[step.icon as keyof typeof STEP_ICONS];
        const isActive = isProcessing && currentStep === index;
        const isCompleted = isProcessing ? currentStep > index : hasResults;
        
        return (
          <div key={index} className="text-center">
            <div 
              className={`
                relative w-10 h-10 md:w-12 md:h-12 mx-auto rounded-full flex items-center justify-center mb-2 transition-all duration-500
                ${isActive 
                  ? `bg-gradient-to-r ${step.color} scale-110 shadow-lg` 
                  : isCompleted
                  ? 'bg-green-500 shadow-md'
                  : 'bg-slate-600/50'
                }
              `}
            >
              <IconComponent 
                className={`w-5 h-5 md:w-6 md:h-6 ${(isActive || isCompleted) ? 'text-white' : 'text-slate-300'}`} 
              />
              {isActive && (
                <>
                  <div className="absolute inset-0 rounded-full border-4 border-white/30 animate-ping" />
                  <div className="absolute inset-0 rounded-full border-2 border-white/50" />
                </>
              )}
              {isCompleted && !isActive && (
                <div className="absolute -top-1 -right-1 w-4 h-4 md:w-5 md:h-5 bg-white rounded-full flex items-center justify-center">
                  <div className="w-1.5 h-1.5 md:w-2 md:h-2 bg-green-500 rounded-full" />
                </div>
              )}
            </div>
            
            <p className={`text-xs font-medium transition-colors mb-1 ${
              (isActive || isCompleted) ? 'text-white' : 'text-slate-300'
            }`}>
              {step.name}
            </p>
            
            {isActive && (
              <div className="mt-2">
                <div className="w-full bg-slate-600 rounded-full h-1">
                  <div className="bg-blue-400 h-1 rounded-full animate-pulse transition-all duration-1000" style={{ width: '60%' }} />
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}