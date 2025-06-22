'use client';

import React from 'react';
import { Play } from 'lucide-react';

interface ProcessButtonProps {
  onProcess: () => void;
  canProcess: boolean;
  isProcessing: boolean;
}

export function ProcessButton({ onProcess, canProcess, isProcessing }: ProcessButtonProps) {
  return (
    <button
      onClick={onProcess}
      disabled={!canProcess}
      className={`
        w-full py-4 px-6 rounded-xl font-semibold text-white transition-all duration-300 shadow-lg
        ${canProcess
          ? 'bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 hover:shadow-xl transform hover:-translate-y-1'
          : 'bg-slate-300 cursor-not-allowed'
        }
      `}
    >
      {isProcessing ? (
        <div className="flex items-center justify-center space-x-2">
          <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
          <span>Traitement en cours...</span>
        </div>
      ) : (
        <div className="flex items-center justify-center space-x-2">
          <Play className="w-5 h-5" />
          <span>Appliquer et Analyser</span>
        </div>
      )}
    </button>
  );
}
