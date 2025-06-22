'use client';

import React from 'react';
import { Target, Download } from 'lucide-react';
import { ProcessingResults } from '@/types/pipeline';
import { formatNumber, formatTime } from '@/lib/utils';

interface ResultsPanelProps {
  results: ProcessingResults;
  onDownload: (type: 'processed' | 'segmentation') => void;
}

export function ResultsPanel({ results, onDownload }: ResultsPanelProps) {
  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 animate-fade-in">
      <h3 className="text-lg font-semibold text-slate-800 mb-6 flex items-center">
        <Target className="w-5 h-5 mr-2 text-blue-600" />
        Résultats d'Analyse
      </h3>
      
      <div className="grid md:grid-cols-2 gap-6">
        {/* Classes Segmentées */}
        <div>
          <h4 className="text-md font-medium text-slate-700 mb-4">Classes Segmentées</h4>
          <div className="space-y-3">
            {results.classes?.map((cls, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors">
                <div className="flex items-center space-x-3">
                  <div 
                    className="w-4 h-4 rounded-full shadow-sm"
                    style={{ backgroundColor: cls.color }}
                  />
                  <span className="font-medium text-slate-700">{cls.name}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-16 bg-slate-200 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${cls.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-sm text-slate-500 min-w-[45px]">
                    {formatNumber(cls.confidence * 100, 1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Métriques de Performance */}
        <div>
          <h4 className="text-md font-medium text-slate-700 mb-4">Métriques de Performance</h4>
          <div className="space-y-3">
            <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
              <div>
                <span className="text-slate-600 font-medium">PSNR</span>
                <p className="text-xs text-slate-500">Peak Signal-to-Noise Ratio</p>
              </div>
              <span className="font-bold text-slate-800">
                {formatNumber(results.metrics.psnr, 1)} dB
              </span>
            </div>
            
            <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
              <div>
                <span className="text-slate-600 font-medium">SSIM</span>
                <p className="text-xs text-slate-500">Structural Similarity Index</p>
              </div>
              <span className="font-bold text-slate-800">
                {formatNumber(results.metrics.ssim, 3)}
              </span>
            </div>
            
            <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
              <div>
                <span className="text-slate-600 font-medium">Score Dice</span>
                <p className="text-xs text-slate-500">Coefficient de similarité</p>
              </div>
              <span className="font-bold text-slate-800">
                {formatNumber(results.metrics.dice, 4)}
              </span>
            </div>
            
            <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
              <div>
                <span className="text-green-600 font-medium">Temps de traitement</span>
                <p className="text-xs text-green-500">Durée totale du pipeline</p>
              </div>
              <span className="font-bold text-green-800">
                {formatTime(results.metrics.processingTime)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Visualisation Segmentation */}
      <div className="mt-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-md font-medium text-slate-700">Masque de Segmentation</h4>
          <button 
            onClick={() => onDownload('segmentation')}
            className="flex items-center space-x-1 text-blue-600 hover:text-blue-700 transition-colors"
          >
            <Download className="w-4 h-4" />
            <span className="text-sm">Télécharger masque</span>
          </button>
        </div>
        
        <div className="relative rounded-xl overflow-hidden bg-slate-100 group">
          <img
            src={results.segmentationMask}
            alt="Segmentation mask"
            className="w-full h-48 object-cover transition-transform group-hover:scale-105"
          />
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-500/10 to-transparent" />
          
          {/* Overlay avec informations */}
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/50 to-transparent p-4">
            <div className="flex justify-between items-center text-white text-sm">
              <span>Classes détectées: {results.classes?.length || 4}</span>
              <span>Résolution améliorée</span>
            </div>
          </div>
        </div>
      </div>

      {/* Actions rapides */}
      <div className="mt-6 pt-4 border-t border-slate-200">
        <div className="flex space-x-3">
          <button 
            onClick={() => onDownload('processed')}
            className="flex-1 flex items-center justify-center space-x-2 py-2 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Download className="w-4 h-4" />
            <span>Image traitée</span>
          </button>
          
          <button 
            onClick={() => onDownload('segmentation')}
            className="flex-1 flex items-center justify-center space-x-2 py-2 px-4 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
          >
            <Download className="w-4 h-4" />
            <span>Masque segmentation</span>
          </button>
        </div>
      </div>
    </div>
  );
}
