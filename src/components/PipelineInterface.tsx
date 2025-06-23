'use client';

import React, { useState, useEffect } from 'react';
import { Upload, Heart, Image as ImageIcon, BarChart3, Target, Play, Clock, Eye, TrendingUp, Zap } from 'lucide-react';
import { usePipeline } from '@/hooks/usePipeline';
import { ImageUploadZone } from './ImageUploadZone';
import { ProcessingPanel } from './ProcessingPanel';

export function PipelineInterface() {
  const {
    uploadedImage,
    processingState,
    results,
    uploadImage,
    processImage,
    downloadResults,
    canProcess,
    hasResults,
  } = usePipeline();

  // Prévenir l'erreur d'hydratation
  const [isClient, setIsClient] = useState(false);
  
  useEffect(() => {
    setIsClient(true);
  }, []);

  return (
    <div className="h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 flex flex-col overflow-hidden">
      {/* Header fixe */}
      <header className="bg-black/20 backdrop-blur-sm border-b border-white/10 px-6 py-4 flex-shrink-0">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-2xl font-bold text-white flex items-center">
            <Heart className="w-6 h-6 mr-3 text-red-400" />
            Pipeline d'Amélioration d'Images Échographiques Cardiaques
          </h1>
          <p className="text-blue-200 mt-1 text-sm">
            IA Avancée • Débruitage • Contraste • Super-Résolution • Segmentation Comparative
          </p>
        </div>
      </header>

      {/* Contenu avec scroll */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto p-6 space-y-6">
          
          {/* Section Upload et Contrôles */}
          <div className="grid lg:grid-cols-3 gap-6">
            {/* Zone d'upload */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Upload className="w-5 h-5 mr-2 text-blue-400" />
                Télécharger une Image Échographique
              </h3>
              <ImageUploadZone 
                onImageUpload={uploadImage}
                hasImage={!!uploadedImage}
              />
            </div>

            {/* Bouton de traitement */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10 flex items-center">
              <button
                onClick={processImage}
                disabled={!canProcess}
                className={`
                  w-full py-4 px-6 rounded-xl font-semibold text-white transition-all duration-300 shadow-lg
                  ${canProcess
                    ? 'bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 hover:shadow-xl transform hover:-translate-y-1'
                    : 'bg-slate-600/50 cursor-not-allowed'
                  }
                `}
              >
                {processingState.isProcessing ? (
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
            </div>

            {/* Pipeline Status */}
            {(processingState.isProcessing || hasResults) && (
              <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <Heart className="w-5 h-5 mr-2 text-red-400" />
                  Pipeline IA Cardiaque
                </h3>
                <ProcessingPanel 
                  processingState={processingState}
                  hasResults={hasResults}
                />
              </div>
            )}
          </div>

          {/* Section Images - 4 colonnes pour la comparaison */}
          <div className="grid lg:grid-cols-4 gap-6">
            
            {/* Image Originale */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 overflow-hidden">
              <div className="bg-gradient-to-r from-slate-700/50 to-slate-600/50 px-6 py-3 border-b border-white/10">
                <h4 className="text-lg font-semibold text-white flex items-center">
                  <ImageIcon className="w-5 h-5 mr-2 text-slate-300" />
                  Original
                </h4>
              </div>
              <div className="h-80 p-4">
                {uploadedImage ? (
                  <div className="relative h-full rounded-xl overflow-hidden bg-slate-800/50 group">
                    <img
                      src={uploadedImage.preview}
                      alt="Original"
                      className="w-full h-full object-contain transition-transform group-hover:scale-105"
                    />
                    <div className="absolute top-4 left-4 bg-black/70 text-white px-3 py-1 rounded-full text-sm">
                      Original
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-400">
                    <div className="text-center">
                      <ImageIcon className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p className="text-slate-400">Image en attente...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Image Améliorée */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 overflow-hidden">
              <div className="bg-gradient-to-r from-green-700/50 to-emerald-600/50 px-6 py-3 border-b border-green-500/20">
                <div className="flex items-center justify-between">
                  <h4 className="text-lg font-semibold text-white flex items-center">
                    <Heart className="w-5 h-5 mr-2 text-green-400" />
                    Améliorée
                  </h4>
                  {results && (
                    <button 
                      onClick={() => downloadResults('processed')}
                      className="text-green-300 hover:text-green-200 text-sm font-medium transition-colors"
                    >
                      ⬇️
                    </button>
                  )}
                </div>
              </div>
              <div className="h-80 p-4">
                {results ? (
                  <div className="relative h-full rounded-xl overflow-hidden bg-slate-800/50 group">
                    <img
                      src={results.processedImage}
                      alt="Processed"
                      className="w-full h-full object-contain transition-transform group-hover:scale-105"
                    />
                    <div className="absolute top-4 left-4 bg-green-600 text-white px-3 py-1 rounded-full text-sm">
                      Pipeline IA
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-400">
                    <div className="text-center">
                      <Heart className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p className="text-slate-400">En attente du traitement...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Segmentation Directe */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 overflow-hidden">
              <div className="bg-gradient-to-r from-orange-700/50 to-red-600/50 px-6 py-3 border-b border-orange-500/20">
                <div className="flex items-center justify-between">
                  <h4 className="text-lg font-semibold text-white flex items-center">
                    <Target className="w-5 h-5 mr-2 text-orange-400" />
                    Segmentation Directe
                  </h4>
                  {results?.segmentationDirect && (
                    <button 
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = results.segmentationDirect!;
                        link.download = `segmentation_direct_${Date.now()}.png`;
                        link.click();
                      }}
                      className="text-orange-300 hover:text-orange-200 text-sm font-medium transition-colors"
                    >
                      ⬇️
                    </button>
                  )}
                </div>
              </div>
              <div className="h-80 p-4">
                {results?.segmentationDirect ? (
                  <div className="relative h-full rounded-xl overflow-hidden bg-slate-800/50 group">
                    <img
                      src={results.segmentationDirect}
                      alt="Segmentation Directe"
                      className="w-full h-full object-contain transition-transform group-hover:scale-105"
                    />
                    <div className="absolute top-4 left-4 bg-orange-600 text-white px-3 py-1 rounded-full text-sm">
                      Sans Pipeline
                    </div>

                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-400">
                    <div className="text-center">
                      <Target className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p className="text-slate-400">Segmentation directe...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Segmentation Pipeline */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 overflow-hidden">
              <div className="bg-gradient-to-r from-purple-700/50 to-pink-600/50 px-6 py-3 border-b border-purple-500/20">
                <div className="flex items-center justify-between">
                  <h4 className="text-lg font-semibold text-white flex items-center">
                    <Zap className="w-5 h-5 mr-2 text-purple-400" />
                    Segmentation  Ameliorée
                  </h4>
                  {results?.segmentationPipeline && (
                    <button 
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = results.segmentationPipeline!;
                        link.download = `segmentation_pipeline_${Date.now()}.png`;
                        link.click();
                      }}
                      className="text-purple-300 hover:text-purple-200 text-sm font-medium transition-colors"
                    >
                      ⬇️
                    </button>
                  )}
                </div>
              </div>
              <div className="h-80 p-4">
                {results?.segmentationPipeline ? (
                  <div className="relative h-full rounded-xl overflow-hidden bg-slate-800/50 group">
                    <img
                      src={results.segmentationPipeline}
                      alt="Segmentation Pipeline"
                      className="w-full h-full object-contain transition-transform group-hover:scale-105"
                    />
                    <div className="absolute top-4 left-4 bg-purple-600 text-white px-3 py-1 rounded-full text-sm">
                      Avec Pipeline
                    </div>

                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-400">
                    <div className="text-center">
                      <Zap className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p className="text-slate-400">Segmentation pipeline...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* MÉTRIQUES - Seulement après traitement et si elles existent */}
          {results && isClient && (
            <div className="space-y-6">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-white mb-2">📊 Résultats du Traitement</h2>
                <p className="text-blue-200">Métriques calculées par le pipeline IA cardiaque</p>
              </div>
              
              {/* Section Comparaison Segmentation */}
              {results.metrics?.segmentation_comparison && (
                <div className="bg-gradient-to-r from-violet-600/20 to-purple-600/20 backdrop-blur-sm rounded-2xl p-6 border border-purple-500/30">
                  <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                    <TrendingUp className="w-6 h-6 mr-2 text-purple-400" />
                    Impact du Pipeline sur la Segmentation
                  </h3>
                  
                  <div className="grid md:grid-cols-2 gap-6">
                    {/* Accord entre segmentations */}
                    <div className="text-center p-4 bg-white/10 rounded-xl">
                      <div className="text-3xl font-bold text-purple-400 mb-2">
                        {(results.metrics.segmentation_comparison.dice_agreement * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-purple-200">Accord Dice</div>
                      <div className="text-xs text-slate-400 mt-1">Similarité entre les deux segmentations</div>
                    </div>
                    
                    {/* Amélioration entropie */}
                    <div className="text-center p-4 bg-white/10 rounded-xl">
                      <div className={`text-3xl font-bold mb-2 ${
                        results.metrics.segmentation_comparison.entropy_improvement > 0 
                          ? 'text-green-400' 
                          : 'text-orange-400'
                      }`}>
                        {results.metrics.segmentation_comparison.entropy_improvement > 0 ? '+' : ''}
                        {(results.metrics.segmentation_comparison.entropy_improvement * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-purple-200">Amélioration Entropie</div>
                      <div className="text-xs text-slate-400 mt-1">Complexité de la segmentation</div>
                    </div>

                  </div>

                  {/* Interprétation */}
                  <div className="mt-6 p-4 bg-white/5 rounded-xl border border-purple-500/20">
                    <h4 className="font-semibold text-white mb-2">💡 Interprétation</h4>
                    <div className="text-sm text-slate-300">
                      {results.metrics.segmentation_comparison.entropy_improvement > 0.1 ? (
                        <span className="text-green-400">✅ Le pipeline améliore significativement la qualité de segmentation</span>
                      ) : results.metrics.segmentation_comparison.entropy_improvement > 0 ? (
                        <span className="text-yellow-400">⚡ Le pipeline apporte une amélioration modérée</span>
                      ) : (
                        <span className="text-orange-400">⚠️ Impact limité du pipeline sur cette image</span>
                      )}
                    </div>
                  </div>
                </div>
              )}
              
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                
                {/* Métriques de Base - TOUJOURS affichées */}
                <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <Eye className="w-5 h-5 mr-2 text-blue-400" />
                    Métriques de Base
                  </h4>
                  <div className="space-y-4">
                    {/* PSNR */}
                    {(results.metrics?.imageQuality?.psnr || results.metrics?.psnr) && (
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <div>
                          <span className="font-medium text-slate-200">PSNR</span>
                          <p className="text-xs text-slate-400">Peak Signal-to-Noise Ratio</p>
                        </div>
                        <span className="text-xl font-bold text-blue-400">
                          {(results.metrics.imageQuality?.psnr || results.metrics.psnr)?.toFixed(1)} dB
                        </span>
                      </div>
                    )}

                    {/* SSIM */}
                    {(results.metrics?.imageQuality?.ssim || results.metrics?.ssim) && (
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <div>
                          <span className="font-medium text-slate-200">SSIM</span>
                          <p className="text-xs text-slate-400">Structural Similarity Index</p>
                        </div>
                        <span className="text-xl font-bold text-blue-400">
                          {(results.metrics.imageQuality?.ssim || results.metrics.ssim)?.toFixed(3)}
                        </span>
                      </div>
                    )}

                    {/* Dice Score */}
                    {(results.metrics?.segmentation?.dice || results.metrics?.dice) && (
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <div>
                          <span className="font-medium text-slate-200">Dice Score</span>
                          <p className="text-xs text-slate-400">Coefficient de segmentation</p>
                        </div>
                        <span className="text-xl font-bold text-purple-400">
                          {(results.metrics.segmentation?.dice || results.metrics.dice)?.toFixed(3)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Classes Segmentation Directe */}
                {results.classesDirect && results.classesDirect.length > 0 && (
                  <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
                    <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
                      <Target className="w-5 h-5 mr-2 text-orange-400" />
                      Segmentation Directe
                    </h4>
                    <div className="space-y-3">
                      {results.classesDirect.filter(cls => 
                        !cls.name.toLowerCase().includes('arriere') && 
                        !cls.name.toLowerCase().includes('arrière') && 
                        cls.confidence > 0
                      ).map((cls, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                          <div className="flex items-center space-x-3">
                            <div 
                              className="w-4 h-4 rounded-full border border-white/20"
                              style={{ backgroundColor: cls.color }}
                            />
                            <span className="font-medium text-slate-200 text-sm">{cls.name.replace(' (Direct)', '')}</span>
                          </div>
                          <span className="text-sm font-semibold text-orange-300">
                            {(cls.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Classes Segmentation Pipeline */}
                {results.classesPipeline && results.classesPipeline.length > 0 && (
                  <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
                    <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
                      <Zap className="w-5 h-5 mr-2 text-purple-400" />
                      Segmentation Ameliorée
                    </h4>
                    <div className="space-y-3">
                      {results.classesPipeline.filter(cls => 
                        !cls.name.toLowerCase().includes('arriere') && 
                        !cls.name.toLowerCase().includes('arrière') && 
                        cls.confidence > 0
                      ).map((cls, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                          <div className="flex items-center space-x-3">
                            <div 
                              className="w-4 h-4 rounded-full border border-white/20"
                              style={{ backgroundColor: cls.color }}
                            />
                            <span className="font-medium text-slate-200 text-sm">{cls.name.replace(' (Pipeline)', '')}</span>
                          </div>
                          <span className="text-sm font-semibold text-purple-300">
                            {(cls.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Performance Temporelle */}
              <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
                <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <Clock className="w-5 h-5 mr-2 text-green-400" />
                  Performance Temporelle
                </h4>
                <div className="text-center p-4 bg-green-500/10 rounded-lg border border-green-500/20">
                  <div className="text-3xl font-bold text-green-400">
                    {results.metrics?.processingTime?.toFixed(1) || '0.0'}s
                  </div>
                  <div className="text-sm text-green-300">Temps Total (incluant les 2 segmentations)</div>
                </div>
              </div>

              {/* Espace en bas pour le scroll */}
              <div className="h-16"></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}