'use client';

import React, { useState, useEffect } from 'react';
import { Upload, Brain, Image as ImageIcon, BarChart3, Target, Play, Clock, Eye } from 'lucide-react';
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
            <Brain className="w-6 h-6 mr-3 text-blue-400" />
            Pipeline d'Amélioration d'Images Échographiques
          </h1>
          <p className="text-blue-200 mt-1 text-sm">
            IA Avancée • Débruitage • Contraste • Super-Résolution • Segmentation
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
                  <Brain className="w-5 h-5 mr-2 text-blue-400" />
                  Pipeline IA
                </h3>
                <ProcessingPanel 
                  processingState={processingState}
                  hasResults={hasResults}
                />
              </div>
            )}
          </div>

          {/* Section Images - 3 colonnes principales */}
          <div className="grid lg:grid-cols-3 gap-6">
            
            {/* Image Originale */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 overflow-hidden">
              <div className="bg-gradient-to-r from-slate-700/50 to-slate-600/50 px-6 py-3 border-b border-white/10">
                <h4 className="text-lg font-semibold text-white flex items-center">
                  <ImageIcon className="w-5 h-5 mr-2 text-slate-300" />
                  Image Originale
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
                    <Brain className="w-5 h-5 mr-2 text-green-400" />
                    Image Améliorée
                  </h4>
                  {results && (
                    <button 
                      onClick={() => downloadResults('processed')}
                      className="text-green-300 hover:text-green-200 text-sm font-medium transition-colors"
                    >
                      Télécharger
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
                      Améliorée
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-400">
                    <div className="text-center">
                      <Brain className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p className="text-slate-400">En attente du traitement...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Segmentation */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 overflow-hidden">
              <div className="bg-gradient-to-r from-purple-700/50 to-pink-600/50 px-6 py-3 border-b border-purple-500/20">
                <div className="flex items-center justify-between">
                  <h4 className="text-lg font-semibold text-white flex items-center">
                    <Target className="w-5 h-5 mr-2 text-purple-400" />
                    Segmentation
                  </h4>
                  {results && (
                    <button 
                      onClick={() => downloadResults('segmentation')}
                      className="text-purple-300 hover:text-purple-200 text-sm font-medium transition-colors"
                    >
                      Télécharger
                    </button>
                  )}
                </div>
              </div>
              <div className="h-80 p-4">
                {results ? (
                  <div className="relative h-full rounded-xl overflow-hidden bg-slate-800/50 group">
                    <img
                      src={results.segmentationMask}
                      alt="Segmentation"
                      className="w-full h-full object-contain transition-transform group-hover:scale-105"
                    />
                    <div className="absolute top-4 left-4 bg-purple-600 text-white px-3 py-1 rounded-full text-sm">
                      Segmentation
                    </div>
                    <div className="absolute bottom-4 left-4 right-4 bg-black/70 text-white px-3 py-2 rounded-lg text-sm">
                      <div className="flex justify-between items-center">
                        <span>{results.classes?.length || 4} classes détectées</span>
                        <span>Résolution 2x</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-400">
                    <div className="text-center">
                      <Target className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p className="text-slate-400">En attente de la segmentation...</p>
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
                <p className="text-blue-200">Métriques calculées par le pipeline IA</p>
              </div>
              
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

                    {/* MSE */}
                    {results.metrics?.imageQuality?.mse && (
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <div>
                          <span className="font-medium text-slate-200">MSE</span>
                          <p className="text-xs text-slate-400">Mean Squared Error</p>
                        </div>
                        <span className="text-xl font-bold text-cyan-400">
                          {results.metrics.imageQuality.mse.toFixed(4)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Classes Détectées */}
                {results.classes && results.classes.length > 0 && (
                  <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
                    <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
                      <Target className="w-5 h-5 mr-2 text-purple-400" />
                      Classes Détectées
                    </h4>
                    <div className="space-y-3">
                      {results.classes.map((cls, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                          <div className="flex items-center space-x-3">
                            <div 
                              className="w-4 h-4 rounded-full border border-white/20"
                              style={{ backgroundColor: cls.color }}
                            />
                            <span className="font-medium text-slate-200">{cls.name}</span>
                          </div>
                          <span className="text-sm font-semibold text-slate-300">
                            {(cls.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Performance Temporelle */}
                <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <Clock className="w-5 h-5 mr-2 text-green-400" />
                    Performance Temporelle
                  </h4>
                  <div className="space-y-4">
                    {/* Temps Total */}
                    <div className="text-center p-4 bg-green-500/10 rounded-lg border border-green-500/20">
                      <div className="text-3xl font-bold text-green-400">
                        {results.metrics?.processingTime?.toFixed(1) || '0.0'}s
                      </div>
                      <div className="text-sm text-green-300">Temps Total</div>
                    </div>

                    {/* Temps par étapes - seulement si disponibles et > 0 */}
                    {results.metrics?.stepTimes && (
                      Object.values(results.metrics.stepTimes).some(time => time && time > 0)
                    ) && (
                      <div className="space-y-2">
                        {results.metrics.stepTimes.denoising && results.metrics.stepTimes.denoising > 0 && (
                          <div className="flex justify-between items-center text-sm">
                            <span className="text-slate-300">Débruitage</span>
                            <span className="text-blue-400">
                              {results.metrics.stepTimes.denoising.toFixed(2)}s
                            </span>
                          </div>
                        )}
                        {results.metrics.stepTimes.contrast && results.metrics.stepTimes.contrast > 0 && (
                          <div className="flex justify-between items-center text-sm">
                            <span className="text-slate-300">Contraste</span>
                            <span className="text-purple-400">
                              {results.metrics.stepTimes.contrast.toFixed(2)}s
                            </span>
                          </div>
                        )}
                        {results.metrics.stepTimes.superResolution && results.metrics.stepTimes.superResolution > 0 && (
                          <div className="flex justify-between items-center text-sm">
                            <span className="text-slate-300">Super-Résolution</span>
                            <span className="text-cyan-400">
                              {results.metrics.stepTimes.superResolution.toFixed(2)}s
                            </span>
                          </div>
                        )}
                        {results.metrics.stepTimes.segmentation && results.metrics.stepTimes.segmentation > 0 && (
                          <div className="flex justify-between items-center text-sm">
                            <span className="text-slate-300">Segmentation</span>
                            <span className="text-orange-400">
                              {results.metrics.stepTimes.segmentation.toFixed(2)}s
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Métriques Avancées - Seulement si elles existent vraiment et > 0 */}
              {(results.metrics?.imageQuality?.snr && results.metrics.imageQuality.snr > 0 || 
                results.metrics?.imageQuality?.cnr && results.metrics.imageQuality.cnr > 0 || 
                results.metrics?.imageQuality?.lpips && results.metrics.imageQuality.lpips > 0 ||
                results.metrics?.imageQuality?.michelson_contrast && results.metrics.imageQuality.michelson_contrast > 0 ||
                results.metrics?.imageQuality?.vif && results.metrics.imageQuality.vif > 0 ||
                results.metrics?.imageQuality?.edge_preservation && results.metrics.imageQuality.edge_preservation > 0 ||
                results.metrics?.segmentation?.iou && results.metrics.segmentation.iou > 0 ||
                results.metrics?.segmentation?.hausdorff_distance && results.metrics.segmentation.hausdorff_distance > 0 ||
                results.metrics?.segmentation?.sensitivity && results.metrics.segmentation.sensitivity > 0 ||
                results.metrics?.segmentation?.precision && results.metrics.segmentation.precision > 0 ||
                results.metrics?.segmentation?.accuracy && results.metrics.segmentation.accuracy > 0 ||
                results.metrics?.segmentation?.volume_similarity && results.metrics.segmentation.volume_similarity > 0 ||
                results.metrics?.segmentation?.matthews_correlation && results.metrics.segmentation.matthews_correlation > 0) && (
                <div className="grid md:grid-cols-2 gap-6">
                  
                  {/* Métriques Avancées d'Image */}
                  {(results.metrics?.imageQuality?.snr && results.metrics.imageQuality.snr > 0 || 
                    results.metrics?.imageQuality?.cnr && results.metrics.imageQuality.cnr > 0 || 
                    results.metrics?.imageQuality?.lpips && results.metrics.imageQuality.lpips > 0 ||
                    results.metrics?.imageQuality?.michelson_contrast && results.metrics.imageQuality.michelson_contrast > 0 ||
                    results.metrics?.imageQuality?.vif && results.metrics.imageQuality.vif > 0 ||
                    results.metrics?.imageQuality?.edge_preservation && results.metrics.imageQuality.edge_preservation > 0) && (
                    <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
                      <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
                        <BarChart3 className="w-5 h-5 mr-2 text-cyan-400" />
                        Métriques Avancées d'Image
                      </h4>
                      <div className="space-y-3">
                        {results.metrics.imageQuality?.snr && results.metrics.imageQuality.snr > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-slate-300">SNR</span>
                            <span className="text-cyan-400 font-semibold">
                              {results.metrics.imageQuality.snr.toFixed(1)} dB
                            </span>
                          </div>
                        )}
                        {results.metrics.imageQuality?.cnr && results.metrics.imageQuality.cnr > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-slate-300">CNR</span>
                            <span className="text-cyan-400 font-semibold">
                              {results.metrics.imageQuality.cnr.toFixed(1)}
                            </span>
                          </div>
                        )}
                        {results.metrics.imageQuality?.lpips && results.metrics.imageQuality.lpips > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-slate-300">LPIPS</span>
                            <span className="text-cyan-400 font-semibold">
                              {results.metrics.imageQuality.lpips.toFixed(4)}
                            </span>
                          </div>
                        )}
                        {results.metrics.imageQuality?.michelson_contrast && results.metrics.imageQuality.michelson_contrast > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-slate-300">Contraste Michelson</span>
                            <span className="text-cyan-400 font-semibold">
                              {results.metrics.imageQuality.michelson_contrast.toFixed(3)}
                            </span>
                          </div>
                        )}
                        {results.metrics.imageQuality?.vif && results.metrics.imageQuality.vif > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-slate-300">VIF</span>
                            <span className="text-cyan-400 font-semibold">
                              {results.metrics.imageQuality.vif.toFixed(3)}
                            </span>
                          </div>
                        )}
                        {results.metrics.imageQuality?.edge_preservation && results.metrics.imageQuality.edge_preservation > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-slate-300">Préservation Contours</span>
                            <span className="text-cyan-400 font-semibold">
                              {results.metrics.imageQuality.edge_preservation.toFixed(3)}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Métriques Avancées de Segmentation */}
                  {(results.metrics?.segmentation?.iou && results.metrics.segmentation.iou > 0 ||
                    results.metrics?.segmentation?.hausdorff_distance && results.metrics.segmentation.hausdorff_distance > 0 ||
                    results.metrics?.segmentation?.sensitivity && results.metrics.segmentation.sensitivity > 0 ||
                    results.metrics?.segmentation?.precision && results.metrics.segmentation.precision > 0 ||
                    results.metrics?.segmentation?.accuracy && results.metrics.segmentation.accuracy > 0 ||
                    results.metrics?.segmentation?.volume_similarity && results.metrics.segmentation.volume_similarity > 0 ||
                    results.metrics?.segmentation?.matthews_correlation && results.metrics.segmentation.matthews_correlation > 0) && (
                    <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
                      <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
                        <Target className="w-5 h-5 mr-2 text-purple-400" />
                        Métriques Avancées de Segmentation
                      </h4>
                      <div className="space-y-3">
                        {results.metrics.segmentation?.iou && results.metrics.segmentation.iou > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-slate-300">IoU/Jaccard</span>
                            <span className="text-purple-400 font-semibold">
                              {results.metrics.segmentation.iou.toFixed(3)}
                            </span>
                          </div>
                        )}
                        {results.metrics.segmentation?.hausdorff_distance && results.metrics.segmentation.hausdorff_distance > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-slate-300">Hausdorff (px)</span>
                            <span className="text-purple-400 font-semibold">
                              {results.metrics.segmentation.hausdorff_distance.toFixed(1)}
                            </span>
                          </div>
                        )}
                        {results.metrics.segmentation?.sensitivity && results.metrics.segmentation.sensitivity > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-slate-300">Sensibilité</span>
                            <span className="text-purple-400 font-semibold">
                              {results.metrics.segmentation.sensitivity.toFixed(3)}
                            </span>
                          </div>
                        )}
                        {results.metrics.segmentation?.precision && results.metrics.segmentation.precision > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-slate-300">Précision</span>
                            <span className="text-purple-400 font-semibold">
                              {results.metrics.segmentation.precision.toFixed(3)}
                            </span>
                          </div>
                        )}
                        {results.metrics.segmentation?.accuracy && results.metrics.segmentation.accuracy > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-slate-300">Accuracy</span>
                            <span className="text-purple-400 font-semibold">
                              {results.metrics.segmentation.accuracy.toFixed(3)}
                            </span>
                          </div>
                        )}
                        {results.metrics.segmentation?.volume_similarity && results.metrics.segmentation.volume_similarity > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-slate-300">Volume Sim.</span>
                            <span className="text-purple-400 font-semibold">
                              {results.metrics.segmentation.volume_similarity.toFixed(3)}
                            </span>
                          </div>
                        )}
                        {results.metrics.segmentation?.matthews_correlation && results.metrics.segmentation.matthews_correlation > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-slate-300">Matthews CC</span>
                            <span className="text-purple-400 font-semibold">
                              {results.metrics.segmentation.matthews_correlation.toFixed(3)}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Métriques par Classe - Seulement si disponibles */}
              {results.metrics?.classMetrics && Object.keys(results.metrics.classMetrics).length > 0 && (
                <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <BarChart3 className="w-5 h-5 mr-2 text-orange-400" />
                    Performance par Classe
                  </h4>
                  <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {Object.entries(results.metrics.classMetrics).map(([className, metrics]) => (
                      <div key={className} className="p-4 bg-white/5 rounded-lg border border-white/5">
                        <h5 className="font-medium text-slate-200 mb-3 text-center">{className}</h5>
                        <div className="space-y-2 text-sm">
                          {metrics.dice && (
                            <div className="flex justify-between">
                              <span className="text-slate-400">Dice:</span>
                              <span className="text-orange-400 font-semibold">{metrics.dice.toFixed(3)}</span>
                            </div>
                          )}
                          {metrics.iou && (
                            <div className="flex justify-between">
                              <span className="text-slate-400">IoU:</span>
                              <span className="text-yellow-400 font-semibold">{metrics.iou.toFixed(3)}</span>
                            </div>
                          )}
                          {metrics.precision && (
                            <div className="flex justify-between">
                              <span className="text-slate-400">Précision:</span>
                              <span className="text-amber-400 font-semibold">{metrics.precision.toFixed(3)}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Espace en bas pour le scroll */}
              <div className="h-16"></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}