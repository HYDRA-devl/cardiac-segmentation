import { useState, useCallback } from 'react';
import { ProcessingParameters, ProcessingResults, ProcessingState, UploadedImage } from '@/types/pipeline';
import { PIPELINE_CONFIG, MESSAGES } from '@/lib/config';
import { delay } from '@/lib/utils';

export function usePipeline() {
  const [uploadedImage, setUploadedImage] = useState<UploadedImage | null>(null);
  const [processingState, setProcessingState] = useState<ProcessingState>({
    isProcessing: false,
    currentStep: 0,
    progress: 0,
  });
  const [results, setResults] = useState<ProcessingResults | null>(null);
  const [parameters, setParameters] = useState<ProcessingParameters>(
    PIPELINE_CONFIG.defaultParameters
  );

  // Fonction pour uploader une image
  const uploadImage = useCallback((file: File) => {
    const preview = URL.createObjectURL(file);
    setUploadedImage({ file, preview });
    setResults(null);
    setProcessingState(prev => ({ ...prev, progress: 0, currentStep: 0, error: undefined }));
  }, []);

  // Fonction pour mettre à jour les paramètres
  const updateParameters = useCallback((newParams: Partial<ProcessingParameters>) => {
    setParameters(prev => ({ ...prev, ...newParams }));
  }, []);

  // Appel de la vraie API Python
  const processImage = useCallback(async () => {
    if (!uploadedImage) {
      throw new Error(MESSAGES.NO_IMAGE_SELECTED);
    }

    console.log('🚀 Début du traitement avec la vraie API');

    setProcessingState({
      isProcessing: true,
      currentStep: 0,
      progress: 0,
    });

    try {
      const totalSteps = PIPELINE_CONFIG.steps.length;
      
      // Animation des étapes pendant le traitement réel
      const progressInterval = setInterval(() => {
        setProcessingState(prev => {
          if (!prev.isProcessing) return prev;
          
          const newStep = Math.min(
            Math.floor((Date.now() % 8000) / 2000), 
            totalSteps - 1
          );
          
          return {
            ...prev,
            currentStep: newStep,
            progress: Math.min(((newStep + 1) / totalSteps) * 100, 90),
          };
        });
      }, 500);

      // Préparer les données pour l'API
      const formData = new FormData();
      formData.append('image', uploadedImage.file);
      formData.append('parameters', JSON.stringify(parameters));

      console.log('📤 Envoi vers l\'API:', parameters);

      // Appeler l'API Next.js qui exécute le pipeline Python
      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Erreur HTTP: ${response.status}`);
      }

      const apiResults = await response.json();
      console.log('📥 Résultats reçus:', apiResults);

      if (!apiResults.success) {
        throw new Error(apiResults.error || 'Erreur lors du traitement');
      }

      // Convertir les résultats de l'API vers le format attendu
      const processedResults: ProcessingResults = {
        processedImage: apiResults.processedImage,
        segmentationMask: apiResults.segmentationMask,
        metrics: {
          processingTime: apiResults.metrics.processingTime || 0,
          stepTimes: apiResults.metrics.stepTimes || {
            denoising: 0,
            contrast: 0,
            superResolution: 0,
            segmentation: 0,
          },
          imageQuality: {
            // Seulement les métriques réellement calculées par le backend
            psnr: apiResults.metrics.psnr || apiResults.metrics.imageQuality?.psnr || 0,
            ssim: apiResults.metrics.ssim || apiResults.metrics.imageQuality?.ssim || 0,
            mse: apiResults.metrics.imageQuality?.mse || 0,
            // Métriques avancées - seulement si elles existent vraiment
            snr: apiResults.metrics.imageQuality?.snr || 0,
            cnr: apiResults.metrics.imageQuality?.cnr || 0,
            lpips: apiResults.metrics.imageQuality?.lpips || 0,
            vif: apiResults.metrics.imageQuality?.vif || 0,
            michelson_contrast: apiResults.metrics.imageQuality?.michelson_contrast || 0,
            rms_contrast: apiResults.metrics.imageQuality?.rms_contrast || 0,
            edge_preservation: apiResults.metrics.imageQuality?.edge_preservation || 0,
            gradient_magnitude: apiResults.metrics.imageQuality?.gradient_magnitude || 0,
          },
          segmentation: {
            dice: apiResults.metrics.dice || apiResults.metrics.segmentation?.dice || 0,
            iou: apiResults.metrics.segmentation?.iou || 0,
            hausdorff_distance: apiResults.metrics.segmentation?.hausdorff_distance || 0,
            avg_surface_distance: apiResults.metrics.segmentation?.avg_surface_distance || 0,
            sensitivity: apiResults.metrics.segmentation?.sensitivity || 0,
            specificity: apiResults.metrics.segmentation?.specificity || 0,
            precision: apiResults.metrics.segmentation?.precision || 0,
            accuracy: apiResults.metrics.segmentation?.accuracy || 0,
            volume_similarity: apiResults.metrics.segmentation?.volume_similarity || 0,
            boundary_f1: apiResults.metrics.segmentation?.boundary_f1 || 0,
            matthews_correlation: apiResults.metrics.segmentation?.matthews_correlation || 0,
          },
          classMetrics: apiResults.metrics.classMetrics || {},
        },
        classes: apiResults.classes || PIPELINE_CONFIG.segmentationClasses,
      };

      setResults(processedResults);
      setProcessingState({
        isProcessing: false,
        currentStep: 0,
        progress: 100,
      });

      console.log('✅ Traitement terminé avec succès');

    } catch (error) {
      console.error('❌ Erreur lors du traitement:', error);
      
      setProcessingState(prev => ({
        ...prev,
        isProcessing: false,
        error: error instanceof Error ? error.message : MESSAGES.PROCESSING_ERROR,
      }));
    }
  }, [uploadedImage, parameters]);

  // Fonction pour réinitialiser l'état
  const reset = useCallback(() => {
    setUploadedImage(null);
    setResults(null);
    setProcessingState({
      isProcessing: false,
      currentStep: 0,
      progress: 0,
    });
    setParameters(PIPELINE_CONFIG.defaultParameters);
  }, []);

  // Fonction pour télécharger les résultats
  const downloadResults = useCallback((type: 'processed' | 'segmentation') => {
    if (!results) return;
    
    const dataUrl = type === 'processed' ? results.processedImage : results.segmentationMask;
    const filename = `ultrasound_${type}_${Date.now()}.png`;
    
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [results]);

  return {
    // État
    uploadedImage,
    processingState,
    results,
    parameters,
    
    // Actions
    uploadImage,
    updateParameters,
    processImage,
    downloadResults,
    reset,
    
    // Helpers
    canProcess: uploadedImage && !processingState.isProcessing,
    hasResults: !!results,
    currentStep: PIPELINE_CONFIG.steps[processingState.currentStep],
  };
}
