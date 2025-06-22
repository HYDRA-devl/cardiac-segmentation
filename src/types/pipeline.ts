// Types pour le pipeline de traitement d'images échographiques

export interface ProcessingParameters {
  contrast: number;
  brightness: number;
  noiseReduction: number;
}

export interface PipelineStep {
  name: string;
  description: string;
  color: string;
  icon: string;
  duration?: number;
}

export interface SegmentationClass {
  name: string;
  color: string;
  confidence: number;
  pixelCount?: number;
  area?: number;
}

// Métriques complètes pour l'évaluation
export interface ImageQualityMetrics {
  // Métriques de base
  psnr: number;          // Peak Signal-to-Noise Ratio (dB)
  ssim: number;          // Structural Similarity Index (0-1)
  mse: number;           // Mean Squared Error
  
  // Métriques avancées de qualité d'image
  snr: number;           // Signal-to-Noise Ratio (dB)
  cnr: number;           // Contrast-to-Noise Ratio
  lpips: number;         // Learned Perceptual Image Patch Similarity
  vif: number;           // Visual Information Fidelity
  
  // Métriques de contraste et netteté
  michelson_contrast: number;    // Contraste de Michelson
  rms_contrast: number;          // RMS Contrast
  edge_preservation: number;     // Edge Preservation Index
  gradient_magnitude: number;    // Gradient Magnitude Similarity
}

export interface SegmentationMetrics {
  // Métriques de base
  dice: number;          // Dice Coefficient/F1-Score (0-1)
  iou: number;           // Intersection over Union / Jaccard Index (0-1)
  
  // Métriques de distance
  hausdorff_distance: number;      // Distance de Hausdorff (pixels)
  avg_surface_distance: number;    // Distance de surface moyenne
  
  // Métriques de classification
  sensitivity: number;   // Recall/True Positive Rate (0-1)
  specificity: number;   // True Negative Rate (0-1)
  precision: number;     // Positive Predictive Value (0-1)
  accuracy: number;      // Accuracy globale (0-1)
  
  // Métriques spécialisées
  volume_similarity: number;       // Similarité volumétrique
  boundary_f1: number;            // F1-Score des contours
  matthews_correlation: number;    // Matthews Correlation Coefficient
}

export interface ProcessingMetrics {
  // Temps et performance
  processingTime: number;         // Temps total (secondes)
  stepTimes: {                   // Temps par étape
    denoising: number;
    contrast: number;
    superResolution: number;
    segmentation: number;
  };
  
  // Métriques de qualité d'image
  imageQuality: ImageQualityMetrics;
  
  // Métriques de segmentation
  segmentation: SegmentationMetrics;
  
  // Métriques par classe
  classMetrics: {
    [className: string]: {
      dice: number;
      iou: number;
      precision: number;
      recall: number;
      volume: number;
    };
  };
  
  // Rétrocompatibilité (deprecated)
  psnr?: number;
  ssim?: number;
  dice?: number;
}

export interface ProcessingResults {
  processedImage: string;
  segmentationMask: string;
  metrics: ProcessingMetrics;
  classes?: SegmentationClass[];
}

export interface ProcessingState {
  isProcessing: boolean;
  currentStep: number;
  progress: number;
  error?: string;
}

export interface UploadedImage {
  file: File;
  preview: string;
  dimensions?: {
    width: number;
    height: number;
  };
}

export type ProcessingStatus = 'idle' | 'processing' | 'completed' | 'error';

export interface PipelineConfig {
  steps: PipelineStep[];
  defaultParameters: ProcessingParameters;
  segmentationClasses: SegmentationClass[];
}