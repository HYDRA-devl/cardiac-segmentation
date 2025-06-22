import { PipelineConfig } from '@/types/pipeline';

export const PIPELINE_CONFIG: PipelineConfig = {
  steps: [
    {
      name: 'Débruitage',
      description: '',
      color: 'from-blue-500 to-cyan-500',
      icon: 'Sparkles',
      duration: 1500,
    },
    {
      name: 'Contraste',
      description: '', 
      color: 'from-purple-500 to-pink-500',
      icon: 'Zap',
      duration: 1200,
    },
    {
      name: 'Super-Résolution',
      description: '',
      color: 'from-green-500 to-emerald-500',
      icon: 'Eye',
      duration: 2000,
    },
    {
      name: 'Segmentation',
      description: '',
      color: 'from-orange-500 to-red-500',
      icon: 'Target',
      duration: 1800,
    }
  ],
  
  defaultParameters: {
    contrast: 50,
    brightness: 50,
    noiseReduction: 75,
  },
  
  // Couleurs ajustées pour correspondre à l'image de segmentation
  segmentationClasses: [
    {
      name: 'VG Endo',
      color: '#8B0000', // Rouge foncé (comme dans l'image)
      confidence: 0.67, // 67% d'après l'interface
    },
    {
      name: 'OG', 
      color: '#00FFFF', // Cyan (comme dans l'image)
      confidence: 0.74, // 74% d'après l'interface
    },
    {
      name: 'VG Epi',
      color: '#FFFF00', // Jaune (comme dans l'image)
      confidence: 0.75, // 75% d'après l'interface
    },
    {
      name: 'Arrière-plan',
      color: '#000080', // Bleu foncé (fond dans l'image)
      confidence: 0.92, // 92% d'après l'interface
    }
  ]
};

// Constantes pour l'interface
export const UI_CONSTANTS = {
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  SUPPORTED_FORMATS: ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'],
  MAX_IMAGE_DIMENSION: 2048,
  ANIMATION_DURATION: 300,
  PROCESSING_TIMEOUT: 60000, // 60 secondes
};

// Messages utilisateur
export const MESSAGES = {
  UPLOAD_SUCCESS: 'Image chargée avec succès',
  UPLOAD_ERROR: 'Erreur lors du chargement de l\'image',
  PROCESSING_START: 'Traitement en cours...',
  PROCESSING_COMPLETE: 'Traitement terminé avec succès',
  PROCESSING_ERROR: 'Erreur lors du traitement',
  FILE_TOO_LARGE: 'Fichier trop volumineux (max 10MB)',
  INVALID_FORMAT: 'Format non supporté',
  NO_IMAGE_SELECTED: 'Aucune image sélectionnée',
};

// Configuration API (pour future intégration backend)
export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000',
  ENDPOINTS: {
    UPLOAD: '/api/upload',
    PROCESS: '/api/process',
    DOWNLOAD: '/api/download',
  },
  TIMEOUT: 30000,
};