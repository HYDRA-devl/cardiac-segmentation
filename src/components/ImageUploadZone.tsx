'use client';

import React, { useCallback, useRef } from 'react';
import { Upload, AlertCircle } from 'lucide-react';
import { isValidImageFile, formatFileSize } from '@/lib/utils';
import { UI_CONSTANTS, MESSAGES } from '@/lib/config';

interface ImageUploadZoneProps {
  onImageUpload: (file: File) => void;
  hasImage: boolean;
}

export function ImageUploadZone({ onImageUpload, hasImage }: ImageUploadZoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const validateFile = useCallback((file: File): string | null => {
    if (file.size > UI_CONSTANTS.MAX_FILE_SIZE) {
      return MESSAGES.FILE_TOO_LARGE;
    }
    
    if (!isValidImageFile(file)) {
      return MESSAGES.INVALID_FORMAT;
    }
    
    return null;
  }, []);

  const handleFile = useCallback((file: File) => {
    const validationError = validateFile(file);
    
    if (validationError) {
      setError(validationError);
      return;
    }
    
    setError(null);
    onImageUpload(file);
  }, [onImageUpload, validateFile]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
    }
  }, [handleFile]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFile(file);
    }
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
  }, []);

  const handleClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return (
    <div className="space-y-4">
      <div
        className={`
          border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 cursor-pointer
          ${hasImage 
            ? 'border-green-300 bg-green-50' 
            : dragActive
            ? 'border-blue-500 bg-blue-50'
            : 'border-slate-300 bg-slate-50 hover:border-blue-400 hover:bg-blue-50'
          }
          ${error ? 'border-red-300 bg-red-50' : ''}
        `}
        onClick={handleClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={UI_CONSTANTS.SUPPORTED_FORMATS.join(',')}
          onChange={handleFileInput}
          className="hidden"
        />
        
        {error ? (
          <div className="space-y-3">
            <div className="w-16 h-16 mx-auto bg-red-100 rounded-full flex items-center justify-center">
              <AlertCircle className="w-8 h-8 text-red-600" />
            </div>
            <p className="text-red-600 font-medium">{error}</p>
            <p className="text-sm text-slate-500">Cliquez pour réessayer</p>
          </div>
        ) : hasImage ? (
          <div className="space-y-3">
            <div className="w-16 h-16 mx-auto bg-green-100 rounded-full flex items-center justify-center">
              <Upload className="w-8 h-8 text-green-600" />
            </div>
            <p className="text-green-600 font-medium">{MESSAGES.UPLOAD_SUCCESS}</p>
            <p className="text-sm text-slate-500">Cliquez pour changer</p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className={`w-16 h-16 mx-auto rounded-full flex items-center justify-center transition-colors ${
              dragActive ? 'bg-blue-100' : 'bg-slate-100'
            }`}>
              <Upload className={`w-8 h-8 transition-colors ${
                dragActive ? 'text-blue-600' : 'text-slate-400'
              }`} />
            </div>
            <div>
              <p className="text-slate-600 font-medium">
                {dragActive ? 'Relâchez pour charger' : 'Glissez votre image ici'}
              </p>
              <p className="text-sm text-slate-500 mt-1">
                ou cliquez pour sélectionner
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Informations sur les formats supportés */}
      <div className="text-center">
        <p className="text-xs text-slate-500">
          Formats supportés: PNG, JPG, BMP, TIFF
        </p>
        <p className="text-xs text-slate-500">
          Taille max: {formatFileSize(UI_CONSTANTS.MAX_FILE_SIZE)}
        </p>
      </div>
    </div>
  );
}
