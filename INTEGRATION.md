# 🔗 Guide d'Intégration Pipeline Python

Ce guide explique comment intégrer votre pipeline Python existant avec l'interface web Next.js.

## 📋 Prérequis

- Node.js 18+
- Python 3.8+
- PyTorch avec CUDA (optionnel)
- Vos modèles entraînés (.pth)

## 🛠️ Intégration Backend

### Option 1: API Flask/FastAPI (Recommandée)

Créez un serveur Python séparé pour traiter les images:

```python
# api_server.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
import base64
from io import BytesIO
from PIL import Image

# Importez vos architectures de modèles
from your_pipeline import (
    SupervisedDenoisingUNet,
    AggressiveContrastNet, 
    FastRealESRGANGenerator,
    LadderNet
)

app = Flask(__name__)

# Chargez vos modèles au démarrage
models = load_all_models('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Récupérer l'image et paramètres
        data = request.get_json()
        image_data = data['image']  # Base64
        parameters = data['parameters']
        
        # Décoder l'image
        image = decode_base64_image(image_data)
        
        # Exécuter votre pipeline
        results = execute_pipeline(image, models, device)
        
        # Encoder les résultats en base64
        processed_image = encode_image_to_base64(results['super_resolution'])
        segmentation_mask = encode_image_to_base64(results['segmentation'])
        
        return jsonify({
            'success': True,
            'processedImage': processed_image,
            'segmentationMask': segmentation_mask,
            'metrics': {
                'psnr': float(results.get('psnr', 0)),
                'ssim': float(results.get('ssim', 0)),
                'dice': float(results.get('dice', 0)),
                'processingTime': float(results.get('time', 0))
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
```

### Option 2: Subprocess (Plus Simple)

Modifiez `src/app/api/process/route.ts`:

```typescript
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs';

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  try {
    // Sauvegarder l'image temporairement
    const formData = await request.formData();
    const image = formData.get('image') as File;
    const parameters = JSON.parse(formData.get('parameters') as string);
    
    const tempDir = path.join(process.cwd(), 'temp');
    const inputPath = path.join(tempDir, `input_${Date.now()}.png`);
    const outputDir = path.join(tempDir, `output_${Date.now()}`);
    
    // Sauvegarder l'image
    const imageBuffer = Buffer.from(await image.arrayBuffer());
    fs.writeFileSync(inputPath, imageBuffer);
    
    // Exécuter votre script Python
    const pythonScript = process.env.PYTHON_SCRIPT_PATH || './pipeline.py';
    const command = `python "${pythonScript}" --input "${inputPath}" --output "${outputDir}" --contrast ${parameters.contrast} --brightness ${parameters.brightness} --noise-reduction ${parameters.noiseReduction}`;
    
    const { stdout, stderr } = await execAsync(command);
    
    // Lire les résultats
    const processedImagePath = path.join(outputDir, 'processed.png');
    const segmentationPath = path.join(outputDir, 'segmentation.png');
    const metricsPath = path.join(outputDir, 'metrics.json');
    
    const processedImage = fs.readFileSync(processedImagePath, 'base64');
    const segmentationMask = fs.readFileSync(segmentationPath, 'base64');
    const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
    
    // Nettoyer les fichiers temporaires
    fs.unlinkSync(inputPath);
    fs.rmSync(outputDir, { recursive: true });
    
    return NextResponse.json({
      success: true,
      processedImage: `data:image/png;base64,${processedImage}`,
      segmentationMask: `data:image/png;base64,${segmentationMask}`,
      metrics
    });
    
  } catch (error) {
    console.error('Erreur:', error);
    return NextResponse.json({ error: 'Erreur de traitement' }, { status: 500 });
  }
}
```

## 🔧 Modification du Pipeline Python

Ajoutez une interface CLI à votre `pipeline.py`:

```python
import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Pipeline échographique')
    parser.add_argument('--input', required=True, help='Chemin image input')
    parser.add_argument('--output', required=True, help='Dossier de sortie')
    parser.add_argument('--contrast', type=int, default=50, help='Contraste 0-100')
    parser.add_argument('--brightness', type=int, default=50, help='Luminosité 0-100')
    parser.add_argument('--noise-reduction', type=int, default=75, help='Réduction bruit 0-100')
    
    args = parser.parse_args()
    
    # Créer dossier de sortie
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Charger l'image
    input_image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    
    # Charger les modèles
    models = load_all_models('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Exécuter le pipeline avec vos paramètres
    results = execute_pipeline(input_image, models, device, {
        'contrast': args.contrast / 100.0,
        'brightness': args.brightness / 100.0, 
        'noise_reduction': args.noise_reduction / 100.0
    })
    
    # Sauvegarder les résultats
    cv2.imwrite(os.path.join(args.output, 'processed.png'), 
                (results['super_resolution'] * 255).astype(np.uint8))
    
    # Sauvegarder la segmentation colorisée
    colored_seg = colorize_segmentation(results['segmentation'])
    cv2.imwrite(os.path.join(args.output, 'segmentation.png'), 
                cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR))
    
    # Sauvegarder les métriques
    metrics = {
        'psnr': float(results.get('psnr', 0)),
        'ssim': float(results.get('ssim', 0)),
        'dice': float(results.get('dice', 0)),
        'processingTime': float(results.get('total_time', 0))
    }
    
    with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()
```

## 🚀 Déploiement

### Développement Local

1. **Démarrer l'interface Next.js:**
   ```bash
   npm run dev
   ```

2. **Démarrer le serveur Python (si Option 1):**
   ```bash
   python api_server.py
   ```

### Production

1. **Construire l'interface:**
   ```bash
   npm run build
   npm start
   ```

2. **Déployer avec PM2:**
   ```bash
   npm install -g pm2
   pm2 start ecosystem.config.js
   ```

## 📝 Configuration

Modifiez `.env.local`:

```env
# Chemin vers votre pipeline Python
PYTHON_SCRIPT_PATH=C:\Users\ASUS\Desktop\Final\pipeline.py

# Chemins des modèles
DENOISING_MODEL_PATH=C:\Users\ASUS\Desktop\Final\models\supervised_denoising_best.pth
CONTRAST_MODEL_PATH=C:\Users\ASUS\Desktop\Final\models\aggressive_contrast_best.pth
REALESRGAN_MODEL_PATH=C:\Users\ASUS\Desktop\Final\models\fast_realesrgan_best.pth
LADDERNET_MODEL_PATH=C:\Users\ASUS\Desktop\Final\models\laddernet_best_dice.pth

# Configuration CUDA
USE_GPU=true
CUDA_VISIBLE_DEVICES=0
```

## 🧪 Test d'Intégration

```bash
# Tester l'API
curl -X POST http://localhost:3000/api/process \
  -F "image=@test_image.png" \
  -F "parameters={\"contrast\":70,\"brightness\":60,\"noiseReduction\":80}"
```

## ⚡ Optimisations

1. **Cache des modèles** - Gardez les modèles chargés en mémoire
2. **Queue de traitement** - Utilisez Redis/Bull pour gérer les tâches
3. **WebSockets** - Mises à jour en temps réel du progrès
4. **GPU sharing** - Gestion intelligente des ressources CUDA

## 🔍 Debug

- Logs Next.js: `npm run dev` puis vérifiez la console
- Logs Python: Ajoutez `print()` dans votre pipeline
- Network: Utilisez les DevTools du navigateur
- Fichiers temp: Vérifiez `./temp/` pour debug

## 📞 Support

Pour toute question sur l'intégration, vérifiez:
1. Les chemins des modèles dans `.env.local`
2. Les permissions d'exécution de Python
3. La compatibilité des versions PyTorch
4. L'espace disque pour les fichiers temporaires
