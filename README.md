# 🔬 Interface Pipeline Échographique

Interface web professionnelle pour le pipeline d'amélioration d'images échographiques utilisant l'intelligence artificielle.

## 🎯 Fonctionnalités

- **Débruitage** - SupervisedDenoisingUNet pour éliminer le bruit
- **Amélioration Contraste** - AggressiveContrastNet pour optimiser le contraste
- **Super-Résolution** - FastRealESRGAN pour augmenter la résolution
- **Segmentation** - LadderNet pour identifier les structures anatomiques

## 🚀 Pipeline de Traitement

1. **Étape 1**: Débruitage avec SupervisedDenoisingUNet
2. **Étape 2**: Amélioration contraste avec AggressiveContrastNet  
3. **Étape 3**: Super-résolution avec FastRealESRGAN
4. **Étape 4**: Segmentation avec LadderNet

## 📋 Prérequis

- Node.js 18+ 
- npm ou yarn
- Navigateur moderne

## 🛠️ Installation

```bash
# Cloner le projet
git clone [URL_DU_REPO]
cd ultrasound-pipeline-interface

# Installer les dépendances
npm install

# Lancer en développement
npm run dev

# Construire pour production
npm run build
npm start
```

## 🎨 Technologies

- **Next.js 14** - Framework React
- **TypeScript** - Typage statique
- **Tailwind CSS** - Styles utilitaires
- **Framer Motion** - Animations
- **Lucide React** - Icônes
- **React Dropzone** - Upload de fichiers

## 📁 Structure

```
src/
├── app/           # Pages Next.js (App Router)
├── components/    # Composants React réutilisables
├── hooks/         # Hooks personnalisés
├── lib/           # Utilitaires et helpers
└── types/         # Types TypeScript
```

## 🎯 Classes de Segmentation

- **VG Endo** - Endocarde ventriculaire gauche
- **OG** - Oreillette gauche  
- **VG Epi** - Épicarde ventriculaire gauche
- **Arrière-plan** - Zone non cardiaque

## 📊 Métriques

- **PSNR** - Peak Signal-to-Noise Ratio
- **SSIM** - Structural Similarity Index
- **Score Dice** - Coefficient de similarité pour segmentation

## 👥 Contribution

1. Fork le projet
2. Créer une branche feature
3. Commit les changements
4. Push vers la branche
5. Ouvrir une Pull Request

## 📄 License

MIT License - voir LICENSE file pour détails.
