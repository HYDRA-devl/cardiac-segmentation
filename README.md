# 🔬 Interface Pipeline Échographique

Interface web professionnelle pour le pipeline d'amélioration d'images échographiques utilisant l'intelligence artificielle.

## 🎯 Fonctionnalités

- **Débruitage** - 
- **Amélioration Contraste** - 
- **Super-Résolution** - 
- **Segmentation** - 



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
