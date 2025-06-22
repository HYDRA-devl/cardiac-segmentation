# 🚀 Guide de Démarrage Rapide

## ⚡ Installation Express (5 minutes)

### 1. Prérequis
- Node.js 18+ ([Télécharger](https://nodejs.org/))
- Git (optionnel)

### 2. Installation
```bash
# Aller dans le dossier du projet
cd ultrasound-pipeline-interface

# Installer les dépendances (Windows)
.\setup.ps1

# OU installer manuellement
npm install
```

### 3. Configuration
```bash
# Copier le fichier de configuration
cp .env.example .env.local

# Modifier les chemins des modèles dans .env.local
# DENOISING_MODEL_PATH=C:\Users\ASUS\Desktop\Final\models\supervised_denoising_best.pth
# CONTRAST_MODEL_PATH=C:\Users\ASUS\Desktop\Final\models\aggressive_contrast_best.pth
# REALESRGAN_MODEL_PATH=C:\Users\ASUS\Desktop\Final\models\fast_realesrgan_best.pth
# LADDERNET_MODEL_PATH=C:\Users\ASUS\Desktop\Final\models\laddernet_best_dice.pth
```

### 4. Démarrage
```bash
npm run dev
```

**🌐 Ouvrir:** http://localhost:3000

---

## 📱 Utilisation

### Interface Principale
1. **📤 Upload** - Glissez une image échographique ou cliquez pour sélectionner
2. **⚙️ Paramètres** - Ajustez contraste, luminosité, réduction bruit
3. **🚀 Traitement** - Cliquez "Appliquer et Analyser"
4. **📊 Résultats** - Visualisez l'image traitée et les métriques

### Pipeline 4 Étapes
1. **🧹 Débruitage** - SupervisedDenoisingUNet
2. **⚡ Contraste** - AggressiveContrastNet  
3. **🔍 Super-Résolution** - FastRealESRGAN
4. **🎯 Segmentation** - LadderNet

---

## 🔧 Intégration Pipeline Python

### Méthode Simple (Subprocess)
Votre pipeline existant sera appelé automatiquement via l'API Next.js.

### Méthode Avancée (Serveur Python)
Créez un serveur Flask/FastAPI séparé pour de meilleures performances.

**📖 Guide complet:** `INTEGRATION.md`

---

## 📂 Structure du Projet

```
ultrasound-pipeline-interface/
├── src/
│   ├── app/                 # Pages Next.js 14 (App Router)
│   │   ├── api/process/     # API de traitement
│   │   ├── globals.css      # Styles globaux
│   │   ├── layout.tsx       # Layout principal
│   │   └── page.tsx         # Page d'accueil
│   ├── components/          # Composants React
│   │   ├── PipelineInterface.tsx    # Interface principale
│   │   ├── ImageUploadZone.tsx      # Zone d'upload
│   │   ├── ParametersPanel.tsx      # Panneau paramètres
│   │   ├── ProcessingPanel.tsx      # Pipeline étapes
│   │   ├── ResultsPanel.tsx         # Affichage résultats
│   │   └── ProcessButton.tsx        # Bouton traitement
│   ├── hooks/               # Hooks personnalisés
│   │   └── usePipeline.ts   # Logique du pipeline
│   ├── lib/                 # Utilitaires
│   │   ├── config.ts        # Configuration
│   │   └── utils.ts         # Fonctions utiles
│   └── types/               # Types TypeScript
│       └── pipeline.ts      # Types du pipeline
├── public/                  # Fichiers statiques
├── package.json            # Dépendances npm
├── tailwind.config.js      # Configuration Tailwind
├── tsconfig.json          # Configuration TypeScript
└── README.md              # Documentation
```

---

## 🛠️ Commandes Utiles

```bash
# Développement
npm run dev          # Démarrer en mode dev (port 3000)
npm run lint         # Vérifier le code

# Production  
npm run build        # Construire l'application
npm run start        # Démarrer en production

# Maintenance
npm install          # Installer dépendances
npm update           # Mettre à jour packages
```

---

## 🎨 Fonctionnalités Interface

### ✨ Design Professionnel
- Interface moderne avec Tailwind CSS
- Animations fluides avec Framer Motion
- Responsive design (mobile/desktop)
- Thème médical avec couleurs appropriées

### 🖱️ Interactions Avancées
- Drag & drop pour upload d'images
- Sliders interactifs pour paramètres
- Indicateurs de progression en temps réel
- Visualisation des résultats étape par étape

### 📊 Métriques Complètes
- **PSNR** - Peak Signal-to-Noise Ratio
- **SSIM** - Structural Similarity Index
- **Score Dice** - Coefficient de similarité segmentation
- **Temps** - Durée de traitement

### 🎯 Segmentation Classes
- **VG Endo** - Endocarde ventriculaire gauche (Rouge)
- **OG** - Oreillette gauche (Bleu)
- **VG Epi** - Épicarde ventriculaire gauche (Vert)
- **Arrière-plan** - Zone non cardiaque (Gris)

---

## 🚨 Dépannage

### Problèmes Courants

**Port 3000 occupé:**
```bash
# Changer le port
npm run dev -- -p 3001
```

**Erreur modules:**
```bash
# Réinstaller
rm -rf node_modules package-lock.json
npm install
```

**Build échoue:**
```bash
# Vérifier TypeScript
npm run type-check
```

### Logs de Debug
- **Browser DevTools** - F12 → Console/Network
- **Terminal** - Messages npm run dev
- **Fichiers temp** - `./temp/` pour debug uploads

---

## 📞 Support & Ressources

### Documentation
- [Next.js 14](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [TypeScript](https://www.typescriptlang.org/docs)

### Personnalisation
- **Couleurs** - `tailwind.config.js`
- **Pipeline** - `src/lib/config.ts`
- **API** - `src/app/api/process/route.ts`

---

## 🎯 Prochaines Étapes

1. **✅ Tester l'interface** - Uploader une image test
2. **🔗 Intégrer Python** - Suivre `INTEGRATION.md`
3. **🎨 Personnaliser** - Adapter couleurs/layout
4. **🚀 Déployer** - Vercel/Netlify pour production

**Temps estimé total:** 30 minutes pour setup + intégration basique

---

## 🏆 Avantages

- **⚡ Performance** - Next.js 14 avec optimisations
- **🎨 UX Moderne** - Interface intuitive et professionnelle
- **🔧 Modulaire** - Code organisé et maintenable
- **📱 Responsive** - Fonctionne sur tous appareils
- **🧠 IA Ready** - Intégration pipeline Python simplifiée
- **📊 Métriques** - Visualisation complète des résultats

**Parfait pour démonstrations, recherche, et usage clinique!**
