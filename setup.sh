#!/bin/bash

# Script d'installation et de configuration pour l'interface Pipeline Échographique
echo "🔬 Installation de l'Interface Pipeline Échographique"
echo "=================================================="

# Vérifier que Node.js est installé
if ! command -v node &> /dev/null; then
    echo "❌ Node.js n'est pas installé. Veuillez installer Node.js 18+ depuis https://nodejs.org/"
    exit 1
fi

# Vérifier la version de Node.js
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18+ requise. Version actuelle: $(node -v)"
    exit 1
fi

echo "✅ Node.js version $(node -v) détectée"

# Installer les dépendances
echo "📦 Installation des dépendances..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de l'installation des dépendances"
    exit 1
fi

# Copier le fichier d'environnement
if [ ! -f .env.local ]; then
    echo "📝 Création du fichier de configuration..."
    cp .env.example .env.local
    echo "✅ Fichier .env.local créé. Modifiez les chemins des modèles selon votre configuration."
else
    echo "⚠️  Fichier .env.local existe déjà"
fi

# Créer les dossiers temporaires
echo "📁 Création des dossiers temporaires..."
mkdir -p temp/uploads
mkdir -p temp/outputs

# Vérification des modèles (optionnel)
echo "🧠 Vérification des modèles..."
MODELS_DIR="C:\Users\ASUS\Desktop\Final\models"
if [ -d "$MODELS_DIR" ]; then
    echo "✅ Dossier des modèles trouvé: $MODELS_DIR"
    
    # Lister les modèles disponibles
    echo "📋 Modèles disponibles:"
    ls -la "$MODELS_DIR"/*.pth 2>/dev/null || echo "⚠️  Aucun fichier .pth trouvé dans $MODELS_DIR"
else
    echo "⚠️  Dossier des modèles non trouvé: $MODELS_DIR"
    echo "   Assurez-vous que vos modèles entraînés sont dans ce dossier"
fi

echo ""
echo "🎉 Installation terminée!"
echo ""
echo "🚀 Pour démarrer l'application:"
echo "   npm run dev"
echo ""
echo "🌐 L'application sera disponible sur:"
echo "   http://localhost:3000"
echo ""
echo "🔧 Configuration:"
echo "   - Modifiez .env.local pour adapter les chemins des modèles"
echo "   - Assurez-vous que Python et PyTorch sont installés pour l'intégration backend"
echo ""
echo "📚 Commandes disponibles:"
echo "   npm run dev      - Démarrer en mode développement"
echo "   npm run build    - Construire pour la production"
echo "   npm run start    - Démarrer en mode production"
echo "   npm run lint     - Vérifier le code"
echo ""
