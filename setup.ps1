# Script d'installation PowerShell pour l'Interface Pipeline Échographique
Write-Host "🔬 Installation de l'Interface Pipeline Échographique" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Vérifier que Node.js est installé
try {
    $nodeVersion = node -v
    Write-Host "✅ Node.js version $nodeVersion détectée" -ForegroundColor Green
    
    # Vérifier la version
    $versionNumber = [int]($nodeVersion -replace "v(\d+)\..*", '$1')
    if ($versionNumber -lt 18) {
        Write-Host "❌ Node.js version 18+ requise. Version actuelle: $nodeVersion" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Node.js n'est pas installé. Veuillez installer Node.js 18+ depuis https://nodejs.org/" -ForegroundColor Red
    exit 1
}

# Installer les dépendances
Write-Host "📦 Installation des dépendances..." -ForegroundColor Yellow
npm install

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors de l'installation des dépendances" -ForegroundColor Red
    exit 1
}

# Copier le fichier d'environnement
if (-not (Test-Path ".env.local")) {
    Write-Host "📝 Création du fichier de configuration..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env.local"
    Write-Host "✅ Fichier .env.local créé. Modifiez les chemins des modèles selon votre configuration." -ForegroundColor Green
} else {
    Write-Host "⚠️  Fichier .env.local existe déjà" -ForegroundColor Yellow
}

# Créer les dossiers temporaires
Write-Host "📁 Création des dossiers temporaires..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "temp\uploads" -Force | Out-Null
New-Item -ItemType Directory -Path "temp\outputs" -Force | Out-Null

# Vérification des modèles
Write-Host "🧠 Vérification des modèles..." -ForegroundColor Yellow
$modelsDir = "C:\Users\ASUS\Desktop\Final\models"
if (Test-Path $modelsDir) {
    Write-Host "✅ Dossier des modèles trouvé: $modelsDir" -ForegroundColor Green
    
    # Lister les modèles disponibles
    Write-Host "📋 Modèles disponibles:" -ForegroundColor Cyan
    $models = Get-ChildItem -Path $modelsDir -Filter "*.pth" -ErrorAction SilentlyContinue
    if ($models.Count -gt 0) {
        $models | ForEach-Object { Write-Host "   - $($_.Name)" -ForegroundColor White }
    } else {
        Write-Host "⚠️  Aucun fichier .pth trouvé dans $modelsDir" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  Dossier des modèles non trouvé: $modelsDir" -ForegroundColor Yellow
    Write-Host "   Assurez-vous que vos modèles entraînés sont dans ce dossier" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🎉 Installation terminée!" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Pour démarrer l'application:" -ForegroundColor Cyan
Write-Host "   npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "🌐 L'application sera disponible sur:" -ForegroundColor Cyan
Write-Host "   http://localhost:3000" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Configuration:" -ForegroundColor Cyan
Write-Host "   - Modifiez .env.local pour adapter les chemins des modèles" -ForegroundColor Gray
Write-Host "   - Assurez-vous que Python et PyTorch sont installés pour l'intégration backend" -ForegroundColor Gray
Write-Host ""
Write-Host "📚 Commandes disponibles:" -ForegroundColor Cyan
Write-Host "   npm run dev      - Démarrer en mode développement" -ForegroundColor Gray
Write-Host "   npm run build    - Construire pour la production" -ForegroundColor Gray
Write-Host "   npm run start    - Démarrer en mode production" -ForegroundColor Gray
Write-Host "   npm run lint     - Vérifier le code" -ForegroundColor Gray
Write-Host ""

# Pause pour que l'utilisateur puisse lire
Write-Host "Appuyez sur une touche pour continuer..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
