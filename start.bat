@echo off
echo.
echo ===============================================
echo  🔬 Interface Pipeline Echographique
echo ===============================================
echo.

REM Vérifier que Node.js est installé
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Node.js n'est pas installé!
    echo Téléchargez-le depuis: https://nodejs.org/
    pause
    exit /b 1
)

REM Afficher la version de Node.js
for /f "tokens=*" %%i in ('node -v') do set NODE_VERSION=%%i
echo ✅ Node.js %NODE_VERSION% détecté

REM Vérifier que les dépendances sont installées
if not exist "node_modules" (
    echo.
    echo 📦 Installation des dépendances...
    npm install
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Erreur lors de l'installation
        pause
        exit /b 1
    )
)

REM Vérifier le fichier de configuration
if not exist ".env.local" (
    echo.
    echo 📝 Création du fichier de configuration...
    copy ".env.example" ".env.local"
    echo ✅ Fichier .env.local créé
    echo ⚠️  Modifiez les chemins des modèles dans .env.local
)

REM Créer les dossiers temporaires
if not exist "temp\uploads" mkdir temp\uploads
if not exist "temp\outputs" mkdir temp\outputs
if not exist "logs" mkdir logs

echo.
echo 🚀 Démarrage de l'application...
echo 🌐 L'interface sera disponible sur: http://localhost:3000
echo.
echo Commandes disponibles:
echo   d = Mode développement (npm run dev)
echo   b = Build production (npm run build)
echo   s = Start production (npm run start)
echo   q = Quitter
echo.

:menu
set /p choice="Votre choix (d/b/s/q): "

if /i "%choice%"=="d" (
    echo.
    echo 🚀 Démarrage en mode développement...
    npm run dev
    goto end
)

if /i "%choice%"=="b" (
    echo.
    echo 🏗️ Construction pour production...
    npm run build
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Build réussi!
        echo.
        set /p startprod="Démarrer en production? (y/n): "
        if /i "!startprod!"=="y" (
            npm run start
        )
    )
    goto end
)

if /i "%choice%"=="s" (
    echo.
    echo 🚀 Démarrage en mode production...
    npm run start
    goto end
)

if /i "%choice%"=="q" (
    goto end
)

echo ❌ Choix invalide. Utilisez d, b, s, ou q.
goto menu

:end
echo.
echo 👋 Merci d'avoir utilisé l'Interface Pipeline Échographique!
pause
