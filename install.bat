@echo off
echo.
echo ===============================================
echo  🔧 Installation via CMD (sans PowerShell)
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

echo.
echo 📦 Installation des dépendances...
npm install

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Erreur lors de l'installation
    pause
    exit /b 1
)

echo.
echo 📝 Configuration...
if not exist ".env.local" (
    copy ".env.example" ".env.local"
    echo ✅ Fichier .env.local créé
)

REM Créer dossiers temporaires
if not exist "temp" mkdir temp
if not exist "temp\uploads" mkdir temp\uploads
if not exist "temp\outputs" mkdir temp\outputs

echo.
echo 🎉 Installation terminée avec succès!
echo.
echo 🚀 Pour démarrer l'application:
echo    npm run dev
echo.
echo 🌐 L'interface sera disponible sur:
echo    http://localhost:3000
echo.
pause
