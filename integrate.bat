@echo off
echo.
echo ===============================================
echo  🔗 INTÉGRATION PIPELINE PYTHON
echo ===============================================
echo.

echo 📋 Cette opération va:
echo   1. Sauvegarder votre pipeline.py original
echo   2. Ajouter l'interface CLI pour l'API web
echo   3. Configurer l'environnement
echo   4. Tester l'intégration
echo.

set /p confirm="Continuer? (y/n): "
if /i NOT "%confirm%"=="y" (
    echo ❌ Opération annulée
    pause
    exit /b 0
)

echo.
echo 🐍 Exécution du script d'intégration Python...
cd C:\Users\ASUS\Desktop\Final
python integrate_pipeline.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Erreur lors de l'intégration!
    echo 💡 Vérifiez que:
    echo    - Python est installé et dans le PATH
    echo    - Le fichier pipeline.py existe
    echo    - Vous avez les permissions d'écriture
    pause
    exit /b 1
)

echo.
echo 🔄 Redémarrage de l'interface web...
cd C:\Users\ASUS\Desktop\Final\ultrasound-pipeline-interface

echo.
echo 🎉 INTÉGRATION TERMINÉE!
echo.
echo 🚀 L'interface va maintenant utiliser votre vrai pipeline Python
echo 📊 Les images seront traitées par vos 4 modèles IA
echo ⏱️ Le traitement prendra quelques secondes par image
echo.
echo 💡 Conseil: Testez avec une petite image d'abord
echo.

set /p start="Démarrer l'interface maintenant? (y/n): "
if /i "%start%"=="y" (
    echo.
    echo 🌐 Ouverture de l'interface...
    npm run dev
) else (
    echo.
    echo 📝 Pour démarrer plus tard: npm run dev
    echo 🌐 Puis ouvrir: http://localhost:3000
)

pause
