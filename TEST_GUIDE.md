# 🧪 GUIDE DE TEST - Pipeline Intégré

## ✅ Étapes de Test Rapide

### 1. 🔗 Intégrer le Pipeline (Une seule fois)

**Option A - Script automatique (Recommandé):**
```cmd
cd C:\Users\ASUS\Desktop\Final\ultrasound-pipeline-interface
integrate.bat
```

**Option B - Manuel:**
```cmd
cd C:\Users\ASUS\Desktop\Final
python integrate_pipeline.py
```

### 2. 🚀 Démarrer l'Interface

```cmd
cd C:\Users\ASUS\Desktop\Final\ultrasound-pipeline-interface
npm run dev
```

### 3. 🧪 Tester le Pipeline

1. **Ouvrir:** http://localhost:3000
2. **Upload:** Glisser une image échographique
3. **Paramètres:** Ajuster si désiré (70% contraste recommandé)  
4. **Traiter:** Cliquer "Appliquer et Analyser"
5. **Attendre:** 3-10 secondes selon votre GPU/CPU
6. **Vérifier:** L'image de droite doit être différente de celle de gauche

## 🔍 Signes de Succès

### ✅ Pipeline Fonctionne
- **Images différentes** - Originale ≠ Traitée
- **4 étapes animées** - Icônes qui passent au vert
- **Métriques réelles** - PSNR, SSIM, Dice scores
- **Segmentation colorée** - Masque avec couleurs distinctes
- **Temps réaliste** - 2-10 secondes selon hardware

### ❌ Pipeline Simulé (à corriger)
- **Images identiques** - Aucun changement visible
- **Temps très rapide** - <1 seconde
- **Métriques fixes** - Toujours les mêmes valeurs

## 🛠️ Dépannage

### Problème: "Erreur lors du traitement"

**Solutions:**
1. **Vérifier Python:**
   ```cmd
   python --version
   # Doit afficher Python 3.8+
   ```

2. **Vérifier PyTorch:**
   ```cmd
   python -c "import torch; print(torch.__version__)"
   # Doit afficher version PyTorch
   ```

3. **Vérifier les modèles:**
   ```cmd
   dir C:\Users\ASUS\Desktop\Final\models\*.pth
   # Doit lister vos 4 modèles .pth
   ```

4. **Vérifier le pipeline:**
   ```cmd
   python C:\Users\ASUS\Desktop\Final\pipeline.py --help
   # Doit afficher l'aide CLI
   ```

### Problème: Images identiques

**Causes possibles:**
- Interface CLI pas ajoutée au pipeline.py
- Erreur Python non visible
- Modèles non chargés correctement

**Solution:**
```cmd
# Relancer l'intégration
cd C:\Users\ASUS\Desktop\Final
python integrate_pipeline.py
```

### Problème: Temps très long (>30s)

**Optimisations:**
- Vérifier GPU disponible: `nvidia-smi`
- Réduire taille d'image test
- Vérifier CUDA dans .env.local

## 📊 Métriques Attendues

### GPU (CUDA)
- **Temps:** 2-5 secondes
- **PSNR:** 25-35 dB
- **SSIM:** 0.75-0.95
- **Dice:** 0.80-0.90

### CPU Seulement  
- **Temps:** 5-15 secondes
- **Qualité:** Identique, juste plus lent

## 🎯 Test avec Image Spécifique

### Image Recommandée
- **Format:** PNG ou JPG
- **Taille:** 256x256 à 512x512 pixels
- **Type:** Échographie cardiaque en niveaux de gris
- **Bruit:** Visible (pour tester le débruitage)

### Paramètres de Test
- **Contraste:** 70%
- **Luminosité:** 60%  
- **Réduction bruit:** 80%

## 📞 Support

### Logs Utiles
```cmd
# Console navigateur (F12 → Console)
# Voir les messages de l'API

# Terminal npm run dev
# Voir les logs du serveur
```

### Fichiers à Vérifier
- `C:\Users\ASUS\Desktop\Final\pipeline.py` - Doit contenir `create_cli_interface`
- `.env.local` - Chemins des modèles corrects
- `temp/` - Dossier créé automatiquement

### Test Manuel CLI
```cmd
# Tester directement le pipeline modifié
cd C:\Users\ASUS\Desktop\Final
python pipeline.py --input test_image.png --output test_output --contrast 0.7
```

---

## 🎉 Succès!

Si vous voyez une **différence claire** entre l'image originale et l'image traitée, avec des **métriques réalistes** et un **masque de segmentation coloré**, alors votre pipeline est parfaitement intégré!

**Vous pouvez maintenant utiliser l'interface pour:**
- Démonstrations professionnelles
- Tests de nouveaux modèles  
- Analyse comparative
- Formation et enseignement

**Félicitations! 🏆**
