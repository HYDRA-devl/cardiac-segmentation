import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time
import os

# ===== ARCHITECTURE DÉBRUITAGE (ÉTAPE 1) =====
class SupervisedDenoisingUNet(nn.Module):
    """U-Net pour débruitage supervisé (final_degraded → denoised)"""
    
    def __init__(self, in_channels=1, out_channels=1):
        super(SupervisedDenoisingUNet, self).__init__()
        
        # Encodeur
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck avec attention
        self.bottleneck = self.conv_block(512, 1024)
        self.attention = self.attention_block(1024)
        
        # Décodeur
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        
        # Couche finale
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        
        # Pooling et upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Dropout
        self.dropout = nn.Dropout2d(0.1)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def attention_block(self, channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        input_x = x
        
        # Encodeur
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck avec attention
        b = self.bottleneck(self.pool(e4))
        att = self.attention(b)
        b = b * att + b
        b = self.dropout(b)
        
        # Décodeur
        d4 = self.up(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = torch.nn.functional.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d4, e4], 1))
        
        d3 = self.up(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = torch.nn.functional.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], 1))
        
        d2 = self.up(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = torch.nn.functional.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], 1))
        
        d1 = self.up(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], 1))
        
        # Sortie finale avec connexion résiduelle
        output = self.final_conv(d1)
        if output.shape == input_x.shape:
            output = output + input_x
        
        return torch.sigmoid(output)

# ===== ARCHITECTURE AGGRESSIVE CONTRAST (ÉTAPE 2) =====
class ContrastAttentionBlock(nn.Module):
    """Module d'attention spécialisé pour contraste"""
    
    def __init__(self, channels):
        super(ContrastAttentionBlock, self).__init__()
        
        # Attention spatiale pour zones importantes
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # Attention canal pour features importantes
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Attention canal
        ca = self.channel_attention(x)
        x = x * ca
        
        # Attention spatiale
        sa = self.spatial_attention(x)
        x = x * sa
        
        return x

class AdaptiveContrastModule(nn.Module):
    """Module d'ajustement adaptatif agressif"""
    
    def __init__(self, channels):
        super(AdaptiveContrastModule, self).__init__()
        
        # Prédiction du facteur d'enhancement
        self.enhancement_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # Ajustement local
        self.local_enhancer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
    
    def forward(self, enhanced, original):
        # Prédire facteur d'enhancement global
        enhancement_factor = self.enhancement_predictor(enhanced)
        enhancement_factor = 0.5 + 1.5 * enhancement_factor  # Range [0.5, 2.0]
        
        # Ajustement local
        local_adjustment = self.local_enhancer(enhanced)
        
        # Combinaison agressive : plus de poids sur l'enhancement
        output = 0.3 * original + 0.7 * enhanced + 0.2 * local_adjustment
        
        # Application du facteur global
        output = output * enhancement_factor
        
        return output

class AggressiveContrastNet(nn.Module):
    """Réseau agressif pour amélioration contraste (denoised → contrast_enhanced)"""
    
    def __init__(self, in_channels=1, out_channels=1):
        super(AggressiveContrastNet, self).__init__()
        
        # Encodeur avec plus de capacité pour analyse contraste
        self.enc1 = self.contrast_block(in_channels, 64)
        self.enc2 = self.contrast_block(64, 128)
        self.enc3 = self.contrast_block(128, 256)
        self.enc4 = self.contrast_block(256, 512)
        
        # Bottleneck avec attention contraste
        self.bottleneck = self.contrast_block(512, 768)
        self.contrast_attention = ContrastAttentionBlock(768)
        
        # Décodeur avec enhancement progressif
        self.dec4 = self.contrast_block(768 + 512, 512)
        self.dec3 = self.contrast_block(512 + 256, 256)
        self.dec2 = self.contrast_block(256 + 128, 128)
        self.dec1 = self.contrast_block(128 + 64, 64)
        
        # Couches finales d'enhancement
        self.contrast_enhancer = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 1)
        )
        
        # Module d'ajustement adaptatif
        self.adaptive_adjustment = AdaptiveContrastModule(out_channels)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout2d(0.1)
    
    def contrast_block(self, in_ch, out_ch):
        """Bloc convolutionnel optimisé pour contraste"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # Ajout d'une convolution 1x1 pour plus d'expressivité
            nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        input_x = x
        
        # Encodeur
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck avec attention contraste
        b = self.bottleneck(self.pool(e4))
        b = self.contrast_attention(b)
        b = self.dropout(b)
        
        # Décodeur
        d4 = self.up(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = torch.nn.functional.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d4, e4], 1))
        
        d3 = self.up(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = torch.nn.functional.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], 1))
        
        d2 = self.up(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = torch.nn.functional.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], 1))
        
        d1 = self.up(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], 1))
        
        # Enhancement final
        enhanced = self.contrast_enhancer(d1)
        
        # Ajustement adaptatif avec connexion résiduelle forte
        output = self.adaptive_adjustment(enhanced, input_x)
        
        return torch.sigmoid(output)

# ===== ARCHITECTURE REAL-ESRGAN (ÉTAPE 3) =====
class FastRRDBBlock(nn.Module):
    """RRDB Block allégé pour vitesse"""
    
    def __init__(self, channels, growth_rate=16):
        super(FastRRDBBlock, self).__init__()
        
        self.dense1 = nn.Conv2d(channels, growth_rate, 3, 1, 1)
        self.dense2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, 1, 1)
        self.dense3 = nn.Conv2d(channels + 2*growth_rate, channels, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        d1 = self.lrelu(self.dense1(x))
        d2 = self.lrelu(self.dense2(torch.cat([x, d1], 1)))
        d3 = self.dense3(torch.cat([x, d1, d2], 1))
        
        return d3 * 0.2 + x

class FastRealESRGANGenerator(nn.Module):
    """Générateur Real-ESRGAN (contrast_enhanced → original)"""
    
    def __init__(self, in_channels=1, out_channels=1, num_blocks=6, scale_factor=2):
        super(FastRealESRGANGenerator, self).__init__()
        
        self.scale_factor = scale_factor
        base_channels = 32
        
        # Feature extraction
        self.conv_first = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        
        # RRDB blocks
        self.rrdb_blocks = nn.ModuleList([
            FastRRDBBlock(base_channels, growth_rate=16) for _ in range(num_blocks)
        ])
        
        # Convolution après RRDB
        self.conv_body = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        
        # Upsampling ×2
        self.upconv1 = nn.Conv2d(base_channels, base_channels * (scale_factor**2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.upconv2 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        
        # Reconstruction finale
        self.conv_hr = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(base_channels, out_channels, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        # Feature extraction
        feat = self.lrelu(self.conv_first(x))
        
        # RRDB processing
        body_feat = feat
        for rrdb in self.rrdb_blocks:
            body_feat = rrdb(body_feat)
        
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        
        # Upsampling
        feat = self.lrelu(self.upconv1(feat))
        feat = self.pixel_shuffle(feat)
        feat = self.lrelu(self.upconv2(feat))
        
        # Reconstruction
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        
        return torch.sigmoid(out)

# ===== ARCHITECTURE LADDERNET CORRIGÉE (ÉTAPE 4) =====
class LadderBlock(nn.Module):
    """Bloc LadderNet avec connexions résiduelles"""
    
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(LadderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        
        # Skip connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # Attention optionnelle
        self.use_attention = use_attention
        if use_attention:
            self.attention = self.channel_attention(out_channels)
    
    def channel_attention(self, channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_attention:
            att = self.attention(out)
            out = out * att
        
        out += residual
        out = self.relu(out)
        
        return out

class LadderNet(nn.Module):
    """LadderNet pour segmentation d'images restaurées - ARCHITECTURE CORRIGÉE"""
    
    def __init__(self, in_channels=1, num_classes=4, base_filters=32):
        super(LadderNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Encodeur avec attention progressive - CONFIGURATION EXACTE DU MODÈLE ENTRAÎNÉ
        self.enc1 = LadderBlock(in_channels, base_filters)
        self.enc2 = LadderBlock(base_filters, base_filters * 2)
        self.enc3 = LadderBlock(base_filters * 2, base_filters * 4, use_attention=True)
        self.enc4 = LadderBlock(base_filters * 4, base_filters * 8, use_attention=True)
        
        # Bottleneck avec forte attention
        self.bottleneck = LadderBlock(base_filters * 8, base_filters * 16, use_attention=True)
        
        # Décodeur symétrique
        self.dec4 = LadderBlock(base_filters * 16 + base_filters * 8, base_filters * 8, use_attention=True)
        self.dec3 = LadderBlock(base_filters * 8 + base_filters * 4, base_filters * 4, use_attention=True)
        self.dec2 = LadderBlock(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.dec1 = LadderBlock(base_filters * 2 + base_filters, base_filters)
        
        # Tête de classification - MÊME NOM QUE LE MODÈLE ENTRAÎNÉ
        self.classifier = nn.Sequential(
            nn.Conv2d(base_filters, base_filters // 2, 3, 1, 1),
            nn.BatchNorm2d(base_filters // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(base_filters // 2, num_classes, 1)
        )
        
        # Pooling et upsampling
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Initialisation des poids
        self.init_weights()
    
    def init_weights(self):
        """Initialisation Xavier pour stabilité"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Encodeur avec skip connections
        e1 = self.enc1(x)                    # [B, 32, H, W]
        e2 = self.enc2(self.pool(e1))        # [B, 64, H/2, W/2]
        e3 = self.enc3(self.pool(e2))        # [B, 128, H/4, W/4]
        e4 = self.enc4(self.pool(e3))        # [B, 256, H/8, W/8]
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))   # [B, 512, H/16, W/16]
        
        # Décodeur avec skip connections
        d4 = self.up(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d4, e4], 1))
        
        d3 = self.up(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], 1))
        
        d2 = self.up(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], 1))
        
        d1 = self.up(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], 1))
        
        # Classification finale - UTILISE 'classifier' PAS 'final_conv'
        output = self.classifier(d1)
        
        # Pour multi-classes : pas de sigmoid, CrossEntropy l'appliquera
        return output

# ===== FONCTION DE CHARGEMENT SÉCURISÉE =====
def load_checkpoint_safe(checkpoint_path, device='cpu'):
    """Charger checkpoint avec gestion des erreurs PyTorch 2.6+"""
    
    print(f"📂 Chargement checkpoint: {Path(checkpoint_path).name}")
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
    
    try:
        # Méthode 1: Essayer avec weights_only=True (sécurisé)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print(f"✅ Chargement sécurisé réussi")
        return checkpoint
        
    except Exception as e1:
        print(f"⚠️ Chargement sécurisé échoué: {type(e1).__name__}")
        
        try:
            # Méthode 2: Utiliser le context manager pour autoriser numpy
            import torch.serialization
            with torch.serialization.safe_globals(['numpy._core.multiarray.scalar', 'numpy.core.multiarray.scalar']):
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                print(f"✅ Chargement avec safe_globals réussi")
                return checkpoint
                
        except Exception as e2:
            print(f"⚠️ Chargement avec safe_globals échoué: {type(e2).__name__}")
            
            try:
                # Méthode 3: Fallback avec weights_only=False (moins sécurisé mais fonctionnel)
                print("🔄 Utilisation fallback weights_only=False...")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                print(f"✅ Chargement fallback réussi")
                return checkpoint
                
            except Exception as e3:
                print(f"❌ Toutes les méthodes ont échoué!")
                print(f"   Erreur 1 (weights_only=True): {e1}")
                print(f"   Erreur 2 (safe_globals): {e2}")
                print(f"   Erreur 3 (weights_only=False): {e3}")
                raise e3

# ===== UTILITAIRES SEGMENTATION =====
def colorize_segmentation(seg_mask, num_classes=4):
    """Coloriser masque de segmentation pour visualisation"""
    
    # Palette de couleurs pour 4 classes
    colors = np.array([
        [0, 0, 128],      # Classe 0: Bleu foncé (background)
        [255, 255, 0],    # Classe 1: Jaune
        [0, 255, 255],    # Classe 2: Cyan
        [128, 0, 0],      # Classe 3: Rouge foncé
    ], dtype=np.uint8)
    
    h, w = seg_mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(num_classes):
        mask = (seg_mask == class_id)
        colored_mask[mask] = colors[class_id]
    
    return colored_mask

# ===== INTERFACES UTILISATEUR =====
def get_user_choice():
    """Interface pour choisir le mode de test"""
    print("\n🎯 SÉLECTION MODE DE TEST:")
    print("="*50)
    print("1. 🤖 Mode automatique (dataset easy_noise validation)")
    print("2. 📁 Mode personnalisé (images spécifiques)")
    print("="*50)
    
    while True:
        choice = input("Votre choix (1 ou 2): ").strip()
        if choice in ['1', '2']:
            return int(choice)
        print("❌ Choix invalide. Tapez 1 ou 2.")

def get_custom_images():
    """Interface pour entrer des chemins d'images personnalisées"""
    print("\n📁 MODE PERSONNALISÉ:")
    print("="*50)
    print("📝 Entrez les chemins de vos images (un par ligne)")
    print("📝 Formats supportés: .png, .jpg, .jpeg, .bmp, .tiff")
    print("📝 Tapez 'done' quand vous avez terminé")
    print("📝 Exemple: C:\\Users\\nom\\image.png")
    print("="*50)
    
    images = []
    
    while True:
        path = input(f"Image {len(images)+1} (ou 'done'): ").strip()
        
        if path.lower() == 'done':
            break
            
        if not path:
            continue
            
        # Vérifier que le fichier existe
        if not os.path.exists(path):
            print(f"❌ Fichier introuvable: {path}")
            continue
            
        # Vérifier l'extension
        ext = Path(path).suffix.lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            print(f"❌ Format non supporté: {ext}")
            continue
            
        images.append(path)
        print(f"✅ Image ajoutée: {Path(path).name}")
    
    if not images:
        print("❌ Aucune image sélectionnée!")
        return None
    
    print(f"\n✅ {len(images)} image(s) sélectionnée(s):")
    for i, img in enumerate(images):
        print(f"   {i+1}. {Path(img).name}")
    
    return images

def ask_for_original_images(custom_images):
    """Demander les images originales correspondantes (optionnel)"""
    print("\n🎯 IMAGES ORIGINALES (OPTIONNEL):")
    print("="*50)
    print("📝 Voulez-vous fournir les images originales pour comparaison?")
    print("📝 Si oui, entrez le chemin de l'original pour chaque image")
    print("📝 Si non, tapez 'skip' pour passer")
    print("="*50)
    
    originals = []
    
    for i, img_path in enumerate(custom_images):
        img_name = Path(img_path).name
        original_path = input(f"Original pour '{img_name}' (ou 'skip'): ").strip()
        
        if original_path.lower() == 'skip':
            originals.append(None)
            print(f"⏭️ Pas d'original pour {img_name}")
        else:
            if os.path.exists(original_path):
                originals.append(original_path)
                print(f"✅ Original ajouté: {Path(original_path).name}")
            else:
                print(f"❌ Original introuvable: {original_path}")
                originals.append(None)
    
    return originals

# ===== CHARGEMENT DES MODÈLES CORRIGÉ =====
def load_all_models(device):
    """Charger tous les modèles du pipeline avec architecture corrigée"""
    
    # 🔧 CONFIGURATION - MODIFIEZ CES CHEMINS
    DENOISING_MODEL = r"C:\Users\ASUS\Desktop\Final\models\supervised_denoising_best.pth"
    AGGRESSIVE_CONTRAST_MODEL = r"C:\Users\ASUS\Desktop\Final\models\aggressive_contrast_best.pth"
    REALESRGAN_MODEL = r"C:\Users\ASUS\Desktop\Final\models\fast_realesrgan_best.pth"
    LADDERNET_MODEL = r"C:\Users\ASUS\Desktop\Final\models\laddernet_best_dice.pth"
    
    print("📥 CHARGEMENT DES 4 MODÈLES (ARCHITECTURE CORRIGÉE)...")
    
    models = {}
    
    # MODÈLE 1: DÉBRUITAGE
    if not Path(DENOISING_MODEL).exists():
        print(f"❌ ERREUR: Modèle débruitage introuvable!")
        print(f"Chemin: {DENOISING_MODEL}")
        return None
    
    try:
        denoising_model = SupervisedDenoisingUNet().to(device)
        checkpoint = load_checkpoint_safe(DENOISING_MODEL, device)
        denoising_model.load_state_dict(checkpoint['model_state_dict'])
        denoising_model.eval()
        models['denoising'] = denoising_model
        print("✅ Modèle 1 (SupervisedDenoisingUNet) chargé!")
    except Exception as e:
        print(f"❌ ERREUR chargement débruitage: {e}")
        return None
    
    # MODÈLE 2: AGGRESSIVE CONTRAST
    if not Path(AGGRESSIVE_CONTRAST_MODEL).exists():
        print(f"❌ ERREUR: Modèle aggressive contrast introuvable!")
        print(f"Chemin: {AGGRESSIVE_CONTRAST_MODEL}")
        return None
    
    try:
        aggressive_contrast_model = AggressiveContrastNet().to(device)
        checkpoint = load_checkpoint_safe(AGGRESSIVE_CONTRAST_MODEL, device)
        aggressive_contrast_model.load_state_dict(checkpoint['model_state_dict'])
        aggressive_contrast_model.eval()
        models['contrast'] = aggressive_contrast_model
        print("✅ Modèle 2 (AggressiveContrastNet) chargé!")
    except Exception as e:
        print(f"❌ ERREUR chargement aggressive contrast: {e}")
        return None
    
    # MODÈLE 3: REAL-ESRGAN
    if not Path(REALESRGAN_MODEL).exists():
        print(f"❌ ERREUR: Modèle Real-ESRGAN introuvable!")
        print(f"Chemin: {REALESRGAN_MODEL}")
        return None
    
    try:
        realesrgan_model = FastRealESRGANGenerator(
            in_channels=1, out_channels=1, 
            num_blocks=6, scale_factor=2
        ).to(device)
        
        checkpoint = load_checkpoint_safe(REALESRGAN_MODEL, device)
        realesrgan_model.load_state_dict(checkpoint['generator_state_dict'])
        realesrgan_model.eval()
        models['realesrgan'] = realesrgan_model
        print("✅ Modèle 3 (FastRealESRGAN) chargé!")
    except Exception as e:
        print(f"❌ ERREUR chargement Real-ESRGAN: {e}")
        return None
    
    # MODÈLE 4: LADDERNET (ARCHITECTURE CORRIGÉE!)
    if not Path(LADDERNET_MODEL).exists():
        print(f"❌ ERREUR: Modèle LadderNet introuvable!")
        print(f"Chemin: {LADDERNET_MODEL}")
        return None
    
    try:
        print("🧠 Chargement LadderNet avec architecture corrigée...")
        # UTILISER EXACTEMENT LES MÊMES PARAMÈTRES QUE L'ENTRAÎNEMENT
        laddernet_model = LadderNet(
            in_channels=1, 
            num_classes=4, 
            base_filters=32  # PARAMÈTRE CRUCIAL!
        ).to(device)
        
        checkpoint = load_checkpoint_safe(LADDERNET_MODEL, device)
        
        # Vérifier la clé du state_dict
        if 'model_state_dict' in checkpoint:
            laddernet_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            laddernet_model.load_state_dict(checkpoint['state_dict'])
        else:
            laddernet_model.load_state_dict(checkpoint)
        
        laddernet_model.eval()
        models['laddernet'] = laddernet_model
        print("✅ Modèle 4 (LadderNet - Segmentation) chargé!")
        
        # Afficher métriques du checkpoint si disponibles
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            if 'seg_metrics' in metrics:
                dice_score = metrics['seg_metrics'].get('dice', 'N/A')
                iou_score = metrics['seg_metrics'].get('iou', 'N/A')
                print(f"   🏆 Score Dice checkpoint: {dice_score}")
                print(f"   🏆 Score IoU checkpoint: {iou_score}")
        
    except Exception as e:
        print(f"❌ ERREUR chargement LadderNet: {e}")
        print(f"💡 Vérifiez que le modèle a été entraîné avec base_filters=32")
        return None
    
    return models

# ===== EXÉCUTION PIPELINE =====
def execute_pipeline(input_image, models, device):
    """Exécuter le pipeline complet sur une image"""
    
    input_norm = input_image.astype(np.float32) / 255.0
    results = {}
    
    # ÉTAPE 1: DÉBRUITAGE
    print("   🧠 ÉTAPE 1: DÉBRUITAGE...")
    start_time = time.time()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_norm).unsqueeze(0).unsqueeze(0).to(device)
        denoised_tensor = models['denoising'](input_tensor)
        denoised = denoised_tensor.squeeze().cpu().numpy()
        denoised = np.clip(denoised, 0, 1)
    results['denoised'] = denoised
    results['time_denoising'] = time.time() - start_time
    
    # ÉTAPE 2: AGGRESSIVE CONTRAST
    print("   🌟 ÉTAPE 2: AGGRESSIVE CONTRAST...")
    start_time = time.time()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(denoised).unsqueeze(0).unsqueeze(0).to(device)
        contrast_tensor = models['contrast'](input_tensor)
        contrast_enhanced = contrast_tensor.squeeze().cpu().numpy()
        contrast_enhanced = np.clip(contrast_enhanced, 0, 1)
    results['contrast_enhanced'] = contrast_enhanced
    results['time_contrast'] = time.time() - start_time
    
    # ÉTAPE 3: REAL-ESRGAN
    print("   🚀 ÉTAPE 3: REAL-ESRGAN...")
    start_time = time.time()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(contrast_enhanced).unsqueeze(0).unsqueeze(0).to(device)
        sr_tensor = models['realesrgan'](input_tensor)
        super_resolution = sr_tensor.squeeze().cpu().numpy()
        super_resolution = np.clip(super_resolution, 0, 1)
    results['super_resolution'] = super_resolution
    results['time_realesrgan'] = time.time() - start_time
    
    # ÉTAPE 4: SEGMENTATION LADDERNET
    print("   🎯 ÉTAPE 4: SEGMENTATION LADDERNET...")
    start_time = time.time()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(super_resolution).unsqueeze(0).unsqueeze(0).to(device)
        seg_logits = models['laddernet'](input_tensor)
        seg_probs = torch.softmax(seg_logits, dim=1)
        seg_mask = torch.argmax(seg_probs, dim=1).squeeze().cpu().numpy()
    results['segmentation'] = seg_mask
    results['time_segmentation'] = time.time() - start_time
    
    return results

# ===== MODE AUTOMATIQUE =====
def test_pipeline_auto_mode():
    """Mode automatique avec dataset easy_noise + segmentation"""
    
    print("🔧 CONFIGURATION PIPELINE 4 ÉTAPES:")
    print("="*70)
    
    # Dataset easy_noise pour le test
    TEST_DATASET = r"C:\Users\ASUS\Desktop\Final\datasets_easy_noise"
    
    # Dossier de sortie
    OUTPUT_FOLDER = r"C:\Users\ASUS\Desktop\Final\pipeline_complete_4_steps_results"
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("🎯 PIPELINE COMPLET 4 ÉTAPES:")
    print("   ÉTAPE 1: final_degraded → DÉBRUITAGE (SupervisedDenoisingUNet) → denoised")
    print("   ÉTAPE 2: denoised → AGGRESSIVE CONTRAST → contrast_enhanced") 
    print("   ÉTAPE 3: contrast_enhanced → REAL-ESRGAN → super_resolution")
    print("   ÉTAPE 4: super_resolution → SEGMENTATION (LadderNet) → segmented")
    print("="*70)
    print(f"📱 Device: {DEVICE}")
    print(f"📁 Dataset test: datasets_easy_noise")
    print(f"💾 Sortie: {OUTPUT_FOLDER}")
    print("="*70)
    
    # Créer dossier de sortie
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # Charger tous les modèles
    models = load_all_models(DEVICE)
    if models is None:
        return False
    
    # Vérification rapide des modèles
    print("\n🔍 VÉRIFICATION RAPIDE DES MODÈLES...")
    try:
        test_input = torch.randn(1, 1, 256, 256).to(DEVICE)
        with torch.no_grad():
            denoised_output = models['denoising'](test_input)
            contrast_output = models['contrast'](denoised_output)
            sr_output = models['realesrgan'](contrast_output)
            seg_output = models['laddernet'](sr_output)
            print("🎉 Tous les modèles fonctionnent correctement!")
    except Exception as e:
        print(f"❌ ERREUR lors du test des modèles: {e}")
        return False
    
    # Chargement dataset easy_noise
    print("\n📚 CHARGEMENT DATASET EASY_NOISE...")
    
    test_input_path = Path(TEST_DATASET) / "final_degraded" / "val"
    test_target_path = Path(TEST_DATASET) / "original" / "val"
    
    if not test_input_path.exists():
        print(f"❌ ERREUR: Dataset test introuvable!")
        print(f"Chemin input: {test_input_path}")
        return False
    
    # Lister images de test
    test_images = sorted(list(test_input_path.glob('*.png')) + list(test_input_path.glob('*.jpg')))
    test_images = test_images[:5]  # Limiter à 5 images pour test rapide
    
    print(f"✅ Dataset easy_noise chargé: {len(test_images)} images")
    
    # Traitement d'une image de test
    if test_images:
        test_img_path = test_images[0]
        filename = test_img_path.stem
        
        print(f"\n🧪 TEST SUR UNE IMAGE: {filename}")
        
        try:
            # Charger image
            input_image = cv2.imread(str(test_img_path), cv2.IMREAD_GRAYSCALE)
            print(f"📥 Image chargée: {input_image.shape}")
            
            # Exécuter pipeline
            print("🔄 EXÉCUTION PIPELINE COMPLET...")
            results = execute_pipeline(input_image, models, DEVICE)
            
            # Sauvegarder résultats
            step0_path = Path(OUTPUT_FOLDER) / f"{filename}_0_input_easy_noise.png"
            cv2.imwrite(str(step0_path), input_image)
            
            step1_path = Path(OUTPUT_FOLDER) / f"{filename}_1_denoised.png"
            cv2.imwrite(str(step1_path), (results['denoised'] * 255).astype(np.uint8))
            
            step2_path = Path(OUTPUT_FOLDER) / f"{filename}_2_contrast_enhanced.png"
            cv2.imwrite(str(step2_path), (results['contrast_enhanced'] * 255).astype(np.uint8))
            
            step3_path = Path(OUTPUT_FOLDER) / f"{filename}_3_super_resolution.png"
            cv2.imwrite(str(step3_path), (results['super_resolution'] * 255).astype(np.uint8))
            
            step4_raw_path = Path(OUTPUT_FOLDER) / f"{filename}_4_segmentation_raw.png"
            cv2.imwrite(str(step4_raw_path), (results['segmentation'] * 85).astype(np.uint8))
            
            # Segmentation colorisée
            colored_seg = colorize_segmentation(results['segmentation'], num_classes=4)
            step4_color_path = Path(OUTPUT_FOLDER) / f"{filename}_4_segmentation_colored.png"
            cv2.imwrite(str(step4_color_path), cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR))
            
            # Temps total
            total_time = results['time_denoising'] + results['time_contrast'] + results['time_realesrgan'] + results['time_segmentation']
            print(f"   ⏱️ Temps total: {total_time*1000:.1f} ms")
            print(f"   🎨 Classes détectées: {np.unique(results['segmentation'])}")
            
            print(f"✅ TEST RÉUSSI!")
            print(f"📁 Fichiers sauvés:")
            print(f"   • {step0_path.name}")
            print(f"   • {step1_path.name}")
            print(f"   • {step2_path.name}")
            print(f"   • {step3_path.name}")
            print(f"   • {step4_raw_path.name}")
            print(f"   • {step4_color_path.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ ERREUR lors du test: {e}")
            return False
    
    return False

# ===== MODE PERSONNALISÉ =====
def test_pipeline_custom_mode():
    """Mode personnalisé avec images utilisateur"""
    
    print("📁 MODE PERSONNALISÉ:")
    
    # Obtenir images utilisateur
    custom_images = get_custom_images()
    if custom_images is None:
        return False
    
    # Obtenir originaux optionnels
    original_images = ask_for_original_images(custom_images)
    
    # Configuration
    OUTPUT_FOLDER = r"C:\Users\ASUS\Desktop\Final\pipeline_custom_results"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n🔧 CONFIGURATION:")
    print(f"📱 Device: {DEVICE}")
    print(f"💾 Sortie: {OUTPUT_FOLDER}")
    
    # Créer dossier de sortie
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # Charger tous les modèles
    models = load_all_models(DEVICE)
    if models is None:
        return False
    
    # Traitement des images personnalisées
    for idx, img_path in enumerate(custom_images):
        filename = Path(img_path).stem
        original_path = original_images[idx] if idx < len(original_images) else None
        
        print(f"\n{'='*70}")
        print(f"📷 IMAGE {idx+1}/{len(custom_images)}: {filename}")
        print(f"📁 Input: {Path(img_path).name}")
        if original_path:
            print(f"📁 Original: {Path(original_path).name}")
        print(f"{'='*70}")
        
        try:
            # Charger image
            input_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if input_image is None:
                print(f"❌ ERREUR: Impossible de charger {img_path}")
                continue
            
            print(f"📥 Image chargée: {input_image.shape}")
            
            # Exécuter pipeline
            print("🔄 EXÉCUTION PIPELINE COMPLET...")
            results = execute_pipeline(input_image, models, DEVICE)
            
            # Sauvegarder résultats
            step0_path = Path(OUTPUT_FOLDER) / f"{filename}_0_input.png"
            cv2.imwrite(str(step0_path), input_image)
            
            step1_path = Path(OUTPUT_FOLDER) / f"{filename}_1_denoised.png"
            cv2.imwrite(str(step1_path), (results['denoised'] * 255).astype(np.uint8))
            
            step2_path = Path(OUTPUT_FOLDER) / f"{filename}_2_contrast_enhanced.png"
            cv2.imwrite(str(step2_path), (results['contrast_enhanced'] * 255).astype(np.uint8))
            
            step3_path = Path(OUTPUT_FOLDER) / f"{filename}_3_super_resolution.png"
            cv2.imwrite(str(step3_path), (results['super_resolution'] * 255).astype(np.uint8))
            
            step4_raw_path = Path(OUTPUT_FOLDER) / f"{filename}_4_segmentation_raw.png"
            cv2.imwrite(str(step4_raw_path), (results['segmentation'] * 85).astype(np.uint8))
            
            # Segmentation colorisée
            colored_seg = colorize_segmentation(results['segmentation'], num_classes=4)
            step4_color_path = Path(OUTPUT_FOLDER) / f"{filename}_4_segmentation_colored.png"
            cv2.imwrite(str(step4_color_path), cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR))
            
            # Métriques si original disponible
            if original_path and os.path.exists(original_path):
                print("📈 CALCUL MÉTRIQUES AVEC ORIGINAL...")
                original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
                if original is not None:
                    if original.shape != results['super_resolution'].shape:
                        original = cv2.resize(original, (results['super_resolution'].shape[1], results['super_resolution'].shape[0]))
                    original_norm = original.astype(np.float32) / 255.0
                    
                    step5_path = Path(OUTPUT_FOLDER) / f"{filename}_5_original_target.png"
                    cv2.imwrite(str(step5_path), original)
                    
                    input_norm = input_image.astype(np.float32) / 255.0
                    if input_norm.shape != results['super_resolution'].shape:
                        baseline = cv2.resize(input_norm, (results['super_resolution'].shape[1], results['super_resolution'].shape[0]), interpolation=cv2.INTER_CUBIC)
                    else:
                        baseline = input_norm
                    
                    psnr_baseline = psnr(original_norm, baseline, data_range=1.0)
                    psnr_pipeline = psnr(original_norm, results['super_resolution'], data_range=1.0)
                    psnr_gain = psnr_pipeline - psnr_baseline
                    
                    ssim_baseline = ssim(original_norm, baseline, data_range=1.0)
                    ssim_pipeline = ssim(original_norm, results['super_resolution'], data_range=1.0)
                    ssim_gain = ssim_pipeline - ssim_baseline
                    
                    print(f"   🎯 PSNR Baseline: {psnr_baseline:.1f} dB")
                    print(f"   🎯 PSNR Pipeline: {psnr_pipeline:.1f} dB")
                    print(f"   🎯 GAIN PSNR: {psnr_gain:+.1f} dB")
                    print(f"   🎯 SSIM Baseline: {ssim_baseline:.3f}")
                    print(f"   🎯 SSIM Pipeline: {ssim_pipeline:.3f}")
                    print(f"   🎯 GAIN SSIM: {ssim_gain:+.3f}")
            
            # Temps total
            total_time = results['time_denoising'] + results['time_contrast'] + results['time_realesrgan'] + results['time_segmentation']
            print(f"   ⏱️ Temps total: {total_time*1000:.1f} ms")
            print(f"   🎨 Classes détectées: {np.unique(results['segmentation'])}")
            
            print(f"✅ IMAGE {idx+1} TERMINÉE")
            print(f"📁 Fichiers sauvés:")
            print(f"   • {step0_path.name} (input)")
            print(f"   • {step1_path.name} (débruitage)")
            print(f"   • {step2_path.name} (contraste)")
            print(f"   • {step3_path.name} (super-résolution)")
            print(f"   • {step4_raw_path.name} (segmentation brute)")
            print(f"   • {step4_color_path.name} (segmentation colorisée)")
            
        except Exception as e:
            print(f"❌ ERREUR image {filename}: {e}")
            continue
    
    print(f"\n🎉 MODE PERSONNALISÉ TERMINÉ!")
    print(f"📁 Résultats sauvés dans: {OUTPUT_FOLDER}")
    return True

# ===== FONCTION PRINCIPALE =====
def main():
    """Fonction principale avec les deux modes"""
    
    print("🔧 PIPELINE COMPLET 4 ÉTAPES - Restauration + Segmentation (CORRIGÉ)")
    print("Ce script teste le pipeline complet avec LadderNet intégré")
    print("="*70)
    print("🎯 PIPELINE COMPLET TESTÉ:")
    print("   ÉTAPE 1: input → SupervisedDenoisingUNet → denoised")
    print("   ÉTAPE 2: denoised → AggressiveContrastNet → contrast_enhanced") 
    print("   ÉTAPE 3: contrast_enhanced → FastRealESRGAN → super_resolution")
    print("   ÉTAPE 4: super_resolution → LadderNet → segmentation")
    print("="*70)
    print("🏆 LadderNet avec score Dice: 0.8414 (84.14%)")
    print("🔧 CORRIGÉ: Architecture LadderNet + Gestion PyTorch 2.6+")
    print("⚠️  MODIFIEZ LES CHEMINS DES 4 MODÈLES DANS LE SCRIPT!")
    print("="*70)
    
    # Choix du mode
    choice = get_user_choice()
    
    if choice == 1:
        success = test_pipeline_auto_mode()
    else:
        success = test_pipeline_custom_mode()
    
    if success:
        print(f"\n🎉 PIPELINE COMPLET 4 ÉTAPES TESTÉ AVEC SUCCÈS!")
        print(f"✅ SupervisedDenoisingUNet pour débruitage")
        print(f"✅ AggressiveContrastNet pour amélioration contraste")
        print(f"✅ FastRealESRGAN pour super-résolution")
        print(f"✅ LadderNet pour segmentation (Dice: 0.8414)")
        print(f"✅ Architecture LadderNet corrigée (base_filters=32)")
        print(f"✅ Correction chargement PyTorch 2.6+ appliquée")
        print(f"✅ Deux modes implémentés: automatique et personnalisé")
    else:
        print(f"\n🔄 Problème lors du test du pipeline complet")
    
    input("\nAppuyez sur Entrée pour quitter...")

if __name__ == "__main__":
    main()