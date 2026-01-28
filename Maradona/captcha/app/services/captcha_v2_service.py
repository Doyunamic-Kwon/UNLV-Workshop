import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import json

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

# Paths are relative to project root usually, but let's make them robust
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "efficientnet_v2_best.pth"
METADATA_PATH = BASE_DIR / "models" / "model_metadata_v2.json"
CLASSES_PATH = BASE_DIR / "models" / "v2_classes.txt"

class CaptchaV2Service:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CaptchaV2Service, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized: return
        
        self.model = None
        self.classes = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.load_model()
        self._initialized = True

    def load_model(self):
        try:
            # Load Classes
            if METADATA_PATH.exists():
                with open(METADATA_PATH, "r") as f:
                    meta = json.load(f)
                    self.classes = meta.get("classes", [])
                    print(f"✅ [V2 Service] Loaded {len(self.classes)} classes from metadata")
            elif CLASSES_PATH.exists():
                with open(CLASSES_PATH, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
                print(f"✅ [V2 Service] Loaded {len(self.classes)} classes from text file")
            
            if not self.classes:
                print("⚠️ [V2 Service] No classes found. Inference disabled.")
                return

            # Load Model
            self.model = models.efficientnet_b0(weights=None)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, len(self.classes))
            
            if MODEL_PATH.exists():
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                self.model.load_state_dict(state_dict)
                self.model.to(DEVICE)
                self.model.eval()
                print(f"✅ [V2 Service] Model loaded from {MODEL_PATH}")
            else:
                print(f"⚠️ [V2 Service] Model file not found at {MODEL_PATH}")
                self.model = None
                
        except Exception as e:
            print(f"❌ [V2 Service] Failed to load model: {e}")
            self.model = None

    def predict(self, img_path: Path):
        """
        Predicts the class of the image at img_path.
        Returns the class name (str) or None if prediction fails.
        """
        if not self.model: 
            # Try reloading if missing (e.g. usage before startup)
            self.load_model()
            if not self.model: return None
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                _, pred_idx = torch.max(outputs, 1)
                
            return self.classes[pred_idx.item()]
        except Exception as e:
            print(f"❌ [V2 Service] Prediction error for {img_path}: {e}")
            return None

# Singleton export
captcha_v2_service = CaptchaV2Service()
