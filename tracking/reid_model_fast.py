# reid_model.py (fragmento modificado)
from torchvision import transforms as T
from fastreid.config import get_cfg
from fastreid.engine.defaults import DefaultPredictor
from PIL import Image
import numpy as np
import torch

class FastReIDModel:
    def __init__(self, config_file, model_weights):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_file)
        self.cfg.MODEL.WEIGHTS = model_weights
        # Forzar CPU, por ejemplo
        self.cfg.MODEL.DEVICE = "cpu"
        
        predictor = DefaultPredictor(self.cfg)
        self.model = predictor.model
        self.model.eval()
        self.model.to(self.cfg.MODEL.DEVICE)

        # Si no están definidos, se usan los valores de ImageNet:
        pixel_mean = getattr(self.cfg.INPUT, "PIXEL_MEAN", [0.485, 0.456, 0.406])
        pixel_std  = getattr(self.cfg.INPUT, "PIXEL_STD", [0.229, 0.224, 0.225])
        
        self.transform = T.Compose([
            T.Resize((128, 256)),
            T.ToTensor(),
            T.Normalize(mean=pixel_mean, std=pixel_std)
        ])

    def extract_features(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img[..., ::-1])  # Convertir BGR a RGB
        img = self.transform(img)
        img = img.unsqueeze(0).to(self.cfg.MODEL.DEVICE)
        with torch.no_grad():
            features = self.model(img)
        return features.cpu().numpy()[0]
