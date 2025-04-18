# reid_model.py
import os
import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as T

# 1) Añadir TransReID al PYTHONPATH
repo_root = os.path.dirname(os.path.abspath(__file__))  # tracking/
transreid_root = os.path.join(repo_root, "TransReID")
if transreid_root not in sys.path:
    sys.path.insert(0, transreid_root)

# 2) Importar según test.py de TransReID
from config import cfg                         # TransReID/config.py
from datasets import make_dataloader          # TransReID/datasets.py
from model import make_model                   # TransReID/model/__init__.py
from utils.checkpoint import load_checkpoint as load_pretrained_weights  # TransReID/utils/checkpoint.py

class TransReIDModel:
    def __init__(self, config_file: str, model_weights: str, device: str = "cpu"):
        # --- Cargar y configurar ---
        cfg.merge_from_file(config_file)
        cfg.MODEL.DEVICE_ID = device
        cfg.freeze()

        # --- Construir dataloader para obtener num_classes, camera_num, view_num ---
        # Solo necesario para construir el modelo
        _, _, _, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

        # --- Montar el modelo y cargar pesos ---
        self.model = make_model(cfg,
                                num_class=num_classes,
                                camera_num=camera_num,
                                view_num=view_num)
        self.model.to(device)
        load_pretrained_weights(self.model, model_weights, device=device)
        self.model.eval()

        # --- Preprocesamiento idéntico a entrenamiento ---
        h = cfg.INPUT.HEIGHT  # p.ej. 256
        w = cfg.INPUT.WIDTH   # p.ej. 128
        mean = cfg.INPUT.PIXEL_MEAN
        std  = cfg.INPUT.PIXEL_STD
        self.transform = T.Compose([
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.device = device

    def extract_features(self, img: np.ndarray) -> np.ndarray:
        # img: BGR numpy array
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(x)  # salida: [1, C]
        feat = F.normalize(feat.squeeze(0), dim=0)
        return feat.cpu().numpy()
