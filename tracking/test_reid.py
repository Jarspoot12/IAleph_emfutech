# test_reid.py
import os
import sys
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forzar CPU


# Agregar el path de fast-reid al sistema
current_dir = os.path.dirname(os.path.abspath(__file__))
fastreid_path = os.path.join(current_dir, "fast-reid")
sys.path.append(fastreid_path)

# Importar clase del módulo personalizado
from reid_model import FastReIDModel
# Inicializar modelo
config_path = os.path.join(current_dir, "fast-reid", "configs", "Market1501", "AGW_R50.yml")
# Construir la ruta absoluta a los pesos; asegúrate que el archivo se encuentre en esa ruta
weights_path = os.path.join(current_dir, "market_agw_R50.pth")

model = FastReIDModel(config_path, weights_path)
# Cargar imagen de prueba
img1 = cv2.imread("/home/jared/Desktop/IAleph/samples/persona_enojada.jpg")
img2 = cv2.imread("/home/jared/Desktop/IAleph/samples/us.jpg")
# Extraer embedding
f1 = model.extract_features(img1)
f2 = model.extract_features(img2)

from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([f1], [f2])[0][0]
print(f"Similaridad entre imágenes: {sim}")
