import torch
import torchvision
from torchvision import transforms as T
import cv2
import numpy as np

# Cargar el modelo preentrenado de Mask R-CNN con backbone ResNet-50
mask_rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
mask_rcnn_model.eval()  # Colocar el modelo en modo evaluación

def segmentar_productos(frame):
    """
    Ejecuta Mask R-CNN sobre el frame para detectar y segmentar productos.
    
    Parámetros:
      - frame: Imagen del frame (numpy array) proveniente del video.
    
    Retorna:
      - productos: Lista de diccionarios, cada uno con la etiqueta, score, bounding box y máscara del objeto detectado.
    """
    # Convertir la imagen de OpenCV (BGR) a un tensor y normalizar a valores entre 0 y 1
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(frame)
    
    # Ejecutar el modelo sin calcular gradientes (para mayor eficiencia)
    with torch.no_grad():
        predictions = mask_rcnn_model([img_tensor])
    
    productos = []
    pred = predictions[0]  # Obtener las predicciones del primer (y único) batch
    umbral = 0.8           # Umbral de confianza para filtrar detecciones
    
    # Iterar sobre cada detección
    for i, score in enumerate(pred["scores"]):
        if score > umbral:
            productos.append({
                "label": int(pred["labels"][i]),    # Clase del objeto detectado
                "score": float(score),               # Puntaje de confianza
                "bbox": pred["boxes"][i].tolist(),   # Coordenadas [x1, y1, x2, y2]
                # Procesar la máscara: convertirla a imagen (multiplicada por 255 y tipo byte)
                "mask": pred["masks"][i].mul(255).byte().cpu().numpy()
            })
    return productos

# Bloque de prueba (se ejecuta solo si se corre este archivo directamente)
if __name__ == "__main__":
    frame = cv2.imread('/home/jared/Desktop/IAleph/samples/imagen_prueba.jpg')
    productos = segmentar_productos(frame)
    print("Productos detectados:", productos)
