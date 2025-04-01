import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8 (versión ligera para mayor velocidad en tiempo real)
yolo_model = YOLO('yolov8n.pt')

def detectar_personas(frame):
    """
    Ejecuta YOLOv8 sobre el frame actual para detectar personas.
    
    Parámetros:
      - frame: Imagen (frame) capturada de la cámara (formato numpy array).
    
    Retorna:
      - detecciones: Lista de bounding boxes [x1, y1, x2, y2] de personas detectadas.
      - results: Objeto completo con la información de las predicciones.
    """
    # Ejecutar el modelo sobre el frame
    results = yolo_model(frame)
    detecciones = []  # Lista para almacenar los bounding boxes de la clase 'persona'
    
    # Recorrer cada resultado (normalmente un resultado por frame)
    for result in results:
        # Recorrer cada bounding box y su clase correspondiente
        for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            # Suponiendo que la clase 0 corresponde a "persona"
            if int(cls) == 0:
                detecciones.append(box)  # Agregar el bounding box [x1, y1, x2, y2]
    
    return detecciones, results

# Bloque de prueba (se ejecuta solo si se corre este archivo directamente)
if __name__ == "__main__":
    # Cargar una imagen de prueba para verificar el detector
    frame = cv2.imread('/home/jared/Desktop/IAleph/samples/imagen_prueba.jpg')
    detecciones, _ = detectar_personas(frame)
    print("Detecciones:", detecciones)
