import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8 (versión ligera para mayor velocidad)
yolo_model = YOLO('yolov8n.pt')

def detectar_personas(frame):
    """
    Ejecuta YOLOv8 sobre el frame actual para detectar personas.
    
    Parámetros:
      - frame: Imagen (frame) capturada de la cámara (numpy array).
    
    Retorna:
      - detecciones: Lista de detecciones en el formato:
          ( [x1, y1, x2, y2], confidence, class_id )
      - results: Objeto completo con la información de las predicciones.
    """
    results = yolo_model(frame)
    detecciones = []  # Lista para almacenar las detecciones de la clase 'persona'
    
    # Recorrer cada resultado (normalmente un resultado por frame)
    for result in results:
        # Recorrer cada bounding box, confianza y clase detectada
        for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                    result.boxes.conf.cpu().numpy(),
                                    result.boxes.cls.cpu().numpy()):
            # Suponiendo que la clase 0 corresponde a "persona"
            if int(cls) == 0 and float(conf) > 0.9:
                # Convertir la caja a lista y empaquetar con la confianza y la clase
                detecciones.append((box.tolist(), float(conf), int(cls)))
    
    return detecciones, results

if __name__ == "__main__":
    # Bloque de prueba: cargar una imagen y verificar las detecciones
    frame = cv2.imread('/home/jared/Desktop/IAleph/samples/imagen_prueba.jpg')
    detecciones, _ = detectar_personas(frame)
    print("Detecciones:", detecciones)
