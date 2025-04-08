import cv2
import numpy as np
import onnxruntime as ort

# Cargar el modelo ONNX generado
session = ort.InferenceSession("yolov8n.onnx")

def preprocess(frame):
    """
    Convierte el frame de OpenCV (BGR) a un tensor adecuado para el modelo ONNX.
    - Redimensiona la imagen a 640x480 (ancho x alto) para que coincida con lo esperado por el modelo.
    - Convierte BGR a RGB.
    - Normaliza la imagen a valores entre 0 y 1.
    - Reordena la imagen a formato [1, 3, H, W].
    """
    # Redimensionar la imagen a 640x480
    frame_resized = cv2.resize(frame, (640, 480))
    
    # Convertir BGR a RGB
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Normalizar a [0, 1]
    rgb = rgb.astype(np.float32) / 255.0
    
    # Reordenar dimensiones: de (H, W, C) a (C, H, W)
    tensor = np.transpose(rgb, (2, 0, 1))
    
    # Agregar dimensión batch: (1, C, H, W)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor

def sigmoid(x):
    """Aplica la función sigmoide."""
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """Aplica la función softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def postprocess(outputs, conf_threshold=0.5):
    """
    Postprocesa la salida del modelo ONNX para extraer las detecciones.
    Se asume que la salida es un tensor de forma [1, N, 85]:
      - Los primeros 4 valores son [x1, y1, x2, y2].
      - El 5º valor es la confianza del objeto (raw score).
      - Los restantes (80 para COCO) son los scores crudos para cada clase.
    
    Retorna:
      - detecciones: Lista de tuplas con (bounding box, confidence, class_id),
        donde 'confidence' es la probabilidad de la clase detectada.
    """
    detecciones = []
    output = outputs[0]  # Se asume que la salida principal es el primer elemento
    
    # Iterar sobre cada detección (suponemos batch_size=1)
    for detection in output[0]:
        # Aplicar sigmoide a la confianza del objeto
        obj_conf = sigmoid(detection[4])
        if obj_conf < conf_threshold:
            continue

        # Aplicar softmax a los scores de las clases (valores crudos)
        class_scores = softmax(detection[5:])
        class_id = np.argmax(class_scores)
        # Usar la probabilidad de la clase detectada
        confidence = class_scores[class_id]
        if confidence < conf_threshold:
            continue

        # Extraer el bounding box (los primeros 4 valores)
        box = detection[:4].tolist()
        # Agregar la detección en el formato (bbox, confidence, class_id)
        detecciones.append((box, float(confidence), int(class_id)))
    
    return detecciones

def detectar_personas(frame, conf_threshold=0.5):
    """
    Ejecuta el modelo ONNX optimizado sobre el frame actual para detectar personas.
    
    Parámetros:
      - frame: Imagen (frame) capturada de la cámara (numpy array).
      - conf_threshold: Umbral de confianza para filtrar detecciones.
    
    Retorna:
      - detecciones: Lista de detecciones en el formato:
          ( [x1, y1, x2, y2], confidence, class_id )
      - outputs: Salida completa del modelo ONNX.
    """
    # Preprocesar el frame para convertirlo en un tensor
    input_tensor = preprocess(frame)
    
    # Ejecutar la inferencia usando ONNX Runtime
    outputs = session.run(None, {"input": input_tensor})
    
    # Postprocesar la salida para extraer las detecciones
    detecciones = postprocess(outputs, conf_threshold)
    
    return detecciones, outputs

if __name__ == "__main__":
    # Bloque de prueba: cargar una imagen de prueba y verificar las detecciones
    frame = cv2.imread('/home/jared/Desktop/IAleph/samples/imagen_prueba.jpg')
    if frame is None:
        print("Error al cargar la imagen.")
    else:
        detecciones, _ = detectar_personas(frame)
        print("Detecciones:", detecciones)