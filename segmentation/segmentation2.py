from ultralytics import YOLO

# Carga el modelo exportado (asegúrate de que la ruta sea correcta)
detector = YOLO("segmentation/best.pt")
# Las etiquetas serán las definidas durante el entrenamiento (por ejemplo, ['bottle','handbag','cell_phone','cup','laptop'])
CLASS_NAMES = detector.names

def segmentar_productos(frame, conf=0.7):
    """
    Ejecuta el modelo entrenado (YOLOv8) sobre la imagen frame y retorna una lista 
    con las detecciones (productos) que tengan una confianza mayor o igual que conf.
    
    Parámetros:
      - frame: imagen (numpy array) que corresponde a la ROI (zona de interés).
      - conf: umbral de confianza (por defecto 0.5).
    
    Retorna:
      - productos: lista de diccionarios, cada uno con la etiqueta y la confianza.
    """
    # Ejecuta el detector sobre la imagen
    resultados = detector(frame, verbose=False)[0]
    productos = [
        {
            "label": CLASS_NAMES[int(cls)]
            # "confidence": float(score)
        }
        for cls, score in zip(resultados.boxes.cls, resultados.boxes.conf)
        if float(score) >= conf
    ]
    return productos