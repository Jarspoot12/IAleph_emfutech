import cv2
import time

# Importar las funciones de cada módulo
from detectors.yolo2 import detectar_personas
from tracking.tracker import actualizar_tracker
from classification.age_gender import clasificar_edad_genero
from classification.emotion import reconocer_emocion
from segmentation.mask_rcnn import segmentar_productos
# -------------------------------
# Configuración de la Captura de Video
# -------------------------------
# Inicializar la cámara (ID 0 es la cámara predeterminada)
cap = cv2.VideoCapture(0)
# Establecer la resolución a 640x480 para optimizar el rendimiento
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break

    # -------------------------------
    # Paso A: Detección de Personas con YOLOv8
    # -------------------------------
    # Ejecutar el detector y obtener la lista de bounding boxes para personas
    detecciones, _ = detectar_personas(frame)

    # -------------------------------
    # Paso B: Actualización del Tracker (Deep SORT)
    # -------------------------------
    # Se asigna un ID único a cada persona detectada para mantener el seguimiento
    personas = actualizar_tracker(detecciones, frame)
    resultados_finales = []  # Lista para almacenar la información combinada por persona

    # Procesar cada persona trackeada
    for persona in personas:
        # Obtener las coordenadas del bounding box y convertirlas a enteros
        x1, y1, x2, y2 = map(int, persona['bbox'])
        # Extraer la región de interés (ROI) correspondiente a la persona
        roi = frame[y1:y2, x1:x2]
        
        # -------------------------------
        # Paso C: Clasificación de Edad y Género (DeepFace)
        # -------------------------------
        result_deepface = None
        try:
            result_deepface = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("Error en conversión de color:", e)
        edad, genero = clasificar_edad_genero(roi)
        
        # -------------------------------
        # Paso D: Reconocimiento de Emociones (FER)
        # -------------------------------
        resultado_emocion = reconocer_emocion(roi)
        emocion = resultado_emocion if resultado_emocion else "Sin detección"
        
        # -------------------------------
        # Paso E: Segmentación de Productos (Mask R-CNN)
        # -------------------------------
        productos = segmentar_productos(roi)
        
        # Crear un registro con toda la información obtenida para la persona actual
        registro = {
            "id": persona['id'],           # ID asignado por el tracker
            "bbox": persona['bbox'],         # Bounding box [x1, y1, x2, y2]
            "edad": edad,                  # Edad predicha
            "genero": genero,              # Género predicho
            "emocion": emocion,            # Emoción detectada
            "productos": productos,        # Lista de productos segmentados en la ROI
            # Agregar un timestamp en formato UTC (ISO 8601)
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        resultados_finales.append(registro)

    # -------------------------------
    # Paso F: Visualización de Resultados en el Frame
    # -------------------------------
    # Dibujar los bounding boxes y etiquetas sobre el frame original
    for r in resultados_finales:
        x1, y1, x2, y2 = map(int, r["bbox"])
        # Dibujar un rectángulo verde alrededor de la persona
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Crear una etiqueta que incluya ID, género, edad y emoción
        etiqueta = f"ID: {r['id']} {r['genero']}, {r['edad']}, {r['emocion']}"
        # Escribir la etiqueta justo arriba del rectángulo
        cv2.putText(frame, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el frame anotado en una ventana
    cv2.imshow("Predicciones en Tiempo Real", frame)
    
    # Esperar 1 ms para detectar la pulsación de 'q' y salir del bucle si se presiona
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Liberar Recursos y Cerrar Ventanas
# -------------------------------
cap.release()            # Liberar la cámara
cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV
